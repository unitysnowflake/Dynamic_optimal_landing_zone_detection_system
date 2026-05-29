from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
import matplotlib.cm as cm
from models.matching import Matching
from models.utils import make_matching_plot_fast, frame2tensor

torch.set_grad_enabled(False)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
MAPS_DIR = SCRIPT_DIR / "DTMs"
CAM_IMAGES_DIR = SCRIPT_DIR / "CAM_images"
RESULTS_DIR = SCRIPT_DIR / "Results"


def extract_phase_token(*values):
    for value in values:
        upper_value = str(value).upper()
        if "P1" in upper_value:
            return "P1"
        if "P2" in upper_value:
            return "P2"
    raise ValueError("Could not determine phase from the provided file names. Expected P1 or P2.")


def resolve_half_range(phase_token):
    if phase_token == "P1":
        return 1790
    if phase_token == "P2":
        return 725
    raise ValueError(f"Unsupported phase token: {phase_token}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue: Locate camera image (image1) on fixed map (image2) with centered coordinates [-1540, 1540]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--map', type=str, required=True,
                        help='Map image file name relative to CAM_unit/SuperGluePretrainedNetwork-master/DTMs, e.g. DTM_P1.png or DTM_P2.png')
    parser.add_argument('--camera', type=str, required=True,
                        help='Camera image file name relative to CAM_unit/SuperGluePretrainedNetwork-master/CAM_images, e.g. CAM_P1.png or CAM_P2.png')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory name relative to CAM_unit/SuperGluePretrainedNetwork-master/Results where visual outputs will be saved')
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='indoor',
                        help='SuperGlue weights')
    parser.add_argument('--max_keypoints', type=int, default=-1,
                        help='Maximum number of keypoints (-1 keeps all)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.005)
    parser.add_argument('--nms_radius', type=int, default=4)
    parser.add_argument('--sinkhorn_iterations', type=int, default=20)
    parser.add_argument('--match_threshold', type=float, default=0.2)
    parser.add_argument('--force_cpu', action='store_true')

    opt = parser.parse_args()

    map_path = MAPS_DIR / opt.map
    camera_path = CAM_IMAGES_DIR / opt.camera
    output_dir = RESULTS_DIR / opt.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_token = extract_phase_token(opt.map, opt.camera, opt.output_dir)

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print(f'Running on device: {device}')

    # Load config
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)

    # Load images
    def load_image(path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img

    map_img = load_image(map_path)      # image2 - fixed map
    cam_img = load_image(camera_path)   # image1 - camera / query

    resize = [640, 480]
    map_img = cv2.resize(map_img, (resize[0], resize[1]))
    cam_img = cv2.resize(cam_img, (resize[0], resize[1]))

    h_map, w_map = map_img.shape[:2]
    h_cam, w_cam = cam_img.shape[:2]

    print(f'Map size: {w_map}x{h_map}')
    print(f'Camera image size: {w_cam}x{h_cam}')

    # Преобразовать в тензор
    tensor_map = frame2tensor(map_img, device)   # image0 = карта
    tensor_cam = frame2tensor(cam_img, device)   # image1 = камера

    # SuperGlue matching
    pred = matching({'image0': tensor_map, 'image1': tensor_cam})

    kpts0 = pred['keypoints0'][0].cpu().numpy()   # keypoints on map
    kpts1 = pred['keypoints1'][0].cpu().numpy()   # keypoints on camera image
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts_map = kpts0[valid]                    # points in map coordinate (pixel)
    mkpts_cam = kpts1[matches[valid]]           # corresponding points in camera image
    matched_confidence = confidence[valid]

    print(f'Found {len(mkpts_map)} good matches')

    if len(mkpts_map) < 8:
        print("Слишком мало совпадений!")

    # Вычисление гомографии
    H, mask = cv2.findHomography(mkpts_cam, mkpts_map, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError("Homography estimation failed. Not enough inliers.")

    if mask is None:
        raise ValueError("Homography mask was not returned.")

    inlier_mask = mask.ravel().astype(bool)
    total_matches = int(len(mkpts_map))
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = (inlier_count / total_matches) if total_matches > 0 else 0.0

    projected_map = cv2.perspectiveTransform(
        mkpts_cam.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)
    reprojection_errors = np.linalg.norm(projected_map - mkpts_map, axis=1)
    reprojection_rmse_px = (
        float(np.sqrt(np.mean(np.square(reprojection_errors[inlier_mask]))))
        if inlier_count > 0 else float('inf')
    )

    mean_match_confidence = float(np.mean(matched_confidence)) if total_matches > 0 else 0.0
    mean_inlier_confidence = (
        float(np.mean(matched_confidence[inlier_mask]))
        if inlier_count > 0 else 0.0
    )
    cam_fusion_confidence = (
        min(1.0, inlier_count / 50.0)
        * inlier_ratio
        * mean_inlier_confidence
        / (1.0 + reprojection_rmse_px / 5.0)
    )

    print("\nГомография H (camera → map):")
    np.set_printoptions(precision=4, suppress=True)
    print(H)
    print(f"Homography inliers: {inlier_count}/{total_matches} ({inlier_ratio:.3f})")
    print(f"Homography reprojection RMSE: {reprojection_rmse_px:.3f} px")

    # Система координат карты
    half_range = resolve_half_range(phase_token)
    scale_x = (2 * half_range) / w_map
    scale_y = (2 * half_range) / h_map

    def pixel_to_map_coords(px, py):
        x_map = (px - w_map / 2) * scale_x
        y_map = - (py - h_map / 2) * scale_y
        return x_map, y_map

    cam_center_cam = np.array([[w_cam / 2, h_cam / 2]], dtype=np.float32).reshape(-1, 1, 2)

    cam_center_map = cv2.perspectiveTransform(cam_center_cam, H)[0][0]

    cam_center_x, cam_center_y = pixel_to_map_coords(cam_center_map[0], cam_center_map[1])

    print("\nРезультат CAM модуля")
    print(f"Центр камеры в СК карты:")
    print(f"   X = {cam_center_x:.2f}   Y = {cam_center_y:.2f}")
    print(f"   (Центр в (0, 0))")

    corners_cam = np.float32([
        [0, 0],
        [w_cam - 1, 0],
        [w_cam - 1, h_cam - 1],
        [0, h_cam - 1]
    ]).reshape(-1, 1, 2)

    corners_map = cv2.perspectiveTransform(corners_cam, H).reshape(-1, 2)

    results_dir = PROJECT_DIR / "CAM_LiDAR_Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"cam_result_coordinates_{phase_token}.txt"

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("sensor = cam\n")
        f.write(f"x = {cam_center_x:.2f}\n")
        f.write(f"y = {cam_center_y:.2f}\n")
        f.write(f"map_keypoints = {len(kpts0)}\n")
        f.write(f"camera_keypoints = {len(kpts1)}\n")
        f.write(f"good_matches = {total_matches}\n")
        f.write(f"homography_inliers = {inlier_count}\n")
        f.write(f"homography_inlier_ratio = {inlier_ratio:.10f}\n")
        f.write(f"mean_match_confidence = {mean_match_confidence:.10f}\n")
        f.write(f"mean_inlier_confidence = {mean_inlier_confidence:.10f}\n")
        f.write(f"homography_reprojection_rmse_px = {reprojection_rmse_px:.10f}\n")
        f.write(f"fusion_confidence = {cam_fusion_confidence:.10f}\n")

    print("\nУглы снимка в СК карты:")
    labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
    for label, (px, py) in zip(labels, corners_map):
        mx, my = pixel_to_map_coords(px, py)
        print(f"  {label:12}:  X={mx:8.2f}   Y={my:8.2f}")

    color = cm.jet(confidence[valid])
    text = ['SuperGlue', 
            f'Keypoints: {len(kpts0)} (map) : {len(kpts1)} (camera)',
            f'Matches: {len(mkpts_map)}']
    small_text = [f'Map center offset: X={cam_center_x:.1f}  Y={cam_center_y:.1f}']

    out = make_matching_plot_fast(
        map_img, cam_img, kpts0, kpts1, mkpts_map, mkpts_cam,
        color, text, path=None, show_keypoints=True, small_text=small_text)

    match_path = output_dir / f'matches_map_vs_camera_{phase_token}.png'
    cv2.imwrite(str(match_path), out)
    print(f'\nMatch visualization saved: {match_path}')

    warped = cv2.warpPerspective(cam_img, H, (w_map, h_map))
    overlay = cv2.addWeighted(map_img, 0.6, warped, 0.4, 0)
    cv2.imwrite(str(output_dir / f'camera_on_map_overlay_{phase_token}.png'), overlay)

    print("\nDone.")
