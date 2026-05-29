import csv
import math
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

W = 512
H = 512
FOV_DEG = 50.0

VOXEL_RATIO = 0.01
MAX_POINTS_RANSAC = 120000
MAX_POINTS_ICP = 120000
RANSAC_MAX_ITERS = 100000
ICP_MAX_ITERS = 80
RANSAC_VOXEL_MULTIPLIERS = (4.0, 2.0, 1.0, 0.5)


# Чтение DTM
def read_xyz_csv(path):
    pts = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"No headers found in {path}")

        headers = [h.strip() for h in reader.fieldnames]
        if not all(k in headers for k in ("X", "Y", "Z")):
            raise RuntimeError(f"CSV must contain X,Y,Z columns. Found: {headers}")

        for row in reader:
            try:
                x = float(row["X"])
                y = float(row["Y"])
                z = float(row["Z"])
            except Exception:
                continue

            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                pts.append([x, y, z])

    arr = np.asarray(pts, dtype=np.float64)
    if len(arr) == 0:
        raise RuntimeError(f"No usable points loaded from {path}")
    return arr


def read_lidar_scan_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"No headers found in {path}")

        headers = [h.strip() for h in reader.fieldnames]
        header_map = {h.strip().lower(): h for h in reader.fieldnames}

        def has(*names):
            return all(name in header_map for name in names)

        if has("cam_x", "cam_y", "cam_z"):
            for row in reader:
                try:
                    cam_x = float(row[header_map["cam_x"]])
                    cam_y = float(row[header_map["cam_y"]])
                    cam_z = float(row[header_map["cam_z"]])
                except Exception:
                    continue

                if np.isfinite(cam_x) and np.isfinite(cam_y) and np.isfinite(cam_z):
                    rows.append([cam_x, cam_y, cam_z])

            arr = np.asarray(rows, dtype=np.float64)
            if len(arr) == 0:
                raise RuntimeError(f"No usable camera-space LiDAR points loaded from {path}")
            return {
                "format": "camera_points",
                "camera_points": arr,
                "headers": headers,
            }

        if has("x", "y", "z"):
            for row in reader:
                try:
                    pixel_x = float(row[header_map["x"]])
                    pixel_y = float(row[header_map["y"]])
                    depth = float(row[header_map["z"]])
                except Exception:
                    continue

                if (
                    np.isfinite(pixel_x)
                    and np.isfinite(pixel_y)
                    and np.isfinite(depth)
                    and depth > 0.0
                ):
                    rows.append([pixel_x, pixel_y, depth])

            arr = np.asarray(rows, dtype=np.float64)
            if len(arr) == 0:
                raise RuntimeError(f"No valid legacy LiDAR scan rows found in {path}")
            return {
                "format": "pixel_depth",
                "pixel_depth": arr,
                "headers": headers,
            }

    raise RuntimeError(
        f"Unsupported LiDAR CSV format in {path}. "
        f"Expected CAM_X,CAM_Y,CAM_Z or legacy X,Y,Z columns."
    )


# LiDAR реконструкция
def pixels_and_depth_to_camera_points(pixel_x, pixel_y, depth, w, h, fov_deg):
    u = ((pixel_x - 0.5) / w) * 2.0 - 1.0
    v = 1.0 - ((pixel_y - 0.5) / h) * 2.0
    tan_half = math.tan(math.radians(fov_deg) * 0.5)

    pts = np.column_stack([
        u * tan_half * depth,
        v * tan_half * depth,
        -depth,
    ])

    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        raise RuntimeError("No valid LiDAR camera-space points reconstructed.")
    return pts, valid


def lidar_scan_to_camera_points(lidar_xyz, w, h, fov_deg):
    return pixels_and_depth_to_camera_points(
        lidar_xyz[:, 0],
        lidar_xyz[:, 1],
        lidar_xyz[:, 2],
        w,
        h,
        fov_deg,
    )


# Open3D вспом
def sample_points(points, max_points, seed):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return points

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def estimate_normals(pcd, radius):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    return pcd


def compute_fpfh(pcd, radius):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100),
    )


def preprocess_cloud(points, voxel):
    pcd = make_pcd(points)
    pcd = pcd.voxel_down_sample(voxel)
    if len(pcd.points) < 30:
        raise RuntimeError("Point cloud too small after downsampling.")

    estimate_normals(pcd, radius=voxel * 2.5)
    fpfh = compute_fpfh(pcd, radius=voxel * 5.0)
    return pcd, fpfh


def transform_point(point_xyz, transform):
    point_h = np.append(np.asarray(point_xyz, dtype=np.float64), 1.0)
    return (transform @ point_h)[:3]


def score_registration(result, voxel):
    if result is None or result.fitness <= 0.0:
        return -np.inf
    return result.fitness / (1.0 + result.inlier_rmse / max(voxel, 1e-9))


def run_multiscale_ransac(src_points, tgt_points, base_voxel):
    best = None
    best_voxel = None
    attempts = []

    for multiplier in RANSAC_VOXEL_MULTIPLIERS:
        voxel = max(base_voxel * multiplier, 1e-6)
        max_corr = voxel * 2.5

        src_down, src_fpfh = preprocess_cloud(src_points, voxel)
        tgt_down, tgt_fpfh = preprocess_cloud(tgt_points, voxel)

        print(f"RANSAC attempt: voxel={voxel:.6f}, src_down={len(src_down.points)}, tgt_down={len(tgt_down.points)}")

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down,
            tgt_down,
            src_fpfh,
            tgt_fpfh,
            False,
            max_corr,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(RANSAC_MAX_ITERS, 0.999),
        )

        attempts.append((voxel, result, len(src_down.points), len(tgt_down.points)))
        if best is None or score_registration(result, voxel) > score_registration(best, best_voxel):
            best = result
            best_voxel = voxel

    if best is None or best.fitness <= 0.0:
        raise RuntimeError("Ошибка RANSAC. Не найдено начального приближения")

    return best, best_voxel, attempts


def resolve_input_path(folder, user_value):
    candidate = Path(user_value)
    if candidate.is_absolute():
        return candidate
    return folder / user_value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match LiDAR scan to DTM and save the phase-specific result file."
    )
    parser.add_argument(
        "--lidar_csv",
        type=str,
        required=True,
        help="LiDAR CSV file name relative to LiDAR_unit/LiDAR_scans, e.g. LiDAR_P1.csv or LiDAR_P2.csv.",
    )
    parser.add_argument(
        "--dtm_csv",
        type=str,
        required=True,
        help="DTM CSV file name relative to LiDAR_unit/DTMs, e.g. DTM_P1.csv or DTM_P2.csv.",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="Result file name relative to CAM_LiDAR_Results, e.g. lidar_result_coordinates_P1.txt or lidar_result_coordinates_P2.txt.",
    )
    return parser.parse_args()


def main():
    opt = parse_args()
    lidar_csv = resolve_input_path(SCRIPT_DIR / "LiDAR_scans", opt.lidar_csv)
    dtm_csv = resolve_input_path(SCRIPT_DIR / "DTMs", opt.dtm_csv)

    lidar_data = read_lidar_scan_csv(lidar_csv)
    dtm_xyz = read_xyz_csv(dtm_csv)

    if lidar_data["format"] == "camera_points":
        src_raw = lidar_data["camera_points"]
        lidar_format_note = "camera-space points from Blender"
    else:
        src_raw, _ = lidar_scan_to_camera_points(lidar_data["pixel_depth"], W, H, FOV_DEG)
        lidar_format_note = (
            "legacy pixel-depth scan reconstructed with hardcoded FOV_DEG; "
            "this mode is less accurate"
        )

    tgt_raw = np.asarray(dtm_xyz, dtype=np.float64)

    src_diag = np.linalg.norm(src_raw.max(axis=0) - src_raw.min(axis=0))
    tgt_diag = np.linalg.norm(tgt_raw.max(axis=0) - tgt_raw.min(axis=0))
    scene_scale = max(src_diag, tgt_diag)
    if scene_scale <= 1e-12:
        raise RuntimeError("Degenerate target bounds.")
    voxel = max(scene_scale * VOXEL_RATIO, 1e-6)
    max_corr = voxel * 2.5

    print(f"Source points: {len(src_raw)}")
    print(f"Target points: {len(tgt_raw)}")
    print(f"LiDAR input mode: {lidar_format_note}")
    print(f"Base voxel size: {voxel:.6f}")

    src_ransac = sample_points(src_raw, MAX_POINTS_RANSAC, seed=1)
    tgt_ransac = sample_points(tgt_raw, MAX_POINTS_RANSAC, seed=2)

    print("Running RANSAC global registration...")
    result_ransac, ransac_voxel, attempts = run_multiscale_ransac(src_ransac, tgt_ransac, voxel)
    src_down, _ = preprocess_cloud(src_ransac, ransac_voxel)
    tgt_down, _ = preprocess_cloud(tgt_ransac, ransac_voxel)

    print(f"RANSAC fitness: {result_ransac.fitness:.6f}")
    print(f"RANSAC rmse:     {result_ransac.inlier_rmse:.6f}")
    print(f"RANSAC voxel:    {ransac_voxel:.6f}")

    src_icp_points = sample_points(src_raw, MAX_POINTS_ICP, seed=3)
    tgt_icp_points = sample_points(tgt_raw, MAX_POINTS_ICP, seed=4)

    src_icp = make_pcd(src_icp_points)
    tgt_icp = make_pcd(tgt_icp_points)
    estimate_normals(src_icp, radius=voxel * 2.5)
    estimate_normals(tgt_icp, radius=voxel * 2.5)

    print("Running ICP refinement...")
    result_icp = o3d.pipelines.registration.registration_icp(
        src_icp,
        tgt_icp,
        max_corr,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERS),
    )

    if result_icp.fitness <= 0.0:
        raise RuntimeError("ICP refinement failed to converge to a usable alignment.")

    transform_camera_to_world = result_icp.transformation
    camera_world_pos = transform_camera_to_world[:3, 3].copy()

    print(f"ICP fitness: {result_icp.fitness:.6f}")
    print(f"ICP rmse:     {result_icp.inlier_rmse:.6f}")

    ransac_rmse_normalized = result_ransac.inlier_rmse / ransac_voxel
    icp_rmse_normalized = result_icp.inlier_rmse / voxel
    lidar_fusion_confidence = (
        result_ransac.fitness
        * result_icp.fitness
        / ((1.0 + ransac_rmse_normalized) * (1.0 + icp_rmse_normalized))
    )

    print("\nЛучшая трансформация (camera -> Blender world):")
    print(transform_camera_to_world)

    print("\nКоординаты LiDAR скана в мировой СК:")
    print(camera_world_pos)

    results_dir = PROJECT_DIR / "CAM_LiDAR_Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    result_file = results_dir / opt.result_file

    x, y, z = camera_world_pos

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("sensor = lidar\n")
        f.write(f"x = {x:.10f}\n")
        f.write(f"y = {y:.10f}\n")
        f.write(f"z = {z:.10f}\n")
        f.write(f"lidar_input_mode = {lidar_data['format']}\n")
        f.write(f"source_points = {len(src_raw)}\n")
        f.write(f"target_points = {len(tgt_raw)}\n")
        f.write(f"voxel_size = {voxel:.10f}\n")
        f.write(f"ransac_voxel_size = {ransac_voxel:.10f}\n")
        f.write(f"ransac_downsampled_points = {len(src_down.points)}\n")
        f.write(f"ransac_fitness = {result_ransac.fitness:.10f}\n")
        f.write(f"ransac_rmse = {result_ransac.inlier_rmse:.10f}\n")
        f.write(f"ransac_rmse_normalized = {ransac_rmse_normalized:.10f}\n")
        f.write(f"icp_fitness = {result_icp.fitness:.10f}\n")
        f.write(f"icp_rmse = {result_icp.inlier_rmse:.10f}\n")
        f.write(f"icp_rmse_normalized = {icp_rmse_normalized:.10f}\n")
        f.write(f"fusion_confidence = {lidar_fusion_confidence:.10f}\n")
        for row_idx, row in enumerate(transform_camera_to_world):
            f.write(
                f"transform_row_{row_idx} = "
                f"{row[0]:.10f}, {row[1]:.10f}, {row[2]:.10f}, {row[3]:.10f}\n"
            )
        for attempt_idx, (attempt_voxel, attempt_result, src_count, tgt_count) in enumerate(attempts, start=1):
            f.write(
                "ransac_attempt_"
                f"{attempt_idx} = voxel:{attempt_voxel:.10f},"
                f"src_down:{src_count},tgt_down:{tgt_count},"
                f"fitness:{attempt_result.fitness:.10f},rmse:{attempt_result.inlier_rmse:.10f}\n"
            )


if __name__ == "__main__":
    main()
