import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse LiDAR and camera coordinates from phase-specific result files."
    )
    parser.add_argument(
        "--lidar_file",
        type=str,
        required=True,
        help="LiDAR result file name relative to CAM_LiDAR_Results, e.g. lidar_result_coordinates_P1.txt or lidar_result_coordinates_P2.txt.",
    )
    parser.add_argument(
        "--cam_file",
        type=str,
        required=True,
        help="Camera result file name relative to CAM_LiDAR_Results, e.g. cam_result_coordinates_P1.txt or cam_result_coordinates_P2.txt.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Fusion output file name relative to CAM_LiDAR_Results, e.g. fused_result_coordinates_P1.txt or fused_result_coordinates_P2.txt.",
    )
    return parser.parse_args()


def parse_sensor_file(file_path):
    """Parse one sensor result file."""
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
    return data


def compute_lidar_confidence(lidar_data):
    """Confidence for LiDAR."""
    fitness = lidar_data.get("icp_fitness", 0.0)
    rmse_norm = lidar_data.get("icp_rmse_normalized", 1.0)

    rmse_score = 1.0 / (rmse_norm ** 0.8 + 0.05)
    confidence = fitness ** 1.1 * rmse_score ** 0.9

    lidar_boost = 20.0
    return max(0.05, confidence * lidar_boost)


def compute_cam_confidence(cam_data):
    """Confidence for camera."""
    good_matches = cam_data.get("good_matches", 0)
    inlier_ratio = cam_data.get("homography_inlier_ratio", 0.0)
    mean_inlier_conf = cam_data.get("mean_inlier_confidence", 0.4)

    match_score = (good_matches ** 0.75) * inlier_ratio * mean_inlier_conf
    return max(0.05, match_score)


def fuse_coordinates(lidar_data, cam_data):
    """Fuse coordinates weighted by confidence."""
    x_l = lidar_data.get("x", 0.0)
    y_l = lidar_data.get("y", 0.0)
    z_l = lidar_data.get("z", 0.0)

    x_c = cam_data.get("x", 0.0)
    y_c = cam_data.get("y", 0.0)
    z_c = cam_data.get("z", z_l)

    conf_l = compute_lidar_confidence(lidar_data)
    conf_c = compute_cam_confidence(cam_data)

    total_conf = conf_l + conf_c

    x_fused = (conf_l * x_l + conf_c * x_c) / total_conf
    y_fused = (conf_l * y_l + conf_c * y_c) / total_conf
    z_fused = (conf_l * z_l + conf_c * z_c) / total_conf

    fusion_confidence = (conf_l * conf_l + conf_c * conf_c) / total_conf

    weight_lidar = conf_l / total_conf
    weight_cam = conf_c / total_conf

    return {
        "x": x_fused,
        "y": y_fused,
        "z": z_fused,
        "fusion_confidence": fusion_confidence,
        "lidar_confidence": conf_l,
        "cam_confidence": conf_c,
        "weight_lidar": weight_lidar,
        "weight_cam": weight_cam,
    }


def main():
    opt = parse_args()
    base_dir = SCRIPT_DIR / "CAM_LiDAR_Results"

    lidar_file = base_dir / opt.lidar_file
    cam_file = base_dir / opt.cam_file
    output_file = base_dir / opt.output_file

    if not lidar_file.exists() or not cam_file.exists():
        print("Error: one of the input sensor result files was not found.")
        return

    lidar_data = parse_sensor_file(lidar_file)
    cam_data = parse_sensor_file(cam_file)

    fused = fuse_coordinates(lidar_data, cam_data)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("sensor = fused\n")
        f.write(f"x = {fused['x']:.10f}\n")
        f.write(f"y = {fused['y']:.10f}\n")
        f.write(f"z = {fused['z']:.10f}\n")
        f.write(f"fusion_confidence = {fused['fusion_confidence']:.10f}\n")
        f.write(f"lidar_confidence = {fused['lidar_confidence']:.10f}\n")
        f.write(f"cam_confidence = {fused['cam_confidence']:.10f}\n")
        f.write(f"weight_lidar = {fused['weight_lidar']:.6f}\n")
        f.write(f"weight_cam = {fused['weight_cam']:.6f}\n")

        f.write("\n# LiDAR metrics:\n")
        f.write(f"icp_fitness = {lidar_data.get('icp_fitness', 0):.10f}\n")
        f.write(f"icp_rmse_normalized = {lidar_data.get('icp_rmse_normalized', 0):.10f}\n")

        f.write("\n# Camera metrics:\n")
        f.write(f"good_matches = {int(cam_data.get('good_matches', 0))}\n")
        f.write(f"homography_inlier_ratio = {cam_data.get('homography_inlier_ratio', 0):.10f}\n")
        f.write(f"homography_reprojection_rmse_px = {cam_data.get('homography_reprojection_rmse_px', 0):.10f}\n")

    print("Fusion completed successfully.")
    print(f"Saved result: {output_file.name}")
    print(f"Position: X={fused['x']:.4f}, Y={fused['y']:.4f}, Z={fused['z']:.4f}")
    print(f"Weights -> LiDAR: {fused['weight_lidar']:.1%} | Camera: {fused['weight_cam']:.1%}")
    print(f"Raw confidences -> LiDAR: {fused['lidar_confidence']:.4f} | Cam: {fused['cam_confidence']:.4f}")


if __name__ == "__main__":
    main()
