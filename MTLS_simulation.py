import bpy
import csv
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from mathutils import Vector

# Config
CAMERA_NAME = "Camera"

DEFAULT_PROJECT_ROOT = Path(r"E:\Applied_Maths_Informatics_Studying\Thesis\Mars_Terrain_Landing_System")
REALTIME_STEP_SECONDS = 1.0

PHASE_1_TRIGGER_ALTITUDE = 5000.0
PHASE_2_TRIGGER_ALTITUDE = 2000.0
PHASE_3_TRIGGER_ALTITUDE = 500.0
LANDING_ALTITUDE = 0.0
LANDING_TOLERANCE = 0.5

INITIAL_DESCENT_SPEED = 150.0
POST_PHASE_1_DESCENT_SPEED = 80.0
POST_PHASE_2_DESCENT_SPEED = 50.0
POST_PHASE_3_DESCENT_SPEED = 10.0

DESCENT_DIRECTION_WORLD = Vector((0.0, 0.0, -1.0))
MOVE_GAIN = 1.0

RENDER_RES_X = 800
RENDER_RES_Y = 800

LIDAR_W = 512
LIDAR_H = 512
MAX_RAY_DISTANCE = 1e6
EPS = 1e-7

WINDOW_SIZE = 48
WINDOW_STEP = 16
MIN_VALID_RATIO = 0.75
SLOPE_LIMIT_DEG = 12.0
ROUGHNESS_LIMIT = 0.08
BOULDER_LIMIT = 0.12
CRATER_LIMIT = 0.12
HAZARD_MOVE_GAIN = 1.0

# Пути
def resolve_project_root():
    if bpy.data.filepath:
        blend_root = Path(bpy.path.abspath("//")).resolve()
        if (blend_root / "LiDAR_unit").exists() and (blend_root / "CAM_unit").exists():
            return blend_root
    return DEFAULT_PROJECT_ROOT


PROJECT_ROOT = resolve_project_root()
LIDAR_UNIT_DIR = PROJECT_ROOT / "LiDAR_unit"
CAM_UNIT_DIR = PROJECT_ROOT / "CAM_unit" / "SuperGluePretrainedNetwork-master"
RESULTS_DIR = PROJECT_ROOT / "CAM_LiDAR_Results"

LIDAR_SCANS_DIR = LIDAR_UNIT_DIR / "LiDAR_scans"
LIDAR_DTMS_DIR = LIDAR_UNIT_DIR / "DTMs"

CAM_IMAGES_DIR = CAM_UNIT_DIR / "CAM_images"
CAM_DTMS_DIR = CAM_UNIT_DIR / "DTMs"
CAM_RESULTS_ROOT = CAM_UNIT_DIR / "Results"

LIDAR_MATCH_SCRIPT = LIDAR_UNIT_DIR / "LiDAR_to_DTM_matching.py"
CAM_MATCH_SCRIPT = CAM_UNIT_DIR / "MTLS_CAM_unit_matching.py"
FUSION_SCRIPT = PROJECT_ROOT / "sensor_fusion_decision.py"

PYTHON_EXE = shutil.which("python") or "python"


# Определение файлов фаз 1 и 2
PHASE_CONFIGS = {
    "P1": {
        "lidar_csv": "LiDAR_P1.csv",
        "dtm_csv": "DTM_P1.csv",
        "lidar_result": "lidar_result_coordinates_P1.txt",
        "cam_map": "DTM_P1.png",
        "cam_image": "CAM_P1.png",
        "cam_output_dir": "P1",
        "cam_result": "cam_result_coordinates_P1.txt",
        "fused_result": "fused_result_coordinates_P1.txt",
    },
    "P2": {
        "lidar_csv": "LiDAR_P2.csv",
        "dtm_csv": "DTM_P2.csv",
        "lidar_result": "lidar_result_coordinates_P2.txt",
        "cam_map": "DTM_P2.png",
        "cam_image": "CAM_P2.png",
        "cam_output_dir": "P2",
        "cam_result": "cam_result_coordinates_P2.txt",
        "fused_result": "fused_result_coordinates_P2.txt",
    },
}

# Вспомогательные функции
def ensure_runtime_dirs():
    for path in (RESULTS_DIR, LIDAR_SCANS_DIR, CAM_IMAGES_DIR, CAM_RESULTS_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def get_scene_and_camera():
    scene = bpy.context.scene
    if CAMERA_NAME in bpy.data.objects:
        scene.camera = bpy.data.objects[CAMERA_NAME]
    if scene.camera is None:
        raise RuntimeError(f"Camera '{CAMERA_NAME}' was not found.")
    return scene, scene.camera


def get_camera_forward_world(camera_obj):
    return (camera_obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()


def raycast_from_camera_center(scene, camera_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera_obj.matrix_world.translation.copy()
    direction = get_camera_forward_world(camera_obj)
    hit, location, normal, face_index, obj, matrix = scene.ray_cast(
        depsgraph, origin, direction, distance=MAX_RAY_DISTANCE
    )
    if not hit:
        return None, None, None
    altitude = (location - origin).length
    return altitude, location, direction


def move_camera_downward(camera_obj, distance):
    if distance <= 0.0:
        return
    camera_obj.location += DESCENT_DIRECTION_WORLD * distance


def read_estimated_xy(result_path):
    text = result_path.read_text(encoding="utf-8")
    x_match = re.search(r"^x\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", text, re.MULTILINE)
    y_match = re.search(r"^y\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", text, re.MULTILINE)
    if not x_match or not y_match:
        raise RuntimeError(f"Could not parse x/y from {result_path}")
    return float(x_match.group(1)), float(y_match.group(1))


def apply_fusion_correction(camera_obj, result_path, gain=MOVE_GAIN):
    x_est, y_est = read_estimated_xy(result_path)
    correction = Vector((-x_est * gain, -y_est * gain, 0.0))
    camera_obj.location += correction
    print(f"Корректировка курса: {correction}")
    return correction


def run_subprocess(cmd, cwd):
    print(f"Запуск: {' '.join(map(str, cmd))}")
    completed = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {cmd}")


def run_parallel_processes(commands_with_cwd):
    processes = []
    try:
        for cmd, cwd in commands_with_cwd:
            print(f"Начало: {' '.join(map(str, cmd))}")
            proc = subprocess.Popen(
                [str(part) for part in cmd],
                cwd=str(cwd),
                shell=False,
            )
            processes.append((cmd, proc))
        for cmd, proc in processes:
            return_code = proc.wait()
            print(f"Завершен ({return_code}): {' '.join(map(str, cmd))}")
            if return_code != 0:
                raise RuntimeError(f"Parallel command failed with exit code {return_code}: {cmd}")
    finally:
        for cmd, proc in processes:
            if proc.poll() is None:
                proc.terminate()

# Рендер + LiDAR
def render_camera_image(scene, phase_name):
    output_path = CAM_IMAGES_DIR / PHASE_CONFIGS[phase_name]["cam_image"]
    scene.render.resolution_x = RENDER_RES_X
    scene.render.resolution_y = RENDER_RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    print(f"Изображение CAM получено: {output_path}")
    return output_path


def run_lidar_scan(scene, camera_obj, output_csv_path):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_data = camera_obj.data

    if cam_data.type not in {"PERSP", "ORTHO"}:
        raise RuntimeError("This script expects a PERSP or ORTHO camera.")

    start_time = time.time()

    frame = cam_data.view_frame(scene=scene)
    bl, br, tr, tl = frame[0], frame[1], frame[2], frame[3]

    origin_world = camera_obj.matrix_world.translation.copy()
    cam_rot = camera_obj.matrix_world.to_3x3()
    cam_forward_world = (cam_rot @ Vector((0.0, 0.0, -1.0))).normalized()

    rows = []

    for j in range(LIDAR_H):
        v = (j + 0.5) / LIDAR_H
        v_vec = bl + (tl - bl) * v

        for i in range(LIDAR_W):
            u = (i + 0.5) / LIDAR_W
            local_pt = v_vec + (br - bl) * u

            if cam_data.type == "PERSP":
                dir_local = local_pt.normalized()
                dir_world = cam_rot @ dir_local
                if dir_world.length < EPS:
                    continue
                dir_world.normalize()
                ray_origin = origin_world
            else:
                ray_origin = camera_obj.matrix_world @ local_pt
                dir_world = cam_forward_world.copy()

            hit, loc, norm, face_index, obj, matrix = scene.ray_cast(
                depsgraph, ray_origin, dir_world, distance=MAX_RAY_DISTANCE
            )

            if not hit:
                continue

            depth = (loc - origin_world).dot(cam_forward_world)
            if depth <= 0.0:
                continue

            if cam_data.type == "PERSP":
                scale = depth / max(-dir_local.z, EPS)
                cam_hit = dir_local * scale
            else:
                cam_hit = local_pt + Vector((0.0, 0.0, -depth))

            rows.append((
                i + 1, j + 1, float(depth),
                float(cam_hit.x), float(cam_hit.y), float(cam_hit.z)
            ))

    if not rows:
        raise RuntimeError("No LiDAR ray hits were recorded.")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PIXEL_X", "PIXEL_Y", "DEPTH", "CAM_X", "CAM_Y", "CAM_Z"])
        writer.writerows(rows)

    duration = time.time() - start_time
    depths = [r[2] for r in rows]
    print(f"LiDAR скан сохранен: {output_csv_path} ({len(rows)} hits, {duration:.2f}s)")
    print(f"Диапазон глубин: min={min(depths):.6f}, max={max(depths):.6f}")

    return output_csv_path

# Фазы 1 и 2
def run_navigation_phase(scene, camera_obj, phase_name):
    ensure_runtime_dirs()
    phase_cfg = PHASE_CONFIGS[phase_name]

    lidar_csv_path = LIDAR_SCANS_DIR / phase_cfg["lidar_csv"]
    render_camera_image(scene, phase_name)
    run_lidar_scan(scene, camera_obj, lidar_csv_path)

    lidar_cmd = [
        PYTHON_EXE,
        LIDAR_MATCH_SCRIPT.name,
        "--lidar_csv",
        phase_cfg["lidar_csv"],
        "--dtm_csv",
        phase_cfg["dtm_csv"],
        "--result_file",
        phase_cfg["lidar_result"],
    ]

    cam_cmd = [
        PYTHON_EXE,
        CAM_MATCH_SCRIPT.name,
        "--map",
        phase_cfg["cam_map"],
        "--camera",
        phase_cfg["cam_image"],
        "--output_dir",
        phase_cfg["cam_output_dir"],
    ]

    run_parallel_processes(
        [
            (lidar_cmd, LIDAR_UNIT_DIR),
            (cam_cmd, CAM_UNIT_DIR),
        ]
    )

    fusion_cmd = [
        PYTHON_EXE,
        FUSION_SCRIPT.name,
        "--lidar_file",
        phase_cfg["lidar_result"],
        "--cam_file",
        phase_cfg["cam_result"],
        "--output_file",
        phase_cfg["fused_result"],
    ]
    run_subprocess(fusion_cmd, PROJECT_ROOT)

    fused_result_path = RESULTS_DIR / phase_cfg["fused_result"]
    apply_fusion_correction(camera_obj, fused_result_path, MOVE_GAIN)
    print(f"{phase_name} завершена")


# Фаза 3
def run_phase_3_scan(scene, camera_obj, output_csv_path):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_data = camera_obj.data
    cam_world = camera_obj.matrix_world.copy()
    cam_inv = cam_world.inverted()
    cam_rot = cam_world.to_3x3()
    origin_world = cam_world.translation.copy()
    cam_forward_world = get_camera_forward_world(camera_obj)

    frame = cam_data.view_frame(scene=scene)
    bl, br, tr, tl = frame[0], frame[1], frame[2], frame[3]

    depth_map = np.full((LIDAR_H, LIDAR_W), np.nan, dtype=np.float32)
    local_x = np.full((LIDAR_H, LIDAR_W), np.nan, dtype=np.float32)
    local_y = np.full((LIDAR_H, LIDAR_W), np.nan, dtype=np.float32)
    local_zf = np.full((LIDAR_H, LIDAR_W), np.nan, dtype=np.float32)
    world_x = np.full((LIDAR_H, LIDAR_W), np.nan, dtype=np.float32)
    world_y = np.full((LIDAR_H, LIDAR_W), np.nan, dtype=np.float32)
    world_z = np.full((LIDAR_H, LIDAR_W), np.nan, dtype=np.float32)

    rows = []
    start_time = time.time()

    for j in range(LIDAR_H):
        v = (j + 0.5) / LIDAR_H
        v_vec = bl.lerp(tl, v)
        for i in range(LIDAR_W):
            u = (i + 0.5) / LIDAR_W
            local_pt = v_vec.lerp(br, u)

            if cam_data.type == "PERSP":
                dir_local = local_pt.normalized()
                dir_world = cam_rot @ dir_local
                dir_world.normalize()
                ray_origin = origin_world
            else:
                ray_origin = cam_world @ local_pt
                dir_world = cam_forward_world.copy()

            hit, location, normal, face_index, obj, matrix = scene.ray_cast(
                depsgraph, ray_origin, dir_world, distance=MAX_RAY_DISTANCE
            )
            if not hit:
                continue

            depth = (location - origin_world).dot(cam_forward_world)
            if depth <= 0.0:
                continue

            location_local = cam_inv @ location
            z_forward = -location_local.z

            depth_map[j, i] = depth
            local_x[j, i] = location_local.x
            local_y[j, i] = location_local.y
            local_zf[j, i] = z_forward
            world_x[j, i] = location.x
            world_y[j, i] = location.y
            world_z[j, i] = location.z

            rows.append((i + 1, j + 1, float(depth)))

    if not rows:
        raise RuntimeError("Phase 3 LiDAR scan returned no hits.")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "Z"])
        writer.writerows(rows)

    duration = time.time() - start_time
    print(f"LiDAR скан фазы 3 сохранен: {output_csv_path} ({len(rows)} hits, {duration:.2f}s)")
    return depth_map, local_x, local_y, local_zf, world_x, world_y, world_z


def fit_plane_and_metrics(xs, ys, zs):
    a_matrix = np.column_stack((xs, ys, np.ones_like(xs)))
    coef, _, _, _ = np.linalg.lstsq(a_matrix, zs, rcond=None)
    a, b, c = coef
    z_hat = a_matrix @ coef
    residuals = zs - z_hat

    slope = math.sqrt(a * a + b * b)
    slope_deg = math.degrees(math.atan(slope))
    roughness = float(np.sqrt(np.mean(residuals ** 2)))
    protrusion = float(max(0.0, -np.min(residuals)))
    depression = float(max(0.0, np.max(residuals)))
    return slope_deg, roughness, protrusion, depression


def analyze_surface(local_x, local_y, local_zf, world_x, world_y, world_z):
    half = WINDOW_SIZE // 2
    best = None
    best_score = float("inf")

    for j in range(half, LIDAR_H - half, WINDOW_STEP):
        for i in range(half, LIDAR_W - half, WINDOW_STEP):
            window_mask = np.isfinite(local_zf[j - half:j + half, i - half:i + half])
            n_valid = int(window_mask.sum())
            total = WINDOW_SIZE * WINDOW_SIZE

            if n_valid < max(30, int(total * MIN_VALID_RATIO)):
                continue

            xs = local_x[j - half:j + half, i - half:i + half][window_mask].ravel()
            ys = local_y[j - half:j + half, i - half:i + half][window_mask].ravel()
            zs = local_zf[j - half:j + half, i - half:i + half][window_mask].ravel()

            slope_deg, roughness, protrusion, depression = fit_plane_and_metrics(xs, ys, zs)

            score = (
                3.0 * (slope_deg / max(SLOPE_LIMIT_DEG, EPS))
                + 2.0 * (roughness / max(ROUGHNESS_LIMIT, EPS))
                + 2.5 * (protrusion / max(BOULDER_LIMIT, EPS))
                + 2.5 * (depression / max(CRATER_LIMIT, EPS))
                + 1.5 * (1.0 - n_valid / total)
            )

            cx = np.nanmean(world_x[j - half:j + half, i - half:i + half][window_mask])
            cy = np.nanmean(world_y[j - half:j + half, i - half:i + half][window_mask])
            cz = np.nanmean(world_z[j - half:j + half, i - half:i + half][window_mask])

            candidate = {
                "center_px": (i, j),
                "centroid_world": Vector((float(cx), float(cy), float(cz))),
                "slope_deg": slope_deg,
                "roughness": roughness,
                "protrusion": protrusion,
                "depression": depression,
                "valid_ratio": n_valid / total,
                "score": float(score),
            }

            if score < best_score:
                best_score = score
                best = candidate

    if best is None:
        raise RuntimeError("No valid hazard-free patch found in phase 3.")

    return best


PHASE_3_MAX_CORRECTION_RADIUS = 30.0  # meters


def move_camera_toward_safe_patch(camera_obj, best_patch):
    current = camera_obj.matrix_world.translation.copy()
    target = best_patch["centroid_world"]

    move_world = Vector((target.x - current.x, target.y - current.y, 0.0))

    move_len = move_world.length
    if move_len > PHASE_3_MAX_CORRECTION_RADIUS:
        move_world *= PHASE_3_MAX_CORRECTION_RADIUS / move_len

    move_world *= HAZARD_MOVE_GAIN

    if camera_obj.parent:
        parent_inv = camera_obj.parent.matrix_world.inverted()
        move_local = parent_inv.to_3x3() @ move_world
        camera_obj.location += move_local
    else:
        camera_obj.location += move_world

    print(f"Маневр увода фазы 3: {move_world}")
    return move_world


def run_phase_3_hazard_avoidance(scene, camera_obj):
    ensure_runtime_dirs()
    lidar_p3_path = LIDAR_SCANS_DIR / "LiDAR_P3.csv"
    scan_data = run_phase_3_scan(scene, camera_obj, lidar_p3_path)
    best_patch = analyze_surface(*scan_data[1:])

    print("Phase 3 best safe patch:")
    print(f"  center_px={best_patch['center_px']}")
    print(f"  slope_deg={best_patch['slope_deg']:.3f}")
    print(f"  roughness={best_patch['roughness']:.5f}")
    print(f"  protrusion={best_patch['protrusion']:.5f}")
    print(f"  depression={best_patch['depression']:.5f}")
    print(f"  valid_ratio={best_patch['valid_ratio']:.3f}")
    print(f"  score={best_patch['score']:.4f}")

    move_camera_toward_safe_patch(camera_obj, best_patch)
    print("Фаза 3 завершена")


# Спуск
class MTLS_OT_three_phase_descent(bpy.types.Operator):
    bl_idname = "mtls.three_phase_descent"
    bl_label = "MTLS Three Phase Descent"

    _timer = None
    state = "INIT"
    last_altitude = None

    def execute_phase(self, context, phase_name):
        scene, camera_obj = get_scene_and_camera()
        run_navigation_phase(scene, camera_obj, phase_name)
        self.last_altitude, _, _ = raycast_from_camera_center(scene, camera_obj)

    def execute_phase_3(self, context):
        scene, camera_obj = get_scene_and_camera()
        run_phase_3_hazard_avoidance(scene, camera_obj)
        self.last_altitude, _, _ = raycast_from_camera_center(scene, camera_obj)

    def cancel(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def finish(self, context):
        self.cancel(context)
        self.report({"INFO"}, "MTLS descent simulation complete.")
        print("Работа системы посадки MTLS завершена")
        return {"FINISHED"}

    def descend_one_step(self, scene, camera_obj, speed, target_altitude=None):
        altitude, _, _ = raycast_from_camera_center(scene, camera_obj)
        if altitude is None:
            raise RuntimeError("Altitude raycast missed the terrain.")

        step_distance = speed * REALTIME_STEP_SECONDS

        if target_altitude is not None:
            step_distance = min(step_distance, max(0.0, altitude - target_altitude))
        else:
            step_distance = min(step_distance, altitude)

        move_camera_downward(camera_obj, step_distance)
        new_altitude, _, _ = raycast_from_camera_center(scene, camera_obj)
        self.last_altitude = new_altitude
        print(
            f"Статус: {self.state} | Скорость: {speed:.1f} м/с | "
            f"Высота={new_altitude:.3f} м"
        )
        return new_altitude

    def modal(self, context, event):
        if event.type == "ESC":
            self.cancel(context)
            self.report({"WARNING"}, "MTLS descent simulation cancelled.")
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        try:
            scene, camera_obj = get_scene_and_camera()

            if self.state == "INIT":
                self.last_altitude, _, _ = raycast_from_camera_center(scene, camera_obj)
                print(f"Начальная высота: {self.last_altitude}")
                self.state = "СНИЖЕНИЕ ДО ФАЗЫ 1"
                return {"RUNNING_MODAL"}

            if self.state == "СНИЖЕНИЕ ДО ФАЗЫ 1":
                altitude = self.descend_one_step(
                    scene, camera_obj, INITIAL_DESCENT_SPEED, PHASE_1_TRIGGER_ALTITUDE
                )
                if altitude is not None and altitude <= PHASE_1_TRIGGER_ALTITUDE + EPS:
                    print("Достигнуто 5000 м. Начало фазы 1")
                    self.state = "ЗАПУСК ФАЗЫ 1"
                return {"RUNNING_MODAL"}

            if self.state == "ЗАПУСК ФАЗЫ 1":
                self.execute_phase(context, "P1")
                self.state = "СНИЖЕНИЕ ДО ФАЗЫ 2"
                return {"RUNNING_MODAL"}

            if self.state == "СНИЖЕНИЕ ДО ФАЗЫ 2":
                altitude = self.descend_one_step(
                    scene, camera_obj, POST_PHASE_1_DESCENT_SPEED, PHASE_2_TRIGGER_ALTITUDE
                )
                if altitude is not None and altitude <= PHASE_2_TRIGGER_ALTITUDE + EPS:
                    print("Достигнуто 2000 м. Начало фазы 2")
                    self.state = "ЗАПУСК ФАЗЫ 2"
                return {"RUNNING_MODAL"}

            if self.state == "ЗАПУСК ФАЗЫ 2":
                self.execute_phase(context, "P2")
                self.state = "СНИЖЕНИЕ ДО ФАЗЫ 3"
                return {"RUNNING_MODAL"}

            if self.state == "СНИЖЕНИЕ ДО ФАЗЫ 3":
                altitude = self.descend_one_step(
                    scene, camera_obj, POST_PHASE_2_DESCENT_SPEED, PHASE_3_TRIGGER_ALTITUDE
                )
                if altitude is not None and altitude <= PHASE_3_TRIGGER_ALTITUDE + EPS:
                    print("Достигнуто 500 м. Начало фазы 3")
                    self.state = "ЗАПУСК ФАЗЫ 3"
                return {"RUNNING_MODAL"}

            if self.state == "ЗАПУСК ФАЗЫ 3":
                self.execute_phase_3(context)
                self.state = "СНИЖЕНИЕ ДО КАСАНИЯ"
                return {"RUNNING_MODAL"}

            if self.state == "СНИЖЕНИЕ ДО КАСАНИЯ":
                altitude = self.descend_one_step(
                    scene, camera_obj, POST_PHASE_3_DESCENT_SPEED, LANDING_ALTITUDE
                )
                if altitude is None or altitude <= LANDING_ALTITUDE + LANDING_TOLERANCE:
                    self.state = "КАСАНИЕ"
                return {"RUNNING_MODAL"}

            if self.state == "КАСАНИЕ":
                return self.finish(context)

            raise RuntimeError(f"Unknown controller state: {self.state}")

        except Exception as exc:
            self.cancel(context)
            self.report({"ERROR"}, str(exc))
            print(f"Ошибка модуля управления системы MTLS: {exc}")
            return {"CANCELLED"}

    def invoke(self, context, event):
        ensure_runtime_dirs()
        self.state = "INIT"
        self.last_altitude = None
        wm = context.window_manager
        self._timer = wm.event_timer_add(REALTIME_STEP_SECONDS, window=context.window)
        wm.modal_handler_add(self)
        print("Старт главного модуля управления системы MTLS")
        print(f"Корень проекта: {PROJECT_ROOT}")
        return {"RUNNING_MODAL"}


def register():
    bpy.utils.register_class(MTLS_OT_three_phase_descent)


def unregister():
    bpy.utils.unregister_class(MTLS_OT_three_phase_descent)


if __name__ == "__main__":
    try:
        unregister()
    except Exception:
        pass
    register()
    bpy.ops.mtls.three_phase_descent("INVOKE_DEFAULT")