#!/usr/bin/env python3

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import signal
import statistics
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _camera_index(value: str) -> int | Path:
    try:
        return int(value)
    except ValueError:
        return Path(value)


def _max_relative_target(value: str) -> float | None:
    if value.lower() == "none":
        return None
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("max-relative-target must be positive or 'none'")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an SO101 with a remote OpenPI policy server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="zjh_follower_arm")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower",
    )
    parser.add_argument("--front-camera", type=_camera_index, default=0)
    parser.add_argument("--wrist-camera", type=_camera_index, default=2)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--control-fps", type=float, default=30.0)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--max-relative-target", type=_max_relative_target, default=5.0)
    parser.add_argument("--task", default="Grab blue battery to the bin")
    parser.add_argument(
        "--latency-log-dir",
        type=Path,
        default=Path.home() / "openpi" / "so101_logs",
        help="Parent directory for one JSONL timing log per deployment run.",
    )
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _format_summary(name: str, values: list[float], unit: str) -> str:
    if not values:
        return f"{name}=n/a"
    return (
        f"{name}: n={len(values)} mean={statistics.fmean(values):.1f}{unit} "
        f"p50={_percentile(values, 50):.1f}{unit} p95={_percentile(values, 95):.1f}{unit}"
    )


def main() -> None:
    args = parse_args()
    if not 1 <= args.actions_per_chunk <= 50:
        raise ValueError("actions-per-chunk must be between 1 and 50")

    calibration_file = args.calibration_dir / f"{args.robot_id}.json"
    if not calibration_file.is_file():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_file}. Use the same robot id as data collection."
        )

    camera_common = {
        "fps": args.camera_fps,
        "width": args.camera_width,
        "height": args.camera_height,
        "fourcc": args.camera_fourcc,
    }
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            calibration_dir=args.calibration_dir,
            max_relative_target=args.max_relative_target,
            disable_torque_on_disconnect=True,
            use_degrees=True,
            cameras={
                "front": OpenCVCameraConfig(index_or_path=args.front_camera, **camera_common),
                "wrist": OpenCVCameraConfig(index_or_path=args.wrist_camera, **camera_common),
            },
        )
    )
    # The first JAX request after a server restart may compile for several minutes.
    # SSH already keeps the tunnel alive, so do not let WebSocket pings abort that request.
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        ping_interval=None,
        ping_timeout=None,
    )
    logging.info("Server metadata: %s", policy.get_server_metadata())

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    robot.connect()
    chunk_index = 0
    period = 1.0 / args.control_fps
    run_dir = args.latency_log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    latency_log_path = run_dir / "latency.jsonl"
    run_info_path = run_dir / "run_info.json"
    run_info_path.write_text(
        json.dumps(
            {
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "server": f"{args.host}:{args.port}",
                "control_fps_target": args.control_fps,
                "actions_per_chunk": args.actions_per_chunk,
                "camera": {"front": str(args.front_camera), "wrist": str(args.wrist_camera)},
                "task": args.task,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.info("Writing deployment timing logs to %s", latency_log_path)
    round_trip_samples: list[float] = []
    policy_samples: list[float] = []
    control_hz_samples: list[float] = []
    try:
        with latency_log_path.open("a", encoding="utf-8", buffering=1) as latency_log:
            while not stop_requested:
                chunk_start = time.perf_counter()
                observation_start = time.perf_counter()
                observation = robot.get_observation()
                observation_ms = (time.perf_counter() - observation_start) * 1000.0
                state = np.asarray([observation[f"{joint}.pos"] for joint in JOINT_NAMES], dtype=np.float32)

                preprocess_start = time.perf_counter()
                request = {
                    "observation/state": state,
                    "observation/image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(observation["front"], 224, 224)
                    ),
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(observation["wrist"], 224, 224)
                    ),
                    "prompt": args.task,
                }
                preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

                infer_start = time.perf_counter()
                response = policy.infer(request)
                round_trip_ms = (time.perf_counter() - infer_start) * 1000.0
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] < len(JOINT_NAMES):
                    raise RuntimeError(f"Unexpected action shape: {actions.shape}")
                if not np.isfinite(actions).all():
                    raise RuntimeError("Policy returned a non-finite action")

                action_count = min(args.actions_per_chunk, actions.shape[0])
                policy_ms = response.get("policy_timing", {}).get("infer_ms")
                server_ms = response.get("server_timing", {}).get("infer_ms")
                transport_ms = round_trip_ms - server_ms if server_ms is not None else None
                action_send_ms: list[float] = []
                execution_start = time.perf_counter()
                executed_actions = 0
                for action in actions[:action_count, : len(JOINT_NAMES)]:
                    if stop_requested:
                        break
                    tick_start = time.perf_counter()
                    robot.send_action({f"{joint}.pos": float(value) for joint, value in zip(JOINT_NAMES, action)})
                    send_ms = (time.perf_counter() - tick_start) * 1000.0
                    action_send_ms.append(send_ms)
                    executed_actions += 1
                    remaining = period - send_ms / 1000.0
                    if remaining > 0:
                        time.sleep(remaining)
                execution_ms = (time.perf_counter() - execution_start) * 1000.0
                actual_hz = executed_actions / (execution_ms / 1000.0) if execution_ms > 0 else 0.0
                chunk_cycle_ms = (time.perf_counter() - chunk_start) * 1000.0

                record = {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    "chunk": chunk_index,
                    "actions_requested": action_count,
                    "actions_executed": executed_actions,
                    "observation_ms": round(observation_ms, 3),
                    "preprocess_ms": round(preprocess_ms, 3),
                    "round_trip_ms": round(round_trip_ms, 3),
                    "server_policy_ms": round(float(policy_ms), 3) if policy_ms is not None else None,
                    "server_total_ms": round(float(server_ms), 3) if server_ms is not None else None,
                    "transport_and_serialization_ms": round(transport_ms, 3) if transport_ms is not None else None,
                    "mean_send_action_ms": round(statistics.fmean(action_send_ms), 3) if action_send_ms else None,
                    "max_send_action_ms": round(max(action_send_ms), 3) if action_send_ms else None,
                    "execution_ms": round(execution_ms, 3),
                    "actual_control_hz": round(actual_hz, 3),
                    "sensor_to_first_command_ms": round(observation_ms + preprocess_ms + round_trip_ms, 3),
                    "chunk_cycle_ms": round(chunk_cycle_ms, 3),
                }
                latency_log.write(json.dumps(record, ensure_ascii=True) + "\n")
                round_trip_samples.append(round_trip_ms)
                if policy_ms is not None:
                    policy_samples.append(float(policy_ms))
                control_hz_samples.append(actual_hz)
                logging.info(
                    "chunk=%d actions=%d obs=%.1fms prep=%.1fms rtt=%.1fms policy=%s "
                    "transport=%s execute=%.1fms hz=%.1f cycle=%.1fms",
                    chunk_index,
                    executed_actions,
                    observation_ms,
                    preprocess_ms,
                    round_trip_ms,
                    f"{policy_ms:.1f}ms" if policy_ms is not None else "n/a",
                    f"{transport_ms:.1f}ms" if transport_ms is not None else "n/a",
                    execution_ms,
                    actual_hz,
                    chunk_cycle_ms,
                )
                chunk_index += 1
    finally:
        logging.info("Timing summary | %s", _format_summary("round_trip", round_trip_samples, "ms"))
        logging.info("Timing summary | %s", _format_summary("policy", policy_samples, "ms"))
        logging.info("Timing summary | %s", _format_summary("control", control_hz_samples, "Hz"))
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
