"""pin_assembly 태스크를 LeRobot v3.0(parquet+h264 mp4)으로 쓴다.
pipeline/to_lerobot_v3.py의 writer 인프라(build_features/RocoFrame/
create_dataset/write_episode)를 그대로 재사용하고, 에피소드 생성만
sim/pin_assembly_openarm_env.py의 run_episode()로 바꿔 끼운다.

사용자 지시 3-1(OpenArm EE 방향 오프셋 + 분기 선택)을 실제로 쓰는 첫
호출부다 -- roco.build_action_sequence()에
ee_frame_offset_mat=roco.OPENARM_EE_DOWNWARD_OFFSET_MAT,
left_initial_rx_branch="positive_pi", right_initial_rx_branch="negative_pi"
를 넘긴다.

사용 예:
    python pipeline/to_lerobot_pin_assembly.py --out-root ./data/lerobot_v3/pin_assembly \
        --n-episodes-per-scenario 25 --verify
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

# PIPELINE.md의 "MUJOCO_GL=osmesa 설정 뒤 mujoco를 먼저 임포트한 뒤
# torch(Adam 등)를 임포트/생성하면 세그폴트" 문제를 피하려고, lerobot
# (torch 의존)을 **mujoco보다 먼저** 임포트한다 -- pipeline/to_lerobot_v3.py
# 가 이미 이 순서(거기도 모듈 최상단에 mujoco를 안 두고 함수 안에서
# 지역 임포트)로 안전하게 동작하는 걸 실측으로 확인했다. 이 파일은
# 모듈 최상단에서 mujoco를 직접 쓰므로, import 순서 자체를 명시적으로
# 뒤집었다(실측: 처음엔 mujoco를 위에 뒀다가 세그폴트 재현, 순서만
# 바꿔서 해결).
from pipeline.to_lerobot_v3 import LEROBOT_AVAILABLE, RocoFrame, _IMPORT_ERROR, create_dataset, write_episode

import mujoco  # noqa: E402 -- 위 주석대로 lerobot(torch) 이후에 임포트

import sim.roco_interface as roco  # noqa: E402
from sim.pin_assembly_openarm_env import run_episode  # noqa: E402
from sim.pin_assembly_openarm_sim import PinAssemblyOpenArmSim  # noqa: E402

SCENARIOS = ("S1", "S2", "S3")


def _make_sim_and_renderer() -> tuple[PinAssemblyOpenArmSim, "mujoco.Renderer"]:
    sim = PinAssemblyOpenArmSim()
    sim.model = roco.build_roco_model(sim.xml_path)
    sim.data = mujoco.MjData(sim.model)
    renderer = mujoco.Renderer(sim.model, height=roco.CAMERA_HEIGHT, width=roco.CAMERA_WIDTH)
    return sim, renderer


_ZERO_IMAGES = {
    cam: np.zeros((roco.CAMERA_HEIGHT, roco.CAMERA_WIDTH, 3), dtype=np.uint8) for cam in roco.CAMERA_NAMES
}


def episode_to_frames(
    scenario: str, seed: int, sim: PinAssemblyOpenArmSim, renderer: "mujoco.Renderer | None"
) -> tuple[list[RocoFrame], dict]:
    """renderer=None이면 실제 카메라 렌더링을 건너뛰고 0으로 채운 이미지를
    쓴다 -- OSMesa 소프트웨어 렌더가 실측 ~1초/프레임(3카메라 합산)으로
    느려서, 시나리오당 수십 개 에피소드를 전부 렌더링하면 시간이 몇 시간
    단위로 든다(완료 보고에 실측치 기록). validate_against_roco.py는
    parquet의 숫자 컬럼만 읽고 비디오/이미지를 전혀 안 보므로, "검증용
    대량 생성"은 renderer=None(빠름)으로, "실제 이미지 포함 데모"는
    renderer를 줘서 소수만 만든다 -- 두 모드를 분리한 이유는 완료 보고에
    그대로 적는다."""
    rng = np.random.default_rng(seed)
    rec, info = run_episode(scenario, rng, sim=sim, renderer=renderer)

    left_pos = np.stack(rec.left_ee_pos)
    left_quat = np.stack(rec.left_ee_quat)
    right_pos = np.stack(rec.right_ee_pos)
    right_quat = np.stack(rec.right_ee_quat)

    left_mapping = roco.default_gripper_mapping(sim.model, "left")
    right_mapping = roco.default_gripper_mapping(sim.model, "right")
    left_grip = np.array([left_mapping.to_roco(v) for v in rec.left_gripper_raw])
    right_grip = np.array([right_mapping.to_roco(v) for v in rec.right_gripper_raw])

    action_seq = roco.build_action_sequence(
        left_pos, left_quat, left_grip, right_pos, right_quat, right_grip,
        ee_frame_offset_mat=roco.OPENARM_EE_DOWNWARD_OFFSET_MAT,
        left_initial_rx_branch="positive_pi",
        right_initial_rx_branch="negative_pi",
    )
    # action[t] = 다음 state에서 도달해야 할 목표(절대 목표 제어 관례,
    # pipeline/to_lerobot_v3.py의 같은 주석 참고) -- 마지막은 자기 반복.
    action_seq = np.concatenate([action_seq[1:], action_seq[-1:]], axis=0)

    n = len(rec)
    states = np.empty((n, 44), dtype=np.float32)
    for i in range(n):
        parts = [
            left_pos[i], left_quat[i], right_pos[i], right_quat[i],
            rec.left_jpos[i], rec.right_jpos[i],
            rec.left_jvel[i], rec.right_jvel[i],
            [left_grip[i]], [right_grip[i]],
        ]
        states[i] = np.concatenate(parts).astype(np.float32)

    frames = []
    for i in range(n):
        frames.append(
            RocoFrame(
                state=states[i],
                action=action_seq[i].astype(np.float32),
                images=rec.images[i] if renderer is not None else _ZERO_IMAGES,
                task=info["language"],
                scenario=info["scenario"],
                failure_type=rec.failure_type[i] or "none",
                phase=rec.phase[i],
                loss_weight=float(rec.loss_weight[i]),
                source="ard_gen_sim",
                episode_success=bool(info["episode_success"]),
            )
        )
    return frames, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=str, default="./data/lerobot_v3/pin_assembly")
    parser.add_argument("--repo-id", type=str, default="ard-gen/pin_assembly")
    parser.add_argument("--n-episodes-per-scenario", type=int, default=25)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--no-render", action="store_true", help="카메라 렌더링을 건너뛰고 0으로 채운 이미지를 쓴다(빠름, episode_to_frames() docstring 참고)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not LEROBOT_AVAILABLE:
        print(f"[to_lerobot_pin_assembly] lerobot import 실패: {_IMPORT_ERROR}")
        return

    if os.path.exists(args.out_root):
        import shutil

        shutil.rmtree(args.out_root)

    dataset = create_dataset(args.repo_id, args.out_root)
    if args.no_render:
        sim = PinAssemblyOpenArmSim()
        sim.model = roco.build_roco_model(sim.xml_path)
        sim.data = mujoco.MjData(sim.model)
        renderer = None
    else:
        sim, renderer = _make_sim_and_renderer()

    stats = {s: {"n": 0, "success": 0, "frames": []} for s in SCENARIOS}
    t0 = time.time()
    for scenario in SCENARIOS:
        for i in range(args.n_episodes_per_scenario):
            seed = args.seed_offset + i
            frames, info = episode_to_frames(scenario, seed, sim, renderer)
            write_episode(dataset, frames)
            stats[scenario]["n"] += 1
            stats[scenario]["success"] += int(info["episode_success"])
            stats[scenario]["frames"].append(info["n_frames"])
            print(
                f"[to_lerobot_pin_assembly] {scenario} ep{i}: {info['n_frames']}프레임 "
                f"success={info['episode_success']} n_assembled={info['n_assembled']} "
                f"({time.time()-t0:.1f}s 경과)"
            )

    dataset.finalize()

    print("\n[to_lerobot_pin_assembly] 시나리오별 결과:")
    for s in SCENARIOS:
        st = stats[s]
        rate = st["success"] / st["n"] if st["n"] else 0.0
        frames = st["frames"]
        print(
            f"  {s}: {st['n']}개 에피소드, 성공률 {rate*100:.1f}%, "
            f"프레임 평균 {np.mean(frames):.1f}(범위 {min(frames)}~{max(frames)})"
        )
    total_frames = sum(sum(st["frames"]) for st in stats.values())
    total_n = sum(st["n"] for st in stats.values())
    print(f"[to_lerobot_pin_assembly] 전체 {total_n}개 에피소드, 총 {total_frames}프레임 -> {args.out_root}")

    if args.verify:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        reloaded = LeRobotDataset(repo_id=args.repo_id, root=args.out_root)
        print(
            f"[to_lerobot_pin_assembly] 재로딩 검증: num_episodes={reloaded.num_episodes} "
            f"num_frames={reloaded.num_frames} fps={reloaded.fps}"
        )
        sample = reloaded[0]
        print(
            f"[to_lerobot_pin_assembly] sample[0]: state.shape={tuple(sample['observation.state'].shape)} "
            f"action.shape={tuple(sample['action'].shape)}"
        )


if __name__ == "__main__":
    main()
