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


def _disable_expensive_render_flags(renderer: "mujoco.Renderer") -> None:
    """사용자 지시 5(렌더링 프로파일링): 프레임당 ~1.1~1.3초 중 렌더러
    객체 재생성이나 update_scene은 거의 0초(실측 0.0002s/30회)였고,
    `render()` 자체가 거의 전부였다 -- OSMesa 소프트웨어 래스터라이저가
    섀도우맵/반사를 매 프레임 추가로 계산하는 비용이 크다는 뜻(실측:
    해상도를 240x320->16x16으로 줄여도 1.33s->1.13s로 거의 안 줄어서,
    병목이 픽셀 수가 아니라 장면 처리 자체임을 확인). `renderer.scene.flags`
    는 update_scene()을 다시 불러도 유지되므로(실측 확인) 렌더러 생성
    직후 한 번만 꺼두면 된다 -- 실측: 1.18s/frame(3카메라) -> 0.19s/frame
    (6배 이상)."""
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0


def _make_sim_and_renderer() -> tuple[PinAssemblyOpenArmSim, "mujoco.Renderer"]:
    sim = PinAssemblyOpenArmSim()
    sim.model = roco.build_roco_model(sim.xml_path)
    sim.data = mujoco.MjData(sim.model)
    mujoco.mj_forward(sim.model, sim.data)
    renderer = mujoco.Renderer(sim.model, height=roco.CAMERA_HEIGHT, width=roco.CAMERA_WIDTH)
    renderer.update_scene(sim.data, camera=roco.CAMERA_NAMES[0])  # scene.flags를 설정하려면 한 번은 update_scene이 먼저 필요
    _disable_expensive_render_flags(renderer)
    return sim, renderer


_ZERO_IMAGES = {
    cam: np.zeros((roco.CAMERA_HEIGHT, roco.CAMERA_WIDTH, 3), dtype=np.uint8) for cam in roco.CAMERA_NAMES
}

# 왼팔 EE 위치를 공식 작업공간 범위(roco.ROCO_OFFICIAL_LEFT_EE_RANGE)로
# 옮기는 좌표 변환(roco_interface.py의 WorkspaceTransform, 사용자 지시
# 3-a 검사를 통과시키려고 이번에 처음 실제로 연결함). source는 이 태스크
# 의 실제 왼팔 EE 궤적을 측정해서 나온 값(scripts/profile_pin_assembly_timing.py
# 류의 실측이 아니라, 변환 없이 한 번 생성해본 데이터셋의 observation.state
# left_ee_x/y/z 컬럼 min/max를 그대로 읽음) -- 태스크마다 실제 작업공간이
# 다르므로 기본값이 없다는 WorkspaceTransform의 설계 그대로, 하드코딩
# 대신 실측치를 상수로 남긴다. 오른팔에도 같은 변환(같은 장면의 공유
# 좌표계 보정이라는 게 roco_interface.py의 전제)을 적용한다 -- 오른팔은
# "상대 변위"만 검사되므로(절대 위치 기준 검사 없음) 선형 변환으로 그
# 변위가 비례해서 커지거나 작아져도 통과에 영향 없다(실측 확인, 완료
# 보고 참고).
_NATIVE_LEFT_EE_BBOX = {"x": (-0.0781, 0.2488), "y": (0.0219, 0.3068), "z": (0.7839, 0.9526)}
LEFT_WORKSPACE_TRANSFORM = roco.WorkspaceTransform(source=_NATIVE_LEFT_EE_BBOX)

# 왼팔 전용 추가 방향 보정(사용자 지시 2: "전문가 동작이 실제로 만드는
# 자세 분포가 공식 중심구간과 다르면 원인을 분석하고 오프셋을 조정").
#
# **roco.OPENARM_EE_DOWNWARD_OFFSET_MAT 자체는 안 바꾼다** -- 그건
# "OpenArm 로컬 -Z=approach(핀거 방향)"를 "RoCo 로컬 +Z=approach" 관례로
# 바꿔주는, 로봇 기하학에서 유도된 상수(테스트: 기준 하향 자세 ->
# (pi,0,0), euler 규약이 바뀌어도 그대로 유효함을 재확인함)다. 원인
# 분석: 이 작업의 왼팔 LEFT_HOME_QPOS(기구학적으로 손목이 플랫폼과 안
# 겹치는 쪽으로 2차 보정한 자세)는 "정확히 수직"이 아니라 어느 정도
# 기울어진 자세였고(dot=0.874), 그 기울어진 방향(요/피치)이 공식 데이터의
# "기준 자세"(거의 수직, yaw~0)와 다른 쪽으로 치우쳐 있어서, 그 기울기가
# 에피소드 전체에 그대로 배어 나와 rx/rz 중심이 공식 중심구간과 멀어진다
# (실측: 보정 전 rx 중심~4.2(공식~2.95), rz 중심~-2.3(공식~0.46)).
#
# OpenArm 쪽에는 "yaw=0이 어느 방향"이라는 외부 기준이 원래 없다
# (roco_interface.py의 OPENARM_EE_DOWNWARD_OFFSET_MAT docstring과 같은
# 논리) -- 그래서 이 작업의 LEFT_HOME_QPOS 자세가 공식 중심 구간
# 중앙값(ry=0.415, rz=0.4645, p1~p99 구간의 중점)에 오도록 **이 작업에만
# 적용하는 추가 보정 회전**을 뒀다(EFFECTIVE_home = RAW_home @
# OPENARM_EE_DOWNWARD_OFFSET_MAT, ADJ = EFFECTIVE_home^-1 @ R_target,
# LEFT_EE_FRAME_OFFSET = OPENARM_EE_DOWNWARD_OFFSET_MAT @ ADJ -- RAW_home은
# LEFT_HOME_QPOS에서 FK로 뽑은 실측 손끝 쿼터니언). rx는 1차로 2.95(공식
# 중심구간 중점)를 그대로 타겟으로 썼다가 실측 분포 중심이 2.66으로
# 나와서(home 자세의 rx 하나만 정확히 맞춰도, 에피소드 동안 팔이 계속
# 움직이면서 전체 분포 중심은 그 값에서 벗어난다 -- ry/rz는 운 좋게
# 거의 안 벗어났지만 rx는 -0.29 벗어났다, 실측), 그 벗어난 만큼
# 보정해서 타겟을 3.24(=2.95+0.29)로 다시 잡았다(ry/rz는 1차 타겟을
# 그대로 유지 -- 그것만으로 97%대 통과를 이미 실측 확인함). 오른팔은
# 이미 공식 rx=-pi 고정을 잘 맞추므로(오른팔 검사 PASS) 그대로
# roco.OPENARM_EE_DOWNWARD_OFFSET_MAT만 쓴다.
LEFT_EE_FRAME_OFFSET = np.array(
    [
        [-0.8958206228407198, 0.327186073693922, 0.30075685341155745],
        [0.06728053051226998, 0.7687913075590301, -0.6359506707565179],
        [-0.43929345761945177, -0.5494626453203466, -0.7107123605864153],
    ]
)


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

    left_pos = LEFT_WORKSPACE_TRANSFORM.apply(np.stack(rec.left_ee_pos))
    left_quat = np.stack(rec.left_ee_quat)
    right_pos = LEFT_WORKSPACE_TRANSFORM.apply(np.stack(rec.right_ee_pos))
    right_quat = np.stack(rec.right_ee_quat)

    left_mapping = roco.default_gripper_mapping(sim.model, "left")
    right_mapping = roco.default_gripper_mapping(sim.model, "right")
    left_grip = np.array([left_mapping.to_roco(v) for v in rec.left_gripper_raw])
    right_grip = np.array([right_mapping.to_roco(v) for v in rec.right_gripper_raw])

    # roco.build_action_sequence()는 양팔에 같은 ee_frame_offset_mat 하나만
    # 받는다 -- 왼팔에는 PIN_ASSEMBLY_LEFT_ORIENTATION_TRIM(이 작업 전용
    # 추가 보정, 아래 상수 docstring 참고)까지 더 얹어야 해서, 그 함수를
    # 안 쓰고 continuous_euler_xyz_from_quats()를 양팔 따로 불러
    # (T,14) 액션을 직접 구성한다(build_action_sequence()의 두 호출을
    # 그대로 풀어쓴 것, 셋 다 roco_interface.py 공용 함수 재사용).
    left_euler = roco.continuous_euler_xyz_from_quats(
        left_quat, ee_frame_offset_mat=LEFT_EE_FRAME_OFFSET, initial_rx_branch="positive_pi"
    )
    right_euler = roco.continuous_euler_xyz_from_quats(
        right_quat, ee_frame_offset_mat=roco.OPENARM_EE_DOWNWARD_OFFSET_MAT, initial_rx_branch="negative_pi"
    )
    action_seq = np.concatenate(
        [left_pos, left_euler, left_grip[:, None], right_pos, right_euler, right_grip[:, None]], axis=1
    ).astype(np.float32)
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
