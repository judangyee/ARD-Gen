"""ARD-Gen -> 진짜 LeRobot v3.0 포맷(parquet + h264 mp4) 변환기.

사용자 지시 "RoCo 공식 시뮬 데이터와 같은 인터페이스로 데이터를 낸다"의
3단계 -- `sim/roco_interface.py`가 만드는 44차원 state/14차원 액션/
head·left_hand·right_hand 카메라 프레임을, 실제 LeRobot v3.0 데이터셋
(`chunk-*/file-*.parquet` + `videos/{cam}/chunk-*/file-*.mp4`)으로 쓴다.

## 기존 pipeline/to_lerobot.py(npz, "v2.1-approx")는 그대로 둔다

그 파일은 이 환경에 `lerobot`/`pandas`/`pyarrow`가 없다는 전제로 만든
근사 포맷(사용자 지시 "기존 npz는 유지하되 새 경로 추가" 그대로) --
이 파일은 그 자리를 대신하지 않고 별도 경로(`--out-root`)에 새로
만든다. 이 세션에는 실제로 `pyarrow`/`av`/`lerobot` 패키지 설치를
시도해서 전부 성공했다(설치 과정은 완료 보고 참고 -- `lerobot` 자체의
의존성 해석이 매우 느려서 `--no-deps`로 하나씩 깔아야 했다).

## 왜 직접 parquet을 쓰지 않고 `lerobot.datasets.lerobot_dataset.LeRobotDataset`를
## 그대로 쓰는가

v3.0 포맷은 생각보다 복잡하다 -- 에피소드 여러 개를 하나의 parquet/mp4
파일에 이어붙이고(사용자 지시에는 없던 세부사항, 설치한 `lerobot` 소스를
직접 읽어서 확인: `chunk-{i:03d}/file-{j:03d}`, 파일 크기 임계값을
넘으면 다음 file-index로), `dataset_from_index`/`dataset_to_index`로
전역 프레임 인덱스를 추적하고, 비디오는 에피소드 경계의 `from_timestamp`
/`to_timestamp`를 따로 기록해야 조회가 된다. 이걸 손으로 다시 구현하면
"비슷하지만 미묘하게 깨진" 포맷이 되기 쉽다(이전 to_lerobot.py가
"호환이지 동일이 아니다"라고 정직하게 밝혔던 바로 그 문제) -- 그래서
이번에는 실제 패키지가 설치돼 있으므로, **그 패키지의 공식 writer를
그대로 쓴다**(참조 구현 자체를 쓰므로 스키마가 항상 정확히 맞고, 읽기도
`LeRobotDataset(repo_id, root=...)`로 그대로 검증된다). `lerobot`이 없는
환경이면 이 파일의 `create_dataset()`이 명확한 ImportError로 멈춘다 --
`pipeline/to_lerobot.py`(npz 근사)를 대신 쓰면 된다.

## 비디오 코덱: h264 (기본값 libsvtav1 아님)

사용자 지시 "h264 mp4" 그대로 -- `LeRobotDataset.create(..., vcodec="h264")`
로 명시한다(기본값은 AV1 계열 `libsvtav1`이라 그대로 두면 안 된다,
`lerobot.datasets.video_utils.VALID_VIDEO_CODECS`에 "h264"가 유효한
값으로 있는 것까지 소스로 확인).

## 추가 열(사용자 지시 그대로)

scenario/failure_type/phase(normal/failure/recovery)/loss_weight/source/
episode_success -- 전부 **프레임 단위**로 받는다(에피소드 내내 같은
값이어도, v3.0의 다른 모든 feature처럼 프레임마다 반복 저장 -- "phase"는
한 에피소드 안에서도 normal->failure->recovery로 바뀔 수 있다는 전제라
애초에 프레임 단위가 맞다). `build_features()`의 EXTRA_FEATURES 참고.

## 프레임 레이트: 10Hz (물리는 100Hz)

sim/peg_in_hole_bimanual_openarm_sim.py의 DT=0.01(100Hz 제어)이지만
RoCo 카메라는 10fps(사용자 지시) -- 데모 에피소드 생성 시 10 제어
스텝마다 한 프레임만 캡처한다(아래 `_generate_demo_episode()` 참고).

사용 예:
    python pipeline/to_lerobot_v3.py --out-root ./data/lerobot_v3/demo --n-episodes 2
    (--verify로 실제 LeRobotDataset 로더로 재로딩까지 확인)
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except ImportError as e:  # pragma: no cover -- 이 세션에서는 설치됨, 환경에 따라 없을 수 있음
    LEROBOT_AVAILABLE = False
    _IMPORT_ERROR = e

import sim.roco_interface as roco

VCODEC = "h264"

# 사용자 지시 그대로의 추가 열 -- 전부 프레임 단위(위 모듈 docstring 참고).
EXTRA_FEATURES: dict[str, dict[str, Any]] = {
    "scenario": {"dtype": "string", "shape": (1,)},
    "failure_type": {"dtype": "string", "shape": (1,)},
    "phase": {"dtype": "string", "shape": (1,)},
    "loss_weight": {"dtype": "float32", "shape": (1,)},
    "source": {"dtype": "string", "shape": (1,)},
    "episode_success": {"dtype": "bool", "shape": (1,)},
}


def build_features() -> dict[str, dict[str, Any]]:
    """sim/roco_interface.py의 STATE_NAMES/ACTION_NAMES/CAMERA_NAMES를 그대로
    LeRobot features 스키마로 옮긴다 -- 이름/차원이 두 곳에서 따로 어긋날
    일이 없도록(싱글 소스), 하드코딩하지 않고 그 모듈 상수를 읽는다."""
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (44,), "names": list(roco.STATE_NAMES)},
        "action": {"dtype": "float32", "shape": (14,), "names": list(roco.ACTION_NAMES)},
    }
    for cam in roco.CAMERA_NAMES:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (roco.CAMERA_HEIGHT, roco.CAMERA_WIDTH, 3),
            "names": ["height", "width", "channel"],
        }
    features.update(EXTRA_FEATURES)
    return features


@dataclasses.dataclass
class RocoFrame:
    """에피소드 하나의 프레임 한 장 -- write_episode()에 이 리스트를 넘긴다."""

    state: np.ndarray  # (44,)
    action: np.ndarray  # (14,)
    images: dict[str, np.ndarray]  # {"head":(H,W,3) uint8, "left_hand":..., "right_hand":...}
    task: str
    scenario: str
    failure_type: str
    phase: str
    loss_weight: float
    source: str
    episode_success: bool


def create_dataset(
    repo_id: str, root: str, fps: int = roco.CAMERA_FPS, robot_type: str = "ard_gen_openarm_bimanual"
) -> "LeRobotDataset":
    if not LEROBOT_AVAILABLE:
        raise ImportError(
            "lerobot 패키지가 없어서 LeRobot v3.0 writer를 못 쓴다 -- "
            "pipeline/to_lerobot.py(npz 근사, v2.1-approx)를 대신 쓰거나 "
            "`pip install lerobot`(+ 의존성, 완료 보고의 설치 로그 참고) 후 재시도. "
            f"원본 ImportError: {_IMPORT_ERROR}"
        )
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=build_features(),
        root=root,
        use_videos=True,
        vcodec=VCODEC,
    )


# lerobot(0.4.4) + 이 환경의 datasets/numpy 조합에서 실측으로 확인한 버그
# 우회: dtype이 숫자형(float32/bool 등)이고 shape=(1,)인 "스칼라" 커스텀
# feature는 add_frame()의 validate_frame()이 "반드시 np.ndarray에 shape
# 정확히 (1,)"를 요구하는데(validate_feature_numpy_array), save_episode()
# 가 그 값을 HF datasets.Value(dtype).encode_example()에 넘기면 내부적으로
# `float(value)`/`bool(value)`를 그 1-원소 배열에 직접 호출한다(소스
# 직접 확인, datasets/features/features.py Value.encode_example) -- 이
# 환경의 numpy(2.4.6, 아주 최신)는 "0차원이 아닌 배열의 float() 변환"을
# 완전히 금지해서(과거 numpy는 DeprecationWarning만 내고 허용했다) 여기서
# TypeError가 난다. "string" dtype(scenario/failure_type/phase/source)은
# str(value)라 이 문제가 없다 -- 실측 확인. validate_frame()을 통과하려면
# add_frame()에는 반드시 shape=(1,) 배열로 넣어야 하므로, save_episode()
# 직전에 내부 버퍼(`episode_buffer`, 공개 속성)의 숫자형 스칼라 필드만
# `.item()`으로 순수 스칼라로 바꿔치기한다 -- lerobot을 패치하지 않고
# 호출부에서만 우회(다른 numpy/datasets 버전 조합이면 이 우회가 필요
#없을 수도 있다, 그때는 이 함수가 조용히 no-op이 되도록 .item() 결과가
# 이미 스칼라면 그대로 둔다).
_SCALAR_NUMPY_EXTRA_FEATURES = ("loss_weight", "episode_success")


def write_episode(dataset: "LeRobotDataset", frames: list[RocoFrame]) -> None:
    """프레임 리스트 하나(에피소드 하나)를 dataset에 기록한다 -- add_frame()을
    프레임마다 부르고 마지막에 save_episode()로 디스크에 확정한다(실제
    parquet/mp4 쓰기는 save_episode()가 함, lerobot 내부 구현)."""
    for f in frames:
        frame_dict: dict[str, Any] = {
            "observation.state": f.state.astype(np.float32),
            "action": f.action.astype(np.float32),
            "task": f.task,
            "scenario": f.scenario,
            "failure_type": f.failure_type,
            "phase": f.phase,
            "loss_weight": np.array([f.loss_weight], dtype=np.float32),
            "source": f.source,
            "episode_success": np.array([f.episode_success], dtype=bool),
        }
        for cam in roco.CAMERA_NAMES:
            frame_dict[f"observation.images.{cam}"] = f.images[cam]
        dataset.add_frame(frame_dict)

    for key in _SCALAR_NUMPY_EXTRA_FEATURES:
        dataset.episode_buffer[key] = [v.item() if hasattr(v, "item") else v for v in dataset.episode_buffer[key]]
    dataset.save_episode()


# ---------------------------------------------------------------------------
# 데모 에피소드 생성 -- 사용자 지시 "태스크 자체는 이번에 만들지 않는다"를
# 지키면서, 이 writer/인터페이스가 실제로 끝까지 도는지 확인하려면 어떤
# 궤적이든 하나 필요하다. 기존 peg_in_hole(OpenArm) 태스크의 admittance
# 컨트롤러를 "궤적 소스"로만 재사용한다 -- 새 시뮬레이션/태스크를 만드는
# 게 아니라, 이미 있는 걸 10Hz로 샘플링해서 RoCo 스키마로 변환만 한다.
#
# 정직하게 기록: peg_in_hole은 오른팔=Actuator(움직임)/왼팔=Stabilizer
# (고정)라서, 공식 RoCo 관례(왼팔=액티브, 오른팔=정지)와 반대다(위
# sim/roco_interface.py 모듈 docstring 참고) -- 이 데모가 보여주는 건
# "파이프라인이 끝까지 돈다"는 것이지 "공식 통계를 통과한다"는 게 아니다.
# ---------------------------------------------------------------------------


def _generate_demo_episode(seed: int) -> tuple[list[RocoFrame], bool]:
    import mujoco

    from sim.peg_in_hole_bimanual_openarm_sim import (
        DT,
        MAX_STEPS,
        _MAX_SANE_DEPTH_M,
        _SUCCESS_HOLD_STEPS,
        BimanualPegInHoleOpenArmSim,
        adaptive_z_rate,
        sample_scene_config,
    )

    rng = np.random.default_rng(seed)
    cfg = sample_scene_config(rng)

    model = roco.build_roco_model("assets/peg_in_hole_bimanual_openarm.xml")
    sim = BimanualPegInHoleOpenArmSim()
    sim.model = model
    sim.data = mujoco.MjData(model)
    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]

    gripper_mappings = {side: roco.default_gripper_mapping(model, side) for side in ("left", "right")}
    renderer = mujoco.Renderer(model, height=roco.CAMERA_HEIGHT, width=roco.CAMERA_WIDTH)

    capture_every = max(1, round(1.0 / roco.CAMERA_FPS / DT))  # 100Hz 물리 -> 10Hz 캡처 = 10

    states: list[np.ndarray] = []
    images_seq: list[dict[str, np.ndarray]] = []
    left_pos_seq: list[np.ndarray] = []
    left_quat_seq: list[np.ndarray] = []
    left_grip_seq: list[float] = []
    right_pos_seq: list[np.ndarray] = []
    right_quat_seq: list[np.ndarray] = []
    right_grip_seq: list[float] = []

    def _capture() -> None:
        states.append(roco.build_state(model, sim.data, gripper_mappings))
        images_seq.append(roco.render_roco_cameras(renderer, sim.data))
        lp, lq = roco.get_ee_pose(model, sim.data, "left")
        rp, rq = roco.get_ee_pose(model, sim.data, "right")
        left_pos_seq.append(lp)
        left_quat_seq.append(lq)
        left_grip_seq.append(gripper_mappings["left"].to_roco(roco.get_gripper_raw(model, sim.data, "left")))
        right_pos_seq.append(rp)
        right_quat_seq.append(rq)
        right_grip_seq.append(gripper_mappings["right"].to_roco(roco.get_gripper_raw(model, sim.data, "right")))

    _capture()  # step 0 (초기 자세)

    prev_dx, prev_dy = 0.0, 0.0
    success = False
    hold_count = 0
    insertion_depth = 0.0
    kp_xy, kd_xy = 0.0012, 0.00024  # tasks/peg_in_hole.yaml x0 스케일과 같은 자리(검색된 게인 재사용 안 함 -- 데모용 고정값)

    for step in range(1, MAX_STEPS + 1):
        cur_peg_tip = sim.get_peg_tip_pos()
        cur_hole_center = sim.get_hole_center_pos()
        cur_dx = float(cur_hole_center[0] - cur_peg_tip[0])
        cur_dy = float(cur_hole_center[1] - cur_peg_tip[1])
        d_dx = (cur_dx - prev_dx) / DT
        d_dy = (cur_dy - prev_dy) / DT
        prev_dx, prev_dy = cur_dx, cur_dy
        delta_xy = np.array([kp_xy * cur_dx + kd_xy * d_dx, kp_xy * cur_dy + kd_xy * d_dy])
        cur_raw_depth = max(0.0, float(cur_hole_center[2] - cur_peg_tip[2]))
        z_rate = adaptive_z_rate(cur_dx, cur_dy, outer_half, cur_raw_depth, target_depth)
        delta = np.array([delta_xy[0], delta_xy[1], -z_rate])
        sim.step(delta)

        if step % capture_every == 0:
            _capture()

        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        xy_ok = abs(dx) < outer_half and abs(dy) < outer_half
        raw_depth = min(_MAX_SANE_DEPTH_M, max(0.0, float(hole_center[2] - peg_tip[2])))
        insertion_depth = raw_depth if xy_ok else 0.0
        if insertion_depth >= target_depth:
            hold_count += 1
        else:
            hold_count = 0
        if hold_count >= _SUCCESS_HOLD_STEPS:
            success = True
            if step % capture_every != 0:
                _capture()
            break

    left_pos_seq = np.stack(left_pos_seq)
    left_quat_seq = np.stack(left_quat_seq)
    right_pos_seq = np.stack(right_pos_seq)
    right_quat_seq = np.stack(right_quat_seq)
    action_seq = roco.build_action_sequence(
        left_pos_seq, left_quat_seq, np.array(left_grip_seq),
        right_pos_seq, right_quat_seq, np.array(right_grip_seq),
    )
    # 액션 관례: action[t] = 이 스텝에서 "도달해야 할" 다음 state의 EE
    # 목표(절대 목표 제어 데이터셋 표준 관례, 위 build_action_sequence()
    # docstring 참고) -- 마지막 프레임은 다음이 없으므로 자기 자신을 반복.
    action_seq = np.concatenate([action_seq[1:], action_seq[-1:]], axis=0)

    frames = []
    n = len(states)
    for i in range(n):
        phase = "normal" if success or i < n - 1 else "failure"
        frames.append(
            RocoFrame(
                state=states[i],
                action=action_seq[i],
                images=images_seq[i],
                task="insert the peg into the hole",
                scenario="peg_in_hole_openarm_demo",
                failure_type="none" if success else "timeout",
                phase=phase,
                loss_weight=1.0,
                source="ard_gen_sim",
                episode_success=success,
            )
        )
    return frames, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=str, default="./data/lerobot_v3/demo")
    parser.add_argument("--repo-id", type=str, default="ard-gen/roco_demo")
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--verify", action="store_true", help="쓴 뒤 LeRobotDataset으로 다시 읽어서 확인")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not LEROBOT_AVAILABLE:
        print(f"[to_lerobot_v3] lerobot을 import할 수 없음: {_IMPORT_ERROR}")
        print("[to_lerobot_v3] pipeline/to_lerobot.py(npz 근사)를 대신 쓰세요.")
        return

    if os.path.exists(args.out_root):
        import shutil

        shutil.rmtree(args.out_root)

    dataset = create_dataset(args.repo_id, args.out_root)
    n_success = 0
    total_frames = 0
    for ep in range(args.n_episodes):
        frames, success = _generate_demo_episode(seed=ep)
        write_episode(dataset, frames)
        total_frames += len(frames)
        n_success += int(success)
        print(f"[to_lerobot_v3] episode {ep}: {len(frames)}프레임, success={success}")

    # finalize()를 꼭 불러야 parquet 푸터 메타데이터가 확정된다 -- 안 부르면
    # LeRobotDataset(repo_id, root=...)로 다시 열 때 "meta/episodes에 parquet
    # 파일이 없다"는 에러가 난다(실측 확인, lerobot_dataset.py의 finalize()
    # docstring도 "이 함수 안 부르면 로드 못 한다"고 명시).
    dataset.finalize()

    print(f"[to_lerobot_v3] {args.n_episodes}개 에피소드, 총 {total_frames}프레임, {n_success}개 성공 -> {args.out_root}")

    if args.verify:
        reloaded = LeRobotDataset(repo_id=args.repo_id, root=args.out_root)
        print(
            f"[to_lerobot_v3] 재로딩 검증: num_episodes={reloaded.num_episodes} "
            f"num_frames={reloaded.num_frames} fps={reloaded.fps}"
        )
        sample = reloaded[0]
        print(
            f"[to_lerobot_v3] sample[0]: state.shape={tuple(sample['observation.state'].shape)} "
            f"action.shape={tuple(sample['action'].shape)} "
            f"keys={sorted(k for k in sample if k.startswith('observation.images'))}"
        )


if __name__ == "__main__":
    main()
