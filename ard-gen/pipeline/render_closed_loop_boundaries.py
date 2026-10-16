"""5단계(ARD-VLA 폐루프 데이터) 서브태스크 경계 이미지를 **별도 프로세스**
에서 같은 시드로 재생해 캡처한다(사용자 지시 그대로) -- 물리 롤아웃
자체(pipeline/generate_closed_loop_episodes.py)는 렌더링 없이 빠르게
돌리고, 이 스크립트가 저장된 에피소드를 다시 돌리면서 top_cam/wide_cam
프레임만 뽑아 같은 npz에 덧붙인다(pipeline/episode_io.py의
add_boundary_images()).

## 재생 방법: 저장된 액션을 그대로 재실행(제어 로직 재계산 안 함)

`right_arm.action`(매 스텝 실제로 실행된 행동, 노이즈/편향/지연까지 이미
반영된 값)을 `sim.step()`에 순서대로 그대로 먹인다 -- controller_step을
다시 불러 재계산하지 않는다(행동 노이즈 같은 건 매번 다른 rng draw를
새로 뽑으면 재현이 안 되지만, 실행된 행동 자체는 이미 결정론적으로
저장돼 있으므로 그걸 재생하면 물리적으로 정확히 같은 경로가 나온다).

## 한계: 폐루프(복구가 일어난) 에피소드는 아직 지원 안 함

R2 복구는 물리 상태를 체크포인트로 되돌리는 **불연속 점프**다 --
`subtask_boundaries`에 그 사실(`recovery_applied`)은 기록돼 있지만, 그
시점의 qpos/qvel 자체는 저장돼 있지 않아서(용량이 커서 안 뒀다) 단순
액션 재생으로는 그 점프를 재현할 수 없다. 그래서 이 스크립트는
`loop_mode == "open"`(복구가 한 번도 안 일어난) 에피소드만 렌더링하고,
`closed`는 건너뛴다 -- 정직하게 로그로 남긴다(PIPELINE.md "5단계" 절
참고, 다음에 이어서 풀 사람에게: 체크포인트 시점 qpos/qvel도 같이
저장하거나, subtask_boundaries에 "jump_to_step" 같은 필드를 추가해서
재생 스크립트가 액션 재생 중간에 그 지점으로 다시 reset+재생하게
만들면 풀린다).

사용법 (osmesa 소프트웨어 렌더러 필요, 이 스크립트는 렌더링만 하고
torch는 전혀 안 써서 PyTorch Adam 세그폴트 조건과 무관하다):
    MUJOCO_GL=osmesa python pipeline/render_closed_loop_boundaries.py --dir data/closed_loop/A
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np

from pipeline.episode_io import add_boundary_images, load_episode
from sim.grasp_slip import ArmSlipConfig, GraspSlipConfig
from sim.peg_in_hole_bimanual_openarm_sim import BimanualPegInHoleOpenArmSim


def _rebuild_grasp_slip_config(disturbance: dict | None) -> GraspSlipConfig | None:
    if disturbance is None or disturbance.get("type") not in ("injected_slip_F2", "injected_slip_F3"):
        return None
    offset = np.array(disturbance["offset_m"])
    arm_cfg = ArmSlipConfig(mode="injected", injected_at_step=disturbance["step"], injected_offset_local=offset)
    return GraspSlipConfig(right=arm_cfg) if disturbance["side"] == "right" else GraspSlipConfig(left=arm_cfg)


def render_episode_boundaries(path: str, width: int = 320, height: int = 240) -> int:
    """하나의 에피소드 npz에 경계 이미지를 덧붙인다. 몇 장을 추가했는지
    돌려준다(0이면 skip된 것 -- loop_mode=='closed')."""
    episode = load_episode(path)
    if episode.get("loop_mode") == "closed":
        print(f"[skip] {path}: loop_mode=closed (복구 발생, 모듈 docstring의 한계 참고)")
        return 0
    boundaries = episode.get("subtask_boundaries")
    if not boundaries:
        print(f"[skip] {path}: subtask_boundaries 없음")
        return 0

    grasp_slip_config = _rebuild_grasp_slip_config(episode.get("disturbance"))
    sim = BimanualPegInHoleOpenArmSim(grasp_slip_config=grasp_slip_config)
    sim.reset(episode["scene_config"])

    renderer = mujoco.Renderer(sim.model, height=height, width=width)

    def capture() -> dict[str, np.ndarray]:
        frames = {}
        for camera in ("top_cam", "wide_cam"):
            renderer.update_scene(sim.data, camera=camera)
            frames[camera] = renderer.render().copy()
        return frames

    actions = episode["right_arm"]["action"]
    images: dict[str, np.ndarray] = {}
    target_steps = sorted({b["start_step"] for b in boundaries} | {b["end_step"] for b in boundaries})
    step_count = 0
    if 0 in target_steps:
        for camera, frame in capture().items():
            images[f"boundary_step0_{camera}"] = frame
    for action in actions:
        sim.step(np.asarray(action, dtype=float))
        step_count += 1
        if step_count in target_steps:
            for camera, frame in capture().items():
                images[f"boundary_step{step_count}_{camera}"] = frame

    renderer.close()
    add_boundary_images(path, images)
    print(f"[OK] {path}: {len(images)}장 캡처 (steps={sorted(target_steps)})")
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=str, required=True, help="episode_*.npz가 들어있는 디렉터리")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "episode_*.npz")))
    rendered = 0
    skipped = 0
    for path in paths:
        n = render_episode_boundaries(path, args.width, args.height)
        if n > 0:
            rendered += 1
        else:
            skipped += 1
    print(f"\n{rendered}개 렌더링, {skipped}개 skip (loop_mode=closed 또는 경계 없음) / 총 {len(paths)}개")


if __name__ == "__main__":
    main()
