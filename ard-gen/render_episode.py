"""4/5단계에서 저장된 완성 에피소드(data/episodes/{task}/episode_*.npz)를
그 안에 저장된 (게인, scene_config) 그대로 다시 실행하면서 mp4로 렌더링한다.

## env.step()을 몽키패치해서 캡처하는 이유

`PegInHoleEnv`/`CapTwistEnv`의 실제 제어 루프(admittance controller,
Stabilizer 동시 실행 등)를 이 파일에 다시 옮겨 적으면 로직이 두 군데로
갈라져서 나중에 한쪽만 고치는 실수가 생기기 쉽다. 대신 `env.step()`을
프레임 캡처가 붙은 버전으로 감싸고 `env.run_episode()`를 그대로 호출한다
-- 그러면 실제 파이프라인(pipeline/filter_episodes.py)이 쓰는 것과
정확히 같은 코드 경로가 실행되면서, 매 스텝마다 프레임이 찍힌다.

cap_twist는 assets/cap_twist.xml에 카메라가 정의돼 있지 않아서(단순
bottle+cap 모델이라 필요 없었음), 자유 시점 카메라를 하나 만들어 쓴다.
peg_in_hole은 assets/peg_in_hole.xml에 이미 정의된 top_cam을 그대로 쓴다.

사용 예:
    MUJOCO_GL=osmesa python render_episode.py --task peg_in_hole \
        --episode-path data/episodes/peg_in_hole/episode_0000.npz --out ./episode_0000.mp4
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import mujoco
import numpy as np

import sys

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.episode_io import load_episode
from sim.task_registry import list_tasks, load_task_config

# 카메라 이름이 있는 태스크만 여기 추가한다 -- 없으면 아래 기본 자유
# 시점 카메라로 대체된다(tacker도 top_cam이 있지만 workpiece가 매 씬 다른
# xy/yaw에 놓이므로 mode="targetbody"가 알아서 따라간다).
_NAMED_CAMERAS = {"peg_in_hole": "top_cam", "tacker": "top_cam"}


def _camera_for(task_name: str, sim):
    if task_name in _NAMED_CAMERAS:
        return _NAMED_CAMERAS[task_name]
    cam = mujoco.MjvCamera()
    cam.lookat = [0, 0, 0.08]
    cam.distance = 0.35
    cam.azimuth = 120
    cam.elevation = -20
    return cam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, required=True, choices=list_tasks())
    parser.add_argument("--episode-path", type=str, required=True)
    parser.add_argument("--out", type=str, default="./episode.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--freeze-frames", type=int, default=20, help="끝난 뒤 결과를 잠깐 정지 화면으로 보여줄 프레임 수")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = load_episode(args.episode_path)
    task = load_task_config(args.task)

    gains = dict(zip(episode["right_arm"]["gain_names"], (float(g) for g in episode["right_arm"]["gains"])))
    sim_cfg = task.to_sim_scene_config(episode["scene_config"])
    print(f"[render_episode] task={args.task} gains={gains}")
    print(f"[render_episode] scene_config(1단계)={episode['scene_config']}")
    print(f"[render_episode] language={episode.get('language')}")

    # use_stabilizer=True를 강제로 넘기지 않는다 -- make_env()는 어차피
    # tasks/{task}.yaml에 stabilizer 절이 있으면 기본값으로 이미 켠다.
    # tacker(OpenArm, 왼팔이 항상 붙어있는 실제 팔이라 그 kwarg 자체가
    # 없음) 같은 태스크의 Env는 이 kwarg를 아예 안 받아서 강제로 넘기면
    # TypeError가 났다(실측 확인).
    env = task.make_env()
    sim = env._sim
    renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)
    camera = _camera_for(args.task, sim)

    frames: list[np.ndarray] = []

    def capture() -> None:
        renderer.update_scene(sim.data, camera=camera)
        frames.append(renderer.render().copy())

    env.reset(sim_cfg)
    capture()  # grasp/approach 완료 직후, Actuator가 아직 시작하기 전

    original_step = env.step

    def step_and_capture(action):
        original_step(action)
        capture()

    env.step = step_and_capture

    result = env.run_episode(gains, sim_cfg)

    for _ in range(args.freeze_frames):
        frames.append(frames[-1])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, quality=8)
    print(
        f"[render_episode] success={result['success']} steps={result['step_count']} "
        f"frames={len(frames)} -> {args.out}"
    )
    renderer.close()


if __name__ == "__main__":
    main()
