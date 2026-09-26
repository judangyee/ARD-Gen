"""ARD-Gen 4단계: 궤적 실행 & 필터링 (태스크 무관, 지금은 Actuator 단독).

Stabilizer(왼팔)는 이번 검증 범위에서 완전히 제외한다 -- PIPELINE.md
"지금은 Actuator 단일 팔로만 파이프라인 검증 중" 참고. 원래 4단계는
"Actuator + Stabilizer 궤적을 같은 시뮬레이션에서 동시에 재생"하는
단계지만, 왼팔이 아직 없으므로 지금은 Actuator 궤적만 실행한다.

흐름: 1단계(태스크의 sample_scene_config)로 씬을 뽑고 -> 2-B(diffusion_gains)
로 그 씬에 맞는 게인을 생성 -> env.run_episode()로 Actuator를 실행 ->
성공 여부(run_episode가 이미 판정)로 필터링 -> 성공한 에피소드만
data/episodes/에 개별 npz로 저장한다.

--task로 태스크를 고른다(sim/task_registry.py) -- 게인 이름/개수, sim
모듈 어느 것도 이 파일에 하드코딩돼 있지 않다.

2-A/2-B와 달리 여기서부터는 **실패 에피소드를 보관하지 않는다** -- 이 뒤
5단계(언어 라벨링)의 출력물이 VLA 학습 데이터 자체이므로, 실패한 시도를
남겨봐야 학습에 쓸 수 없어서다.

사용 예:
    python pipeline/filter_episodes.py --task peg_in_hole --n-scenes 100 --scene-seed 42 \
        --diffusion-path ./data/bootstrap/diffusion_gains.pt \
        --out-dir ./data/episodes
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from pipeline.diffusion_gains import load_checkpoint, sample_gains
from pipeline.episode_io import save_episode
from sim.task_registry import list_tasks, load_task_config


# run_episode() 결과에서 이미 episode dict의 정해진 자리로 옮겨지는 필드들
# -- 그 외 스칼라 필드는 아래 main()에서 그대로 통과시킨다(태스크별 언어
# 라벨링 필드용, 위 import 아래 docstring 참고 없음 -- episode_io.py 참고).
_KNOWN_RESULT_FIELDS = {
    "trajectory", "ee_poses", "actions", "force_profile", "torque_profile",
    "forces", "torques", "insertion_depth", "final_distance", "max_force",
    "step_count", "success", "reward", "gains", "scene_config",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="peg_in_hole", choices=list_tasks())
    parser.add_argument("--n-scenes", type=int, default=100)
    parser.add_argument("--scene-seed", type=int, default=42)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="diffusion reverse-sampling(torch)의 노이즈 시드 -- 안 고정하면 씬 시드가 같아도 "
        "매 실행마다 뽑히는 게인이 달라져 성공/실패 결과가 재현되지 않는다",
    )
    parser.add_argument("--diffusion-path", type=str, default="./data/bootstrap/diffusion_gains.pt")
    parser.add_argument("--out-dir", type=str, default="./data/episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_task_config(args.task)
    os.makedirs(args.out_dir, exist_ok=True)

    # 이전 실행의 episode_*.npz가 남아있으면 이번 결과와 섞여서 개수/내용이
    # 헷갈리므로, 매 실행마다 out-dir을 비우고 새로 채운다.
    for stale in glob.glob(os.path.join(args.out_dir, "episode_*.npz")):
        os.remove(stale)

    torch.manual_seed(args.sample_seed)
    ckpt = load_checkpoint(args.diffusion_path)
    env = task.make_env()
    rng = np.random.default_rng(args.scene_seed)

    n_success = 0
    for i in range(args.n_scenes):
        scene_cfg = task.sample_scene_config(rng)
        gains = sample_gains(ckpt, task, scene_cfg)
        sim_cfg = task.to_sim_scene_config(scene_cfg)
        result = env.run_episode(gains, sim_cfg)

        if not result["success"]:
            continue

        # ee_poses/forces/torques는 물리 시뮬레이션 태스크(peg-in-hole 등)만
        # 있다 -- cap_twist처럼 궤적/힘을 다루지 않는 순수 스칼라 태스크는
        # 없어도 되므로 없으면 빈 배열로 채운다(episode_io.py는 (0,) 배열도
        # 그대로 저장/복원한다).
        episode: dict = {
            "task": task.name,
            "right_arm": {
                "traj": result.get("ee_poses", np.zeros((0,), dtype=np.float32)),
                "gain_names": task.gain_names,
                "gains": task.gains_to_vector(gains),
                "force": result.get("forces", np.zeros((0, 3), dtype=np.float32)),
                "torque": result.get("torques", np.zeros((0, 3), dtype=np.float32)),
                "role": "actuator",
            },
            "scene_config": scene_cfg,
            "success": True,
        }
        if result.get("insertion_depth") is not None:
            episode["insertion_depth"] = result["insertion_depth"]
        if result.get("forces") is not None and len(result["forces"]) > 0:
            episode["force_max"] = float(np.linalg.norm(result["forces"], axis=1).max())

        # 그 외 태스크별 스칼라 필드(예: 회전 태스크의 "direction", "quantity")를
        # 이름 하드코딩 없이 그대로 실어 나른다 -- pipeline/episode_io.py의
        # extra_* 자동 저장/복원, pipeline/language_labeling.py의
        # direction_words/quantity_words 참고.
        for key, value in result.items():
            if key in _KNOWN_RESULT_FIELDS or key in episode:
                continue
            if isinstance(value, (int, float, bool, str)):
                episode[key] = value

        out_path = os.path.join(args.out_dir, f"episode_{n_success:04d}.npz")
        save_episode(out_path, episode)
        n_success += 1

    print(
        f"[filter_episodes] task={task.name} 씬 {args.n_scenes}개 중 {n_success}개 성공 "
        f"({n_success / args.n_scenes:.1%}) -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
