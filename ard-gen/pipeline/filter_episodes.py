"""ARD-Gen 4단계: 궤적 실행 & 필터링 (태스크 무관, 3단계 완료 후 양팔).

흐름: 1단계(태스크의 sample_scene_config)로 씬을 뽑고 -> 2-B(diffusion_gains)
로 그 씬에 맞는 Actuator 게인을 생성 -> env.run_episode()가 같은 물리
시뮬레이션 안에서 Stabilizer(3단계, tasks/*.yaml에 stabilizer 절이 있으면
sim.task_registry.TaskConfig.make_env()가 자동으로 붙임)와 Actuator를
동시에 실행 -> 성공 여부로 필터링 -> 성공한 에피소드만 저장한다.

## success 판정은 Actuator 것 하나뿐이고, 그걸로 충분한 이유

`env.run_episode()`가 돌려주는 `success`는 Actuator(right_arm)의 자체
판정(예: insertion_depth>=target, 회전 목표 도달)이다 -- Stabilizer
전용 성공 조건은 따로 없고 필요도 없다. 왼팔과 오른팔이 **같은
MjModel/MjData를 공유**해서 물리적으로 이미 하나로 얽혀 있기 때문에,
Stabilizer가 물체를 제대로 못 붙잡으면 물체가 밀리거나 같이 돌아가고,
그 결과가 곧바로 Actuator 쪽 성공 판정(삽입 깊이/회전 진행도)에 반영된다
-- 3단계 검증에서 실측으로 확인한 그대로다(자유물체 전환만으로 성공률이
떨어지고 Stabilizer를 붙이면 다시 올라가는 걸 직접 측정했다, PIPELINE.md
3단계 참고). 즉 "Stabilizer 실패로 인한 간접 실패"를 따로 코드로 챙길
필요가 없다 -- 이미 물리를 통해 자연스럽게 반영된다.

--task로 태스크를 고른다(sim/task_registry.py) -- 게인 이름/개수, sim
모듈 어느 것도 이 파일에 하드코딩돼 있지 않다.

2-A/2-B와 달리 여기서부터는 **실패 에피소드를 보관하지 않는다** -- 이 뒤
5단계(언어 라벨링)의 출력물이 VLA 학습 데이터 자체이므로, 실패한 시도를
남겨봐야 학습에 쓸 수 없어서다.

## 태스크별 디렉터리로 분리하는 이유

`--out-dir`를 안 주면 `./data/episodes/{task}/`에 저장한다(태스크별로
섞이지 않게) -- 여러 태스크를 순서대로 돌려서 하나의 데이터셋을 만들 때,
이전 태스크의 episode_*.npz가 다음 태스크 실행 때 지워지는(아래 "매
실행마다 out-dir을 비운다" 참고) 사고를 막는다.

사용 예:
    python pipeline/filter_episodes.py --task peg_in_hole --n-scenes 260 --scene-seed 42 \
        --diffusion-path ./data/bootstrap/diffusion_gains.pt
    # -> ./data/episodes/peg_in_hole/episode_*.npz
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
    "left_arm_traj",  # 3단계(Stabilizer) -- 위에서 episode["left_arm"]로 이미 옮김
    # 조인트 공간 state/action(공식 그리퍼 반영 태스크만, 아래서
    # episode["right_arm"]/episode["left_arm"]로 이미 옮김) -- 배열이라
    # 아래 스칼라 전용 extra_* 통과 루프에 걸리면 TypeError가 난다.
    "right_joint_pos", "right_joint_action", "right_gripper_action",
    "left_joint_pos", "left_gripper_action",
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
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="생략하면 ./data/episodes/{task}/ (태스크별 분리, 위 docstring 참고)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_task_config(args.task)
    out_dir = args.out_dir or os.path.join("./data/episodes", args.task)
    os.makedirs(out_dir, exist_ok=True)

    # 이전 실행의 episode_*.npz가 남아있으면 이번 결과와 섞여서 개수/내용이
    # 헷갈리므로, 매 실행마다 out-dir을 비우고 새로 채운다.
    for stale in glob.glob(os.path.join(out_dir, "episode_*.npz")):
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
                "action": result.get("actions", np.zeros((0, 1), dtype=np.float32)),
                "gain_names": task.gain_names,
                "gains": task.gains_to_vector(gains),
                "force": result.get("forces", np.zeros((0, 3), dtype=np.float32)),
                "torque": result.get("torques", np.zeros((0, 3), dtype=np.float32)),
                "role": "actuator",
            },
            "scene_config": scene_cfg,
            "success": True,
        }
        # 3단계(Stabilizer, 왼팔): env.run_episode()가 "left_arm_traj"를
        # 돌려줬으면(tasks/*.yaml에 stabilizer 절이 있는 태스크) 그대로
        # left_arm으로 실어 나른다 -- 같은 물리 시뮬레이션 안에서 오른팔과
        # 동시에 기록된 궤적이다(sim/stabilizer.py 참고).
        if result.get("left_arm_traj") is not None:
            episode["left_arm"] = {"traj": result["left_arm_traj"], "role": "stabilizer"}
        # 조인트 공간 state/action(공식 그리퍼 반영 태스크만) -- 있으면만
        # 실어 나른다. 없는 태스크(cap_twist 등)는 그냥 생략된다(episode_io.py
        # 의 save_episode()도 .get()으로 같은 방식으로 optional 처리한다).
        if result.get("right_joint_pos") is not None:
            episode["right_arm"]["joint_pos"] = result["right_joint_pos"]
        if result.get("right_joint_action") is not None:
            episode["right_arm"]["joint_action"] = result["right_joint_action"]
        if result.get("right_gripper_action") is not None:
            episode["right_arm"]["gripper_action"] = result["right_gripper_action"]
        if episode.get("left_arm") is not None:
            if result.get("left_joint_pos") is not None:
                episode["left_arm"]["joint_pos"] = result["left_joint_pos"]
            if result.get("left_gripper_action") is not None:
                episode["left_arm"]["gripper_action"] = result["left_gripper_action"]
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

        out_path = os.path.join(out_dir, f"episode_{n_success:04d}.npz")
        save_episode(out_path, episode)
        n_success += 1

    print(
        f"[filter_episodes] task={task.name} 씬 {args.n_scenes}개 중 {n_success}개 성공 "
        f"({n_success / args.n_scenes:.1%}) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
