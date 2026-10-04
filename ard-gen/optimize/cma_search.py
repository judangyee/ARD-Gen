"""ARD-Gen 0단계(Seed 확보): CMA-ES로 태스크의 컨트롤러 게인을 찾는다.

--task로 태스크를 고르면(sim/task_registry.py의 TASK_REGISTRY, 실제 게인
이름/탐색 범위/평가 시나리오/성공 임계값은 tasks/{task}.yaml), 각
세대마다 CMA-ES가 제안하는 게인 후보들을 그 태스크의 평가 시나리오들에
대해 env.run_episode()로 실행하고 평균 리워드로 평가한다. 목표
리워드(threshold)에 도달하거나 최대 세대 수에 도달하면 멈추고, 최종
게인값 + 그 게인으로 다시 실행한 대표(첫 번째) 시나리오의 궤적을
seed_trajectory.npz로 저장한다.

리팩토링 이전에는 이 파일이 peg-in-hole 하나만 다뤘다(Kp_xy/Kd_xy,
EVAL_SCENE_CONFIGS 하드코딩) -- 그 값들은 이제 tasks/peg_in_hole.yaml로
옮겨졌고, --task peg_in_hole(기본값)로 실행하면 리팩토링 전과 동일한
결과가 나온다(tests/test_regression_peg_in_hole.py로 확인).

사용 예:
    python optimize/cma_search.py --task peg_in_hole --max-generations 100 --threshold 40.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cma
import numpy as np

from sim.task_registry import TaskConfig, list_tasks, load_task_config


def _build_eval_scene_configs(task: TaskConfig) -> list[dict]:
    """task.eval_scenarios(각각 default_scene_config() 위에 덮어쓸 필드
    dict)로부터 완전한 scene_config 리스트를 만든다. 원본
    optimize/cma_search.py의 EVAL_SCENE_CONFIGS 구성 방식과 동일."""
    configs = []
    for overrides in task.eval_scenarios:
        cfg = task.default_scene_config()
        cfg.update(overrides)
        configs.append(cfg)
    return configs


def evaluate_gains(envs: list, values: list[float], task: TaskConfig, eval_scene_configs: list[dict]) -> float:
    """대표 시나리오들에 대한 평균 리워드를 계산한다. gains는 평가 전에
    task.clip_gains()로 bounds 안쪽으로 자른다 -- 아래 main()의 opts에
    "bounds"를 안 넘기는 이유와 같은 근거(바로 아래 docstring) 참고."""
    gains = task.clip_gains(task.gains_from_vector(values))
    rewards = [env.run_episode(gains, cfg)["reward"] for env, cfg in zip(envs, eval_scene_configs)]
    return float(np.mean(rewards))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="peg_in_hole", choices=list_tasks())
    parser.add_argument("--max-generations", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="이 평균 리워드에 도달하면 조기 종료 (생략 시 태스크 설정의 success.reward_threshold)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-path", type=str, default="./seed_trajectory.npz")
    parser.add_argument("--curve-path", type=str, default="./convergence.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_task_config(args.task)
    threshold = args.threshold if args.threshold is not None else task.success_reward_threshold

    eval_scene_configs = _build_eval_scene_configs(task)
    primary_scene_config = eval_scene_configs[0]

    # bounds는 CMA-ES(opts["bounds"]) 쪽에 안 맡기고 evaluate_gains()/아래
    # best_overall 갱신에서 task.clip_gains()로 직접 클리핑한다 -- 레거시
    # optimize/screw_driving_cma_search.py가 이미 실측으로 발견/문서화한
    # 것과 같은 cma 라이브러리 버그(1차원에 가까운 게인 + bounds 조합이
    # 특정 세대에서 CMA의 내부 step-size가 bounds 쪽으로 수렴하면
    # es.tell() 내부 _stds_into_limits()에서 "array is 0-dimensional"
    # IndexError로 죽음, cma 4.5.0에서도 재현됨)를 범용 cma_search.py에서도
    # 그대로 재현했다(screw_driving 태스크로 실측 확인 -- cap_twist처럼
    # landscape가 매끈해서 CMA가 한 세대 안에 조기 종료되는 태스크는 이
    # 버그를 드러낼 만큼 충분히 오래 안 돌아서 지금까지 숨어있었을 뿐,
    # screw_driving의 절벽형 landscape(문턱값 아래/위로 성공/실패가
    # 갈림)에서 여러 세대를 돌리면 누구나 재현된다). bounds를 CMA-ES에서
    # 빼는 쪽이 모든 태스크에 안전한 근본 수정이라 여기서 고쳤다(기존
    # peg_in_hole/cap_twist/tacker 회귀 테스트로 동작 변화 없음을 확인).
    opts = {
        "CMA_stds": task.sigma0,
        "popsize": args.popsize,
        "maxiter": args.max_generations,
        "seed": args.seed,
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(task.x0, 1.0, opts)

    # run_episode()가 매번 MjModel을 새로 컴파일하지 않도록, 평가 시나리오
    # 개수만큼 Env를 한 번만 만들어 재사용한다 (훨씬 빠름).
    envs = [task.make_env() for _ in eval_scene_configs]

    best_reward_per_gen = []
    best_overall = {"reward": -np.inf, "gains": None}

    generation = 0
    while not es.stop() and generation < args.max_generations:
        generation += 1
        candidates = es.ask()
        rewards = [evaluate_gains(envs, values, task, eval_scene_configs) for values in candidates]
        es.tell(candidates, [-r for r in rewards])  # cma는 최소화하므로 부호 반전

        gen_best_idx = int(np.argmax(rewards))
        gen_best_reward = rewards[gen_best_idx]
        best_reward_per_gen.append(gen_best_reward)

        gen_best_gains = task.clip_gains(task.gains_from_vector(candidates[gen_best_idx]))
        if gen_best_reward > best_overall["reward"]:
            best_overall["reward"] = gen_best_reward
            best_overall["gains"] = gen_best_gains

        gains_str = ",".join(f"{v:.5f}" for v in task.gains_to_vector(gen_best_gains))
        print(
            f"[cma_search] gen {generation:3d}: best_reward_this_gen={gen_best_reward:7.3f} "
            f"best_overall={best_overall['reward']:7.3f} "
            f"gains({','.join(task.gain_names)})={gains_str}"
        )

        if best_overall["reward"] >= threshold:
            print(f"[cma_search] 목표 리워드({threshold}) 도달 -> 조기 종료 (gen {generation})")
            break
    else:
        if generation >= args.max_generations:
            print(f"[cma_search] 최대 세대 수({args.max_generations}) 도달 -> 종료")

    print(f"[cma_search] 최종 게인: {best_overall['gains']}, 평균 리워드: {best_overall['reward']:.3f}")

    # 최종 게인으로 대표(첫 번째) 시나리오를 다시 실행해 궤적을 뽑는다.
    primary_env = task.make_env()
    final_result = primary_env.run_episode(best_overall["gains"], primary_scene_config)
    print(
        f"[cma_search] 메인 시나리오 재실행: success={final_result['success']} "
        f"insertion_depth={final_result.get('insertion_depth', float('nan')):.4f} "
        f"max_force={final_result.get('max_force', float('nan')):.2f} reward={final_result['reward']:.2f}"
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    # ee_poses/actions/forces/torques는 물리 시뮬레이션 태스크(peg-in-hole
    # 등)만 있다 -- cap_twist처럼 궤적을 다루지 않는 순수 스칼라 태스크는
    # 없어도 되므로 없으면 빈 배열로 채운다.
    empty = np.zeros((0,), dtype=np.float32)
    np.savez(
        args.out_path,
        task=np.array(task.name),
        gain_names=np.array(json.dumps(task.gain_names)),
        gains=np.array(task.gains_to_vector(best_overall["gains"]), dtype=np.float32),
        ee_poses=final_result.get("ee_poses", empty),
        actions=final_result.get("actions", empty),
        forces=final_result.get("forces", empty),
        torques=final_result.get("torques", empty),
        scene_config=np.array(json.dumps(primary_scene_config)),
        success=np.array(final_result["success"]),
        insertion_depth=np.array(final_result.get("insertion_depth", np.nan), dtype=np.float32),
        reward=np.array(final_result["reward"], dtype=np.float32),
        best_reward_per_gen=np.array(best_reward_per_gen, dtype=np.float32),
    )
    print(f"[cma_search] seed_trajectory 저장: {args.out_path}")

    # 수렴 곡선 (세대별 최고 리워드)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(best_reward_per_gen) + 1), best_reward_per_gen, marker="o", markersize=3)
        plt.axhline(threshold, color="r", linestyle="--", label=f"threshold={threshold}")
        plt.xlabel("generation")
        plt.ylabel("best reward (this generation)")
        plt.title(f"CMA-ES convergence ({task.name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.curve_path, dpi=120)
        print(f"[cma_search] 수렴 곡선 저장: {args.curve_path}")
    except ImportError:
        pass

    print("[cma_search] 세대별 최고 리워드:", [round(r, 2) for r in best_reward_per_gen])


if __name__ == "__main__":
    main()
