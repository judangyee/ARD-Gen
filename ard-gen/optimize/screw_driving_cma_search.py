"""screw driving admittance controller 게인(kp_torque, kd_torque)을 CMA-ES로 찾는다.

optimize/cma_search.py(peg_in_hole)와 같은 구조: 몇 가지 대표 시나리오에
대해 sim/screw_driving_sim.run_episode()를 돌려서 평균 리워드로 게인을
평가하고, 세대를 거듭하며 CMA-ES가 더 나은 게인을 찾아가게 한다.

## 이 탐색이 실제로 찾아낸 것 (미리 밝혀두는 정직한 결론)

돌리기 전에 먼저 실측한 사실: kp_torque를 0 -> 2.0까지 올려도 max_torque는
거의 안 변한다(1.38 -> 1.37, 오차 수준). 이 모델의 저항 토크는 회전
"속도"가 아니라 사이클(=삽입 깊이)에 달려 있어서(뒤 사이클일수록 저항이
누적돼서 커짐 -- sim/screw_driving_sim.py 상단 docstring 참고), 느리게
돈다고 그 순간의 저항 자체가 줄어들지 않는다. 오히려 kp_torque를 올리면
스텝 수가 늘어나서(더 오래 걸려서) mean_abs_torque가 살짝 올라간다(저항이
큰 뒷부분 사이클에 상대적으로 더 오래 머무르므로). 그래서 리워드를
mean_abs_torque 기준으로 정직하게 설계하면(mean_torque가 아니라 max_torque로
페널티를 줬다면 이 효과가 가려졌을 것) CMA-ES는 아마 kp_torque를 0에 가깝게
수렴시킬 것이다 -- 이건 버그가 아니라 "이 모델에서는 회전 속도를 늦추는
방식의 admittance가 저항을 줄여주지 않는다"는 실측 결론이다. RATE_MIN>0을
둬서 최소 속도는 보장하니 안전(완전 정지/역전 없음)하다는 점은 여전히
유효하다.

사용 예:
    python optimize/screw_driving_cma_search.py --max-generations 10 --threshold 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cma
import numpy as np

from sim.screw_driving_sim import ScrewDrivingSim, _default_scene_config, _run_episode_with_sim

# 대표 평가 시나리오 2개: 기본 저항과, 실측상 저항에 유의미한 영향을 주는
# hinge_friction_scale을 키운 "더 뻑뻑한 나사" 시나리오. seat_friction_scale은
# scene_config에서 뺐다(0.02~100배를 흔들어도 저항이 전혀 안 변한다는 걸
# 실측으로 확인, sim/screw_driving_sim.py 참고).
EVAL_SCENE_CONFIGS: list[dict] = []
for _hinge_scale in [1.0, 5.0]:
    _cfg = _default_scene_config()
    _cfg["hinge_friction_scale"] = _hinge_scale
    EVAL_SCENE_CONFIGS.append(_cfg)

PRIMARY_SCENE_CONFIG = EVAL_SCENE_CONFIGS[0]

KP_BOUNDS = (0.0, 3.0)
KD_BOUNDS = (0.0, 1.0)


def evaluate_gains(sims: list[ScrewDrivingSim], kp_torque: float, kd_torque: float) -> float:
    """대표 시나리오들에 대한 평균 리워드를 계산한다."""
    gains = {"kp_torque": kp_torque, "kd_torque": kd_torque}
    rewards = []
    for sim, cfg in zip(sims, EVAL_SCENE_CONFIGS):
        result = _run_episode_with_sim(sim, gains, cfg)
        rewards.append(result["reward"])
    return float(np.mean(rewards))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=1e9, help="이 평균 리워드에 도달하면 조기 종료")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-path", type=str, default="./screw_driving_gains.npz")
    parser.add_argument("--curve-path", type=str, default="./screw_driving_convergence.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x0 = [0.5, 0.1]
    sigma0 = 1.0

    opts = {
        "bounds": [[KP_BOUNDS[0], KD_BOUNDS[0]], [KP_BOUNDS[1], KD_BOUNDS[1]]],
        "CMA_stds": [0.5, 0.15],
        "popsize": args.popsize,
        "maxiter": args.max_generations,
        "seed": args.seed,
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    sims = [ScrewDrivingSim() for _ in EVAL_SCENE_CONFIGS]

    best_reward_per_gen = []
    best_overall = {"reward": -np.inf, "gains": None}

    generation = 0
    while not es.stop() and generation < args.max_generations:
        generation += 1
        candidates = es.ask()
        rewards = [evaluate_gains(sims, kp, kd) for kp, kd in candidates]
        es.tell(candidates, [-r for r in rewards])

        gen_best_idx = int(np.argmax(rewards))
        gen_best_reward = rewards[gen_best_idx]
        best_reward_per_gen.append(gen_best_reward)

        if gen_best_reward > best_overall["reward"]:
            best_overall["reward"] = gen_best_reward
            best_overall["gains"] = {
                "kp_torque": float(candidates[gen_best_idx][0]),
                "kd_torque": float(candidates[gen_best_idx][1]),
            }

        print(
            f"[screw_driving_cma_search] gen {generation:3d}: "
            f"best_reward_this_gen={gen_best_reward:8.3f} best_overall={best_overall['reward']:8.3f} "
            f"gains(kp,kd)={candidates[gen_best_idx][0]:.4f},{candidates[gen_best_idx][1]:.4f}"
        )

        if best_overall["reward"] >= args.threshold:
            print(f"[screw_driving_cma_search] 목표 리워드({args.threshold}) 도달 -> 조기 종료 (gen {generation})")
            break
    else:
        if generation >= args.max_generations:
            print(f"[screw_driving_cma_search] 최대 세대 수({args.max_generations}) 도달 -> 종료")

    print(f"[screw_driving_cma_search] 최종 게인: {best_overall['gains']}, 평균 리워드: {best_overall['reward']:.3f}")

    primary_sim = ScrewDrivingSim()
    final_result = _run_episode_with_sim(primary_sim, best_overall["gains"], PRIMARY_SCENE_CONFIG)
    print(
        f"[screw_driving_cma_search] 메인 시나리오 재실행: success={final_result['success']} "
        f"insertion_depth_mm={final_result['insertion_depth'] * 1000:.2f} "
        f"mean_abs_torque={final_result['mean_abs_torque']:.4f} "
        f"max_torque={final_result['max_torque']:.3f} "
        f"step_count={final_result['step_count']} reward={final_result['reward']:.2f}"
    )

    # 비교용: kp_torque=0(admittance 없음)일 때 같은 시나리오 결과.
    baseline_sim = ScrewDrivingSim()
    baseline_result = _run_episode_with_sim(baseline_sim, {"kp_torque": 0.0, "kd_torque": 0.0}, PRIMARY_SCENE_CONFIG)
    print(
        f"[screw_driving_cma_search] kp_torque=0 베이스라인: success={baseline_result['success']} "
        f"mean_abs_torque={baseline_result['mean_abs_torque']:.4f} "
        f"step_count={baseline_result['step_count']} reward={baseline_result['reward']:.2f}"
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    np.savez(
        args.out_path,
        depth_profile=final_result["depth_profile"],
        torque_profile=final_result["torque_profile"],
        rate_profile=final_result["rate_profile"],
        gains=np.array(
            [best_overall["gains"]["kp_torque"], best_overall["gains"]["kd_torque"]], dtype=np.float32
        ),
        scene_config=np.array(json.dumps(PRIMARY_SCENE_CONFIG)),
        success=np.array(final_result["success"]),
        insertion_depth=np.array(final_result["insertion_depth"], dtype=np.float32),
        mean_abs_torque=np.array(final_result["mean_abs_torque"], dtype=np.float32),
        reward=np.array(final_result["reward"], dtype=np.float32),
        best_reward_per_gen=np.array(best_reward_per_gen, dtype=np.float32),
    )
    print(f"[screw_driving_cma_search] 게인/궤적 저장: {args.out_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(best_reward_per_gen) + 1), best_reward_per_gen, marker="o", markersize=3)
        plt.xlabel("generation")
        plt.ylabel("best reward (this generation)")
        plt.title("screw_driving CMA-ES convergence")
        plt.tight_layout()
        plt.savefig(args.curve_path, dpi=120)
        print(f"[screw_driving_cma_search] 수렴 곡선 저장: {args.curve_path}")
    except ImportError:
        pass

    print("[screw_driving_cma_search] 세대별 최고 리워드:", [round(r, 2) for r in best_reward_per_gen])


if __name__ == "__main__":
    main()
