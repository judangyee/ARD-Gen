"""screw driving 토크 리미터 게인(torque_limit)을 CMA-ES로 찾는다.

optimize/cma_search.py(peg_in_hole)와 같은 구조: 몇 가지 대표 시나리오에
대해 sim/screw_driving_sim.run_episode()를 돌려서 평균 리워드로 게인을
평가하고, 세대를 거듭하며 CMA-ES가 더 나은 게인을 찾아가게 한다.

## 이 탐색이 실제로 찾아낸 것 (미리 밝혀두는 정직한 결론)

처음 만든 컨트롤러는 "저항 토크가 크면 회전 속도를 늦추는" PD형
admittance(kp_torque, kd_torque)였는데, 그 버전으로 이 스크립트를 먼저
돌려봤더니 CMA-ES가 두 게인을 전부 거의 0으로 수렴시켰다(kp_torque=0
베이스라인과 리워드가 완전히 동일). 실측 추적 결과: 이 모델의 저항
토크는 회전 "속도"가 아니라 사이클(=삽입 깊이)에 달려 있어서, 느리게
돈다고 그 순간의 저항이 줄지 않았다 -- 그래서 속도 조절은 버리고, 실제
전동 드라이버처럼 "설정 토크(torque_limit)에 닿으면 그 사이클의 돌리기를
즉시 멈추는" 토크 리미터로 컨트롤 법칙 자체를 다시 짰다
(sim/screw_driving_sim.py 상단 docstring 참고).

이 리미터는 실측해보니 뚜렷한 문턱값을 보인다: torque_limit이 자연
저항의 최대치(이 시나리오에서 약 1.35~1.4) *미만*이면, 매 사이클
돌리기가 항상 조기 종료돼서 삽입이 영원히 멈춘다(15000스텝이든
40000스텝이든 깊이가 똑같은 데서 멈춤 -- 진짜 데드락이지 "느린 게"
아니다, 실제 클러치가 나사를 끝까지 못 박고 계속 미끄러지기만 하는 것과
같은 고장 모드). 문턱값 이상이면 리미터가 사실상 안 걸려서 "무제한"과
정확히 같은 스텝 수/깊이로 끝난다. 즉 이 게인 탐색의 진짜 목표는 "그
문턱값에 최대한 가깝게(그러나 위에서) 안전마진을 찾는" 것이고, 리워드가
성공(완료)에 큰 가중치를 주고 그 안에서 낮은 max_torque를 선호하게
설계돼 있어서 CMA-ES가 자연스럽게 그 경계로 수렴하도록 했다.

사용 예:
    python optimize/screw_driving_cma_search.py --max-generations 10
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
# hinge_friction_scale을 키운 "더 뻑뻑한 나사" 시나리오 -- 마찰이 클수록
# 자연 저항의 문턱값도 높아지므로, 단일 torque_limit 게인이 두 시나리오
# 모두에서 안전(완료)하려면 더 뻑뻑한 쪽 문턱값까지 커버해야 한다는 긴장을
# 만든다. seat_friction_scale은 scene_config에서 뺐다(0.02~100배를
# 흔들어도 저항이 전혀 안 변한다는 걸 실측으로 확인, sim/screw_driving_sim.py
# 참고).
EVAL_SCENE_CONFIGS: list[dict] = []
for _hinge_scale in [1.0, 5.0]:
    _cfg = _default_scene_config()
    _cfg["hinge_friction_scale"] = _hinge_scale
    EVAL_SCENE_CONFIGS.append(_cfg)

PRIMARY_SCENE_CONFIG = EVAL_SCENE_CONFIGS[0]

TORQUE_LIMIT_BOUNDS = (0.1, 4.0)


def evaluate_gains(sims: list[ScrewDrivingSim], torque_limit: float) -> float:
    """대표 시나리오들에 대한 평균 리워드를 계산한다."""
    gains = {"torque_limit": torque_limit}
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

    # 일부러 문턱값(~1.4)보다 훨씬 낮은 값에서 시작한다 -- CMA-ES가 몇 세대에
    # 걸쳐 "이 값들은 전부 완료를 못 한다"는 걸 겪으며 값을 끌어올려서 실제
    # 안전 문턱을 찾아가는 과정을 보여주기 위함.
    x0 = [0.5]
    sigma0 = 0.8

    opts = {
        "bounds": [[TORQUE_LIMIT_BOUNDS[0]], [TORQUE_LIMIT_BOUNDS[1]]],
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
        rewards = [evaluate_gains(sims, cand[0]) for cand in candidates]
        es.tell(candidates, [-r for r in rewards])

        gen_best_idx = int(np.argmax(rewards))
        gen_best_reward = rewards[gen_best_idx]
        best_reward_per_gen.append(gen_best_reward)

        if gen_best_reward > best_overall["reward"]:
            best_overall["reward"] = gen_best_reward
            best_overall["gains"] = {"torque_limit": float(candidates[gen_best_idx][0])}

        print(
            f"[screw_driving_cma_search] gen {generation:3d}: "
            f"best_reward_this_gen={gen_best_reward:8.3f} best_overall={best_overall['reward']:8.3f} "
            f"torque_limit={candidates[gen_best_idx][0]:.4f}"
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

    # 비교용: torque_limit 없음(리미터 없음)일 때 같은 시나리오 결과.
    baseline_sim = ScrewDrivingSim()
    baseline_result = _run_episode_with_sim(
        baseline_sim, {"torque_limit": float("inf")}, PRIMARY_SCENE_CONFIG
    )
    print(
        f"[screw_driving_cma_search] 리미터 없음 베이스라인: success={baseline_result['success']} "
        f"max_torque={baseline_result['max_torque']:.3f} "
        f"step_count={baseline_result['step_count']} reward={baseline_result['reward']:.2f}"
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    np.savez(
        args.out_path,
        depth_profile=final_result["depth_profile"],
        torque_profile=final_result["torque_profile"],
        rate_profile=final_result["rate_profile"],
        gains=np.array([best_overall["gains"]["torque_limit"]], dtype=np.float32),
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
        plt.title("screw_driving CMA-ES convergence (torque_limit)")
        plt.tight_layout()
        plt.savefig(args.curve_path, dpi=120)
        print(f"[screw_driving_cma_search] 수렴 곡선 저장: {args.curve_path}")
    except ImportError:
        pass

    print("[screw_driving_cma_search] 세대별 최고 리워드:", [round(r, 2) for r in best_reward_per_gen])


if __name__ == "__main__":
    main()
