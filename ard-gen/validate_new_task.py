"""RoboTwin 2.0 이식 Part 3-4: 새 태스크 자동 검증 루프 (0단계 -> 1단계 -> 2-A단계).

scaffold_task.py(Part 3-2)로 만든 새 태스크가 lint(Part 3-3)만 통과하는
수준을 넘어, 실제로 "그럭저럭 쓸만한 난이도"인지까지 자동으로 확인한다.

1단계(공유 씬 샘플링)는 별도 실행 파일이 없다 -- sim.task_registry.
TaskConfig.sample_scene_config()이 그 자체이고, 2-A(pipeline/bootstrap.py)
가 매 trial마다 그걸 호출하므로 이 스크립트를 실행하면 자동으로 함께
실행된다(그래서 제목이 "0단계 -> 2-A단계"가 아니라 "0 -> 1 -> 2-A"다).

흐름:
  0단계: optimize/cma_search.py로 seed 게인을 짧게 탐색한다(검증
         목적이라 완전 수렴시키지 않는다 -- --max-generations/--popsize를
         낮게 둔다).
  2-A단계: pipeline/bootstrap.py로 그 seed 주변에 무작위 노이즈를 줘서
           N회 평가, 성공률을 집계한다(이 호출 자체가 1단계
           sample_scene_config()를 매 trial 반복 호출한다).
  판정: 성공률이 20% 미만이면 "거의 항상 실패"(게인/씬 범위가 너무
        가혹함), 95% 초과면 "거의 항상 성공"(게인/씬 범위가 너무
        쉬움) -- 둘 다 2-B단계 diffusion이 성공/실패 경계를 학습할
        신호가 부족하다는 뜻이라 경고한다(에러는 아님 -- 새 태스크의
        tasks/{name}.yaml 게인 bounds나 scene_config 범위를 사람이
        다시 조정하라는 신호).

사용법:
    python validate_new_task.py --task demo_block
    python validate_new_task.py --task demo_block --n-trials 60 --max-generations 15
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from sim.task_registry import list_tasks

_LOW_SUCCESS_THRESHOLD = 0.20
_HIGH_SUCCESS_THRESHOLD = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=list_tasks())
    parser.add_argument("--max-generations", type=int, default=15, help="0단계 CMA-ES 세대 수(검증용이라 짧게)")
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument("--n-trials", type=int, default=60, help="2-A단계(bootstrap) 평가 횟수")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_stage0(task_name: str, max_generations: int, popsize: int, seed: int, seed_out_path: str) -> None:
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "optimize", "cma_search.py"),
        "--task", task_name,
        "--max-generations", str(max_generations),
        "--popsize", str(popsize),
        "--seed", str(seed),
        "--out-path", seed_out_path,
        "--curve-path", os.path.join(tempfile.gettempdir(), f"{task_name}_convergence_validate.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"0단계(cma_search.py) 실패 (task={task_name!r})")


def run_stage2a(task_name: str, seed_path: str, n_trials: int, seed: int, out_path: str) -> float:
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "pipeline", "bootstrap.py"),
        "--task", task_name,
        "--seed-path", seed_path,
        "--n-trials", str(n_trials),
        "--seed", str(seed),
        "--out-path", out_path,
        "--log-every", str(max(n_trials, 1)),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"2-A단계(bootstrap.py) 실패 (task={task_name!r})")
    data = np.load(out_path, allow_pickle=False)
    return float(data["success"].mean())


def validate(task_name: str, max_generations: int, popsize: int, n_trials: int, seed: int) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        seed_path = os.path.join(tmpdir, "seed_trajectory.npz")
        bootstrap_path = os.path.join(tmpdir, "bootstrap_dataset.npz")

        print(f"[validate_new_task] task={task_name} -- 0단계(CMA-ES seed 탐색) 시작")
        run_stage0(task_name, max_generations, popsize, seed, seed_path)

        print(f"[validate_new_task] task={task_name} -- 1단계+2-A단계(bootstrap, n_trials={n_trials}) 시작")
        success_rate = run_stage2a(task_name, seed_path, n_trials, seed, bootstrap_path)

    print(f"[validate_new_task] task={task_name} 성공률: {success_rate:.1%}")
    if success_rate < _LOW_SUCCESS_THRESHOLD:
        print(
            f"[validate_new_task] 경고: 성공률({success_rate:.1%})이 {_LOW_SUCCESS_THRESHOLD:.0%} 미만 -- "
            "게인 bounds/x0가 너무 빡빡하거나 scene_config 범위가 너무 가혹할 수 있다. "
            "tasks/{0}.yaml의 gains.bounds/x0나 sample_scene_config() 범위를 조정해볼 것.".format(task_name)
        )
    elif success_rate > _HIGH_SUCCESS_THRESHOLD:
        print(
            f"[validate_new_task] 경고: 성공률({success_rate:.1%})이 {_HIGH_SUCCESS_THRESHOLD:.0%} 초과 -- "
            "태스크가 너무 쉬워서 diffusion이 성공/실패 경계를 학습할 신호가 부족할 수 있다. "
            "scene_config 범위를 넓히거나 success 기준을 더 엄격히 할 것.".format(task_name)
        )
    else:
        print(f"[validate_new_task] 성공률이 적정 범위({_LOW_SUCCESS_THRESHOLD:.0%}~{_HIGH_SUCCESS_THRESHOLD:.0%}) 안에 있음 -- 통과")
    return success_rate


def main() -> None:
    args = parse_args()
    validate(args.task, args.max_generations, args.popsize, args.n_trials, args.seed)


if __name__ == "__main__":
    main()
