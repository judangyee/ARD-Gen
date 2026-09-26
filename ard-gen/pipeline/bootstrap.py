"""ARD-Gen 2-A단계: 컨트롤러 게인 부트스트래핑 (태스크 무관).

0단계에서 CMA-ES로 찾은 seed 게인 하나만으로는 2-B단계의 diffusion
모델을 학습시킬 수 없다 -- 학습 데이터가 1개뿐이기 때문이다. 그래서 이
스크립트가 seed 게인 주변에 넓은 무작위 노이즈를 줘서 시뮬레이션을
수백~수천 번 반복 실행하고, (씬 조건, 게인, 성공여부, force_profile)
기록을 대량으로 쌓는다. 이 기록이 2-B단계 diffusion 모델의 학습
데이터가 된다.

성공/실패 에피소드를 **둘 다** 저장한다 -- diffusion이 "어떤 게인이
실패하는지"도 학습해야, 새 씬 조건에서 성공 확률 높은 쪽으로 샘플링할
수 있기 때문이다.

--task로 태스크를 고른다(sim/task_registry.py). 게인 이름/개수는 태스크
설정(tasks/{task}.yaml)에서 오므로, Kp_xy/Kd_xy 같은 이름을 이 파일에
하드코딩하지 않는다 -- 저장되는 npz는 게인을 (N, n_gains) 배열 +
gain_names로, 씬 조건은 필드 이름별 배열로 담는다(diffusion_gains.py가
그대로 읽음).

사용 예:
    python pipeline/bootstrap.py --task peg_in_hole --seed-path ./seed_trajectory.npz \
        --n-trials 1000 --seed 0 --out-path ./data/bootstrap/bootstrap_dataset.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sim.task_registry import list_tasks, load_task_config

# seed 게인에 곱할 노이즈 배율 범위. 게인마다 독립적으로 적용한다(같은
# 배율을 모든 게인에 동시에 곱하면 탐색이 seed 방향의 1차원 직선으로만
# 퍼져서, diffusion 학습 데이터로 쓰기엔 다양성이 부족해진다).
_GAIN_NOISE_RANGE = (0.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="peg_in_hole", choices=list_tasks())
    parser.add_argument("--seed-path", type=str, default="./seed_trajectory.npz")
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="씬/게인 노이즈 샘플링용 RNG 시드")
    parser.add_argument("--out-path", type=str, default="./data/bootstrap/bootstrap_dataset.npz")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _load_seed_gains(seed_path: str, task) -> dict[str, float]:
    seed_data = np.load(seed_path, allow_pickle=False)
    if "gain_names" in seed_data.files:
        names = json.loads(str(seed_data["gain_names"]))
    else:
        # 리팩토링 전에 저장된 seed_trajectory.npz(이름 없이 값 순서만
        # 있음) 호환 -- 태스크 설정의 게인 순서와 같다고 가정한다.
        names = task.gain_names
    values = seed_data["gains"]
    return {name: float(v) for name, v in zip(names, values)}


def main() -> None:
    args = parse_args()
    task = load_task_config(args.task)

    seed_gains = _load_seed_gains(args.seed_path, task)
    print(f"[bootstrap] task={task.name} seed 게인: {seed_gains}")
    print(f"[bootstrap] 노이즈 범위: seed x [{_GAIN_NOISE_RANGE[0]}, {_GAIN_NOISE_RANGE[1]}] (게인별 독립)")

    rng = np.random.default_rng(args.seed)
    env = task.make_env()

    # 씬 조건 필드는 태스크마다 이름/개수/모양이 다르므로, 첫 샘플로
    # 각 필드의 shape을 재서 그 이름 그대로 컬럼 배열을 만든다.
    first_scene = task.sample_scene_config(rng)
    scene_columns: dict[str, np.ndarray] = {}
    for field_name, value in first_scene.items():
        shape = (args.n_trials,) if not isinstance(value, (tuple, list)) else (args.n_trials, len(value))
        scene_columns[field_name] = np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(args.seed)  # 위에서 하나 소모했으니 다시 시작

    gains_matrix = np.zeros((args.n_trials, len(task.gain_names)), dtype=np.float32)
    successes = np.zeros(args.n_trials, dtype=bool)
    force_max = np.zeros(args.n_trials, dtype=np.float32)
    force_mean = np.zeros(args.n_trials, dtype=np.float32)
    insertion_depths = np.full(args.n_trials, np.nan, dtype=np.float32)

    n_success_so_far = 0
    for i in range(args.n_trials):
        shared_cfg = task.sample_scene_config(rng)
        sim_cfg = task.to_sim_scene_config(shared_cfg)

        gains = task.clip_gains(
            {name: value * rng.uniform(*_GAIN_NOISE_RANGE) for name, value in seed_gains.items()}
        )

        result = env.run_episode(gains, sim_cfg)

        for field_name, value in shared_cfg.items():
            scene_columns[field_name][i] = value
        gains_matrix[i] = task.gains_to_vector(gains)
        successes[i] = result["success"]
        if "forces" in result:
            force_mag = np.linalg.norm(result["forces"], axis=1)
            force_max[i] = float(force_mag.max())
            force_mean[i] = float(force_mag.mean())
        if result.get("insertion_depth") is not None:
            insertion_depths[i] = result["insertion_depth"]

        if result["success"]:
            n_success_so_far += 1

        if (i + 1) % args.log_every == 0:
            print(f"[bootstrap] {i + 1}/{args.n_trials}  누적 성공률={n_success_so_far / (i + 1):.1%}")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    np.savez(
        args.out_path,
        task=np.array(task.name),
        seed_gain_names=np.array(json.dumps(task.gain_names)),
        seed_gains=np.array(task.gains_to_vector(seed_gains), dtype=np.float32),
        gain_names=np.array(json.dumps(task.gain_names)),
        gains=gains_matrix,
        success=successes,
        force_max=force_max,
        force_mean=force_mean,
        insertion_depth=insertion_depths,
        **scene_columns,
    )
    print(f"[bootstrap] 저장: {args.out_path} ({args.n_trials}건)")

    _print_summary(successes, gains_matrix, task.gain_names, scene_columns)


def _print_summary(
    successes: np.ndarray,
    gains_matrix: np.ndarray,
    gain_names: list[str],
    scene_columns: dict[str, np.ndarray],
) -> None:
    n = len(successes)
    n_success = int(successes.sum())
    print()
    print("=" * 60)
    print(f"전체 성공률: {n_success / n:.1%} ({n_success}/{n})")

    if n_success > 0:
        print()
        print("성공 게인 분포:")
        for j, name in enumerate(gain_names):
            col = gains_matrix[successes, j]
            print(f"  {name}: 평균={col.mean():.6g}  표준편차={col.std():.6g}")

    fail = ~successes
    if fail.sum() > 0 and n_success > 0 and "clearance_m" in scene_columns:
        print()
        print("실패가 어떤 조건에서 몰리는지 (clearance 중심):")
        clearances = scene_columns["clearance_m"]
        quartile_edges = np.quantile(clearances, [0.0, 0.25, 0.5, 0.75, 1.0])
        for lo, hi in zip(quartile_edges[:-1], quartile_edges[1:]):
            mask = (clearances >= lo) & (clearances <= hi)
            if mask.sum() == 0:
                continue
            rate = successes[mask].mean()
            print(f"  clearance [{lo * 1000:.2f}~{hi * 1000:.2f}]mm ({mask.sum()}건): 성공률 {rate:.1%}")
        print(f"  성공군 clearance 평균: {clearances[successes].mean() * 1000:.2f}mm")
        print(f"  실패군 clearance 평균: {clearances[fail].mean() * 1000:.2f}mm")
    print("=" * 60)


if __name__ == "__main__":
    main()
