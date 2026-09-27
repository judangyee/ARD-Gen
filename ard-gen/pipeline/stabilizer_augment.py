"""ARD-Gen 3단계: MimicGen 스타일 Stabilizer(왼팔) 궤적 기하 증강.

sim/stabilizer.py의 Stabilizer는 처음부터 순수 기하 계산(물체의 world
포즈 + tasks/*.yaml의 grasp_offset)만으로 목표를 정한다 -- 물리를 다시
풀거나 게인을 탐색할 필요가 없다(PIPELINE.md 3단계 설계: "물리 시뮬레이션이
필요 없어 계산 비용이 가장 낮다"). 그래서 seed 에피소드에서 실제로 기록한
왼팔 궤적(world-frame EE 위치, sim/{peg_in_hole,cap_twist}_env.py의
run_episode()가 반환하는 "left_arm_traj")을 물체 기준 좌표계로 한 번 저장해
두면, 새로운 씬의 물체 포즈가 주어졌을 때 SE(3) 변환(회전+평행이동) 하나로
새 궤적을 즉시 만들 수 있다 -- 물리 재시뮬레이션도, Stabilizer를 다시
실행하는 것도 필요 없다.

## 지금 당장은 "기하 변환 = 라이브 재계산"과 결과가 같다

sim/stabilizer.py의 Stabilizer.reset()이 이미 "물체 현재 포즈 +
grasp_offset"이라는 순수 기하 공식으로 매 에피소드 목표를 새로 계산하므로,
지금 시점에는 이 모듈로 만든 "증강된" 궤적과 Stabilizer를 씬에 맞춰
그냥 다시 돌린 결과가 수학적으로 동일하다. 이 모듈이 필요한 이유는 (1)
PIPELINE.md가 명시한 "물리 재시뮬레이션 없이 SE(3) 변환만으로 왼팔 궤적을
만든다"는 절차 자체를 실제로 구현/검증해 두는 것, (2) 앞으로 Stabilizer가
더 복잡해지면(PIPELINE.md 3단계 설계 원칙: "접촉력이 threshold를 넘는
구간에서 반발 방향 보정을 추가해 버티는 반응을 표현" 등) 궤적이 더 이상
"물체 포즈 하나로 정해지는 상수"가 아니라 매 스텝 값이 달라지는 실제
시계열이 될 텐데, 이때도 seed에서 기록해둔 시계열 형태를 그대로 SE(3)
변환하면 되도록 지금부터 "기록 -> 변환" 구조로 짜 두는 것이다.

## 두 태스크에서 이 변환이 실제로 하는 일이 다르다 (정직하게 밝힘)

- peg_in_hole: hole_socket의 world 위치가 씬마다 실제로 다르다(hole_pos_xy,
  1단계 무작위화 대상) -- 그래서 SE(3) 변환이 매번 실제로 다른 궤적을
  만들어낸다(아래 데모의 핵심 사례).
- cap_twist: bottle의 world 위치는 현재 씬 무작위화 대상이 아니다
  (resistance_torque/target_turns/disengage_ratio만 무작위화된다 --
  PIPELINE.md 1단계, tasks/cap_twist.yaml 참고). 즉 이 태스크에서는 변환이
  항상 항등(identity)이라 데모가 재미없다 -- 이건 버그가 아니라 지금
  스코프의 정직한 반영이다(bottle 위치 무작위화 자체를 추가하는 건 이번
  3단계 요청 범위 밖이라 손대지 않았다).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sim.task_registry import list_tasks, load_task_config


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """quat=(w,x,y,z) -> 3x3 회전행렬(world_from_local)."""
    import mujoco

    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def to_object_frame(world_traj: np.ndarray, object_pos: np.ndarray, object_quat: np.ndarray) -> np.ndarray:
    """world_traj (T,3, world-frame EE 위치) -> 물체 기준 좌표계 (T,3).

    local = R^T @ (world - object_pos), R=world_from_local(object_quat)."""
    r = _quat_to_mat(object_quat)
    return (world_traj - object_pos) @ r  # (v @ R) == (R.T @ v.T).T, 행벡터라 이렇게 쓴다


def to_world_frame(local_traj: np.ndarray, object_pos: np.ndarray, object_quat: np.ndarray) -> np.ndarray:
    """to_object_frame()의 역변환: local(물체 기준) -> world."""
    r = _quat_to_mat(object_quat)
    return local_traj @ r.T + object_pos


def save_stabilizer_seed(
    left_arm_traj_world: np.ndarray,
    object_pos_ref: np.ndarray,
    object_quat_ref: np.ndarray,
    out_path: str,
) -> None:
    """seed 에피소드의 왼팔 궤적(world-frame)을 물체 기준 좌표계로 변환해서
    저장한다. object_pos_ref/object_quat_ref는 그 궤적을 기록할 당시
    물체의 world 포즈(보통 에피소드 시작 시점, Actuator가 아직 아무것도
    안 건드린 때) -- sim/{peg_in_hole,cap_twist}_env.py의 reset()이 반환/
    보관하는 값을 그대로 쓰면 된다."""
    local_traj = to_object_frame(left_arm_traj_world, object_pos_ref, object_quat_ref)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        local_traj=local_traj.astype(np.float32),
        object_pos_ref=np.asarray(object_pos_ref, dtype=np.float32),
        object_quat_ref=np.asarray(object_quat_ref, dtype=np.float32),
    )


def load_stabilizer_seed(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "local_traj": data["local_traj"],
        "object_pos_ref": data["object_pos_ref"],
        "object_quat_ref": data["object_quat_ref"],
    }


def augment_to_new_object_pose(
    seed: dict[str, np.ndarray], new_object_pos: np.ndarray, new_object_quat: np.ndarray
) -> np.ndarray:
    """저장된(물체 기준) 궤적에 새 물체 포즈로의 SE(3) 변환을 적용한다 --
    물리 시뮬레이션도 Stabilizer 재실행도 필요 없이 순수 좌표 계산 하나로
    끝난다. 반환값은 새 world-frame 궤적(T,3)."""
    return to_world_frame(seed["local_traj"], new_object_pos, new_object_quat)


# ----------------------------------------------------------------------
def _object_pos_of(env: Any, object_body_name: str) -> tuple[np.ndarray, np.ndarray]:
    model, data = env._sim.model, env._sim.data
    body_id = model.body(object_body_name).id
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def demo(task_name: str, out_dir: str) -> None:
    """seed 에피소드 하나를 실제로 돌려서 왼팔 궤적을 기록 -> 물체 기준으로
    저장 -> 새 무작위 씬 몇 개의 물체 포즈로 SE(3) 변환 -> "라이브
    재계산"(Stabilizer를 그 씬에서 그냥 다시 돌린 결과)과 정확히 일치하는지
    검증한다(모듈 docstring의 "지금은 둘이 같아야 한다" 주장을 실측으로
    확인)."""
    task = load_task_config(task_name)
    object_body = task.raw["stabilizer"]["object_body"]

    seed_env = task.make_env(use_stabilizer=True)
    seed_cfg = task.default_scene_config()
    seed_gains = task.gains_from_vector(task.x0)
    # reset()을 먼저 명시적으로 불러야 data.xpos가 실제로 계산된다 -- 갓
    # 만든 MjData는 forward kinematics를 한 번도 안 돌려서 xpos가 전부
    # 0으로 남아있다(실측으로 발견: object_pos_ref가 [0,0,0]으로 찍히는
    # 버그였다). run_episode()도 내부에서 다시 reset()하므로 중복이지만
    # 씬 설정이 같아 결과에는 영향이 없다.
    seed_env.reset(seed_cfg)
    seed_pos_ref, seed_quat_ref = _object_pos_of(seed_env, object_body)
    seed_result = seed_env.run_episode(seed_gains, seed_cfg)
    if "left_arm_traj" not in seed_result:
        raise RuntimeError("stabilizer가 붙지 않았다 -- tasks/*.yaml에 stabilizer 절이 있는지 확인")

    out_path = os.path.join(out_dir, f"{task_name}_stabilizer_seed.npz")
    save_stabilizer_seed(seed_result["left_arm_traj"], seed_pos_ref, seed_quat_ref, out_path)
    print(f"[stabilizer_augment] task={task_name} seed 저장: {out_path} "
          f"(궤적 {seed_result['left_arm_traj'].shape[0]}스텝, object_pos_ref={seed_pos_ref})")

    seed = load_stabilizer_seed(out_path)

    rng = np.random.default_rng(7)
    max_err_mm = 0.0
    for i in range(5):
        shared_cfg = task.sample_scene_config(rng)
        sim_cfg = task.to_sim_scene_config(shared_cfg)

        live_env = task.make_env(use_stabilizer=True)
        live_result = live_env.run_episode(seed_gains, sim_cfg)
        # 에피소드 종료 후 새로 reset()해서, run_episode() 시작 시점에
        # grasp target 계산에 실제로 쓰인 것과 동일한 "Actuator가 아직 아무
        # 것도 안 건드린" 물체 포즈를 다시 확보한다(재현성 보장).
        live_env.reset(sim_cfg)
        live_pos_ref, live_quat_ref = _object_pos_of(live_env, object_body)

        augmented_traj = augment_to_new_object_pose(seed, live_pos_ref, live_quat_ref)

        # 두 궤적은 접근(approach) 단계 길이가 다를 수 있다(시작점-목표
        # 거리가 씬마다 달라서 수렴에 걸리는 스텝 수가 다름) -- 그래서
        # 스텝 인덱스를 그대로 맞춰 비교하면 전이 구간에서 앞뒤로 어긋나
        # 가짜 오차가 생긴다. 의미 있는 비교는 "유지(hold)" 단계, 즉 각
        # 궤적의 마지막(정지) 위치가 실제로 grasp target과 일치하는지다.
        aug_final = augmented_traj[-1]
        live_final = live_result["left_arm_traj"][-1]
        err_mm = float(np.linalg.norm(aug_final - live_final)) * 1000
        max_err_mm = max(max_err_mm, err_mm)
        print(
            f"  씬 {i}: object_pos={live_pos_ref} 증강 최종위치={aug_final} "
            f"라이브재계산 최종위치={live_final} 오차={err_mm:.4f}mm success(live)={live_result['success']}"
        )

    print(f"[stabilizer_augment] 5개 씬 전체 최대 오차: {max_err_mm:.4f}mm "
          f"(0에 가까울수록 '기하 변환만으로 재계산 없이 재현 가능'이 실측 확인됨)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="peg_in_hole", choices=list_tasks())
    parser.add_argument("--out-dir", type=str, default="./data/stabilizer_seeds")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo(args.task, args.out_dir)
