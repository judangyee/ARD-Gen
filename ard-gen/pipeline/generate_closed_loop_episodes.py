"""ARD-VLA 폐루프 Judge 학습/실험에 쓸 소규모 데이터를 생성한다.

**대규모 생성은 하지 않는다**(사용자 지시) -- 교란 세트(B/C/A)별로
수십 개만 만들어서 통합 데이터 스키마(pipeline/episode_io.py의
closed_loop/subtask_boundaries/disturbance 확장, 5단계)·분할·렌더링이
끝까지 동작하는지만 확인한다.

## 경계 이미지: 방법 A (생성 중 바로 캡처, `--render`)

`pipeline/render_closed_loop_boundaries.py`(별도 프로세스, 저장된 액션을
재생)는 R2 복구의 불연속 점프를 재현할 수 없어서 `loop_mode=="closed"`
에피소드를 전부 skip한다 -- 폐루프 후속 작업에서 가장 중요한 "복구가
일어난" 에피소드가 바로 이미지가 없는 쪽이라는 문제. 그래서
`--render`를 주면 이 스크립트가 서브태스크 경계(+ R2 점프) 시점에
`PegInHoleClosedLoopEnv.render_frames()`로 바로 캡처한다(어댑터에 이미
있던 `_render_camera()` 재사용) -- 재생이 아니라 실제 생성 중 캡처라
점프도 그대로 커버됨. 이 방식을 쓰려면 생성 프로세스가 torch를
import하면 안 된다(`MUJOCO_GL=osmesa` + PyTorch Adam 생성 조합이
세그폴트를 낸다, PIPELINE.md 참고) -- `assert_no_torch_imported()`가
그 전제를 확인한다. 키 형식은 `episode_io.py`가 이미 문서화해둔
"boundary_{idx}_{camera}_{start|end}"(subtask_boundaries 리스트
인덱스 기준 -- 점프로 전역 스텝 수가 꼬여도 유일하다).

closedloop/peg_in_hole_adapter.py(4단계)의 `PegInHoleClosedLoopEnv`/
`ScriptedOraclePolicy`를 재사용한다 -- ARD-VLA의 `DecisionLogic`은 안
쓴다. 이 스크립트 자체가 아주 단순한 로컬 규칙(실패 감지 시 R2 복구를
최대 1회)으로 폐루프 흐름을 만든다 -- 외부 저장소 없이도 독립적으로
돌아가야 하기 때문(사용자 지시 "Judge 학습/실험 비교는 범위 밖" -- 이
스크립트는 그 실험에 쓸 *데이터*를 만들 뿐, ARD-VLA의 결정 로직을 그대로
구동하는 게 목적이 아니다. 실제 ARD-VLA 결정 로직 구동 데모는
tests/test_closed_loop_adapter.py의 통합 테스트가 이미 보여준다).

## 교란 세트

- **B(학습용)**: 매 에피소드 `_B_DISTURBANCE_TYPES` 중 하나를 무작위로
  골라 적용(행동 노이즈/xy 목표 편향/나쁜 게인/좁은 clearance/높은
  마찰/주입 미끄러짐 F2 또는 F3).
- **C(테스트 전용, 학습에 안 씀)**: B에 없는 범위(마찰/clearance를 B보다
  더 극단으로, 행동 지연) + `--render --visual-disturbance`를 같이 주면
  렌더링 단계 시각 교란(조명 방향/세기, top_cam/wide_cam 위치, 구멍/페그
  색 -- `VisualDisturbanceConfig`/`apply_visual_disturbance()`)도 C에만
  적용된다(기본 꺼짐, A/B 이미지는 그대로). Part 1의 전체 시각
  randomization(텍스처/배경 등)은 여전히 `ard-gen-v2` 브랜치 전용이고
  이번 작업은 병합하지 않는다(사용자 지시, 0단계 브랜치 관계 절 참고) --
  여기서는 "C만 추가로 더 흔든다"는 좁은 의미의 시각 교란만 다룬다.
- **A(자연 실패)**: 교란 없음, 씬 난이도만 높임(peg_init_offset_xy
  반경을 0.009~0.020m·각도를 전체 원으로 넓힘 -- PIPELINE.md "5단계"
  절에 실측 기록된 대로 개루프 실패율 20% 근방).

## 복구 (이 스크립트의 단순화 규칙)

서브태스크가 자연 완료가 아니라 실패/timeout으로 끝나면, 그 노드를 R2
(시작 자세+그립 복원)로 **최대 1회**만 복구를 시도한다(소규모 생성이라
재시도를 더 깊게 보진 않음) -- 그래도 실패하면 에피소드를 실패로 끝낸다.
`loop_mode`는 복구가 실제로 한 번이라도 일어났으면 "closed", 아니면
"open"으로 기록한다.

## K-분기(candidate_chunks, 기본 꺼짐)

`--k-branch`를 주면 각 서브태스크 시작 시점에 체크포인트를 뜨고, Kp_xy를
조금씩 바꾼 K개 분기를 짧게(최대 50스텝) 실행해서 "그 스텝 수 안에 얼마나
진행됐는지"를 기록한 뒤 체크포인트로 되돌아가 원래 게인으로 실제 기록을
이어간다(비용이 커서 기본 꺼짐, 사용자 지시 그대로).

사용법:
    python pipeline/generate_closed_loop_episodes.py --split B --n 24 --out data/closed_loop/B --seed 1
    python pipeline/generate_closed_loop_episodes.py --split C --n 16 --out data/closed_loop/C --seed 2
    python pipeline/generate_closed_loop_episodes.py --split A --n 20 --out data/closed_loop/A --seed 3
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# mujoco는 MUJOCO_GL을 "자기 모듈이 import되는 시점"에 한 번 읽어서 렌더
# 백엔드를 고정한다(그 뒤에 바꿔도 효과 없음 -- tests/
# test_closed_loop_data_schema.py에서 실측 확인한 그대로). 이 스크립트는
# --render가 없을 때는 MUJOCO_GL을 절대 안 건드려야 하므로(사용자 지시
# "렌더링 안 하는 스크립트에서는 MUJOCO_GL=osmesa 설정 금지"), argparse가
# 돌기도 전에 sys.argv를 직접 보고 결정한다 -- 아래 `from closedloop...`
# import가 mujoco를 끌어오기 **전에** 반드시 끝나야 한다.
if "--render" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import yaml

from closedloop.peg_in_hole_adapter import PegInHoleClosedLoopEnv, ScriptedOraclePolicy
from pipeline.episode_io import save_episode
from sim.grasp_slip import ArmSlipConfig, GraspSlipConfig
from sim.peg_in_hole_bimanual_openarm_sim import BimanualPegInHoleOpenArmSim


def assert_no_torch_imported() -> None:
    """후속 작업 "작업 1" 방법 A의 전제 조건 -- MUJOCO_GL=osmesa 상태에서
    PyTorch Adam을 만들면 세그폴트가 난다(PIPELINE.md 참고). 이 생성
    스크립트는 torch를 아예 import하지 않지만(실측 확인: `python3 -c
    "import pipeline.generate_closed_loop_episodes; print('torch' in
    sys.modules)"` -> False), 혹시 다른 호출부가 이 모듈을 import하기
    **전에** torch를 먼저 import해버리는 경우까지 막으려고 명시적으로
    확인한다(`tests/test_closed_loop_data_schema.py`의
    `test_assert_no_torch_imported`가 이 함수 자체를 테스트)."""
    assert "torch" not in sys.modules, (
        "torch가 이미 import돼 있다 -- MUJOCO_GL=osmesa(렌더링)와 PyTorch Adam 생성을 "
        "같은 프로세스에 두면 세그폴트가 난다(PIPELINE.md 참고). 렌더링 생성과 학습 "
        "코드를 같은 프로세스에서 돌리지 말 것."
    )

_GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "tasks", "peg_in_hole.graph.yaml")
_NOMINAL_GAINS = {"Kp_xy": 0.124630, "Kd_xy": 0.001125}
_BAD_GAINS = {"Kp_xy": 0.01, "Kd_xy": 0.0001}  # tasks/peg_in_hole.yaml의 CMA-ES x0(의도적으로 나쁜 시작점)
_TARGET_DEPTH = 0.055 / 3
_MAX_ITERATIONS = 2000
_MAX_ATTEMPTS_PER_NODE = 2  # 소규모 생성이라 복구는 최대 1회만 시도

_B_DISTURBANCE_TYPES = [
    "action_noise", "xy_target_bias", "bad_gains", "narrow_clearance", "high_friction",
    "injected_slip_F2", "injected_slip_F3",
]
_C_DISTURBANCE_TYPES = ["wide_friction", "very_narrow_clearance", "action_delay"]


def _load_graph() -> dict:
    with open(_GRAPH_PATH) as f:
        return yaml.safe_load(f)


def _base_scene(rng: np.random.Generator, angle_range: tuple[float, float], radius_range: tuple[float, float]) -> dict:
    angle = rng.uniform(*angle_range)
    radius = rng.uniform(*radius_range)
    return {
        "friction": float(rng.uniform(0.2, 0.8)),
        "clearance_m": float(rng.uniform(0.0025, 0.0035)),
        "peg_init_offset_xy": (float(radius * np.cos(angle)), float(radius * np.sin(angle))),
        "target_insertion_depth": _TARGET_DEPTH,
    }


def sample_scene_A(rng: np.random.Generator) -> dict:
    """자연 실패 세트 -- 명목 clearance/friction은 그대로 두고 오프셋
    반경·각도만 넓혀서 실패율을 올린다(PIPELINE.md "5단계" 절 실측:
    radius U(0.009,0.020), angle U(-pi,pi) -> 개루프 실패율 20.0%,
    N=60/100 두 시드에서 재확인)."""
    return _base_scene(rng, (-np.pi, np.pi), (0.009, 0.020))


def sample_scene_and_disturbance_B(rng: np.random.Generator) -> tuple[dict, dict, dict, GraspSlipConfig | None]:
    """(scene_config, disturbance_meta, gains, grasp_slip_config)를 돌려준다.
    disturbance_meta는 그대로 episode["disturbance"]에 저장된다."""
    dtype = str(rng.choice(_B_DISTURBANCE_TYPES))
    cfg = _base_scene(rng, (-np.pi / 4, np.pi / 4), (0.009, 0.014))
    gains = dict(_NOMINAL_GAINS)
    grasp_slip_config = None
    meta: dict[str, Any] = {"type": dtype}

    if dtype == "narrow_clearance":
        cfg["clearance_m"] = float(rng.uniform(0.0012, 0.0020))
        meta["clearance_m"] = cfg["clearance_m"]
    elif dtype == "high_friction":
        cfg["friction"] = float(rng.uniform(1.0, 1.8))
        meta["friction"] = cfg["friction"]
    elif dtype == "bad_gains":
        gains = dict(_BAD_GAINS)
        meta["gains"] = gains
    elif dtype == "action_noise":
        meta["std_m"] = 0.0008
    elif dtype == "xy_target_bias":
        bias = rng.uniform(-0.004, 0.004, size=2)
        meta["bias_m"] = bias.tolist()
    elif dtype in ("injected_slip_F2", "injected_slip_F3"):
        side = "right" if dtype.endswith("F2") else "left"
        inj_step = int(rng.integers(60, 150))
        sign = float(rng.choice([-1.0, 1.0]))
        offset = np.array([0.015 * sign, 0.0, 0.0])
        arm_cfg = ArmSlipConfig(mode="injected", injected_at_step=inj_step, injected_offset_local=offset)
        grasp_slip_config = GraspSlipConfig(right=arm_cfg) if side == "right" else GraspSlipConfig(left=arm_cfg)
        meta.update({"side": side, "step": inj_step, "offset_m": offset.tolist()})
    return cfg, meta, gains, grasp_slip_config


def sample_scene_and_disturbance_C(rng: np.random.Generator) -> tuple[dict, dict, int]:
    """(scene_config, disturbance_meta, action_delay_steps)."""
    dtype = str(rng.choice(_C_DISTURBANCE_TYPES))
    cfg = _base_scene(rng, (-np.pi / 4, np.pi / 4), (0.009, 0.014))
    meta: dict[str, Any] = {"type": dtype}
    delay_steps = 0
    if dtype == "wide_friction":
        # B의 high_friction(1.0-1.8) 바로 위 -- 실측(N=20, 명목 오프셋/
        # clearance): 1.8-2.2는 1/20, 1.4-1.8(B와 겹침, 참고용)는 7/20
        # 성공. 전부 0%는 아니지만 거의 항상 실패한다는 걸 그대로
        # 반영한 범위(PIPELINE.md "5단계" 절 참고).
        cfg["friction"] = float(rng.uniform(1.8, 2.2))
        meta["friction"] = cfg["friction"]
    elif dtype == "very_narrow_clearance":
        # 실측(N=20): 0.0010-0.0014는 0/20, 0.0014-0.0018는 1/20.
        cfg["clearance_m"] = float(rng.uniform(0.0012, 0.0018))
        meta["clearance_m"] = cfg["clearance_m"]
    elif dtype == "action_delay":
        delay_steps = 5
        meta["delay_steps"] = delay_steps
    return cfg, meta, delay_steps


@dataclass
class VisualDisturbanceConfig:
    """후속 작업 "작업 1"의 렌더링 단계 시각 교란 -- C세트에만 적용하고
    기본 꺼짐(사용자 지시 "C-only, default OFF"). 물리에는 전혀 영향
    없음(model.light_*/cam_pos/geom_rgba만 건드림) -- `_apply_clearance()`
    와 같은 "컴파일된 모델 인스턴스를 직접 변형"패턴(각
    `BimanualPegInHoleOpenArmSim()` 호출이 자기 모델을 새로 컴파일하므로
    인스턴스 간 간섭 없음)."""

    enabled: bool = False
    light_dir_jitter_rad: float = 0.15
    light_diffuse_jitter: float = 0.25
    cam_pos_jitter_m: float = 0.03
    hole_rgba_jitter: float = 0.15
    peg_rgba_jitter: float = 0.15


def apply_visual_disturbance(sim: BimanualPegInHoleOpenArmSim, cfg: VisualDisturbanceConfig, rng: np.random.Generator) -> dict[str, Any]:
    """`cfg.enabled`가 False면 아무것도 안 바꾸고 빈 dict. True면 조명
    방향/세기, top_cam/wide_cam 위치, 구멍/페그 색을 지터링하고 실제
    적용값을 돌려준다(에피소드 메타데이터 기록용, 사용자 지시 "범위와
    실제 적용값을 에피소드 메타데이터에 기록")."""
    if not cfg.enabled:
        return {}
    model = sim.model
    applied: dict[str, Any] = {}

    light_deltas = rng.uniform(-cfg.light_dir_jitter_rad, cfg.light_dir_jitter_rad, size=(model.nlight, 3))
    diffuse_mults = rng.uniform(1.0 - cfg.light_diffuse_jitter, 1.0 + cfg.light_diffuse_jitter, size=model.nlight)
    for i in range(model.nlight):
        model.light_dir[i] = model.light_dir[i] + light_deltas[i]
        model.light_diffuse[i] = np.clip(model.light_diffuse[i] * diffuse_mults[i], 0.0, 1.0)
    applied["light_dir_delta"] = light_deltas.tolist()
    applied["light_diffuse_mult"] = diffuse_mults.tolist()

    cam_deltas: dict[str, list] = {}
    for cam_name in ("top_cam", "wide_cam"):
        cam_id = model.camera(cam_name).id
        delta = rng.uniform(-cfg.cam_pos_jitter_m, cfg.cam_pos_jitter_m, size=3)
        model.cam_pos[cam_id] = model.cam_pos[cam_id] + delta
        cam_deltas[cam_name] = delta.tolist()
    applied["cam_pos_delta_m"] = cam_deltas

    hole_mult = 1.0 + rng.uniform(-cfg.hole_rgba_jitter, cfg.hole_rgba_jitter, size=3)
    for name in ("hole_wall_px", "hole_wall_nx", "hole_wall_py", "hole_wall_ny", "hole_floor"):
        gid = model.geom(name).id
        model.geom_rgba[gid][:3] = np.clip(model.geom_rgba[gid][:3] * hole_mult, 0.0, 1.0)
    applied["hole_rgba_mult"] = hole_mult.tolist()

    peg_mult = 1.0 + rng.uniform(-cfg.peg_rgba_jitter, cfg.peg_rgba_jitter, size=3)
    for name in ("peg_shaft", "peg_tip_ball"):
        gid = model.geom(name).id
        model.geom_rgba[gid][:3] = np.clip(model.geom_rgba[gid][:3] * peg_mult, 0.0, 1.0)
    applied["peg_rgba_mult"] = peg_mult.tolist()

    return applied


def _empty_grasp_hist() -> dict[str, list]:
    return {"weld_active": [], "weld_force": [], "slipped": [], "cumulative_shift": []}


def _append_grasp(hist: dict[str, list], state) -> None:
    if state is None:
        hist["weld_active"].append(True)
        hist["weld_force"].append(0.0)
        hist["slipped"].append(False)
        hist["cumulative_shift"].append(np.zeros(3, dtype=np.float32))
    else:
        hist["weld_active"].append(bool(state.weld_active))
        hist["weld_force"].append(float(state.weld_force))
        hist["slipped"].append(bool(state.slipped_this_step))
        hist["cumulative_shift"].append(state.cumulative_shift_pos.astype(np.float32))


def _run_k_branch(env: PegInHoleClosedLoopEnv, k: int, horizon: int, rng: np.random.Generator) -> list[dict]:
    """모듈 docstring "K-분기" 절 그대로 -- 체크포인트에서 K개 게인 변형을
    짧게 실행해 진행도만 기록하고 체크포인트로 복귀한다."""
    checkpoint = (env.sim.data.qpos.copy(), env.sim.data.qvel.copy(), dict(env.sim._virtual_qpos))
    base_kp = env.gains["Kp_xy"]
    results = []
    for k_idx in range(k):
        kp_mult = float(rng.uniform(0.5, 1.5))
        env.gains = {"Kp_xy": base_kp * kp_mult, "Kd_xy": _NOMINAL_GAINS["Kd_xy"]}
        policy = ScriptedOraclePolicy(env)
        start_progress = None
        end_progress = 0.0
        for step in range(horizon):
            actions, decisions = policy.predict_action_chunk_with_decisions(None)
            if start_progress is None:
                start_progress = decisions["progress"]
            env.execute(actions[0, 0])
            end_progress = decisions["progress"]
            if env.subtask_finished():
                break
        results.append(
            {
                "kp_mult": kp_mult,
                "start_progress": float(start_progress or 0.0),
                "end_progress": float(end_progress),
                "steps_run": step + 1,
            }
        )
        # 체크포인트로 복귀 -- 다음 분기도 같은 시작점에서.
        env.sim.data.qpos[:] = checkpoint[0]
        env.sim.data.qvel[:] = checkpoint[1]
        env.sim._virtual_qpos = dict(checkpoint[2])
        import mujoco

        mujoco.mj_forward(env.sim.model, env.sim.data)
        env._step_in_subtask = 0
        env._hold_count = 0
    env.gains = {"Kp_xy": base_kp, "Kd_xy": _NOMINAL_GAINS["Kd_xy"]}
    return results


def run_one_episode(
    rng: np.random.Generator,
    split: str,
    seed: int,
    k_branch: bool = False,
    k_branch_count: int = 4,
    k_branch_horizon: int = 50,
    render: bool = False,
    visual_disturbance: VisualDisturbanceConfig | None = None,
) -> dict[str, Any]:
    """`render=True`면 후속 작업 "작업 1" 방법 A대로 서브태스크 경계
    (+ R2 복구 점프)마다 top_cam/wide_cam을 생성 중에 바로 캡처한다 --
    `render_closed_loop_boundaries.py`처럼 별도 프로세스에서 액션을
    재생하지 않으므로 복구 점프도 그대로 커버된다. `visual_disturbance`는
    C세트에서만 뜻이 있다(사용자 지시 "C-only")."""
    if render:
        assert_no_torch_imported()
    disturbance: dict[str, Any] | None = None
    grasp_slip_config: GraspSlipConfig | None = None
    gains = dict(_NOMINAL_GAINS)
    action_noise_std = 0.0
    action_delay_steps = 0
    xy_bias = np.zeros(2)

    if split == "A":
        cfg = sample_scene_A(rng)
    elif split == "B":
        cfg, disturbance, gains, grasp_slip_config = sample_scene_and_disturbance_B(rng)
        if disturbance["type"] == "action_noise":
            action_noise_std = disturbance["std_m"]
        elif disturbance["type"] == "xy_target_bias":
            xy_bias = np.array(disturbance["bias_m"])
    elif split == "C":
        cfg, disturbance, action_delay_steps = sample_scene_and_disturbance_C(rng)
    else:
        raise ValueError(f"unknown split {split!r}")

    sim = BimanualPegInHoleOpenArmSim(grasp_slip_config=grasp_slip_config)
    if split == "C" and visual_disturbance is not None:
        visual_applied = apply_visual_disturbance(sim, visual_disturbance, rng)
        if visual_applied:
            disturbance = dict(disturbance or {"type": "none"})
            disturbance["visual"] = visual_applied
    env = PegInHoleClosedLoopEnv(sim, _load_graph(), cfg, gains, render=render)
    env.reset_episode("align")
    policy = ScriptedOraclePolicy(env)

    if k_branch:
        candidate_chunks = [{"node": "align", "branches": _run_k_branch(env, k_branch_count, k_branch_horizon, rng)}]
        env.reset_episode("align")  # K-분기가 episode 상태를 흩트렸을 수 있으니 완전히 다시 시작.
        policy = ScriptedOraclePolicy(env)
    else:
        candidate_chunks = None

    boundary_images: dict[str, np.ndarray] = {}
    # 다음에 완료될 경계의 "start" 프레임 -- 점프(R2 복구) 없이 자연히
    # 이어지면 직전 경계의 "end"를 그대로 재사용하고, 점프가 있으면
    # run_recovery() 뒤에 다시 찍는다(사용자 지시 "점프 없으면 다음
    # 경계의 start로 복제, 점프 있으면 run_recovery() 뒤에 새로 캡처").
    pending_start_frames: dict[str, np.ndarray] | None = env.render_frames() if render else None

    joint_hist = [sim.joint_observation_16dof()]
    ee_poses_hist = [sim.get_ee_pose()]
    left_arm_traj_hist = [sim.get_left_ee_pos()]
    actions_hist: list[np.ndarray] = []
    forces_hist: list[np.ndarray] = []
    torques_hist: list[np.ndarray] = []
    progress_hist: list[float] = []
    subtask_id_hist: list[str] = []
    grasp_right_hist = _empty_grasp_hist()
    grasp_left_hist = _empty_grasp_hist()
    subtask_boundaries: list[dict[str, Any]] = []
    action_queue: list[np.ndarray] = []

    attempt = 1
    total_steps = 0
    success_final = False
    recovery_happened = False

    for _ in range(_MAX_ITERATIONS):
        actions, decisions = policy.predict_action_chunk_with_decisions(None)
        action = actions[0, 0].astype(np.float64).copy()
        if action_noise_std > 0:
            action = action + rng.normal(0.0, action_noise_std, size=3)
        action[:2] += xy_bias
        if action_delay_steps > 0:
            action_queue.append(action.copy())
            action = action_queue.pop(0) if len(action_queue) > action_delay_steps else np.zeros(3)

        env.execute(action)
        total_steps += 1

        joint_hist.append(sim.joint_observation_16dof())
        ee_poses_hist.append(sim.get_ee_pose())
        left_arm_traj_hist.append(sim.get_left_ee_pos())
        force, torque = sim.get_force_torque()
        forces_hist.append(force)
        torques_hist.append(torque)
        actions_hist.append(action.astype(np.float32))
        progress_hist.append(float(decisions["progress"]))
        subtask_id_hist.append(env.current_node)
        rs, ls = sim.get_grasp_states()
        _append_grasp(grasp_right_hist, rs)
        _append_grasp(grasp_left_hist, ls)

        if env.subtask_finished():
            env.settle()
            node = env.current_node
            start_step = total_steps - env._step_in_subtask
            # 방금 끝난 결과를 정확히 읽으려면 오라클을 한 번 더 불러야 한다
            # -- 위에서 받은 `decisions`는 *이번* execute() 이전(즉 저번
            # 스텝의 끝) 상태를 반영한 것이라 한 박자 늦다(ARD-VLA
            # closed_loop.py 모듈 docstring이 명시적으로 경고하는, "실행
            # 전 관측으로 판정하는" 버그와 똑같은 함정 -- 4단계 어댑터
            # docstring의 "판정은 방금 끝난 서브태스크를 본다" 절도 같은
            # 이유로 이 호출이 꼭 필요하다고 설명함). `_subtask_done_
            # naturally`/`_subtask_timed_out`는 바로 위 `subtask_finished()`
            # 가 이미 갱신해뒀으므로, 이 호출의 `decisions`는 정확하다.
            _, fresh_decisions = policy.predict_action_chunk_with_decisions(None)
            boundary = {
                "node": node,
                "start_step": start_step,
                "end_step": total_steps,
                "finished_naturally": bool(env._subtask_done_naturally),
                "failure_type": fresh_decisions["failure_type_label"],
                "attempt": attempt,
                "recovery_applied": None,
            }
            end_frames = env.render_frames() if render else None
            if render:
                idx = len(subtask_boundaries)
                if pending_start_frames is not None:
                    boundary_images[f"boundary_{idx}_top_cam_start"] = pending_start_frames["top_cam"]
                    boundary_images[f"boundary_{idx}_wide_cam_start"] = pending_start_frames["wide_cam"]
                boundary_images[f"boundary_{idx}_top_cam_end"] = end_frames["top_cam"]
                boundary_images[f"boundary_{idx}_wide_cam_end"] = end_frames["wide_cam"]
            if env._subtask_done_naturally:
                subtask_boundaries.append(boundary)
                if node == "insert":
                    success_final = True
                    break
                env.set_subtask("insert")
                attempt = 1
                if render:
                    pending_start_frames = end_frames  # 점프 없음 -- 끝난 순간 그대로가 다음 경계의 시작.
            elif attempt < _MAX_ATTEMPTS_PER_NODE:
                boundary["recovery_applied"] = "R2"
                subtask_boundaries.append(boundary)
                env.run_recovery("R2")
                recovery_happened = True
                attempt += 1
                if render:
                    pending_start_frames = env.render_frames()  # 체크포인트로 점프했으니 다시 찍어야 함.
            else:
                subtask_boundaries.append(boundary)
                success_final = False
                break

    episode: dict[str, Any] = {
        "task": "peg_in_hole",
        "right_arm": {
            "traj": np.stack(ee_poses_hist).astype(np.float32),
            "action": np.stack(actions_hist).astype(np.float32) if actions_hist else np.zeros((0, 3), dtype=np.float32),
            "gain_names": ["Kp_xy", "Kd_xy"],
            "gains": [gains["Kp_xy"], gains["Kd_xy"]],
            "force": np.stack(forces_hist).astype(np.float32) if forces_hist else np.zeros((0, 3), dtype=np.float32),
            "torque": np.stack(torques_hist).astype(np.float32) if torques_hist else np.zeros((0, 3), dtype=np.float32),
            "role": "actuator",
        },
        "left_arm": {"traj": np.stack(left_arm_traj_hist).astype(np.float32), "role": "stabilizer"},
        "scene_config": cfg,
        "success": success_final,
        "closed_loop": {
            "joint_16dof": np.stack(joint_hist).astype(np.float32),
            "progress": np.array(progress_hist, dtype=np.float32),
            "subtask_id": np.array(subtask_id_hist),
            "grasp_right": {k: np.stack(v) if k == "cumulative_shift" else np.array(v) for k, v in grasp_right_hist.items()},
            "grasp_left": {k: np.stack(v) if k == "cumulative_shift" else np.array(v) for k, v in grasp_left_hist.items()},
        },
        "subtask_boundaries": subtask_boundaries,
        "seed": seed,
        # 분할 이름은 "train/val/test_B/test_C/test_A" 중 하나(5단계
        # 사용자 지시 그대로) -- B만 train/val로 더 쪼갠다(학습용), C/A는
        # 전부 test_*(테스트 전용 -- C는 "학습에 쓰지 않음"이 지시사항
        # 원문, A는 자연 실패라 애초에 평가용).  같은 에피소드 시드는
        # 한 분할에만 속한다(에피소드 단위로만 나누므로 당연히 성립).
        "split": ("train" if split == "B" and rng.random() < 0.8 else "val") if split == "B" else f"test_{split}",
        "loop_mode": "closed" if recovery_happened else "open",
    }
    if disturbance is not None:
        episode["disturbance"] = disturbance
    if candidate_chunks is not None:
        episode["candidate_chunks"] = candidate_chunks
    if render:
        episode["boundary_images"] = boundary_images
    return episode


def main() -> None:
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["B", "C", "A"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-branch", action="store_true", help="처음 episode 하나에만 K-분기를 켠다(비용이 커서)")
    parser.add_argument(
        "--render", action="store_true",
        help="후속 작업 '작업 1' 방법 A -- 서브태스크 경계(+R2 복구 점프)마다 생성 중에 top_cam/wide_cam을 바로 캡처한다",
    )
    parser.add_argument(
        "--visual-disturbance", action="store_true",
        help="C세트에서만 뜻이 있음 -- 조명/카메라/색 렌더링 단계 시각 교란을 추가 적용(기본 꺼짐)",
    )
    args = parser.parse_args()

    if args.render:
        assert_no_torch_imported()
    visual_cfg = VisualDisturbanceConfig(enabled=args.visual_disturbance) if args.split == "C" else None

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_success = 0
    total_boundaries = 0
    render_start = time.perf_counter()
    for i in range(args.n):
        episode_seed = args.seed * 10_000 + i
        episode = run_one_episode(
            rng, args.split, episode_seed, k_branch=(args.k_branch and i == 0),
            render=args.render, visual_disturbance=visual_cfg,
        )
        n_success += int(episode["success"])
        total_boundaries += len(episode["subtask_boundaries"])
        path = os.path.join(args.out, f"episode_{i:04d}.npz")
        save_episode(path, episode)
        print(
            f"[{args.split}] episode {i}: success={episode['success']} "
            f"loop_mode={episode['loop_mode']} disturbance={episode.get('disturbance', {}).get('type')} "
            f"steps={len(episode['closed_loop']['subtask_id'])} -> {path}"
        )
    elapsed = time.perf_counter() - render_start
    print(f"\n[{args.split}] {n_success}/{args.n} succeeded ({n_success/args.n*100:.1f}%)")
    if args.render:
        print(
            f"[{args.split}] render timing: {elapsed:.1f}s total / {args.n}ep "
            f"({elapsed/args.n:.3f}s/ep) / {total_boundaries} boundaries "
            f"({elapsed/max(total_boundaries, 1):.3f}s/boundary)"
        )


if __name__ == "__main__":
    main()
