"""RoboTwin 2.0 이식 Part 3-2: scaffold_task.py가 생성하는 새 태스크의
실행 엔진 (Level 1 자동화).

기존 3개 태스크(peg_in_hole/cap_twist/tacker)의 OpenArm 7-DOF
control-loop(resolved-rate IK, 널스페이스, 관절 한계 freezing 등)는
여기서 전혀 재사용하지 않는다 -- 그건 "로봇이 어떻게 움직이는가"의
영역이라 이번 요청이 명시적으로 건드리지 않기로 한 범위다. 대신 이
레포에 이미 있는, 기구학이 필요 없는 범용 메커니즘 두 가지만 재사용한다:

1. sim/stabilizer.py:Stabilizer -- 3-슬라이드 조인트 가상 엔드이펙터 +
   weld로 물체를 "쥐는" 로직(IK 불필요, world 좌표를 그대로 ctrl에
   넣으면 끝). 원래는 왼팔(Stabilizer) 역할이었지만, 여기서는 그 자체를
   **구동되는 Actuator**로도 쓴다(쥔 뒤에 ctrl을 계속 바꿔서 끌고
   다닌다) -- 3-슬라이드 조인트는 애초에 "방향 없는 점 하나를 아무
   데서나 다른 아무 데로 옮기는" 범용 장치라 position-correction/
   impact-recoil 두 패턴 모두에 그대로 맞는다.
2. 힌지 + 위치 액추에이터 + jointactuatorfrc 센서(torque-reactive 패턴,
   cap_twist의 "cap" 힌지와 완전히 같은 메커니즘, 팔 기구학 자체가
   없음) -- patterns/torque_reactive.py의 mjcf_substructure()가 이미
   이 구조를 만든다.

패턴별 compute_reward()/is_success()는 patterns/{position_correction,
torque_reactive,impact_recoil}.py를 그대로 호출한다(새로 만든 reward
공식이 아니라, Part 3-1에서 기존 3개 태스크로부터 뽑아낸 그 템플릿).

## "Level 1" 범위 안에서의 단순화

실제 peg_in_hole/cap_twist/tacker는 각각 실측 튜닝(관절 한계 페널티,
stuck-detection, 3단계 변위 분리 등)이 많이 들어가 있다 -- 이 범용
엔진은 그런 튜닝 없이 패턴의 "핵심 동작"만 구현한다(목표로 PD 이동,
토크-의존 damping, 접근+정착+후퇴). scaffold_task.py로 만든 새 태스크가
바로 생산 품질 데이터를 뽑는 게 목적이 아니라, "패턴에 새 물체를
끼우면 바로 돌아간다"를 보여주는 게 목적이기 때문이다(요청 "완전
자동화가 아니다" 문구 참고).
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from patterns import impact_recoil, position_correction, torque_reactive
from sim.base_task_env import BaseTaskEnv
from sim.stabilizer import Stabilizer, run_approach_phase

MAX_STEPS = 1200  # cap_twist(sim/cap_twist_env.py)의 MAX_STEPS와 동일 -- 0.002s 타임스텝
# 기준 2.4s, OMEGA_BASE(=3rad/s)로 1바퀴(2pi) 돌리기엔 그래도 빠듯하므로
# torque_reactive 기본 target_rotation은 1바퀴보다 작게 잡는다(scaffold_
# task.py의 기본 scene_config 참고) -- Level 1 데모 목적이라 "1바퀴"
# 자체를 재현할 필요는 없다.
SUCCESS_HOLD_STEPS = 5

# peg_in_hole_bimanual_openarm_sim.py의 N_SUBSTEPS(=5, "mj_step 호출당
# 제어 주기 dt=0.01s")와 같은 이유로 position-correction에도 그대로
# 쓴다 -- 실측으로 발견: 매 물리 스텝(timestep=0.002s)마다 D항을
# 계산하면 1/dt=500배로 증폭돼 Kd가 0.01만 돼도 발산했다(아래
# ScaffoldedPositionCorrectionEnv.run_episode 참고). 제어 주기를
# 0.01s로 늘리면(N_SUBSTEPS번 물리를 진행한 뒤에만 D항을 갱신) 실제
# peg_in_hole과 같은 스케일의 Kp/Kd 값으로 안정적으로 수렴한다.
POSITION_CORRECTION_N_SUBSTEPS = 5


# BaseTaskEnv.apply_visual_config()는 self._sim.model을 기대한다(3개
# 기존 태스크와 같은 패턴) -- 이 엔진의 Env들은 sim 래퍼 클래스가 따로
# 없어서, model/data를 직접 들고 있는 자기 자신을 _sim으로도 쓴다
# (아래 각 Env의 __init__에서 self._sim = self).


# ----------------------------------------------------------------------
# position-correction
# ----------------------------------------------------------------------
class ScaffoldedPositionCorrectionEnv(BaseTaskEnv):
    """moving_object(가상 EE가 weld로 쥠) -> target_site(고정 목표점)로
    PD 이동. gains: Kp_pos, Kd_pos(3축 동일 게인 -- 실제 peg_in_hole처럼
    xy/z를 분리하지 않는다, Level 1 단순화)."""

    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._sim = self
        self._moving_body_id = self.model.body("moving_object").id
        self._target_site_id = self.model.site("target_site").id
        self._stab_act_ids = [self.model.actuator(n).id for n in ("stab_x", "stab_y", "stab_z")]
        self._target_depth = 0.0
        self._initial_distance = 0.0
        self._stabilizer: Stabilizer | None = None

    def reset(self, scene_config: dict[str, Any]) -> float:
        mujoco.mj_resetData(self.model, self.data)
        start_offset = np.asarray(scene_config.get("start_offset", (0.0, 0.0, 0.0)), dtype=float)
        self.data.qpos[self.model.joint("moving_object_free").qposadr[0]: self.model.joint("moving_object_free").qposadr[0] + 3] += start_offset
        mujoco.mj_forward(self.model, self.data)

        self._stabilizer = Stabilizer(
            self.model, self.data, "moving_object", scene_config.get("grasp_offset", (0.0, 0.0, 0.0))
        )
        self._stabilizer.reset()
        run_approach_phase(self._stabilizer, self.model, self.data, n_substeps=1)

        obj_pos = self.data.xpos[self._moving_body_id]
        target_pos = self.data.site_xpos[self._target_site_id]
        self._initial_distance = float(np.linalg.norm(target_pos - obj_pos))
        success_radius = float(scene_config.get("success_radius_m", 0.01))
        self._target_depth = max(0.0, self._initial_distance - success_radius)
        return self._initial_distance

    def step(self, delta_xyz: np.ndarray) -> None:
        for act_id, d in zip(self._stab_act_ids, delta_xyz):
            self.data.ctrl[act_id] = float(self.data.ctrl[act_id] + d)
        mujoco.mj_step(self.model, self.data, nstep=POSITION_CORRECTION_N_SUBSTEPS)

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        return position_correction.compute_reward(episode_result, position_correction.DEFAULT_COEFS)

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        return position_correction.is_success(episode_result, {"target_depth": self._target_depth})

    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        self.reset(scene_config)
        kp = float(gains["Kp_pos"])
        kd = float(gains["Kd_pos"])
        dt = float(self.model.opt.timestep) * POSITION_CORRECTION_N_SUBSTEPS

        prev_err = self.data.site_xpos[self._target_site_id] - self.data.xpos[self._moving_body_id]
        ee_poses = [self.data.xpos[self._moving_body_id].copy()]
        actions: list[np.ndarray] = []
        success = False
        hold = 0
        step_count = 0
        final_distance = float(np.linalg.norm(prev_err))
        insertion_depth = 0.0

        for step_count in range(1, MAX_STEPS + 1):
            err = self.data.site_xpos[self._target_site_id] - self.data.xpos[self._moving_body_id]
            d_err = (err - prev_err) / dt
            prev_err = err
            delta = kp * err + kd * d_err
            self.step(delta)
            ee_poses.append(self.data.xpos[self._moving_body_id].copy())
            actions.append(delta.copy())

            final_distance = float(np.linalg.norm(err))
            insertion_depth = max(0.0, self._initial_distance - final_distance)
            if self.is_success({"insertion_depth": insertion_depth}):
                hold += 1
            else:
                hold = 0
            if hold >= SUCCESS_HOLD_STEPS:
                success = True
                break

        episode_result = {
            "insertion_depth": insertion_depth,
            "final_distance": final_distance,
            "max_force": 0.0,
            "step_count": step_count,
            "success": success,
        }
        reward = self.compute_reward(episode_result)
        return {
            **episode_result,
            "reward": reward,
            "gains": dict(gains),
            "scene_config": scene_config,
            "ee_poses": np.stack(ee_poses).astype(np.float32),
            "actions": np.stack(actions).astype(np.float32) if actions else np.zeros((0, 3), dtype=np.float32),
        }


# ----------------------------------------------------------------------
# torque-reactive
# ----------------------------------------------------------------------
_OMEGA_BASE = 3.0  # rad/s -- cap_twist의 OMEGA_BASE와 동일 상수(저항 없을 때 기준 속도)


class ScaffoldedTorqueReactiveEnv(BaseTaskEnv):
    """main_hinge를 target_rotation까지 돌린다. 저항은 dof_damping =
    resistance_torque / OMEGA_BASE로 모델링(cap_twist와 동일 변환).
    gains: Kp_tau(측정 토크에 비례해 각속도를 낮추는 admittance 게인)."""

    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._sim = self
        self._hinge_qposadr = self.model.joint("main_hinge").qposadr[0]
        self._hinge_dofadr = self.model.joint("main_hinge").dofadr[0]
        self._actuator_id = self.model.actuator("main_hinge_drive").id
        self._torque_adr = self.model.sensor("main_hinge_torque").adr[0]
        self._target_rotation = 0.0

    def reset(self, scene_config: dict[str, Any]) -> None:
        mujoco.mj_resetData(self.model, self.data)
        resistance_torque = float(scene_config.get("resistance_torque", 0.3))
        self.model.dof_damping[self._hinge_dofadr] = resistance_torque / _OMEGA_BASE
        self.data.qpos[self._hinge_qposadr] = 0.0
        self.data.ctrl[self._actuator_id] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._target_rotation = float(scene_config.get("target_rotation", 2.0 * np.pi))

    def step(self, target_angle: float) -> None:
        self.data.ctrl[self._actuator_id] = float(target_angle)
        mujoco.mj_step(self.model, self.data)

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        return torque_reactive.compute_reward(episode_result, torque_reactive.DEFAULT_COEFS)

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        return torque_reactive.is_success(episode_result, torque_reactive.DEFAULT_COEFS)

    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        self.reset(scene_config)
        target_rotation = self._target_rotation
        direction = 1.0 if target_rotation >= 0 else -1.0
        target_mag = abs(target_rotation)
        kp_tau = float(gains["Kp_tau"])
        dt = float(self.model.opt.timestep)

        angle = 0.0  # 액추에이터에 보내는 명령 각도(open-loop 적분) -- 성공 판정은
        # 이게 아니라 실제 측정 qpos로 한다(cap_twist run_episode()의
        # "actual_angle = sim.get_angle()" 과 같은 이유 -- 명령 각도로
        # 판정하면 액추에이터가 못 따라가도(kp 부족, damping 과다) 항상
        # 성공으로 잘못 보일 수 있다, 실측으로 발견한 버그를 여기서도
        # 똑같이 피한다).
        max_torque = 0.0
        torque = 0.0
        forward_progress = 0.0
        step_count = 0
        for step_count in range(1, MAX_STEPS + 1):
            torque = float(self.data.sensordata[self._torque_adr])
            max_torque = max(max_torque, abs(torque))
            omega = max(0.1, _OMEGA_BASE - kp_tau * abs(torque))
            angle += direction * omega * dt
            self.step(angle)
            actual_angle = float(self.data.qpos[self._hinge_qposadr])
            forward_progress = max(0.0, actual_angle * direction)
            if forward_progress >= target_mag:
                break

        episode_result = {
            "target_rotation": target_rotation,
            "forward_progress": forward_progress,
            "max_torque": max_torque,
            "step_count": step_count,
            "disengaged": False,
            "current_torque": abs(torque),
        }
        episode_result["success"] = self.is_success(episode_result)
        reward = self.compute_reward(episode_result)
        episode_result["reward"] = reward
        episode_result["gains"] = dict(gains)
        episode_result["scene_config"] = scene_config
        return episode_result


# ----------------------------------------------------------------------
# impact-recoil
# ----------------------------------------------------------------------
class ScaffoldedImpactRecoilEnv(BaseTaskEnv):
    """가상 EE(= 공구, "stabilizer_ee" 바디)가 workpiece로 접근 ->
    strike_distance 이내 도달 시 발사(fired) -> 정착 -> 후퇴 -> 최종
    정착. 반동 자체는 제어하지 않는다(workpiece는 freejoint 자유
    바디라 접근/후퇴에 의한 접촉으로만 밀린다 -- impact-recoil 패턴의
    핵심, 모듈 docstring 참고). gains: Kp_approach(접근 속도, m/s급)."""

    _N_SETTLE = 20
    _N_RETRACT = 30
    _N_FINAL_SETTLE = 20
    _RETRACT_DISTANCE_M = 0.08

    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._sim = self
        self._workpiece_body_id = self.model.body("workpiece").id
        self._ee_body_id = self.model.body("stabilizer_ee").id
        self._act_ids = [self.model.actuator(n).id for n in ("stab_x", "stab_y", "stab_z")]
        self._success_displacement_m = impact_recoil.DEFAULT_COEFS["default_success_displacement_m"]

    def reset(self, scene_config: dict[str, Any]) -> None:
        mujoco.mj_resetData(self.model, self.data)
        start_pos = np.asarray(scene_config.get("ee_start_pos", (-0.12, 0.0, 0.05)), dtype=float)
        for act_id, v in zip(self._act_ids, start_pos):
            self.data.ctrl[act_id] = float(v)
        for name, v in zip(("stab_x", "stab_y", "stab_z"), start_pos):
            self.data.qpos[self.model.joint(name).qposadr[0]] = float(v)
        mujoco.mj_forward(self.model, self.data)
        self._success_displacement_m = float(
            scene_config.get("success_displacement_m", impact_recoil.DEFAULT_COEFS["default_success_displacement_m"])
        )

    def step(self, ctrl_target: np.ndarray) -> None:
        for act_id, v in zip(self._act_ids, ctrl_target):
            self.data.ctrl[act_id] = float(v)
        mujoco.mj_step(self.model, self.data)

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        return impact_recoil.compute_reward(episode_result, impact_recoil.DEFAULT_COEFS)

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        return impact_recoil.is_success(episode_result, impact_recoil.DEFAULT_COEFS)

    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        self.reset(scene_config)
        approach_rate = float(gains["Kp_approach"])
        dt = float(self.model.opt.timestep)
        strike_distance = float(scene_config.get("strike_distance_m", 0.02))

        workpiece_init = self.data.xpos[self._workpiece_body_id].copy()
        ee_pos = self.data.ctrl[self._act_ids].copy()
        step_count = 0
        fired = False
        min_distance = float("inf")

        for step_count in range(1, MAX_STEPS + 1):
            target_vec = self.data.xpos[self._workpiece_body_id] - self.data.xpos[self._ee_body_id]
            dist = float(np.linalg.norm(target_vec))
            min_distance = min(min_distance, dist)
            if dist <= strike_distance:
                fired = True
                break
            direction = target_vec / max(dist, 1e-9)
            ee_pos = ee_pos + direction * approach_rate * dt
            self.step(ee_pos)

        for _ in range(self._N_SETTLE):
            self.step(ee_pos)
            step_count += 1
        displacement = float(np.linalg.norm(self.data.xpos[self._workpiece_body_id] - workpiece_init))

        before_retract = self.data.xpos[self._workpiece_body_id].copy()
        retract_dir = ee_pos - self.data.xpos[self._workpiece_body_id]
        retract_dir = retract_dir / max(float(np.linalg.norm(retract_dir)), 1e-9)
        retract_target = ee_pos + retract_dir * self._RETRACT_DISTANCE_M
        for i in range(self._N_RETRACT):
            frac = (i + 1) / self._N_RETRACT
            self.step(ee_pos + (retract_target - ee_pos) * frac)
            step_count += 1
        retract_bump = float(np.linalg.norm(self.data.xpos[self._workpiece_body_id] - before_retract))

        before_final = self.data.xpos[self._workpiece_body_id].copy()
        for _ in range(self._N_FINAL_SETTLE):
            self.step(retract_target)
            step_count += 1
        final_bump = float(np.linalg.norm(self.data.xpos[self._workpiece_body_id] - before_final))
        retracted = True

        episode_result = {
            "final_distance": min_distance,
            "fired": fired,
            "displacement": displacement,
            "retract_bump": retract_bump,
            "final_bump": final_bump,
            "overshoot_penalty": 0.0,
            "retracted": retracted,
            "step_count": step_count,
            "success_displacement_m": self._success_displacement_m,
        }
        episode_result["success"] = self.is_success(episode_result)
        reward = self.compute_reward(episode_result)
        episode_result["reward"] = reward
        episode_result["gains"] = dict(gains)
        episode_result["scene_config"] = scene_config
        return episode_result
