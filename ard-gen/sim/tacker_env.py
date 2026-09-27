"""tacker(타카): "위치 정확도 + 1회성 발사" 태스크.

peg_in_hole/cap_twist는 Actuator가 에피소드 내내 연속적으로 힘/토크를
조절해야 하는 태스크였다(admittance PD 루프). tacker는 Actuator가 할 일이
"목표 지점까지 정확히 접근"뿐이고, 도달 즉시 자동으로 발사되는 **이산적
이벤트**다 -- 그래서 이 태스크가 검증하려는 물리적 핵심이 Actuator가 아니라
Stabilizer로 넘어간다: 발사 반동(impulse)을 Stabilizer가 얼마나 잘
흡수하느냐가 성공/실패를 가른다(assets/tacker.xml 상단 docstring 참고).

## 게인이 Kp_approach 하나뿐인 이유, 그리고 그 역할

peg_in_hole의 Kp_xy/Kd_xy(접촉힘 PD)나 cap_twist의 Kp_tau(저항 토크 PD)는
"매 스텝 관측되는 물리량(힘/토크)에 반응해 궤적을 실시간으로 보정"하는
피드백 게인이었다. tacker는 접근 중에 반응할 물리 신호가 없다(발사 전까지는
접촉도 저항도 없음) -- 그래서 Kp_approach는 순수하게 "목표까지 얼마나
공격적으로 접근할 것인가"를 정하는 위치 오차 비례 게인이다:

    delta = clip(Kp_approach * (nail_pos - ee_pos), MAX_APPROACH_STEP_M)

너무 작으면 MAX_STEPS 안에 tolerance에 못 들어와 타임아웃 실패, 너무 크면
tolerance 반경(근접 구간)에서도 큰 스텝으로 움직여 "발사 조건을 만족하는
순간을 건너뛰거나(oscillation)" 접근 속도 자체가 과도해진다(overshoot_penalty로
페널티) -- peg_in_hole/cap_twist와 같은 "너무 작아도 커도 나쁜" 단일축
탐색 문제지만, 그 원인이 힘 피드백이 아니라 순수 기하학적 접근 프로파일이라는
점이 다르다.

## action이 delta position인 이유, wrist 없이 3차원으로 충분한 이유

peg_in_hole처럼 action을 절대 목표가 아니라 매 스텝 위치 변화량(delta)으로
정의했다 -- LeRobotDataset의 action 컬럼이 "이번 스텝에 실제로 내린 제어
명령"이라는 다른 태스크와의 관례를 그대로 따른다(cap_twist만 예외적으로
각속도를 저장하는데, 그것도 "이번 스텝의 명령"이라는 점은 같다). wrist
회전은 필요 없다고 판단했다 -- 이 태스크는 타카를 표면에 수직으로 대고
누르는 동작이라 삽입 방향 정렬(peg_in_hole처럼 그리퍼 각도가 삽입축과
일치해야 함)이나 회전 진행도(cap_twist) 같은 "각도가 성공 조건에 들어가는"
요소가 전혀 없다 -- 성공은 순수하게 위치(발사 지점 도달) + 발사 후 변위로만
정의되므로, 3차원 위치만으로 충분하다.

## 발사 트리거: 사람이 아니라 위치 조건이 스스로 발사시킨다

매 스텝 이동 후 EE와 nail_site 사이 거리를 확인해서, FIRE_TOLERANCE_M
이내로 들어오는 바로 그 스텝에서 자동으로 발사한다(gains나 action이
"발사해라"를 명시하지 않는다) -- 실제 타카가 눌리는 순간 자동으로
발사되는 것과 같은 동작이다. 발사는 에피소드당 정확히 한 번만 일어난다
(fired 플래그로 막음).

## 발사 반동 구현: workpiece에 직접 velocity kick

impulse = F*dt를 dt->0 극한으로 이상화해서, 발사 순간 workpiece의 freejoint
선속도에 직접 킥을 더한다(qfrc_applied로 여러 스텝에 걸쳐 힘을 적분하는
대신 -- 순간적 이벤트라는 태스크 설계 의도에 더 잘 맞고, 구현도 결정론적
이라 재현하기 쉽다). 수평 성분이 주(recoil_angle 방향)고 수직 성분은
작게(0.2배) 섞었다 -- 순수 수직 킥은 중력+테이블 접촉만으로도 금방
멈추므로 Stabilizer 유무가 거의 안 갈리고(테이블이 이미 버텨줌), 수평
킥이라야 "자유물체가 테이블 마찰만으로 밀려나느냐 vs Stabilizer가
붙잡아주느냐"가 실제로 갈린다 -- 이게 이 태스크를 만든 목적 그 자체다.
발사 직후 SETTLE_STEPS만큼 더 진행해서 반동이 가라앉은 뒤의 최종 변위로
성공을 판정한다(발사 순간의 순간 속도가 아니라 "결국 얼마나 밀렸는가"가
실제로 중요하기 때문).
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

from sim.base_task_env import BaseTaskEnv
from sim.stabilizer import Stabilizer, run_approach_phase

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "tacker.xml")

N_SUBSTEPS = 5  # mj_step 호출당 substep 수 (timestep=0.002 -> 제어 주기 dt=0.01s)
DT = N_SUBSTEPS * 0.002

MAX_STEPS = 400  # 접근 예산(발사 전까지)
SETTLE_STEPS = 30  # 발사 후 반동이 가라앉는 걸 관찰하는 스텝 수

FIRE_TOLERANCE_M = 0.003  # 이 이내로 들어오면 자동 발사 (요청의 "tolerance 반경 3mm")
MAX_APPROACH_STEP_M = 0.01  # 한 스텝에 허용하는 최대 이동량(물리적 속도 상한)
_NEAR_ZONE_MULT = 4.0  # tolerance의 이 배수 이내를 "근접 구간"으로 보고 오버슈트를 추적
_SAFE_NEAR_STEP_M = 2.0 * FIRE_TOLERANCE_M  # 근접 구간에서 이보다 큰 스텝이면 오버슈트로 페널티

SUCCESS_DISPLACEMENT_M = 0.006  # 발사 후 이 이내로 workpiece가 밀리면 성공(기본값, scene_config로 덮어쓸 수 있음)

_WORKPIECE_SETTLE_STEPS = 20  # reset() 직후 테이블 접촉 침투 해소용


def default_scene_config() -> dict[str, Any]:
    return {
        "workpiece_pos_xy": (0.0, 0.0),
        "workpiece_yaw": 0.0,
        "nail_offset_xy": (0.0, 0.0),
        "actuator_init_offset": (0.03, -0.03, 0.05),
        "recoil_strength": 0.3,
        "recoil_angle": 0.0,
        "success_displacement_m": SUCCESS_DISPLACEMENT_M,
    }


def sample_scene_config(rng: np.random.Generator | None = None) -> dict[str, Any]:
    """이미 sim 레벨(reset()이 바로 받는) 스키마를 그대로 뽑는다 -- peg_in_hole
    OpenArm 버전과 같은 이유로 별도의 "1단계 공유 씬" 변환 단계가 없다
    (to_sim_scene_config()는 항등 함수). 요청의 "nail_site 위치, workpiece
    초기 자세, 반동 강도" 세 가지를 전부 무작위화한다:
      - workpiece_pos_xy/workpiece_yaw: workpiece 자체가 테이블 위 어디에
        어떤 각도로 놓여 있는지(초기 자세).
      - nail_offset_xy: 그 workpiece 표면 위에서 정확히 어디를 조준해야
        하는지(조준점의 개체 로컬 오프셋 -- workpiece가 회전해도 같이 돈다).
      - actuator_init_offset: EE가 얼마나 멀리서/어느 방향에서 접근을
        시작하는지(월드 기준, nail_site 기준 상대 오프셋).
      - recoil_strength/recoil_angle: 발사 반동의 세기/방향 -- 이게 이
        태스크의 진짜 난이도 축이다(Stabilizer가 버텨야 하는 부하).
    """
    if rng is None:
        rng = np.random.default_rng()

    wp_angle = rng.uniform(-np.pi, np.pi)
    wp_radius = rng.uniform(0.0, 0.03)
    nail_angle = rng.uniform(-np.pi, np.pi)
    nail_radius = rng.uniform(0.0, 0.015)
    approach_angle = rng.uniform(-np.pi, np.pi)
    approach_radius = rng.uniform(0.03, 0.06)
    approach_height = rng.uniform(0.04, 0.07)

    return {
        "workpiece_pos_xy": (float(wp_radius * np.cos(wp_angle)), float(wp_radius * np.sin(wp_angle))),
        "workpiece_yaw": float(rng.uniform(-np.pi, np.pi)),
        "nail_offset_xy": (float(nail_radius * np.cos(nail_angle)), float(nail_radius * np.sin(nail_angle))),
        "actuator_init_offset": (
            float(approach_radius * np.cos(approach_angle)),
            float(approach_radius * np.sin(approach_angle)),
            float(approach_height),
        ),
        "recoil_strength": float(rng.uniform(0.15, 0.45)),
        "recoil_angle": float(rng.uniform(-np.pi, np.pi)),
        "success_displacement_m": SUCCESS_DISPLACEMENT_M,
    }


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(shared_cfg)


class TackerSim:
    """MjModel/MjData를 재사용하는 시뮬레이션 래퍼(다른 태스크의 *Sim 클래스와
    같은 역할)."""

    def __init__(self, xml_path: str | None = None):
        self.xml_path = xml_path or _DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self._workpiece_body_id = self.model.body("workpiece").id
        self._nail_site_id = self.model.site("nail_site").id
        self._tip_site_id = self.model.site("tacker_tip_site").id
        self._workpiece_qposadr = self.model.joint("workpiece_free").qposadr[0]
        self._workpiece_dofadr = self.model.joint("workpiece_free").dofadr[0]

        self._act_actuator_ids = [self.model.actuator(n).id for n in ("act_x", "act_y", "act_z")]
        self._act_qposadr = [self.model.joint(n).qposadr[0] for n in ("act_x", "act_y", "act_z")]

    def reset(self, scene_config: dict[str, Any]) -> None:
        mujoco.mj_resetData(self.model, self.data)

        wp_xy = scene_config["workpiece_pos_xy"]
        yaw = float(scene_config["workpiece_yaw"])
        qadr = self._workpiece_qposadr
        self.data.qpos[qadr : qadr + 3] = [wp_xy[0], wp_xy[1], 0.01]
        half = yaw / 2.0
        self.data.qpos[qadr + 3 : qadr + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]

        nail_xy = scene_config["nail_offset_xy"]
        self.model.site_pos[self._nail_site_id] = [nail_xy[0], nail_xy[1], 0.01]

        mujoco.mj_forward(self.model, self.data)
        # workpiece가 정확히 테이블면에 얹힌 값으로 배치했지만, yaw/xy가
        # 바뀌면 부동소수점 수준의 잔류 침투가 생길 수 있어 cap_twist와
        # 같은 이유로 짧게 정착시킨다.
        for _ in range(_WORKPIECE_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        # EE 시작 위치 = nail_site의 (정착 이후) 실제 world 위치 + 월드 기준
        # 오프셋. tacker_ee는 world에 직접 매인 조인트라 qpos에 world 좌표를
        # 그대로 쓰면 그 자리에 즉시 놓인다(sim/stabilizer.py가 stabilizer_ee에
        # 쓰는 것과 같은 성질) -- 에피소드 시작을 "이미 그 자리"로 명확히
        # 정의하기 위해 접근 자체를 물리로 시뮬레이션하지 않고 직접 배치한다.
        nail_world = self.data.site_xpos[self._nail_site_id].copy()
        ee_start = nail_world + np.asarray(scene_config["actuator_init_offset"], dtype=float)
        for adr, val in zip(self._act_qposadr, ee_start):
            self.data.qpos[adr] = val
        for act_id, val in zip(self._act_actuator_ids, ee_start):
            self.data.ctrl[act_id] = float(val)
        mujoco.mj_forward(self.model, self.data)

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_site_id].copy()

    def get_nail_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._nail_site_id].copy()

    def get_workpiece_pos(self) -> np.ndarray:
        return self.data.xpos[self._workpiece_body_id].copy()

    def get_force_torque(self) -> tuple[np.ndarray, np.ndarray]:
        """workpiece(freejoint)에 실제로 작용 중인 구속력 합 -- XML <sensor>
        force/torque를 안 쓰는 이유는 assets/tacker.xml 상단 docstring
        "workpiece 반력 로깅" 절 참고(freejoint에서는 그 센서가 구조적으로
        항상 0을 반환한다는 걸 실측으로 발견했다). qfrc_constraint는
        접촉+equality(Stabilizer weld 포함) 전부의 합력이라, 발사 반동을
        Stabilizer가 흡수하는 순간의 진짜 반력 스파이크를 그대로 반영한다."""
        dofadr = self._workpiece_dofadr
        force = self.data.qfrc_constraint[dofadr : dofadr + 3].copy()
        torque = self.data.qfrc_constraint[dofadr + 3 : dofadr + 6].copy()
        return force, torque

    def apply_recoil(self, kick_vector: np.ndarray) -> None:
        """workpiece의 freejoint 선속도에 직접 킥을 더한다(모듈 docstring
        "발사 반동 구현" 참고). 회전(각속도)은 건드리지 않는다 -- 순수
        선형 임팩트로 단순화."""
        dofadr = self._workpiece_dofadr
        self.data.qvel[dofadr : dofadr + 3] += np.asarray(kick_vector, dtype=float)

    def step(self, target_pos: np.ndarray) -> None:
        for act_id, val in zip(self._act_actuator_ids, target_pos):
            self.data.ctrl[act_id] = float(val)
        mujoco.mj_step(self.model, self.data, nstep=N_SUBSTEPS)


class TackerEnv(BaseTaskEnv):
    """stabilizer_config/use_stabilizer: 다른 태스크와 동일한 규약(3단계, 왼팔
    지원) -- 자세한 설명은 sim/peg_in_hole_env.py:PegInHoleEnv docstring 참고."""

    def __init__(
        self,
        xml_path: str | None = None,
        stabilizer_config: dict[str, Any] | None = None,
        use_stabilizer: bool = True,
    ):
        self._sim = TackerSim(xml_path=xml_path)
        self._target_pos = np.zeros(3)
        self._success_displacement_m = SUCCESS_DISPLACEMENT_M
        self._stabilizer: Stabilizer | None = None
        if use_stabilizer and stabilizer_config is not None:
            self._stabilizer = Stabilizer(
                self._sim.model, self._sim.data,
                stabilizer_config["object_body"], stabilizer_config["grasp_offset"],
            )

    # ------------------------------------------------------------------
    def reset(self, scene_config: dict[str, Any]) -> None:
        self._sim.reset(scene_config)
        self._target_pos = self._sim.get_ee_pos().copy()
        self._success_displacement_m = float(scene_config.get("success_displacement_m", SUCCESS_DISPLACEMENT_M))
        if self._stabilizer is not None:
            self._stabilizer.reset()

    def step(self, action: np.ndarray) -> None:
        """action: 이번 스텝의 위치 변화량(delta, m). 누적해서 절대 목표를
        유지하고 그걸 위치 액추에이터에 명령한다(모듈 docstring 참고)."""
        self._target_pos = self._target_pos + np.asarray(action, dtype=float)
        self._sim.step(self._target_pos)

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        """거리 페널티(접근 실패 시) + 발사 성공 보너스(고정) - 변위 초과
        페널티 - 오버슈트 페널티 - 스텝 페널티 + 최종 성공 보너스. peg_in_hole의
        "insertion_depth 자리"에 "발사 성공 여부"(이산값)가, "force 페널티
        자리"에 "발사 후 변위 초과분"이 들어간 구조다."""
        success_disp = episode_result.get("success_displacement_m", SUCCESS_DISPLACEMENT_M)
        reward = (
            -2.0 * episode_result["final_distance"]
            + 20.0 * (1.0 if episode_result.get("fired") else 0.0)
            - 500.0 * max(0.0, episode_result.get("displacement", 0.0) - success_disp)
            - 20.0 * episode_result.get("overshoot_penalty", 0.0)
            - 0.01 * episode_result["step_count"]
        )
        if episode_result.get("success"):
            reward += 50.0
        return float(reward)

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        """발사가 실제로 일어났고(fired), 그 후 workpiece 변위가 허용치
        이내면 성공."""
        if not episode_result.get("fired"):
            return False
        success_disp = episode_result.get("success_displacement_m", SUCCESS_DISPLACEMENT_M)
        return bool(episode_result.get("displacement", float("inf")) <= success_disp)

    # ------------------------------------------------------------------
    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        sim = self._sim
        self.reset(scene_config)
        kp_approach = float(gains["Kp_approach"])

        # Stabilizer가 있으면, 접근을 시작하기 전에 먼저 확실히 쥐게 한다
        # (sim/stabilizer.py의 run_approach_phase() docstring 참고 -- 발사가
        # 언제 일어날지 몰라서, 붙잡기 전에 발사돼버리는 경쟁을 원천적으로
        # 막는다).
        left_arm_traj: list[np.ndarray] = run_approach_phase(
            self._stabilizer, sim.model, sim.data, N_SUBSTEPS
        )

        ee_poses = [sim.get_ee_pos()]
        actions: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        torques: list[np.ndarray] = []

        fired = False
        pre_fire_workpiece_pos: np.ndarray | None = None
        overshoot_penalty = 0.0
        final_dist = float("nan")
        step_count = 0

        for step_count in range(1, MAX_STEPS + 1):
            ee_pos = sim.get_ee_pos()
            nail_pos = sim.get_nail_pos()
            err = nail_pos - ee_pos
            dist = float(np.linalg.norm(err))
            final_dist = dist

            delta = kp_approach * err
            step_norm = float(np.linalg.norm(delta))
            if step_norm > MAX_APPROACH_STEP_M:
                delta = delta * (MAX_APPROACH_STEP_M / step_norm)
                step_norm = MAX_APPROACH_STEP_M
            if dist < FIRE_TOLERANCE_M * _NEAR_ZONE_MULT:
                overshoot_penalty = max(overshoot_penalty, step_norm - _SAFE_NEAR_STEP_M)

            self.step(delta)
            if self._stabilizer is not None:
                left_arm_traj.append(self._stabilizer.ee_pos)

            actions.append(delta.astype(np.float32))
            force_now, torque_now = sim.get_force_torque()
            forces.append(force_now)
            torques.append(torque_now)
            ee_poses.append(sim.get_ee_pos())

            new_dist = float(np.linalg.norm(sim.get_nail_pos() - sim.get_ee_pos()))
            if new_dist <= FIRE_TOLERANCE_M:
                fired = True
                final_dist = new_dist
                pre_fire_workpiece_pos = sim.get_workpiece_pos().copy()
                recoil_strength = float(scene_config["recoil_strength"])
                recoil_angle = float(scene_config.get("recoil_angle", 0.0))
                kick = recoil_strength * np.array([np.cos(recoil_angle), np.sin(recoil_angle), 0.2])
                sim.apply_recoil(kick)

                for _ in range(SETTLE_STEPS):
                    mujoco.mj_step(sim.model, sim.data, nstep=N_SUBSTEPS)
                    if self._stabilizer is not None:
                        left_arm_traj.append(self._stabilizer.ee_pos)
                    force_now, torque_now = sim.get_force_torque()
                    forces.append(force_now)
                    torques.append(torque_now)
                    ee_poses.append(sim.get_ee_pos())
                    actions.append(np.zeros(3, dtype=np.float32))
                break

        displacement = 0.0
        if fired and pre_fire_workpiece_pos is not None:
            displacement = float(np.linalg.norm(sim.get_workpiece_pos() - pre_fire_workpiece_pos))

        episode_result = {
            "final_distance": final_dist,
            "fired": fired,
            "displacement": displacement,
            "success_displacement_m": self._success_displacement_m,
            "overshoot_penalty": float(max(0.0, overshoot_penalty)),
            "step_count": step_count,
        }
        success = self.is_success(episode_result)
        episode_result["success"] = success
        reward = self.compute_reward(episode_result)

        result = {
            "trajectory": {"ee_poses": np.stack(ee_poses).astype(np.float32)},
            "ee_poses": np.stack(ee_poses).astype(np.float32),
            "actions": np.stack(actions).astype(np.float32) if actions else np.zeros((0, 3), dtype=np.float32),
            "forces": np.stack(forces).astype(np.float32) if forces else np.zeros((0, 3), dtype=np.float32),
            "torques": np.stack(torques).astype(np.float32) if torques else np.zeros((0, 3), dtype=np.float32),
            "final_distance": final_dist,
            "fired": fired,
            "displacement": float(displacement),
            "step_count": step_count,
            "success": success,
            "reward": reward,
            "gains": dict(gains),
            "scene_config": scene_config,
            "stabilizer_grasped": bool(self._stabilizer.grasped) if self._stabilizer is not None else None,
        }
        if self._stabilizer is not None:
            result["left_arm_traj"] = np.stack(left_arm_traj).astype(np.float32)
        return result
