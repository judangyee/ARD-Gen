"""ARD-Gen 검증 태스크 2(나사 조이기)용 admittance controller.

peg_in_hole_sim.py와 같은 스타일(Sim 클래스 + run_episode(gains, scene_config))로
맞췄다. 다른 점은 peg-in-hole의 admittance는 "힘 오차 -> xy 위치 보정"인데,
여기서는 "저항 토크 -> 손목 회전 속도 감속"이다 -- 둘 다 "저항이 커지면
움직임을 줄인다"는 같은 admittance 철학이고, 여기 맞는 자유도(회전 속도)에
적용한 것뿐이다.

## 왜 완료 판정은 토크가 아니라 여전히 기하학(bolt_slide)인가

처음엔 "저항 토크가 ~0.05로 확 뛰면 다 조여진 신호"로 쓸 생각이었는데
(render_screw_driving.py의 옛 주석 참고), 실제로 다회전 시퀀스 전체에
걸쳐 torque 시계열을 실측해보니 그렇게 깔끔하지 않았다 -- bolt_slide가
아직 목표의 1/3도 안 됐을 때(cycle 6, 12mm/34mm)부터 이미 plateau
토크가 껑충 뛰고(0.001 -> 0.077), 그 뒤로도 사이클마다 계속 커진다
(cycle19 즈음엔 액추에이터 forcerange 한계 ±2까지 찍는다). 이건 볼트가
block 벽에 눌리면서 생기는 누적 마찰 효과로 보이는데, 고정 임계값으로는
"덜 조여진 상태"와 "다 조여진 상태"를 안정적으로 구분할 수 없다는 뜻이라
완료 판정용으로 못 쓴다고 판단했다(실측 확인). 그래서 peg_in_hole.py와
똑같은 방식으로: 토크 피드백은 컨트롤 법칙(회전 속도 감속)에만 쓰고,
성공/완료 판정은 여전히 기하학적 신호(bolt_slide 깊이)로 한다.

## 다회전(turn/rewind) 사이클과 admittance의 관계

wrist_rotate 물리 한계(±pi) 때문에 여러 번 "돌리기(engaged)/되감기
(disengaged)"를 반복해야 하는 건 render_screw_driving.py에서 이미 확립된
구조 그대로다. admittance가 붙는 지점은 "돌리기" 구간의 회전 *속도*뿐이다
-- 되감기 구간은 disengaged(볼트에 저항이 안 걸림)라 modulation이 의미
없다.

## ctrl을 "목표값 한 번에 대입 + 오래 정착"이 아니라 "매 스텝 조금씩 램프"로 바꾼 이유

render_screw_driving.py의 다회전 데모는 "ctrl을 목표 극단값으로 한 번에
설정하고 500스텝 동안 정착시키는" 방식을 쓰는데(그게 아니면 실제로 안
움직인다는 걸 그 파일에서 실측으로 확인했다), 그건 torque 신호를 매
스텝 관찰하면서 속도를 조절하는 admittance 컨트롤러와는 안 맞는다(목표를
한 번에 던지면 "지금 얼마나 빨리 돌리고 있는지"를 컨트롤 법칙이 조절할
방법이 없다). 그래서 여기서는 매 컨트롤 틱마다 ctrl을 `rate * dt`만큼만
전진시키는 진짜 속도 제어로 바꿨다 -- 실측으로 확인한 wrist_rotate
액추에이터(kp=7)의 정상상태 추종 특성: rate=1~3 rad/s 구간에서는 위상
지연(steady-state lag)이 rate에 선형 비례(약 0.1rad per rad/s)하고
발산하지 않는다(4초 연속 램프에서 지연이 커지지 않고 일정하게 유지됨,
실측 확인) -- 그래서 이 정도 rate라면 "매 스텝 조금씩 램프"가 실제로
안정적으로 작동한다.
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "screw_driving.xml")

N_SUBSTEPS = 5  # mj_step 호출당 substep 수 (timestep=0.002 -> 제어 주기 dt=0.01s)
DT = N_SUBSTEPS * 0.002

TARGET_DEPTH = 0.034  # bolt_slide 최대 범위(완전 삽입)
PITCH_PER_RAD = 0.002 / (2 * np.pi)  # 나사산 피치 2mm/rev

# 다회전 사이클 범위 (render_screw_driving.py와 동일한 근거: wrist_rotate
# 물리 한계 ±pi에 여유를 둔 구간).
TURN_LOW = -2.8
TURN_HIGH = 2.8

# admittance 속도 제어 파라미터. NOMINAL_RATE=2.5rad/s는 위 정상상태 지연
# 실측(1~3rad/s 구간에서 안정)에 맞춰 고른 값. RATE_MIN>0으로 둬서 저항이
# 아무리 커도 전진이 완전히 멈추거나(역전은 더더욱) 하지 않게 한다 --
# "느려지지만 절대 멈추지 않는다"는 admittance 철학.
NOMINAL_RATE = 2.5  # rad/s, 무저항 시 기본 회전 속도
RATE_MIN = 0.15  # rad/s, 최대 저항 시에도 보장하는 최소 속도

MAX_CONTROL_STEPS = 15000  # 제어 틱 기준 최대 스텝 (안전장치, dt=0.01 -> 150s).
# 6000(60s)으로 처음 테스트했더니 admittance 감속 때문에(저항이 커질수록
# 느려짐) 목표 깊이의 60%(20.6mm/34mm)에서 스텝이 끝나버렸다(실측 확인) --
# 저항이 커지는 뒷부분 사이클이 빨리 도는 앞부분보다 훨씬 오래 걸리는 걸
# 감안해서 넉넉하게 늘렸다.

_ARM_FOLLOW_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle"]
_JAC_DAMPING = 1e-4

_HOME_QPOS = {
    "waist": 0.0,
    "shoulder": -0.89237,
    "elbow": 1.05339,
    "forearm_roll": 0.0,
    "wrist_angle": 1.40932,
    "wrist_rotate": 0.0,
}
_ARM_JOINTS_HOME = list(_HOME_QPOS.keys())


# block_wall_*(seat) 접촉의 friction 계수를 스케일하는 "seat_friction_scale"도
# scene_config 후보로 시도했었다 -- 실측해보니 sliding(index 0)만 스케일하든
# [sliding, torsional, rolling] 세 성분을 다 같이 스케일하든, 0.02배~100배까지
# 흔들어도 max_torque/step_count가 완전히 그대로였다(byte-identical). bolt_hinge
# frictionloss는(아래) 20배 스케일에서 실제로 차이가 났던 것과 대조적이다. 즉 이
# 모델에서는 저항 토크가 seat 접촉의 마찰 계수가 아니라 거의 전적으로
# bolt_hinge의 frictionloss(+ 사이클이 누적될수록 커지는 다른 요인, 아직 정확한
# 원인 불명)에서 나온다는 뜻이라 -- 효과가 없는 손잡이를 scene_config에 넣어두는
# 건 오해만 부르므로 뺐다.
_NOMINAL_HINGE_FRICTIONLOSS = 0.015  # bolt_hinge 기본값


def _default_scene_config() -> dict[str, Any]:
    return {
        "target_depth": TARGET_DEPTH,
        "hinge_friction_scale": 1.0,  # bolt_hinge frictionloss 스케일 (실측상 유의미하게 저항에 영향을 줌)
    }


class ScrewDrivingSim:
    """MjModel/MjData를 재사용하는 시뮬레이션 래퍼 (PegInHoleSim과 같은 역할)."""

    def __init__(self, xml_path: str | None = None):
        self.xml_path = xml_path or _DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self._arm_qposadr = {name: self.model.joint(name).qposadr[0] for name in _ARM_JOINTS_HOME}
        self._arm_dofadr = {name: self.model.joint(name).dofadr[0] for name in _ARM_FOLLOW_JOINTS}
        self._arm_actuator_ids = {name: self.model.actuator(name).id for name in _ARM_JOINTS_HOME}

        self._wrist_qposadr = self.model.joint("wrist_rotate").qposadr[0]
        self._bolt_hinge_qposadr = self.model.joint("bolt_hinge").qposadr[0]
        self._bolt_slide_qposadr = self.model.joint("bolt_slide").qposadr[0]
        self._bolt_hinge_drive_id = self.model.actuator("bolt_hinge_drive").id
        self._bolt_slide_drive_id = self.model.actuator("bolt_slide_drive").id
        self._gripper_actuator_id = self.model.actuator("gripper").id
        self._grasp_weld_id = self.model.equality("driver_grasp").id

        self._tip_site_id = self.model.site("driver_tip_site").id
        self._head_site_id = self.model.site("bolt_head_site").id
        self._jac_ref_site_id = self.model.site("gripper_tip_ref").id
        self._torque_adr = self.model.sensor("bolt_drive_torque").adr[0]

        self._bolt_hinge_dofadr = self.model.joint("bolt_hinge").dofadr[0]

        self._prev_torque = 0.0

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        self._target_offset = np.zeros(3)
        self._phase = "turn"  # "turn"(engaged, 저항 걸림) / "rewind"(disengaged)
        self._engage_wrist_ref = 0.0
        self._engage_hinge_ref = 0.0
        self._frozen_hinge_ctrl = 0.0

    # ------------------------------------------------------------------
    def reset(self, scene_config: dict[str, Any] | None = None) -> None:
        """드라이버를 이미 쥐고 홈 자세에 있는 상태로 초기화한다. 픽업
        애니메이션 자체는 다루지 않는다(peg_in_hole도 grasp 애니메이션은
        다루지 않음, render_screw_driving.py에만 있는 시각화용 시퀀스다).

        처음엔 render_screw_driving.py처럼 "픽업 자세에서 weld 켜고 홈
        자세로 이동"을 흉내내려다 mj_forward만 두 번 부른 버전을 짰는데,
        그게 완전히 틀렸다(실측으로 발견): weld 같은 equality 제약은
        mj_step(물리 적분/제약 솔버)이 돌아야 실제로 자유 바디를 끌고
        오는데, mj_forward는 현재 qpos에서 파생값(xpos 등)만 계산할 뿐
        제약을 풀어서 qpos를 옮기지 않는다. 그래서 팔만 순간이동하듯 홈
        자세로 가고 driver는 원래(픽업) 자리에 그대로 남아서, tip-head
        거리가 11.5mm가 아니라 113mm로 나오는 버그를 실측으로 잡았다.

        고친 방법: 픽업 자세 시뮬레이션 자체를 건너뛰고, 홈 자세로 바로
        forward kinematics를 돌린 뒤 gripper_link의 그 자세를 이용해서
        driver의 자유조인트 qpos(위치+쿼터니언)를 "원래 고정 오프셋
        (0.13,0,0, 무회전)"에 맞게 직접 계산해서 대입한다 -- weld의
        relpose와 정확히 같은 값이라 물리를 한 스텝도 안 돌려도 이미
        제약이 만족된 상태로 시작한다."""
        cfg = _default_scene_config()
        if scene_config:
            cfg.update(scene_config)
        self.model.dof_frictionloss[self._bolt_hinge_dofadr] = (
            _NOMINAL_HINGE_FRICTIONLOSS * cfg["hinge_friction_scale"]
        )

        mujoco.mj_resetData(self.model, self.data)
        for name, val in _HOME_QPOS.items():
            self.data.qpos[self._arm_qposadr[name]] = val
            self.data.ctrl[self._arm_actuator_ids[name]] = val
        mujoco.mj_forward(self.model, self.data)

        gripper_id = self.model.body("gripper_link").id
        driver_qposadr = self.model.joint("driver_free").qposadr[0]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.xquat[gripper_id])
        rot = rot.reshape(3, 3)
        self.data.qpos[driver_qposadr : driver_qposadr + 3] = (
            self.data.xpos[gripper_id] + rot @ np.array([0.13, 0.0, 0.0])
        )
        self.data.qpos[driver_qposadr + 3 : driver_qposadr + 7] = self.data.xquat[gripper_id]
        self.data.eq_active[self._grasp_weld_id] = 1
        self.data.ctrl[self._gripper_actuator_id] = 0.021  # closed
        mujoco.mj_forward(self.model, self.data)

        self._target_offset = (
            self.data.site_xpos[self._tip_site_id] - self.data.site_xpos[self._head_site_id]
        ).copy()

        self._phase = "turn"
        self._engage_wrist_ref = self.data.qpos[self._wrist_qposadr]
        self._engage_hinge_ref = self.data.qpos[self._bolt_hinge_qposadr]
        self._prev_torque = self.get_torque()
        self.data.ctrl[self._arm_actuator_ids["wrist_rotate"]] = TURN_LOW

    def get_torque(self) -> float:
        return float(self.data.sensordata[self._torque_adr])

    def get_insertion_depth(self) -> float:
        return float(self.data.qpos[self._bolt_slide_qposadr])

    # ------------------------------------------------------------------
    def _z_track_step(self) -> None:
        """볼트 slide 구동(나사산 이상화) + 팔 z-추종. gripper_tip_ref
        사이트에서 Jacobian을 가져오는 이유는 render_screw_driving.py에서
        실측으로 확인한 버그(자유바디가 된 driver_tip_site 자체로 Jacobian을
        구하면 팔 관절 쪽이 전부 0이 나온다 -- weld 같은 등호 제약은
        mj_jacSite가 보는 강체 트리에 안 잡힌다) 때문이다."""
        self.data.ctrl[self._bolt_slide_drive_id] = (
            PITCH_PER_RAD * self.data.qpos[self._bolt_hinge_qposadr]
        )
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self._jac_ref_site_id)
        current_offset = self.data.site_xpos[self._tip_site_id] - self.data.site_xpos[self._head_site_id]
        error = self._target_offset - current_offset
        jac_arm = self._jacp[:, [self._arm_dofadr[n] for n in _ARM_FOLLOW_JOINTS]]
        jjt = jac_arm @ jac_arm.T + _JAC_DAMPING * np.eye(3)
        dq = jac_arm.T @ np.linalg.solve(jjt, error * 0.8)
        for i, name in enumerate(_ARM_FOLLOW_JOINTS):
            qadr = self._arm_qposadr[name]
            self.data.ctrl[self._arm_actuator_ids[name]] = self.data.qpos[qadr] + dq[i]

    def step(self, gains: dict[str, float]) -> dict[str, float]:
        """admittance 제어 틱 하나를 진행한다. peg_in_hole의 Kp_xy/Kd_xy와
        같은 PD 구조: 저항 토크 자체(Kp)뿐 아니라 저항이 얼마나 빠르게
        커지고 있는지(Kd, d(torque)/dt)까지 봐서 속도를 줄인다 -- 저항이
        갑자기 급증하는 상황(예: seat에 막 닿는 순간)에 Kp만으로는 한 틱
        늦게 반응하는데, Kd가 있으면 그 변화율에 먼저 반응해서 더 빨리
        늦출 수 있다.

        gains: {"kp_torque": 저항 토크 1단위당 속도 감소량(rad/s per N*m),
                "kd_torque": 저항 변화율 1단위당 속도 감소량(rad/s per
                N*m/s)}. rate = NOMINAL_RATE - kp_torque*max(0,torque) -
                kd_torque*max(0,d_torque), RATE_MIN 이하로는 안 내려간다.

        반환: {"torque": float, "rate": float, "wrist_ctrl": float,
               "phase": "turn"|"rewind"} -- 이번 틱에서 실제로 쓰인 값들
               (에피소드 로깅용)."""
        kp_torque = float(gains.get("kp_torque", 0.0))
        kd_torque = float(gains.get("kd_torque", 0.0))
        wrist_act = self._arm_actuator_ids["wrist_rotate"]
        current_wrist_ctrl = float(self.data.ctrl[wrist_act])
        torque = self.get_torque()
        d_torque = (torque - self._prev_torque) / DT
        self._prev_torque = torque

        if self._phase == "turn":
            resistance = kp_torque * max(0.0, abs(torque)) + kd_torque * max(0.0, d_torque)
            rate = max(RATE_MIN, NOMINAL_RATE - resistance)
            new_ctrl = min(current_wrist_ctrl + rate * DT, TURN_HIGH)
            self.data.ctrl[wrist_act] = new_ctrl
            self.data.ctrl[self._bolt_hinge_drive_id] = (
                self._engage_hinge_ref + (self.data.qpos[self._wrist_qposadr] - self._engage_wrist_ref)
            )
            if new_ctrl >= TURN_HIGH:
                self._phase = "rewind"
                self._frozen_hinge_ctrl = float(self.data.ctrl[self._bolt_hinge_drive_id])
        else:  # rewind: disengaged, 저항이 안 걸리니 admittance 감속 없이 nominal rate로
            rate = NOMINAL_RATE
            new_ctrl = max(current_wrist_ctrl - rate * DT, TURN_LOW)
            self.data.ctrl[wrist_act] = new_ctrl
            self.data.ctrl[self._bolt_hinge_drive_id] = self._frozen_hinge_ctrl
            if new_ctrl <= TURN_LOW:
                self._phase = "turn"
                self._engage_wrist_ref = self.data.qpos[self._wrist_qposadr]
                self._engage_hinge_ref = self.data.qpos[self._bolt_hinge_qposadr]

        self._z_track_step()
        mujoco.mj_step(self.model, self.data, nstep=N_SUBSTEPS)

        return {"torque": torque, "rate": rate, "wrist_ctrl": new_ctrl, "phase": self._phase}


def run_episode(gains: dict[str, float], scene_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """admittance controller로 한 에피소드(나사 하나 완전 삽입 또는
    MAX_CONTROL_STEPS 도달)를 실행한다.

    Args:
        gains: {"kp_torque": float, "kd_torque": float}
        scene_config: {"target_depth": float, "hinge_friction_scale": float}
            (없는 키는 _default_scene_config() 기본값)

    Returns:
        dict with: depth_profile, torque_profile, rate_profile, phase_profile,
        insertion_depth(float), success(bool), reward(float), step_count(int),
        max_torque(float).
    """
    cfg = _default_scene_config()
    if scene_config:
        cfg.update(scene_config)
    sim = ScrewDrivingSim()
    return _run_episode_with_sim(sim, gains, cfg)


def _run_episode_with_sim(
    sim: ScrewDrivingSim, gains: dict[str, float], cfg: dict[str, Any]
) -> dict[str, Any]:
    sim.reset(cfg)
    target_depth = float(cfg["target_depth"])

    depth_profile = []
    torque_profile = []
    rate_profile = []
    phase_profile = []
    max_torque = 0.0
    success = False
    step_count = 0

    for step_count in range(1, MAX_CONTROL_STEPS + 1):
        info = sim.step(gains)
        depth = sim.get_insertion_depth()
        depth_profile.append(depth)
        torque_profile.append(info["torque"])
        rate_profile.append(info["rate"])
        phase_profile.append(info["phase"])
        max_torque = max(max_torque, abs(info["torque"]))

        if depth >= target_depth * 0.99:
            success = True
            break

    final_depth = depth_profile[-1] if depth_profile else 0.0
    mean_abs_torque = float(np.mean(np.abs(torque_profile))) if torque_profile else 0.0
    # max_torque가 아니라 mean_abs_torque로 페널티를 준다 -- 실측해보니
    # kp_torque를 0->2.0까지 올려도 max_torque는 거의 그대로였다(1.38->1.37,
    # 오차 수준). 이 모델의 저항은 회전 "속도"가 아니라 사이클(=깊이)에 달려
    # 있어서(뒤 사이클일수록 저항이 누적돼서 커짐, assets/screw_driving.xml
    # 참고), 느리게 돈다고 그 순간의 저항 자체가 줄지는 않는다 -- 다만
    # kp_torque를 올리면 스텝 수가 늘어나서(더 오래 걸려서) 오히려
    # mean_abs_torque가 살짝 올라간다(더 오래 저항 구간에 머무르므로). 즉
    # 이 컨트롤 법칙은 (이 모델 한정) "저항을 줄여주는" 효과가 없고, 순전히
    # "느려지는 비용"만 있다는 게 실측으로 드러난 결론이다 -- 정직하게
    # mean_abs_torque를 그대로 페널티에 반영해서 CMA-ES가 이 트레이드오프를
    # 있는 그대로 보고 선택하게 한다.
    reward = 20.0 * final_depth - 2.0 * mean_abs_torque - 0.001 * step_count
    if success:
        reward += 50.0

    return {
        "depth_profile": np.array(depth_profile, dtype=np.float32),
        "torque_profile": np.array(torque_profile, dtype=np.float32),
        "rate_profile": np.array(rate_profile, dtype=np.float32),
        "phase_profile": phase_profile,
        "insertion_depth": float(final_depth),
        "max_torque": float(max_torque),
        "mean_abs_torque": mean_abs_torque,
        "step_count": step_count,
        "success": success,
        "reward": float(reward),
        "gains": dict(gains),
        "scene_config": cfg,
    }


if __name__ == "__main__":
    result = run_episode({"kp_torque": 1.2, "kd_torque": 0.0})
    print(
        f"[screw_driving_sim] success={result['success']} "
        f"depth={result['insertion_depth'] * 1000:.2f}mm "
        f"steps={result['step_count']} max_torque={result['max_torque']:.3f}"
    )
