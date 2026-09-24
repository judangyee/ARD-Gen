"""ARD-Gen 검증 태스크 2(나사 조이기)용 토크 리미터 컨트롤러.

peg_in_hole_sim.py와 같은 스타일(Sim 클래스 + run_episode(gains, scene_config))로
맞췄다.

## 속도 조절형 admittance에서 토크 리미터로 바꾼 이유 (실측으로 확인)

처음엔 peg-in-hole의 "힘 오차 -> xy 위치 보정" admittance를 그대로 본떠서
"저항 토크가 크면 회전 속도를 늦춘다"(rate = NOMINAL_RATE -
kp_torque*torque)는 PD 컨트롤러를 만들었다. 그런데 optimize/screw_driving_cma_search.py로
실제 게인 탐색을 돌려보니 CMA-ES가 kp_torque, kd_torque를 전부 거의
0으로 수렴시켰다(리워드가 kp_torque=0 베이스라인과 완전히 동일) -- 즉
CMA-ES가 "느려질 이유가 없다"고 스스로 판단한 것이다. 원인을 실측으로
추적해보니: kp_torque를 0->2.0까지 올려도 max_torque가 거의 그대로였다
(1.380 -> 1.368, 오차 수준). 이 모델의 저항 토크는 회전 *속도*가 아니라
*사이클(=삽입 깊이)*에 달려 있어서(뒤 사이클일수록 저항이 누적돼서 커짐),
느리게 돈다고 그 순간의 저항 자체가 줄어들지 않는다 -- 오히려 느리게
돌면 저항이 큰 뒷부분 사이클에 상대적으로 더 오래 머무르니 평균 저항이
살짝 올라가기까지 했다.

이게 바로 실제 전동 드라이버가 "속도를 조절하는" 방식이 아니라 "설정
토크에 도달하면 클러치가 미끄러지며 멈추는" 토크 리미터 방식을 쓰는
이유와 같은 결론이라고 보고, 컨트롤 법칙 자체를 토크 리미터로
다시 짰다: 속도는 항상 NOMINAL_RATE로 일정하게 돌리고, 저항 토크가
`torque_limit`을 넘는 순간 그 사이클의 "돌리기"를 즉시 멈추고 되감기로
전환한다. 이건 "그 순간의 저항을 줄이는" 게 아니라 "저항이 일정 수준을
넘으면 즉시 손을 뗀다"는, 속도 조절과는 본질적으로 다른 메커니즘이라
실제로 max_torque를 낮출 수 있다(step()의 `limited` 반환값으로 이 파일
__main__ 데모에서 확인 가능).

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
똑같은 방식으로: 토크 피드백은 컨트롤 법칙(토크 리미터)에만 쓰고,
성공/완료 판정은 여전히 기하학적 신호(bolt_slide 깊이)로 한다.

## 다회전(turn/rewind) 사이클과 토크 리미터의 관계

wrist_rotate 물리 한계(±pi) 때문에 여러 번 "돌리기(engaged)/되감기
(disengaged)"를 반복해야 하는 건 render_screw_driving.py에서 이미 확립된
구조 그대로다. 토크 리미터가 붙는 지점은 "돌리기" 구간뿐이다 -- 되감기
구간은 disengaged(볼트에 저항이 안 걸림)라 리밋이 걸릴 일이 없다.
torque_limit에 걸리면 그 사이클의 "돌리기"가 원래 도달했어야 할
TURN_HIGH까지 못 가고 일찍 끝나므로, 사이클당 회전량(=삽입 진행량)이
줄어서 완료까지 사이클 수가 늘어난다 -- 이게 이 컨트롤러가 다루는
진짜 트레이드오프다(안전하게 낮은 한계 vs 빠르게 끝내기).

## ctrl을 "목표값 한 번에 대입 + 오래 정착"이 아니라 "매 스텝 조금씩 램프"로 바꾼 이유

render_screw_driving.py의 다회전 데모는 원래 "ctrl을 목표 극단값으로 한
번에 설정하고 500스텝 동안 정착시키는" 방식을 썼는데(그게 아니면 실제로
안 움직인다는 걸 그 파일에서 실측으로 확인했다), 그건 매 스텝 토크를
읽어서 반응해야 하는 컨트롤러와는 안 맞는다(목표를 한 번에 던지면 중간에
끼어들 방법이 없다). 그래서 여기서는 매 컨트롤 틱마다 ctrl을
`NOMINAL_RATE * dt`만큼만 전진시키는 진짜 속도 제어로 바꿨다 -- 실측으로
확인한 wrist_rotate 액추에이터(kp=7)의 정상상태 추종 특성: rate=1~3 rad/s
구간에서는 위상 지연(steady-state lag)이 rate에 선형 비례(약 0.1rad per
rad/s)하고 발산하지 않는다(4초 연속 램프에서 지연이 커지지 않고 일정하게 유지됨,
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

# 회전 속도(토크 리미터가 걸리지 않는 동안은 항상 이 속도). 2.5rad/s는 위
# 정상상태 지연 실측(1~3rad/s 구간에서 안정)에 맞춰 고른 값.
NOMINAL_RATE = 2.5  # rad/s

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
        """제어 틱 하나를 진행한다.

        속도 조절형 admittance(rate = NOMINAL_RATE - kp*torque)는 CMA-ES로
        실측 검증해본 결과 이 모델에서는 실질적 이득이 없었다(kp_torque를
        0->2.0까지 올려도 max_torque가 1.380->1.368로 거의 그대로, CMA-ES도
        독립적으로 kp_torque~0으로 수렴함 -- optimize/screw_driving_cma_search.py
        참고). 이유: 이 모델의 저항은 회전 속도가 아니라 사이클(깊이)에
        달려 있어서, 느리게 돈다고 그 순간의 저항이 줄지 않는다.

        그래서 실제 전동 드라이버의 토크 리미터(클러치)처럼 재설계했다:
        속도는 항상 NOMINAL_RATE로 일정하게 돌리고, 저항 토크가
        `torque_limit`을 넘으면(클러치가 미끄러지는 것과 같은 상황) 그
        사이클의 "돌리기"를 즉시 멈추고 되감기로 넘어간다 -- 속도를
        조절하는 게 아니라 "이 이상은 안 된다"는 상한선 자체를 강제하는
        방식이라, 속도 조절과 달리 실제로 max_torque를 낮출 수 있다(실측
        검증: 이 파일 __main__ 참고).

        gains: {"torque_limit": float} -- 이 값을 넘는 순간 그 사이클의
               돌리기를 중단한다. 너무 낮으면(그 사이클에서 필요한 저항보다
               낮으면) 회전을 거의 못 하고 매번 멈춰서 완료까지 사이클
               수가 크게 늘어난다 -- 그 트레이드오프를 게인 탐색이 실제로
               보게 하려는 목적.

        반환: {"torque": float, "rate": float, "wrist_ctrl": float,
               "phase": "turn"|"rewind", "limited": bool} -- 이번 틱에서
               실제로 쓰인 값들(에피소드 로깅용). limited=True면 이번
               틱에서 torque_limit에 걸려 강제로 되감기로 전환됐다는 뜻."""
        torque_limit = float(gains.get("torque_limit", np.inf))
        wrist_act = self._arm_actuator_ids["wrist_rotate"]
        current_wrist_ctrl = float(self.data.ctrl[wrist_act])
        torque = self.get_torque()
        limited = False

        if self._phase == "turn":
            if abs(torque) >= torque_limit:
                limited = True
                self._phase = "rewind"
                self._frozen_hinge_ctrl = float(self.data.ctrl[self._bolt_hinge_drive_id])
                new_ctrl = current_wrist_ctrl
            else:
                new_ctrl = min(current_wrist_ctrl + NOMINAL_RATE * DT, TURN_HIGH)
                self.data.ctrl[wrist_act] = new_ctrl
                self.data.ctrl[self._bolt_hinge_drive_id] = (
                    self._engage_hinge_ref + (self.data.qpos[self._wrist_qposadr] - self._engage_wrist_ref)
                )
                if new_ctrl >= TURN_HIGH:
                    self._phase = "rewind"
                    self._frozen_hinge_ctrl = float(self.data.ctrl[self._bolt_hinge_drive_id])
            rate = NOMINAL_RATE if not limited else 0.0
        else:  # rewind: disengaged, 저항이 안 걸리니 항상 nominal rate로 되감는다
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

        return {"torque": torque, "rate": rate, "wrist_ctrl": new_ctrl, "phase": self._phase, "limited": limited}


def run_episode(gains: dict[str, float], scene_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """토크 리미터 컨트롤러로 한 에피소드(나사 하나 완전 삽입 또는
    MAX_CONTROL_STEPS 도달)를 실행한다.

    Args:
        gains: {"torque_limit": float}
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
    # 처음엔 success 여부와 상관없이 항상 -3.0*max_torque를 뺐는데, 그렇게
    # CMA-ES를 돌려보니 torque_limit=0.15(자연 문턱값 ~1.4의 1/10)로
    # 수렴해버렸다(실측 확인: reward=-15.5, "리미터 없음" 베이스라인의
    # +36.9보다 훨씬 나쁜데도!). 원인: 실패(데드락) 구간에서는 torque_limit을
    # 낮출수록 max_torque도 같이 낮아지니까, "무조건 낮은 torque_limit이
    # 리워드에 유리"해 보이는 가짜 경사가 생겨서 CMA-ES가 성공 문턱값과는
    # 반대 방향(더 낮게)으로 끌려갔다 -- 정작 성공 여부(depth, +50 보너스)가
    # 주는 진짜 신호보다 이 가짜 경사가 훨씬 강했다. 그래서 max_torque
    # 페널티는 "완료된 경우에만" 적용하도록 고쳤다 -- 실패한 시도의 토크가
    # 낮다고 보상해줄 이유가 없고(어차피 나사를 못 박았으니), 이렇게 해야
    # "일단 문턱값을 넘겨서 완료하고, 그 안에서 torque_limit을 낮춘다"는
    # 원래 의도한 트레이드오프만 남는다.
    reward = 20.0 * final_depth - 0.001 * step_count
    if success:
        reward += 50.0 - 3.0 * max_torque

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
    # torque_limit 몇 개를 비교해서 실제로 max_torque를 낮추는지(속도 조절형
    # admittance와 다르게) 눈으로 확인한다.
    for torque_limit in [np.inf, 1.0, 0.6, 0.3]:
        result = run_episode({"torque_limit": torque_limit})
        label = "no limit" if np.isinf(torque_limit) else f"limit={torque_limit}"
        print(
            f"[screw_driving_sim] {label:12s} success={result['success']} "
            f"depth={result['insertion_depth'] * 1000:.2f}mm "
            f"steps={result['step_count']} max_torque={result['max_torque']:.3f}"
        )
