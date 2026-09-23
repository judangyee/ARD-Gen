"""screw_driving.xml 모델 검증/데모 렌더링 스크립트.

sim/screw_driving_sim.py의 admittance controller(토크 피드백으로 회전
속도를 조절)를 그대로 가져다 써서 렌더링한다 -- 컨트롤 로직은 sim 모듈이
소스이고(_KP_TORQUE 등 게인도 거기서 import), 이 스크립트는 카메라
캡처/픽업 애니메이션만 더한 것이다.

## 왜 "여러 바퀴(다회전)" 시퀀스가 필요한가

피치 2mm/rev로 목표 삽입 깊이(34mm)까지 박으려면 17바퀴가 필요한데,
wrist_rotate는 실제 VX300s처럼 관절 범위가 ±180도(반바퀴)뿐이다. 그래서
실제 드라이버 작업처럼 "한계까지 돌리기 -> 놓고 손목만 되감기(볼트는 그
자리에 고정) -> 다시 물고 이어서 돌리기"를 여러 번 반복한다(완료 조건은
bolt_slide가 목표 깊이에 도달하는 것, _MAX_CONTROL_TICKS는 안전장치).
이전 버전은 wrist_rotate를 0->pi까지 딱 한 번만 돌리고 끝냈는데, 그때는
사실 나사산 관계 자체가 버그(중력만으로도 회전 없이 미끄러지는 문제)로
깨져 있어서 "적은 회전으로도 끝까지 박히는" 것처럼 보였을 뿐이다 -- 그
버그를 실제 액추에이터 기반으로 고치고 나니(assets/screw_driving.xml
참고) 반바퀴로는 1mm도 안 들어간다는 실제 물리가 드러났고, 그래서 이
다회전 시퀀스를 제대로 구현하게 됐다.

## admittance: 저항 토크로 회전 속도를 조절

처음(다회전만 구현했을 때) 버전은 "돌리기" 구간마다 ctrl을 목표 극단값
(-2.8/2.8)으로 한 번에 던지고 500스텝 동안 정착시키는 방식이었다 --
wrist_rotate 액추에이터(kp=7)가 즉각적인 큰 램프를 못 따라간다는 걸
실측으로 확인하고 고른 방법이었는데, 이 방식은 "지금 얼마나 빨리
돌리고 있는지"를 컨트롤 법칙이 조절할 방법이 없어서 진짜 admittance와
안 맞았다. 그래서 매 컨트롤 틱(dt=0.01s)마다 ctrl을 `rate*dt`만큼만
전진시키는 진짜 속도 제어로 바꿨다: `rate = max(RATE_MIN, NOMINAL_RATE -
kp_torque*저항토크)` -- 저항이 커질수록 느려지되(admittance), RATE_MIN
아래로는 안 내려가서(전진이 완전히 멈추지 않음) 항상 목표에 수렴한다.
wrist_rotate의 정상상태 추종 지연이 이 rate 범위(1~3rad/s)에서 rate에
선형 비례하고 발산하지 않는다는 걸 실측으로 확인했다(sim/screw_driving_sim.py
docstring 참고). "되감기" 구간은 disengaged라 저항이 안 걸리니 admittance
감속 없이 항상 nominal rate로 되감는다.

완료 판정은 (peg_in_hole과 마찬가지로) 토크가 아니라 기하학(bolt_slide
깊이)로 한다 -- 다회전 전체에 걸쳐 저항 토크를 실측해보니 아직 목표
깊이의 1/3도 안 됐을 때부터 이미 plateau 토크가 크게 뛰고 사이클마다
계속 커져서(block 벽에 눌리며 생기는 누적 마찰로 보임), 고정 임계값으로
"덜 조여짐"과 "다 조여짐"을 구분할 수 없었다(실측 확인). 토크 피드백은
컨트롤 법칙(속도 조절)에만 쓰고, 완료 판정은 여전히 기하학 신호를 쓴다.

## 팔이 볼트를 따라 내려가야 하는 이유 (실측으로 발견한 문제)

볼트가 다 조여지는 동안 bolt_slide가 약 34mm 움직이는데, 팔이 전혀
움직이지 않으면 드라이버(그리퍼에 고정)는 제자리에 남아있고 볼트만
아래로 내려가버려서, 영상 후반부에는 드라이버와 볼트 머리 사이 간격이
처음(약 12mm)보다 훨씬 크게 벌어진다 -- 실제로 렌더링해서 확인한 문제다.

그래서 매 스텝 "이번 스텝에 볼트가 내려간 만큼(delta bolt_slide)"을
Jacobian 기반으로 팔에 반영해서, 드라이버가 볼트를 계속 따라 내려가게
했다(wrist_rotate는 회전 전용으로 남겨두고, 나머지 5개 관절
waist/shoulder/elbow/forearm_roll/wrist_angle로 위치를 보정한다). 이건
진짜 admittance controller가 아니라 순전히 "데모 영상이 끝까지 붙어있게"
만드는 임시 추종 보정이다 -- 나중에 토크 기반 컨트롤러를 만들 때 이
z-추종 역할을 그 컨트롤러가 대신하게 된다.

## 드라이버를 바닥에서 집는 것부터 시작 (grasp 단순화)

원래 driver는 gripper_link의 자식으로 강체 고정돼 있었는데(peg_in_hole의
peg와 같은 패턴), "드라이버를 바닥에 두고 줍는 것부터 시작하자"는 요청으로
assets/screw_driving.xml에서 자유 바디(freejoint)로 바꿨다. 실제 손가락
접촉/마찰로 쥐는 물리는 다루지 않고(이 프로젝트 전체에 걸친 기존
단순화 방향과 동일하게), 팔이 픽업 자세에 도달해서 손가락을 닫는 순간
코드에서 weld equality(driver_grasp)를 active=1로 켜서 "쥐었다"를
흉내낸다. weld의 relpose가 원래 고정 오프셋(0.13,0,0, 무회전)과 정확히
같고, 드라이버의 초기 자유 바디 위치도 그 오프셋을 픽업 자세에서
forward-kinematics로 역산해 박아넣은 값이라, weld를 켜는 순간 스냅(튕김)
없이 자연스럽게 "잡힌다"(실측 확인: 300스텝 방치해도 드러버 위치 변화
<0.001mm).

픽업 자세(waist=-1.2, 나머지 관절은 홈 자세와 동일)에서 waist만 0으로
되돌리는 것만으로 홈 자세(기존 나사 조이기 시퀀스가 시작하던 자리)로
정확히 돌아온다 -- 그래서 픽업+운반 단계는 별도 Jacobian IK 없이 두
qpos 딕셔너리 사이를 직접 ctrl로 오가는 것만으로 충분하다.

사용 예:
    MUJOCO_GL=osmesa python render_screw_driving.py --out-dir .
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sim.screw_driving_sim import DT, N_SUBSTEPS, NOMINAL_RATE, PITCH_PER_RAD, RATE_MIN, TURN_HIGH, TURN_LOW

_HOME_QPOS = {
    "waist": 0.0,
    "shoulder": -0.89237,
    "elbow": 1.05339,
    "forearm_roll": 0.0,
    "wrist_angle": 1.40932,
    "wrist_rotate": 0.0,
}
# 픽업 자세: 홈 자세와 waist만 다르다(-1.2rad). assets/screw_driving.xml의
# driver 초기 위치가 바로 이 자세에서 forward-kinematics로 역산된 값이라,
# 팔이 여기 도달하면 그리퍼가 정확히 드라이버 자리에 와 있다.
_PICKUP_QPOS = {
    "waist": -1.2,
    "shoulder": -0.89237,
    "elbow": 1.05339,
    "forearm_roll": 0.0,
    "wrist_angle": 1.40932,
    "wrist_rotate": 0.0,
}
# 시작 자세: 픽업 자세도 홈 자세도 아닌 임의의 자세에서 출발해서, 팔이
# 실제로 "다가가서 집는" 모습이 보이게 한다.
_REST_QPOS = {
    "waist": 0.6,
    "shoulder": -0.2,
    "elbow": 0.3,
    "forearm_roll": 0.0,
    "wrist_angle": 0.0,
    "wrist_rotate": 0.0,
}
_GRIPPER_OPEN = 0.057
_GRIPPER_CLOSED = 0.021
_STEPS_APPROACH = 500
_STEPS_CLOSE = 100
_STEPS_GRASP_HOLD = 50
_STEPS_TRANSPORT = 500
_ARM_FOLLOW_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle"]
_JAC_DAMPING = 1e-4
_TARGET_DEPTH = 0.034  # bolt_slide 최대 범위(완전 삽입)

# 다회전(turn/rewind) + admittance 속도 제어 파라미터는 sim/screw_driving_sim.py
# 것을 그대로 가져다 쓴다 -- 이 렌더 스크립트가 실제로 쓰는 컨트롤러와 다른
# 로직을 보여주면 "검증"의 의미가 없기 때문이다. 이전 버전(커밋 이력 참고)은
# 여기서 자체적으로 "ctrl 목표값 한 번에 대입 + 500스텝 정착"하는 방식을
# 썼는데, 그건 admittance(토크로 속도를 조절)와 근본적으로 안 맞는 방식이라
# (목표를 한 번에 던지면 "지금 얼마나 빨리 돌릴지"를 조절할 방법이 없다)
# sim 모듈 쪽에서 매 컨트롤 틱마다 `rate*dt`만큼만 전진시키는 진짜 속도
# 제어로 바꿨고, 이 렌더 스크립트도 그걸 그대로 따라간다.
_KP_TORQUE = 1.2  # sim/screw_driving_sim.py에서 실측 검증된 기본 게인
_MAX_CONTROL_TICKS = 15000
_CAPTURE_EVERY = 15


def _overlay_depth_readout(frame: np.ndarray, depth_m: float) -> np.ndarray:
    """영상만 보면 사이클마다 손목이 크게 왔다갔다 흔드는 동작에 묻혀서
    실제 삽입 진행(사이클당 ~1.7mm)이 잘 안 보인다는 피드백을 받고 추가한
    번인(burn-in) 오버레이. 화면 어디를 보든 숫자/막대로 진행률이 명확하게
    드러나게 한다."""
    pct = max(0.0, min(1.0, depth_m / _TARGET_DEPTH))
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    text = f"depth {depth_m * 1000:5.1f} / {_TARGET_DEPTH * 1000:.0f} mm ({pct * 100:4.1f}%)"
    draw.rectangle([4, 4, 250, 24], fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 220, 0))
    bar_x0, bar_x1, bar_y0, bar_y1 = 8, 246, 28, 36
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(40, 40, 40))
    draw.rectangle([bar_x0, bar_y0, bar_x0 + int((bar_x1 - bar_x0) * pct), bar_y1], fill=(80, 200, 80))
    return np.array(img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default="./assets/screw_driving.xml")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--freeze-frames", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    m = mujoco.MjModel.from_xml_path(args.model_path)
    d = mujoco.MjData(m)
    for name, val in _REST_QPOS.items():
        d.qpos[m.joint(name).qposadr[0]] = val
    mujoco.mj_forward(m, d)
    for name, val in _REST_QPOS.items():
        d.ctrl[m.actuator(name).id] = val
    d.ctrl[m.actuator("gripper").id] = _GRIPPER_OPEN

    arm_dofadr = [m.joint(n).dofadr[0] for n in _ARM_FOLLOW_JOINTS]
    arm_qposadr = [m.joint(n).qposadr[0] for n in _ARM_FOLLOW_JOINTS]
    wrist_act = m.actuator("wrist_rotate").id
    wrist_qpos = m.joint("wrist_rotate").qposadr[0]
    bolt_drive_act = m.actuator("bolt_hinge_drive").id
    bolt_slide_drive_act = m.actuator("bolt_slide_drive").id
    bolt_hinge_qpos = m.joint("bolt_hinge").qposadr[0]
    bolt_slide_qpos = m.joint("bolt_slide").qposadr[0]
    torque_adr = m.sensor("bolt_drive_torque").adr[0]
    tip_site_id = m.site("driver_tip_site").id
    head_site_id = m.site("bolt_head_site").id
    jac_ref_site_id = m.site("gripper_tip_ref").id
    grasp_weld_id = m.equality("driver_grasp").id

    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))

    top_r = mujoco.Renderer(m, height=args.height, width=args.width)
    wrist_r = mujoco.Renderer(m, height=args.height, width=args.width)
    closeup_r = mujoco.Renderer(m, height=args.height, width=args.width)
    top_frames: list = []
    wrist_frames: list = []
    closeup_frames: list = []

    def capture() -> None:
        depth_m = d.qpos[bolt_slide_qpos]
        top_r.update_scene(d, camera="top_cam")
        top_frames.append(_overlay_depth_readout(top_r.render().copy(), depth_m))
        wrist_r.update_scene(d, camera="wrist_cam")
        wrist_frames.append(_overlay_depth_readout(wrist_r.render().copy(), depth_m))
        closeup_r.update_scene(d, camera="closeup_cam")
        closeup_frames.append(_overlay_depth_readout(closeup_r.render().copy(), depth_m))

    capture()

    def settle_arm(target_qpos: dict, steps: int) -> None:
        """픽업/운반 단계용: Jacobian 추종 없이 6개 팔 관절 ctrl을 목표
        qpos로 그냥 대입해두고 충분한 스텝 동안 실제로 수렴하게 둔다
        (wrist_rotate 다회전 시퀀스에서 검증된 "램프 대신 목표값+충분한
        정착 스텝" 패턴과 동일)."""
        for name, val in target_qpos.items():
            d.ctrl[m.actuator(name).id] = val
        for t in range(steps):
            mujoco.mj_step(m, d)
            if (t + 1) % _CAPTURE_EVERY == 0:
                capture()

    # 1) 접근: 임의의 시작 자세(_REST_QPOS)에서 픽업 자세로 이동. 이 시점엔
    #    아직 weld가 꺼져 있어서(active=false) 드라이버는 gravcomp로 픽업
    #    스탠드 자리에 가만히 떠 있는다.
    settle_arm(_PICKUP_QPOS, _STEPS_APPROACH)

    # 2) 손가락 닫기: 실제 접촉/마찰로 쥐는 물리는 다루지 않고, 시각적으로만
    #    닫는다 (이 시점에 그리퍼가 정확히 드라이버 자리에 와 있다 --
    #    driver 초기 위치가 이 픽업 자세에서 역산된 값이기 때문).
    d.ctrl[m.actuator("gripper").id] = _GRIPPER_CLOSED
    for t in range(_STEPS_CLOSE):
        mujoco.mj_step(m, d)
        if (t + 1) % _CAPTURE_EVERY == 0:
            capture()

    # 3) grasp: weld를 켜서 "쥐었다"를 흉내낸다. relpose가 지금 실제
    #    상대 자세와 정확히 일치하도록 맞춰뒀기 때문에 스냅(튕김) 없이
    #    조용히 붙는다(실측 확인, 300스텝 방치해도 위치 변화 <0.001mm).
    d.eq_active[grasp_weld_id] = 1
    for t in range(_STEPS_GRASP_HOLD):
        mujoco.mj_step(m, d)
        if (t + 1) % _CAPTURE_EVERY == 0:
            capture()

    # 4) 운반: 픽업 자세 -> 홈 자세(waist만 원위치로). weld가 켜져 있으니
    #    드라이버가 그리퍼를 따라 그대로 딸려온다.
    settle_arm(_HOME_QPOS, _STEPS_TRANSPORT)

    # 목표 상대 오프셋(드라이버 팁 - 볼트 머리)을 시작 시점에 한 번만 기록해두고,
    # 매 스텝 "이번 스텝 변화량만" ctrl에 누적"하는 방식은 실측해보니 650스텝
    # 동안 오차가 계속 쌓여 center가 최대 6mm까지 벌어졌다. 그 다음 시도한
    # "오차*게인을 매 스텝 ctrl에 누적"하는 폐루프도 적분(integrator)처럼
    # 작동해서 91mm까지 발산했다. 최종적으로는 ctrl을 "누적"하지 않고
    # "현재 실제 qpos + 오차 보정"으로 매 스텝 새로 계산해서 대입하는 방식
    # (peg_in_hole_sim.py의 반복 IK와 같은 패턴)으로 안정화했다 -- 650스텝
    # 뒤 오차가 시작 시점 대비 0.03mm까지 수렴하는 걸 확인했다.
    target_offset = d.site(tip_site_id).xpos - d.site(head_site_id).xpos

    def follow_tick() -> None:
        """볼트 slide 구동 + 팔 z-추종. 매 컨트롤 틱(N_SUBSTEPS만큼 물리
        서브스텝)마다 호출한다.

        driver가 gripper_link의 자식이 아니라 weld로 붙은 자유 바디가 된
        뒤로는, driver_tip_site 자체의 Jacobian을 쓰면 안 된다 --
        실측해보니 팔 관절에 대한 Jacobian이 전부 0이었다(weld 같은 등호
        제약은 mj_jacSite가 보는 강체 트리 구조에 안 잡힌다). 그래서
        "팔을 움직이면 실제로 뭐가 얼마나 움직이는가"는 gripper_link 위의
        참조 사이트(gripper_tip_ref, 자식이었을 때의 tip 오프셋과 동일)로
        구하고, 실제 오차(웰드의 잔류 컴플라이언스까지 포함한 진짜 위치
        차이)는 여전히 진짜 tip/head 사이트로 계산한다."""
        d.ctrl[bolt_slide_drive_act] = PITCH_PER_RAD * d.qpos[bolt_hinge_qpos]

        mujoco.mj_jacSite(m, d, jacp, jacr, jac_ref_site_id)
        current_offset = d.site(tip_site_id).xpos - d.site(head_site_id).xpos
        error = target_offset - current_offset
        jac_arm = jacp[:, arm_dofadr]
        jjt = jac_arm @ jac_arm.T + _JAC_DAMPING * np.eye(3)
        dq = jac_arm.T @ np.linalg.solve(jjt, error * 0.8)
        for name, qpos_adr, dof_local in zip(_ARM_FOLLOW_JOINTS, arm_qposadr, range(len(_ARM_FOLLOW_JOINTS))):
            d.ctrl[m.actuator(name).id] = d.qpos[qpos_adr] + dq[dof_local]

        mujoco.mj_step(m, d, nstep=N_SUBSTEPS)

    # admittance 다회전 시퀀스: sim/screw_driving_sim.py의 ScrewDrivingSim.step()과
    # 정확히 같은 컨트롤 법칙. "돌리기(engaged)" 구간에서만 저항 토크로 속도를
    # 늦추고(admittance), "되감기(disengaged)" 구간은 저항이 안 걸리니 항상
    # nominal rate로 되감는다.
    phase = "turn"
    engage_wrist_ref = d.qpos[wrist_qpos]
    engage_hinge_ref = d.qpos[bolt_hinge_qpos]
    d.ctrl[wrist_act] = TURN_LOW

    for tick in range(1, _MAX_CONTROL_TICKS + 1):
        current_wrist_ctrl = float(d.ctrl[wrist_act])
        torque = float(d.sensordata[torque_adr])

        if phase == "turn":
            rate = max(RATE_MIN, NOMINAL_RATE - _KP_TORQUE * max(0.0, abs(torque)))
            new_ctrl = min(current_wrist_ctrl + rate * DT, TURN_HIGH)
            d.ctrl[wrist_act] = new_ctrl
            d.ctrl[bolt_drive_act] = engage_hinge_ref + (d.qpos[wrist_qpos] - engage_wrist_ref)
            if new_ctrl >= TURN_HIGH:
                phase = "rewind"
                frozen_hinge_ctrl = float(d.ctrl[bolt_drive_act])
        else:
            new_ctrl = max(current_wrist_ctrl - NOMINAL_RATE * DT, TURN_LOW)
            d.ctrl[wrist_act] = new_ctrl
            d.ctrl[bolt_drive_act] = frozen_hinge_ctrl
            if new_ctrl <= TURN_LOW:
                phase = "turn"
                engage_wrist_ref = d.qpos[wrist_qpos]
                engage_hinge_ref = d.qpos[bolt_hinge_qpos]

        follow_tick()
        if tick % _CAPTURE_EVERY == 0:
            capture()

        if d.qpos[bolt_slide_qpos] >= _TARGET_DEPTH * 0.99:
            break

    for _ in range(args.freeze_frames):
        capture()

    tip = d.site("driver_tip_site").xpos
    head = d.site("bolt_head_site").xpos
    print(
        f"[render_screw_driving] final bolt_slide={d.qpos[bolt_slide_qpos]:.5f} "
        f"tip-head distance={np.linalg.norm(tip - head) * 1000:.2f}mm "
        f"torque={d.sensordata[torque_adr]:.4f}"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    top_path = os.path.join(args.out_dir, "screw_driving_top.mp4")
    wrist_path = os.path.join(args.out_dir, "screw_driving_wrist.mp4")
    closeup_path = os.path.join(args.out_dir, "screw_driving_closeup.mp4")
    imageio.mimsave(top_path, top_frames, fps=args.fps, quality=8)
    imageio.mimsave(wrist_path, wrist_frames, fps=args.fps, quality=8)
    imageio.mimsave(closeup_path, closeup_frames, fps=args.fps, quality=8)
    print(f"[render_screw_driving] saved: {top_path}, {wrist_path}, {closeup_path} ({len(top_frames)} frames)")

    closeup_r.close()
    top_r.close()
    wrist_r.close()


if __name__ == "__main__":
    main()
