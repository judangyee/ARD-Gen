"""screw_driving.xml 모델 검증/데모 렌더링 스크립트.

아직 진짜 admittance controller(토크 피드백 기반)는 없다 -- 이 스크립트는
"모델 자체가 물리적으로 말이 되는가"를 눈으로 확인하기 위한 것으로,
wrist_rotate를 0->pi까지 그냥 계속 돌리기만 한다(bolt_hinge_drive가 매 스텝
그 각도를 그대로 복사해서 볼트를 구동).

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

_HOME_QPOS = {
    "waist": 0.0,
    "shoulder": -0.89237,
    "elbow": 1.05339,
    "forearm_roll": 0.0,
    "wrist_angle": 1.40932,
    "wrist_rotate": 0.0,
}
_ARM_FOLLOW_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle"]
_JAC_DAMPING = 1e-4
_N_DRIVE_STEPS = 650
_MAX_WRIST_TARGET = np.pi  # VX300s의 실제 wrist_rotate 관절 한계 (±180도)


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
    for name, val in _HOME_QPOS.items():
        d.qpos[m.joint(name).qposadr[0]] = val
    mujoco.mj_forward(m, d)
    for name, val in _HOME_QPOS.items():
        d.ctrl[m.actuator(name).id] = val

    arm_dofadr = [m.joint(n).dofadr[0] for n in _ARM_FOLLOW_JOINTS]
    wrist_act = m.actuator("wrist_rotate").id
    wrist_qpos = m.joint("wrist_rotate").qposadr[0]
    bolt_drive_act = m.actuator("bolt_hinge_drive").id
    bolt_slide_qpos = m.joint("bolt_slide").qposadr[0]
    torque_adr = m.sensor("bolt_drive_torque").adr[0]
    tip_site_id = m.site("driver_tip_site").id

    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))

    top_r = mujoco.Renderer(m, height=args.height, width=args.width)
    wrist_r = mujoco.Renderer(m, height=args.height, width=args.width)
    closeup_r = mujoco.Renderer(m, height=args.height, width=args.width)
    top_frames: list = []
    wrist_frames: list = []
    closeup_frames: list = []

    def capture() -> None:
        top_r.update_scene(d, camera="top_cam")
        top_frames.append(top_r.render().copy())
        wrist_r.update_scene(d, camera="wrist_cam")
        wrist_frames.append(wrist_r.render().copy())
        closeup_r.update_scene(d, camera="closeup_cam")
        closeup_frames.append(closeup_r.render().copy())

    capture()
    prev_slide = d.qpos[bolt_slide_qpos]

    for step in range(_N_DRIVE_STEPS):
        d.ctrl[wrist_act] = min(step * 0.005, _MAX_WRIST_TARGET)
        d.ctrl[bolt_drive_act] = d.qpos[wrist_qpos]

        mujoco.mj_jacSite(m, d, jacp, jacr, tip_site_id)
        slide_now = d.qpos[bolt_slide_qpos]
        delta_slide = slide_now - prev_slide
        prev_slide = slide_now
        target_delta_world = np.array([0.0, 0.0, -delta_slide])
        jac_arm = jacp[:, arm_dofadr]
        jjt = jac_arm @ jac_arm.T + _JAC_DAMPING * np.eye(3)
        dq = jac_arm.T @ np.linalg.solve(jjt, target_delta_world)
        for name, dof_local in zip(_ARM_FOLLOW_JOINTS, range(len(_ARM_FOLLOW_JOINTS))):
            d.ctrl[m.actuator(name).id] += dq[dof_local]

        mujoco.mj_step(m, d)
        if step % 3 == 0:
            capture()

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
