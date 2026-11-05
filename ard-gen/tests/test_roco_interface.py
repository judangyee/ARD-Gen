"""sim/roco_interface.py(태스크 무관 RoCo 스키마/변환 모듈) 단위 테스트.

pytest 없이 plain assert로 짰다 -- 이 저장소의 다른 테스트들과 같은 스타일
(직접 실행, 실패하면 AssertionError로 죽는다).

이 파일이 확인하는 것: STATE_NAMES/ACTION_NAMES 이름·순서 정확 일치,
Euler 각도 연속성(래핑 없음), 좌표 변환(WorkspaceTransform) 왕복,
그리퍼 매핑 왕복, build_state/build_action_sequence 스키마(shape/순서),
build_roco_model이 카메라를 추가하면서 기존 바디/조인트/액추에이터 id를
그대로 유지하는지.

이 파일이 확인하지 않는 것(roco_interface.py 모듈 docstring 참고): 어느
팔이 "액티브"인지, 특정 태스크의 제어 루프 -- 그건 범위 밖(scripts/
validate_against_roco.py가 공식 통계 비교를 담당).

렌더링 테스트(test_build_roco_model_cameras_render)만 MUJOCO_GL=osmesa가
필요해서, 다른 렌더링 스크립트들과 같은 패턴으로 모듈 최상단에서
mujoco를 임포트하기 전에 os.environ.setdefault로 깔아둔다 -- 이 파일은
torch를 전혀 임포트하지 않으므로 PIPELINE.md에 기록된 "MUJOCO_GL=osmesa
이후 Adam 생성 시 세그폴트" 문제와는 무관하다(그 문제는 osmesa+Adam
조합에서만 재현됨, 렌더링만 하는 이 파일은 안전).

사용법:
    python tests/test_roco_interface.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import numpy as np

import sim.roco_interface as roco
from sim.peg_in_hole_bimanual_openarm_sim import (
    _DEFAULT_XML,
    BimanualPegInHoleOpenArmSim,
    _default_scene_config,
)


def test_state_names_exact() -> None:
    """사용자 지시 공식 그대로 재구성한 리스트와 완전히 (이름+순서) 일치해야
    한다."""
    expected = (
        [f"{s}_ee_{a}" for s in ("left", "right") for a in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        + [f"{s}_jpos_{i}" for s in ("left", "right") for i in range(7)]
        + [f"{s}_jvel_{i}" for s in ("left", "right") for i in range(7)]
        + ["left_gripper", "right_gripper"]
    )
    assert roco.STATE_NAMES == expected
    assert len(roco.STATE_NAMES) == 44
    assert roco.STATE_NAMES[-2:] == ["left_gripper", "right_gripper"]
    print("[OK] STATE_NAMES exact 44-name/order match")


def test_action_names_exact() -> None:
    expected = [
        f"{s}_{a}"
        for s in ("left", "right")
        for a in ("x", "y", "z", "rx", "ry", "rz", "gripper")
    ]
    assert roco.ACTION_NAMES == expected
    assert len(roco.ACTION_NAMES) == 14
    print("[OK] ACTION_NAMES exact 14-name/order match")


def test_euler_mat_roundtrip() -> None:
    """짐벌락에서 먼 임의 각도들에서 mat->euler->mat 왕복이 원래 행렬을
    복원하는지(각도 자체의 왕복이 아니라, 회전행렬 기준 왕복 -- 각도
    표현은 여러 개가 같은 행렬일 수 있어도 행렬은 유일해야 한다)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        euler = rng.uniform([-3.0, -1.0, -3.0], [3.0, 1.0, 3.0])
        mat = roco.euler_xyz_to_mat(euler)
        recovered_euler = roco.euler_xyz_from_mat(mat)
        recovered_mat = roco.euler_xyz_to_mat(recovered_euler)
        assert np.allclose(mat, recovered_mat, atol=1e-9), (euler, mat, recovered_mat)
    print("[OK] euler_xyz_to_mat <-> euler_xyz_from_mat round-trips at the matrix level")


def test_quat_mat_roundtrip() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        mat = roco.quat_wxyz_to_mat(q)
        q2 = roco.quat_wxyz_from_mat(mat)
        # 쿼터니언은 q와 -q가 같은 회전 -- 둘 중 하나와 가까우면 통과.
        assert np.allclose(q, q2, atol=1e-6) or np.allclose(q, -q2, atol=1e-6), (q, q2)
    print("[OK] quat_wxyz_to_mat <-> quat_wxyz_from_mat round-trips")


def test_continuous_euler_no_jump() -> None:
    """합성 시퀀스: rz를 -3*pi에서 +3*pi까지(2바퀴 이상 회전) 촘촘히
    회전시키는 쿼터니언 시퀀스를 만들어서, continuous_euler_xyz_from_quats()
    결과가 (a) 연속 프레임 간 pi 이상 점프가 0건이고 (b) unwrap된 rz가
    실제 누적 회전각과 거의 같은 값으로 계속 증가하는지 확인한다 -- 사용자
    지시 "연속 프레임 간 pi 이상 점프 0건"을 직접 검증."""
    true_rz = np.linspace(-3 * np.pi, 3 * np.pi, 400)
    quats = np.stack(
        [roco.quat_wxyz_from_mat(roco.euler_xyz_to_mat(np.array([0.0, 0.0, rz]))) for rz in true_rz]
    )
    euler_seq = roco.continuous_euler_xyz_from_quats(quats)
    assert euler_seq.shape == (400, 3)

    jumps = np.abs(np.diff(euler_seq, axis=0))
    assert np.all(jumps < np.pi), f"max per-axis jump={jumps.max()} (pi={np.pi})"

    # rz는 true_rz와 거의 같은 값으로(부호/오프셋 없이) 복원돼야 한다 --
    # 시작점에서 euler_xyz_from_mat 주분기가 true_rz[0]=-3pi와 2*pi*k만큼
    # 다를 수 있으므로, 그 오프셋을 보정한 뒤 비교한다.
    offset = euler_seq[0, 2] - true_rz[0]
    assert abs(offset % (2 * np.pi)) < 1e-6 or abs(offset % (2 * np.pi) - 2 * np.pi) < 1e-6, offset
    assert np.allclose(euler_seq[:, 2] - offset, true_rz, atol=1e-6)
    print("[OK] continuous_euler_xyz_from_quats: 0 jumps >= pi across 2+ full turns, rz unwraps correctly")


def test_workspace_transform_roundtrip() -> None:
    source = {"x": (0.0, 1.0), "y": (-0.5, 0.5), "z": (0.0, 2.0)}
    transform = roco.WorkspaceTransform(source=source)
    assert transform.target == roco.ROCO_OFFICIAL_LEFT_EE_RANGE

    rng = np.random.default_rng(2)
    # source 범위 안/밖(외삽) 모두 테스트 -- apply()가 클리핑하지 않는다는
    # roco_interface.py의 설계를 그대로 확인.
    pts = rng.uniform(-2.0, 3.0, size=(30, 3))
    mapped = transform.apply(pts)
    recovered = transform.inverse(mapped)
    assert np.allclose(pts, recovered, atol=1e-9), np.abs(pts - recovered).max()

    # source 범위의 양 끝은 target 범위의 양 끝으로 정확히 매핑돼야 한다.
    lo = np.array([source[a][0] for a in ("x", "y", "z")])
    hi = np.array([source[a][1] for a in ("x", "y", "z")])
    mapped_lo = transform.apply(lo)
    mapped_hi = transform.apply(hi)
    expected_lo = np.array([roco.ROCO_OFFICIAL_LEFT_EE_RANGE[a][0] for a in ("x", "y", "z")])
    expected_hi = np.array([roco.ROCO_OFFICIAL_LEFT_EE_RANGE[a][1] for a in ("x", "y", "z")])
    assert np.allclose(mapped_lo, expected_lo, atol=1e-9)
    assert np.allclose(mapped_hi, expected_hi, atol=1e-9)
    print("[OK] WorkspaceTransform.apply <-> inverse round-trips (incl. extrapolation), endpoints map exactly")


def test_gripper_mapping_roundtrip() -> None:
    mapping = roco.GripperMapping(closed_value=0.0, open_value=1.2)
    for raw in (-0.5, 0.0, 0.3, 1.2, 1.5):
        roco_val = mapping.to_roco(raw)
        lo, hi = mapping.target_range
        assert lo - 1e-9 <= roco_val <= hi + 1e-9, roco_val
        recovered_raw = mapping.from_roco(roco_val)
        clipped_raw = float(np.clip(raw, mapping.closed_value, mapping.open_value))
        assert abs(recovered_raw - clipped_raw) < 1e-9, (raw, recovered_raw, clipped_raw)
    print("[OK] GripperMapping.to_roco <-> from_roco round-trips (within target_range, clips raw)")


def test_default_gripper_mapping_matches_model() -> None:
    model = mujoco.MjModel.from_xml_path(_DEFAULT_XML)
    for side in ("left", "right"):
        mapping = roco.default_gripper_mapping(model, side)
        joint = model.joint(roco.gripper_joint_name(side))
        lo, hi = float(joint.range[0]), float(joint.range[1])
        assert {mapping.open_value, mapping.closed_value} == {lo, hi}
        assert abs(mapping.open_value) <= abs(mapping.closed_value)
    print("[OK] default_gripper_mapping reads model joint range, open=near-zero end")


def test_build_state_shape_and_order() -> None:
    sim = BimanualPegInHoleOpenArmSim()
    sim.reset(_default_scene_config())
    mujoco.mj_forward(sim.model, sim.data)
    mappings = {side: roco.default_gripper_mapping(sim.model, side) for side in ("left", "right")}
    state = roco.build_state(sim.model, sim.data, mappings)
    assert state.shape == (44,)
    assert state.dtype == np.float32

    # 쿼터니언은 정규화되어 있어야 한다(ee pose 블록, 인덱스 3..6과 10..13).
    left_quat = state[3:7]
    right_quat = state[10:14]
    assert abs(float(np.linalg.norm(left_quat)) - 1.0) < 1e-3, left_quat
    assert abs(float(np.linalg.norm(right_quat)) - 1.0) < 1e-3, right_quat

    # 그리퍼 값(마지막 두 개)은 target_range 안에 있어야 한다.
    for side_idx, side in enumerate(("left", "right")):
        lo, hi = mappings[side].target_range
        val = state[42 + side_idx]
        assert lo - 1e-6 <= val <= hi + 1e-6, (side, val)
    print("[OK] build_state: shape=(44,), float32, normalized quats, grippers within target_range")


def test_build_action_sequence_shape_and_order() -> None:
    T = 6
    rng = np.random.default_rng(3)
    left_pos = rng.uniform(-1, 1, size=(T, 3))
    right_pos = rng.uniform(-1, 1, size=(T, 3))

    def random_quats(n: int) -> np.ndarray:
        q = rng.normal(size=(n, 4))
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    left_quat = random_quats(T)
    right_quat = random_quats(T)
    left_grip = rng.uniform(0.06, 0.30, size=T)
    right_grip = rng.uniform(0.06, 0.30, size=T)

    action = roco.build_action_sequence(left_pos, left_quat, left_grip, right_pos, right_quat, right_grip)
    assert action.shape == (T, 14)
    assert action.dtype == np.float32

    # 열 순서: left x,y,z,rx,ry,rz,gripper, right x,y,z,rx,ry,rz,gripper.
    assert np.allclose(action[:, 0:3], left_pos, atol=1e-5)
    assert np.allclose(action[:, 6], left_grip, atol=1e-5)
    assert np.allclose(action[:, 7:10], right_pos, atol=1e-5)
    assert np.allclose(action[:, 13], right_grip, atol=1e-5)

    # 회전 열(3:6, 10:13)은 continuous_euler_xyz_from_quats()와 정확히 같아야 한다.
    assert np.allclose(action[:, 3:6], roco.continuous_euler_xyz_from_quats(left_quat), atol=1e-5)
    assert np.allclose(action[:, 10:13], roco.continuous_euler_xyz_from_quats(right_quat), atol=1e-5)
    print("[OK] build_action_sequence: shape=(T,14), column order matches ACTION_NAMES")


def test_build_roco_model_adds_cameras_preserves_ids() -> None:
    """build_roco_model()이 head/left_hand/right_hand 카메라를 추가하면서
    기존 바디/조인트/액추에이터 id를 하나도 바꾸지 않는지 확인 -- 이게
    깨지면 roco_interface.py를 쓰는 다른 모든 get_* 함수(모델 id를
    캐싱하는 호출자 코드)가 조용히 잘못된 값을 읽게 된다."""
    original = mujoco.MjModel.from_xml_path(_DEFAULT_XML)
    roco_model = roco.build_roco_model(_DEFAULT_XML)

    cam_names = {roco_model.camera(i).name for i in range(roco_model.ncam)}
    assert set(roco.CAMERA_NAMES) <= cam_names, cam_names

    assert roco_model.nbody == original.nbody
    assert roco_model.njnt == original.njnt
    assert roco_model.nu == original.nu
    for i in range(original.nbody):
        assert roco_model.body(i).name == original.body(i).name
    for i in range(original.njnt):
        assert roco_model.joint(i).name == original.joint(i).name
    for i in range(original.nu):
        assert roco_model.actuator(i).name == original.actuator(i).name

    # idempotent: 두 번째 호출도 카메라 3개만 있고 더 늘어나지 않아야 한다.
    roco_model_again = roco.build_roco_model(_DEFAULT_XML)
    assert roco_model_again.ncam == roco_model.ncam
    print("[OK] build_roco_model: adds 3 cameras, preserves body/joint/actuator ids+order, idempotent")


def test_build_roco_model_cameras_render() -> None:
    model = roco.build_roco_model(_DEFAULT_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=roco.CAMERA_HEIGHT, width=roco.CAMERA_WIDTH)
    frames = roco.render_roco_cameras(renderer, data)
    assert set(frames.keys()) == set(roco.CAMERA_NAMES)
    for name, frame in frames.items():
        assert frame.shape == (roco.CAMERA_HEIGHT, roco.CAMERA_WIDTH, 3), (name, frame.shape)
        assert frame.dtype == np.uint8
    print("[OK] render_roco_cameras: head/left_hand/right_hand all render at 320x240x3")


if __name__ == "__main__":
    test_state_names_exact()
    test_action_names_exact()
    test_euler_mat_roundtrip()
    test_quat_mat_roundtrip()
    test_continuous_euler_no_jump()
    test_workspace_transform_roundtrip()
    test_gripper_mapping_roundtrip()
    test_default_gripper_mapping_matches_model()
    test_build_state_shape_and_order()
    test_build_action_sequence_shape_and_order()
    test_build_roco_model_adds_cameras_preserves_ids()
    test_build_roco_model_cameras_render()
    print()
    print("ALL ROCO_INTERFACE TESTS PASSED")
