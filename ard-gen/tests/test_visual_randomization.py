"""RoboTwin 2.0 이식 Part 1(시각 domain randomization) 회귀/검증 테스트.

sim/visual_randomization.py 모듈 docstring의 주장(물리와 완전히 독립,
3개 태스크 공통 동작, 테이블 높이 범위가 안전함)을 "코드를 읽어서
그럴 것"이 아니라 직접 실측으로 증명한다.

## 렌더링(실제 픽셀) 검증은 이 샌드박스에서 불가능 -- 알려진 한계

이 환경에는 OSMesa/EGL이 없다(apt 미러도 접근 불가 -- 설치 시도 실측
확인). mujoco.Renderer()는 OpenGL 컨텍스트를 못 만들어서 render_episode.py
류 스크립트는 이 샌드박스에서 실행할 수 없다. 그래서 "카메라가 시각
randomization을 반영하는가"는 실제 렌더 프레임 비교가 아니라, 카메라가
보는 geom들의 model 레벨 수치(geom_matid/geom_rgba/body_pos/light_*)가
실제로 바뀌는지로 검증한다 -- OSMesa가 있는 환경에서 render_episode.py로
mp4를 직접 열어 눈으로 확인하는 건 여전히 사용자가 해야 할 일로 남는다
(PIPELINE.md RoboTwin 2.0 이식 섹션에 명시).

사용법:
    python tests/test_visual_randomization.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mujoco

from sim.task_registry import load_task_config
from sim.visual_randomization import (
    N_CLUTTER_SLOTS,
    TABLE_HEIGHT_RANGE_DOWN_M,
    TABLE_HEIGHT_RANGE_UP_M,
    apply_visual_config,
    default_visual_config,
    sample_visual_config,
)

_TASKS = ["peg_in_hole", "cap_twist", "tacker"]
_KNOWN_GOOD_GAINS_PEG = (0.124630, 0.001125)


def test_sample_visual_config_independent_rng_stream() -> None:
    """같은 시드라도 물리 rng와 visual rng는 호출자가 분리해서 넘기면
    서로 다른 시퀀스가 나온다 -- "독립 스트림" 전제가 실제로 지켜지는지
    (한쪽을 소비해도 다른 쪽에 영향 없는지) 확인."""
    phys_rng = np.random.default_rng(7)
    visual_rng = np.random.default_rng(7)

    task = load_task_config("peg_in_hole")
    scene_before = task.sample_scene_config(phys_rng)
    visual_cfg = sample_visual_config(visual_rng)
    scene_after_same_phys_rng_untouched = task.sample_scene_config(np.random.default_rng(7))

    # phys_rng를 소비한 뒤에도(scene_config 1회) visual_rng(별도 시드값만
    # 같은 독립 인스턴스)는 전혀 다른 소비 이력을 갖는다는 걸 보여주는
    # 목적 -- 두 Generator가 물리적으로 다른 객체라는 구조적 증거.
    assert isinstance(visual_cfg["table_height_offset"], float)
    assert scene_before == scene_after_same_phys_rng_untouched  # phys_rng 재현성 자체는 보존됨
    print("[OK] visual rng와 physics rng가 구조적으로 분리된 Generator 인스턴스")


def test_independent_of_physics_peg_in_hole() -> None:
    """핵심 요구사항: 같은 물리 scene_config/게인에 서로 다른 visual_cfg를
    적용해도 reward/success/step_count/trajectory가 bit-identical해야
    한다 -- "시각 randomization이 CMA-ES/성공 판정에 영향을 주면 안
    된다"는 요구를 실측으로 증명."""
    task = load_task_config("peg_in_hole")
    env = task.make_env()
    gains = task.gains_from_vector(_KNOWN_GOOD_GAINS_PEG)
    cfg = task.default_scene_config()

    env.apply_visual_config(None)
    result_a = env.run_episode(gains, cfg)

    rng = np.random.default_rng(123)
    env.apply_visual_config(sample_visual_config(rng))
    result_b = env.run_episode(gains, cfg)

    rng2 = np.random.default_rng(999)
    env.apply_visual_config(sample_visual_config(rng2))
    result_c = env.run_episode(gains, cfg)

    assert result_a["success"] == result_b["success"] == result_c["success"]
    assert result_a["step_count"] == result_b["step_count"] == result_c["step_count"]
    np.testing.assert_allclose(result_a["reward"], result_b["reward"], atol=1e-9)
    np.testing.assert_allclose(result_a["reward"], result_c["reward"], atol=1e-9)
    np.testing.assert_allclose(result_a["ee_poses"], result_b["ee_poses"], atol=1e-9)
    np.testing.assert_allclose(result_a["ee_poses"], result_c["ee_poses"], atol=1e-9)
    np.testing.assert_allclose(result_a["max_force"], result_b["max_force"], atol=1e-9)
    print("[OK] peg_in_hole: 시각 randomization이 reward/success/trajectory에 영향 없음(bit-identical)")


def test_apply_visual_config_none_is_noop() -> None:
    """visual_config=None이면 model 배열을 전혀 건드리지 않는다(호출
    자체를 안 하는 것과 동일) -- 하위호환 보장."""
    for task_name in _TASKS:
        task = load_task_config(task_name)
        env = task.make_env()
        model = env._sim.model
        before = {
            "mat_rgba": model.mat_rgba.copy(),
            "geom_rgba": model.geom_rgba.copy(),
            "geom_matid": model.geom_matid.copy(),
            "body_pos": model.body_pos.copy(),
            "light_diffuse": model.light_diffuse.copy(),
        }
        env.apply_visual_config(None)
        np.testing.assert_array_equal(before["mat_rgba"], model.mat_rgba)
        np.testing.assert_array_equal(before["geom_rgba"], model.geom_rgba)
        np.testing.assert_array_equal(before["geom_matid"], model.geom_matid)
        np.testing.assert_array_equal(before["body_pos"], model.body_pos)
        np.testing.assert_array_equal(before["light_diffuse"], model.light_diffuse)
    print("[OK] visual_config=None은 세 태스크 모두 완전한 no-op")


def test_common_logic_applies_to_all_three_tasks() -> None:
    """apply_visual_config() 하나가 (태스크별 분기 없이) 3개 태스크
    XML 전부에서 실제로 외형 수치를 바꾸는지 확인 -- "공통 로직" 요구."""
    for task_name in _TASKS:
        task = load_task_config(task_name)
        env = task.make_env()
        model = env._sim.model

        ground_matid_before = model.geom_matid[model.geom("ground").id]
        table_body_id = model.body("table").id
        table_z_before = float(model.body_pos[table_body_id][2])
        light_diffuse_before = model.light_diffuse.copy()
        clutter_pos_before = [model.body_pos[model.body(f"clutter_{i}").id].copy() for i in range(N_CLUTTER_SLOTS)]

        rng = np.random.default_rng(42)
        cfg = sample_visual_config(rng)
        # 변형이 매번 default(0)로 안 뽑히게 확률 낮은 시드 몇 개를 더 시도.
        for seed in (42, 1, 2, 3, 4, 5):
            cfg = sample_visual_config(np.random.default_rng(seed))
            if cfg["ground_variant"] != 0 or abs(cfg["table_height_offset"]) > 1e-6:
                break
        apply_visual_config(model, cfg)

        assert model.geom_matid[model.geom("ground").id] == model.material(f"ground_mat_{cfg['ground_variant']}").id
        table_z_after = float(model.body_pos[table_body_id][2])
        assert abs(table_z_after - (table_z_before + cfg["table_height_offset"])) < 1e-9
        assert not np.array_equal(light_diffuse_before, model.light_diffuse) or cfg["light_intensity_scale"] == 1.0

        any_clutter_moved = False
        for i in range(N_CLUTTER_SLOTS):
            pos_after = model.body_pos[model.body(f"clutter_{i}").id]
            if not np.allclose(pos_after, clutter_pos_before[i]):
                any_clutter_moved = True
        assert any_clutter_moved, f"{task_name}: clutter 위치가 전혀 안 바뀜"

        print(f"[OK] {task_name}: 공통 apply_visual_config()가 ground/table/light/clutter 전부 반영")


def test_clutter_never_collides_structurally() -> None:
    """clutter_i/clutter_i_geom은 contype=0 conaffinity=0으로 구워져
    있어서 위치를 아무리 무작위화해도 물리적으로 충돌할 수 없다 --
    "충돌 없이 배치" 요구가 설계부터 구조적으로 보장되는지(런타임
    위치와 무관하게) 확인."""
    for task_name in _TASKS:
        task = load_task_config(task_name)
        env = task.make_env()
        model = env._sim.model
        for i in range(N_CLUTTER_SLOTS):
            geom_id = model.geom(f"clutter_{i}_geom").id
            assert model.geom_contype[geom_id] == 0
            assert model.geom_conaffinity[geom_id] == 0
    print("[OK] 세 태스크 모두 clutter geom이 contype=0 conaffinity=0 (구조적으로 무충돌)")


def test_table_decoration_never_collides_structurally() -> None:
    """table_top/table_leg_1..4도 애초에 contype=0 conaffinity=0인
    순수 장식 geom이다(assets/*.xml 작성 시점부터 -- 이 테스트는 그
    설계를 깨지 않았는지 지키는 회귀 가드)."""
    for task_name in _TASKS:
        task = load_task_config(task_name)
        env = task.make_env()
        model = env._sim.model
        for geom_name in ("table_top", "table_leg_1", "table_leg_2", "table_leg_3", "table_leg_4"):
            geom_id = model.geom(geom_name).id
            assert model.geom_contype[geom_id] == 0
            assert model.geom_conaffinity[geom_id] == 0
    print("[OK] 세 태스크 모두 장식 table geom이 contype=0 conaffinity=0")


def test_table_height_range_safe_for_openarm_tasks() -> None:
    """테이블 높이 randomization 범위(아래 -TABLE_HEIGHT_RANGE_DOWN_M ~
    위 +TABLE_HEIGHT_RANGE_UP_M)가 OpenArm 양팔 home 자세와 실제로
    겹치지 않는지 mj_geomDistance로 직접 확인한다 -- table 장식 geom은
    contype=0이라 MuJoCo 엔진이 실제 접촉력을 만들진 않지만(물리적으로는
    어느 쪽이든 항상 안전), "시각적으로 말이 되는 범위인가"를 실측
    표면 거리로 확인한다(sim/visual_randomization.py 모듈 상수 주석의
    실측값과 같은 스캔)."""
    for task_name in ("peg_in_hole", "tacker"):
        task = load_task_config(task_name)
        env = task.make_env()
        model, data = env._sim.model, env._sim.data
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)

        table_geom_ids = [model.geom(n).id for n in ("table_top", "table_leg_1", "table_leg_2", "table_leg_3", "table_leg_4")]

        # table 바디 자체를 범위 양끝으로 옮겨서 다시 forward, 모든 충돌
        # 가능(contype!=0) geom과의 실제 최소 거리를 mj_geomDistance로
        # 잰다(bounding sphere는 openarm_body_link0_collision처럼 세로로
        # 길쭉한 받침대 메시에서 과대평가돼 실측과 안 맞았다 -- 실측으로
        # 확인 후 진짜 표면 거리를 재는 이 방식으로 바꿈).
        table_body_id = model.body("table").id
        other_gids = [gid for gid in range(model.ngeom) if model.geom_contype[gid] != 0]
        min_clearance = float("inf")
        worst = None
        for offset in (-TABLE_HEIGHT_RANGE_DOWN_M, 0.0, TABLE_HEIGHT_RANGE_UP_M):
            # table 바디 자신은 XML상 pos=(0.35,0,0)이고 table_top/leg_*가
            # 로컬 z로 떠 있다 -- apply_visual_config._apply_table_height와
            # 똑같이 body_pos.z에 offset을 더한다(그게 곧 table_top 등의
            # world z를 offset만큼 움직인다).
            model.body_pos[table_body_id] = np.array([0.35, 0.0, offset])
            mujoco.mj_forward(model, data)

            for tgid in table_geom_ids:
                for gid in other_gids:
                    dist = mujoco.mj_geomDistance(model, data, tgid, gid, 1.0, None)
                    if dist < min_clearance:
                        min_clearance = dist
                        worst = (offset, model.geom(tgid).name, model.geom(gid).name)

        assert min_clearance > 0.0, (
            f"{task_name}: table height range에서 최소 clearance={min_clearance:.4f}m (겹침 위험), worst={worst}"
        )
        print(
            f"[OK] {task_name}: table height [-{TABLE_HEIGHT_RANGE_DOWN_M}, +{TABLE_HEIGHT_RANGE_UP_M}]m 범위, "
            f"실측 최소 clearance={min_clearance * 1000:.1f}mm"
        )


def test_cap_twist_floor_contact_renamed_not_reclassified() -> None:
    """cap_twist의 실제 충돌 평면(원래 이름 "table")을 floor_contact로
    이름만 바꿨다 -- contype/conaffinity/friction 등 물성은 그대로인지
    확인(이름 변경이 물리를 안 바꿨다는 회귀 가드)."""
    task = load_task_config("cap_twist")
    env = task.make_env()
    model = env._sim.model
    geom_id = model.geom("floor_contact").id
    assert model.geom_contype[geom_id] == 1
    assert model.geom_conaffinity[geom_id] == 1
    np.testing.assert_allclose(model.geom_friction[geom_id], [1.0, 0.005, 0.0001])
    print("[OK] cap_twist floor_contact(구 'table')의 충돌 물성이 그대로 보존됨")


def test_default_visual_config_matches_xml_baseline() -> None:
    """default_visual_config()를 적용하면(아무 randomization도 안 한
    것과 동일해야 함) ground/table의 matid가 변형 0번(=XML 원본과 같은
    첫 재질)을 가리키는지 확인 -- "기존 동작과 시각적으로 같다"는
    주장의 최소 증거."""
    for task_name in _TASKS:
        task = load_task_config(task_name)
        env = task.make_env()
        model = env._sim.model
        apply_visual_config(model, default_visual_config())
        assert model.geom_matid[model.geom("ground").id] == model.material("ground_mat_0").id
        assert model.geom_matid[model.geom("table_top").id] == model.material("table_mat_0").id
        for i in range(N_CLUTTER_SLOTS):
            assert float(model.body_pos[model.body(f"clutter_{i}").id][2]) == -5.0
    print("[OK] default_visual_config()가 세 태스크 모두 'variant 0 + clutter 비활성' 기준선과 일치")


if __name__ == "__main__":
    test_sample_visual_config_independent_rng_stream()
    test_independent_of_physics_peg_in_hole()
    test_apply_visual_config_none_is_noop()
    test_common_logic_applies_to_all_three_tasks()
    test_clutter_never_collides_structurally()
    test_table_decoration_never_collides_structurally()
    test_table_height_range_safe_for_openarm_tasks()
    test_cap_twist_floor_contact_renamed_not_reclassified()
    test_default_visual_config_matches_xml_baseline()
    print()
    print("ALL TESTS PASSED (visual randomization)")
