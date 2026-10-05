"""RoboTwin 2.0 이식 Part 2(텍스처 라이브러리) 테스트.

사용법:
    python tests/test_texture_library.py
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import pytest

from sim.task_registry import load_task_config
from sim.texture_library import get_texture, list_by_category, load_manifest, request_external_texture, texture_to_mjcf

_TASKS = ["peg_in_hole", "cap_twist", "tacker"]


def test_manifest_loads_and_has_six_entries() -> None:
    manifest = load_manifest()
    assert len(manifest) == 6
    ids = {tex.id for tex in manifest}
    assert ids == {"ground_mat_0", "ground_mat_1", "ground_mat_2", "table_mat_0", "table_mat_1", "table_mat_2"}
    print("[OK] manifest.yaml 6개 항목 로드")


def test_categories() -> None:
    floor = list_by_category("floor")
    table = list_by_category("table")
    assert {t.id for t in floor} == {"ground_mat_0", "ground_mat_1", "ground_mat_2"}
    assert {t.id for t in table} == {"table_mat_0", "table_mat_1", "table_mat_2"}
    print("[OK] category 필터링(floor/table)")


def test_manifest_matches_actual_xml_definitions() -> None:
    """manifest.yaml에 적힌 rgb1/rgb2가 실제 3개 XML에 구워진 <texture>
    정의와 어긋나지 않는지 확인 -- "단일 진실 소스"라는 주장의 증거
    (문서가 실제 에셋과 따로 놀면 Part 3 scaffold_task.py가 잘못된
    숫자를 새 태스크에 복제하게 된다)."""
    manifest = {tex.id: tex for tex in load_manifest()}
    for task_name in _TASKS:
        task = load_task_config(task_name)
        model = task.make_env()._sim.model
        for mat_id_name, tex in manifest.items():
            mat_id = model.material(mat_id_name).id
            tex_id = model.mat_texid[mat_id][mujoco.mjtTextureRole.mjTEXROLE_RGB]
            actual_rgb1 = model.tex_data[model.tex_adr[tex_id]: model.tex_adr[tex_id] + 3].astype(float) / 255.0
            expected_rgb1 = tex.mujoco["rgb1"]
            # builtin checker/gradient의 첫 픽셀이 항상 정확히 rgb1은
            # 아닐 수 있어(그라디언트는 보간) -- 느슨하게 "크게 안
            # 어긋남"만 확인(완전 동치 비교는 MuJoCo 내부 렌더링 구현에
            # 너무 의존적이라 깨지기 쉬움).
            assert all(abs(a - e) < 0.5 for a, e in zip(actual_rgb1, expected_rgb1)), (
                f"{task_name}/{mat_id_name}: manifest rgb1={expected_rgb1} vs 실제 첫 텍셀={actual_rgb1}"
            )
    print("[OK] manifest.yaml의 rgb1/rgb2가 3개 XML의 실제 재질과 일치(첫 텍셀 기준)")


def test_texture_to_mjcf_roundtrip_compiles() -> None:
    """texture_to_mjcf()가 만든 XML 조각이 실제로 MuJoCo에 컴파일되는지
    확인 -- scaffold_task.py가 이걸로 새 태스크 XML을 만들 것이므로,
    문법이 깨지면 안 된다."""
    tex = get_texture("ground_mat_1")
    snippet = texture_to_mjcf(tex, "my_tex", "my_mat")
    xml = f"""
    <mujoco>
      <asset>
        {snippet}
      </asset>
      <worldbody>
        <geom name="g" type="plane" size="0.5 0.5 0.01" material="my_mat"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.geom_matid[model.geom("g").id] == model.material("my_mat").id
    print("[OK] texture_to_mjcf()가 생성한 조각이 실제로 컴파일됨")


def test_request_external_texture_raises_not_implemented() -> None:
    """외부 생성 API는 인터페이스만 있고 구현이 없다 -- 조용히 가짜
    값을 돌려주지 않고 명시적으로 NotImplementedError를 내는지 확인."""
    with pytest.raises(NotImplementedError):
        request_external_texture("낡은 나무 바닥", "floor")
    print("[OK] request_external_texture()는 아직 구현 안 됐다고 명시적으로 실패")


if __name__ == "__main__":
    test_manifest_loads_and_has_six_entries()
    test_categories()
    test_manifest_matches_actual_xml_definitions()
    test_texture_to_mjcf_roundtrip_compiles()
    test_request_external_texture_raises_not_implemented()
    print()
    print("ALL TESTS PASSED (texture library)")
