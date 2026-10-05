"""RoboTwin 2.0 이식 Part 3-3(lint_task.py) 테스트.

이 레포의 실제 과거 버그 4종을 lint가 "진짜로 잡아내는지"를, 일부러
그 버그를 주입한 최소 모델로 확인한다(기존 3개 태스크 + scaffold로
만든 데모가 전부 통과하는 것만 보면 "항상 통과하는 lint"와 구분이
안 되므로, 실패 사례도 같이 검증).

사용법:
    python tests/test_lint_task.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco

import lint_task


def test_existing_tasks_pass_lint() -> None:
    for task_name in ("peg_in_hole", "cap_twist", "tacker"):
        errors = lint_task.lint_task(task_name)
        assert errors == [], f"{task_name}: lint가 기존(검증된) 태스크에서 false positive를 냄: {errors}"
    print("[OK] 기존 3개 태스크 전부 lint 통과(false positive 없음)")


def test_catches_missing_implicitfast() -> None:
    xml = """
    <mujoco>
      <option/>
      <worldbody>
        <body name="a"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="0.1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    errors = lint_task.check_implicitfast(model, "<string>")
    assert len(errors) == 1 and "버그 1" in errors[0]
    print("[OK] implicitfast 누락을 실제로 잡아냄")


def test_passes_with_implicitfast() -> None:
    xml = """
    <mujoco>
      <option integrator="implicitfast"/>
      <worldbody>
        <body name="a"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="0.1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    assert lint_task.check_implicitfast(model, "<string>") == []
    print("[OK] implicitfast 있으면 통과")


def test_catches_missing_contact_exclude() -> None:
    xml = """
    <mujoco>
      <worldbody>
        <body name="a"><freejoint/><geom name="ga" type="box" size="0.05 0.05 0.05" mass="0.1"/></body>
        <body name="b" pos="0.02 0 0"><freejoint/><geom name="gb" type="box" size="0.05 0.05 0.05" mass="0.1"/></body>
      </worldbody>
      <equality><weld name="w" body1="a" body2="b" active="true"/></equality>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    errors = lint_task.check_contact_excludes(model, data)
    assert len(errors) == 1 and "버그 3" in errors[0]
    print("[OK] weld로 고정된 겹치는 바디의 충돌 제외 누락을 실제로 잡아냄")


def test_passes_with_contact_exclude() -> None:
    xml = """
    <mujoco>
      <worldbody>
        <body name="a"><freejoint/><geom name="ga" type="box" size="0.05 0.05 0.05" mass="0.1"/></body>
        <body name="b" pos="0.02 0 0"><freejoint/><geom name="gb" type="box" size="0.05 0.05 0.05" mass="0.1" contype="0" conaffinity="0"/></body>
      </worldbody>
      <equality><weld name="w" body1="a" body2="b" active="true"/></equality>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    assert lint_task.check_contact_excludes(model, data) == []
    print("[OK] 충돌 비활성화(contype=0)면 통과")


def test_catches_unknown_task_registry_entry() -> None:
    errors = lint_task.check_task_registry_wiring("this_task_does_not_exist_xyz")
    assert len(errors) == 1 and "버그 4" in errors[0]
    print("[OK] TASK_REGISTRY에 없는 태스크를 실제로 잡아냄")


def test_scaffolded_demos_pass_lint() -> None:
    """scaffold_task.py로 만든 3개 패턴 데모 전부 lint 통과 확인 --
    이 테스트가 돌 때 demo_block/demo_peg/demo_tack이 이미 생성돼
    있어야 한다(scaffold_task.py를 먼저 실행했거나, 이 테스트 스위트가
    시작하기 전에 생성된 상태)."""
    from sim.task_registry import TASK_REGISTRY

    for task_name in ("demo_block", "demo_peg", "demo_tack"):
        if task_name not in TASK_REGISTRY:
            print(f"[SKIP] {task_name}이 아직 scaffold되지 않음 -- scaffold_task.py를 먼저 실행할 것")
            continue
        errors = lint_task.lint_task(task_name)
        assert errors == [], f"{task_name}: {errors}"
        print(f"[OK] {task_name}: lint 통과")


if __name__ == "__main__":
    test_existing_tasks_pass_lint()
    test_catches_missing_implicitfast()
    test_passes_with_implicitfast()
    test_catches_missing_contact_exclude()
    test_passes_with_contact_exclude()
    test_catches_unknown_task_registry_entry()
    test_scaffolded_demos_pass_lint()
    print()
    print("ALL TESTS PASSED (lint_task)")
