"""RoboTwin 2.0 이식 Part 3-3: 새로 생성된(또는 기존) 태스크 자산을
이 레포가 실측으로 겪은 과거 버그 4종에 대해 검사한다.
scaffold_task.py가 매 생성 직후 자동으로 돌리고(Part 3-2), 기존
3개 태스크에 대해서도 수동으로 돌려 "진짜 버그가 있었을 때 이 lint가
잡아냈을까"를 확인할 수 있다.

검사 항목(전부 PIPELINE.md에 기록된 실측 버그 그대로):

1. **implicitfast 누락**: 강한 위치 액추에이터 + 항시-활성 weld 조합에서
   기본 적분기(Euler)는 20mm+/0.3s(tacker) ~ 69mm/2000스텝(peg_in_hole)
   드리프트를 낸다 -- <option integrator="implicitfast">가 있는지 확인.
2. **weld eq_data 레이아웃**: `[anchor(3), relpos(3), relquat(4),
   torquescale(1)]`(sim/stabilizer.py 모듈 docstring 참고, relpos가
   0이 아니라 3부터 시작한다는 걸 몰라서 생긴 버그) -- 활성 weld의
   relquat 슬롯(6:10)이 단위 사원수(norm~=1)인지로 레이아웃이 뒤집히지
   않았는지 간접 확인한다.
3. **충돌 제외 누락**: weld로 서로 고정된 두 바디의 geom이 실제로도
   충돌에 참여할 수 있는 조합(둘 다 contype!=0 and conaffinity!=0)인데
   `<contact><exclude>`가 없으면, tacker에서 겪은 "자기 손과 거짓
   접촉"(최대 2.7cm 파고듦) 버그가 재발한다.
4. **TASK_REGISTRY/드라이버 매칭**: 태스크가 TASK_REGISTRY에 등록돼
   있고, 가리키는 모듈:클래스가 실제로 import/인스턴스화되는지(
   screw_driving이 한동안 잘못된/구버전 시뮬레이터를 가리켰던 버그 계열).

사용법:
    python lint_task.py --task demo_block
    python lint_task.py --task peg_in_hole   # 기존 태스크에도 쓸 수 있음
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import mujoco
import numpy as np


class LintError(Exception):
    pass


def _find_xml_path(task_name: str) -> str:
    candidate = os.path.join(os.path.dirname(__file__), "assets", f"{task_name}.xml")
    if os.path.exists(candidate):
        return candidate
    # 일부 기존 태스크는 다른 파일명 규칙을 쓴다(예: peg_in_hole ->
    # peg_in_hole_bimanual_openarm.xml) -- TASK_REGISTRY가 가리키는
    # Env의 _DEFAULT_XML/xml_path를 통해 찾는다.
    from sim.task_registry import load_task_config

    task = load_task_config(task_name)
    env = task.make_env()
    xml_path = getattr(env._sim, "xml_path", None)
    if xml_path is None:
        raise LintError(f"{task_name}: XML 경로를 찾을 수 없음(assets/{task_name}.xml도 없고 env._sim.xml_path도 없음)")
    return xml_path


def check_implicitfast(model: mujoco.MjModel, xml_path: str) -> list[str]:
    errors = []
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        errors.append(
            f"[버그 1: implicitfast 누락] {xml_path}: integrator가 implicitfast가 아님 "
            f"(현재={model.opt.integrator!r}) -- 강한 위치 액추에이터 + 항시-활성 weld 조합에서 "
            f"드리프트 버그가 재발할 수 있다(tacker 20mm+/0.3s, peg_in_hole 69mm/2000스텝 실측)."
        )
    return errors


def check_weld_eq_data_layout(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    errors = []
    mujoco.mj_forward(model, data)
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_WELD:
            continue
        if not data.eq_active[i]:
            continue  # 런타임에 동적으로 활성화되는 weld(예: Stabilizer)는 아직 eq_data가 안 채워져 있어 검사 대상이 아님
        relquat = model.eq_data[i, 6:10]
        norm = float(np.linalg.norm(relquat))
        if abs(norm - 1.0) > 1e-3:
            name = model.eq(i).name or f"eq#{i}"
            errors.append(
                f"[버그 2: weld eq_data 레이아웃] {name}: relquat 슬롯(eq_data[6:10])의 norm={norm:.4f} "
                f"(단위 사원수여야 함, ~1.0) -- [anchor(3),relpos(3),relquat(4),torquescale(1)] 레이아웃이 "
                f"뒤집혔을 가능성(relpos를 0:3에, relquat을 3:7에 쓰는 실수, sim/stabilizer.py 모듈 docstring 참고)."
            )
    return errors


def _body_geoms(model: mujoco.MjModel, body_id: int) -> list[int]:
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]


def check_contact_excludes(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    """`<exclude>`가 "선언돼 있는가"가 아니라 "weld로 고정된 바디끼리
    실제로 파고들지 않는가"를 mj_geomDistance로 직접 잰다 -- exclude를
    선언했는지 자체보다 결과(거짓 접촉이 없는가)가 중요하고, MuJoCo
    공개 API가 exclude 쌍을 body id로 바로 조회하는 방법을 제공하지
    않는다."""
    errors = []
    mujoco.mj_forward(model, data)
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_WELD or not data.eq_active[i]:
            continue
        b1, b2 = model.eq_obj1id[i], model.eq_obj2id[i]
        geoms1 = [g for g in _body_geoms(model, b1) if model.geom_contype[g] != 0]
        geoms2 = [g for g in _body_geoms(model, b2) if model.geom_conaffinity[g] != 0]
        for g1 in geoms1:
            for g2 in geoms2:
                if (model.geom_contype[g1] & model.geom_conaffinity[g2]) == 0:
                    continue
                dist = mujoco.mj_geomDistance(model, data, g1, g2, 0.05, None)
                if dist < 0.0:
                    name = model.eq(i).name or f"eq#{i}"
                    errors.append(
                        f"[버그 3: 충돌 제외 누락] {name}로 고정된 {model.body(b1).name}/{model.body(b2).name}의 "
                        f"geom {model.geom(g1).name}/{model.geom(g2).name}이 {-dist * 1000:.1f}mm 파고들어 "
                        f"있음(contact exclude 필요, tacker의 '자기 손과 거짓 접촉' 버그 계열)."
                    )
    return errors


def check_task_registry_wiring(task_name: str) -> list[str]:
    errors = []
    from sim.task_registry import TASK_REGISTRY, load_task_config

    if task_name not in TASK_REGISTRY:
        errors.append(f"[버그 4: TASK_REGISTRY 누락] '{task_name}'이 TASK_REGISTRY에 없음.")
        return errors
    try:
        task = load_task_config(task_name)
        env = task.make_env()
        if not hasattr(env, "run_episode"):
            errors.append(f"[버그 4: 드라이버 매칭] '{task_name}'의 Env가 run_episode()를 구현하지 않음.")
        if not task.gain_names:
            errors.append(f"[버그 4: 드라이버 매칭] '{task_name}'의 gain_names가 비어 있음(CMA-ES가 탐색할 대상이 없음).")
    except Exception as exc:  # noqa: BLE001 -- lint는 어떤 종류의 import/생성 실패든 다 잡아야 한다
        errors.append(f"[버그 4: 드라이버 매칭] '{task_name}' 로드 실패: {type(exc).__name__}: {exc}")
    return errors


def lint_task(task_name: str) -> list[str]:
    errors: list[str] = []
    errors += check_task_registry_wiring(task_name)

    try:
        xml_path = _find_xml_path(task_name)
    except LintError as exc:
        errors.append(str(exc))
        return errors

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[컴파일 실패] {xml_path}: {type(exc).__name__}: {exc}")
        return errors

    errors += check_implicitfast(model, xml_path)
    errors += check_weld_eq_data_layout(model, data)
    errors += check_contact_excludes(model, data)
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = lint_task(args.task)
    if errors:
        print(f"[lint_task] '{args.task}': {len(errors)}건 발견")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"[lint_task] '{args.task}': 4개 검사 항목 전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
