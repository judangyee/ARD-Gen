"""RoboTwin 2.0 이식 Part 3-2: 새 태스크 스캐폴딩 (Level 1 자동화).

"물체 설명(모양/치수) + 패턴 이름"을 받아서 assets/{name}.xml +
sim/{name}_env.py(패턴별 범용 엔진, sim/generic_pattern_env.py의
Env를 그대로 가리키는 1줄 서브클래스 + scene_config 함수 3개) +
tasks/{name}.yaml을 생성하고, sim/task_registry.py의 TASK_REGISTRY에
등록한다. 생성 직후 lint_task.py(Part 3-3)를 자동으로 돌려서 이
레포의 실제 과거 버그 4종이 없는지 확인한다(실패하면 생성을 롤백하지는
않지만 경고를 명확히 띄운다 -- "채워 넣은 숫자가 틀렸다"는 신호).

"Level 1"이라는 뜻: 3개 검증된 패턴(patterns/{position_correction,
torque_reactive,impact_recoil}.py)의 수치만 채워 넣는다 -- LLM이
매번 물리를 새로 설계/작성하지 않는다. 자유형 XML 작성은 없다: 이
파일의 모든 문자열 조립은 고정된 틀에 숫자를 꽂아 넣는 f-string이고,
구조(바디/조인트/액추에이터 종류와 배치) 자체는 patterns/*.py와
assets/tacker_openarm.xml의 Part 1 장식 블록에서 그대로 가져온다.

사용 예:
    python scaffold_task.py --name demo_block --pattern torque_reactive \
        --object-shape box --object-size 0.03 0.03 0.03 --object-mass 0.1

    python scaffold_task.py --name demo_peg --pattern position_correction \
        --object-shape box --object-size 0.01 0.01 0.03 --object-mass 0.04

    python scaffold_task.py --name demo_tack --pattern impact_recoil \
        --object-shape box --object-size 0.02 0.02 0.02 --object-mass 0.08
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

from patterns import impact_recoil, position_correction, torque_reactive

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
_TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasks")
_SIM_DIR = os.path.join(os.path.dirname(__file__), "sim")
_REGISTRY_PATH = os.path.join(_SIM_DIR, "task_registry.py")

# RoboTwin 2.0 이식 Part 1(시각 domain randomization)의 공통 장식 블록을
# 그대로 재사용한다(assets/tacker_openarm.xml과 이름까지 동일) --
# sim/visual_randomization.py의 apply_visual_config()가 새 태스크에도
# 자동으로 적용되려면 이 이름 규칙을 그대로 지켜야 한다(Part 3-4 검증
# 항목).
_VISUAL_ASSET_BLOCK = """
    <texture name="ground_tex_0" type="2d" builtin="checker" rgb1="0.5 0.5 0.52" rgb2="0.4 0.4 0.42" width="64" height="64"/>
    <material name="ground_mat_0" texture="ground_tex_0" texrepeat="4 4" reflectance="0.1"/>
    <texture name="ground_tex_1" type="2d" builtin="checker" rgb1="0.45 0.42 0.38" rgb2="0.3 0.28 0.25" width="64" height="64"/>
    <material name="ground_mat_1" texture="ground_tex_1" texrepeat="6 6" reflectance="0.05"/>
    <texture name="ground_tex_2" type="2d" builtin="gradient" rgb1="0.55 0.6 0.65" rgb2="0.25 0.28 0.32" width="64" height="64"/>
    <material name="ground_mat_2" texture="ground_tex_2" reflectance="0.0"/>

    <texture name="table_tex_0" type="2d" builtin="checker" rgb1="0.55 0.4 0.3" rgb2="0.45 0.32 0.24" width="64" height="64"/>
    <material name="table_mat_0" texture="table_tex_0" texrepeat="3 3" reflectance="0.1"/>
    <texture name="table_tex_1" type="2d" builtin="checker" rgb1="0.6 0.55 0.5" rgb2="0.4 0.36 0.3" width="64" height="64"/>
    <material name="table_mat_1" texture="table_tex_1" texrepeat="2 2" reflectance="0.15"/>
    <texture name="table_tex_2" type="2d" builtin="gradient" rgb1="0.5 0.35 0.25" rgb2="0.25 0.18 0.12" width="64" height="64"/>
    <material name="table_mat_2" texture="table_tex_2" reflectance="0.05"/>
"""

_VISUAL_WORLDBODY_BLOCK = """
    <light pos="0.4 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="0.7 -0.4 1.0" dir="-0.4 0.4 -0.7" diffuse="0.4 0.4 0.4"/>

    <geom name="ground" type="plane" pos="0 0 0" size="0.8 0.8 0.01" contype="0" conaffinity="0" material="ground_mat_0" rgba="0.5 0.5 0.52 1"/>

    <body name="clutter_0" pos="-0.55 -0.55 0">
      <geom name="clutter_0_geom" type="box" size="0.03 0.03 0.03" contype="0" conaffinity="0" rgba="0.5 0.5 0.5 1"/>
    </body>
    <body name="clutter_1" pos="-0.55 0.55 0">
      <geom name="clutter_1_geom" type="box" size="0.03 0.03 0.03" contype="0" conaffinity="0" rgba="0.5 0.5 0.5 1"/>
    </body>
    <body name="clutter_2" pos="0.55 -0.55 0">
      <geom name="clutter_2_geom" type="box" size="0.03 0.03 0.03" contype="0" conaffinity="0" rgba="0.5 0.5 0.5 1"/>
    </body>

    <body name="table" pos="0.35 0 0">
      <geom name="table_top" type="box" pos="0 0 0.38" size="0.25 0.25 0.015" contype="0" conaffinity="0" material="table_mat_0" rgba="0.55 0.4 0.3 1"/>
      <geom name="table_leg_1" type="box" pos="0.22 0.22 0.19" size="0.015 0.015 0.19" contype="0" conaffinity="0" rgba="0.35 0.25 0.18 1"/>
      <geom name="table_leg_2" type="box" pos="-0.22 0.22 0.19" size="0.015 0.015 0.19" contype="0" conaffinity="0" rgba="0.35 0.25 0.18 1"/>
      <geom name="table_leg_3" type="box" pos="0.22 -0.22 0.19" size="0.015 0.015 0.19" contype="0" conaffinity="0" rgba="0.35 0.25 0.18 1"/>
      <geom name="table_leg_4" type="box" pos="-0.22 -0.22 0.19" size="0.015 0.015 0.19" contype="0" conaffinity="0" rgba="0.35 0.25 0.18 1"/>
    </body>
"""

# sim/stabilizer.py:Stabilizer가 요구하는 고정 이름(모듈 docstring 참고) --
# position-correction/impact-recoil 두 패턴 모두 이 블록을 그대로 쓴다.
_VIRTUAL_EE_BLOCK = """
    <body name="stabilizer_ee" pos="0 0 0">
      <joint name="stab_x" type="slide" axis="1 0 0" damping="90"/>
      <joint name="stab_y" type="slide" axis="0 1 0" damping="90"/>
      <joint name="stab_z" type="slide" axis="0 0 1" damping="90"/>
      <inertial pos="0 0 0" mass="0.1" diaginertia="1e-5 1e-5 1e-5"/>
      <geom name="stabilizer_ee_geom" type="sphere" size="0.012" contype="0" conaffinity="0" rgba="0.2 0.8 0.3 1"/>
      <site name="stabilizer_ee_site" pos="0 0 0" size="0.005"/>
    </body>
"""
_VIRTUAL_EE_ACTUATORS = """
    <position name="stab_x" joint="stab_x" kp="20000" forcerange="-3000 3000"/>
    <position name="stab_y" joint="stab_y" kp="20000" forcerange="-3000 3000"/>
    <position name="stab_z" joint="stab_z" kp="20000" forcerange="-3000 3000"/>
"""

# impact-recoil 전용: _VIRTUAL_EE_BLOCK과 거의 같지만 geom이 충돌에
# 참여한다(contype=1 conaffinity=1) -- 이 패턴은 "쥐고 옮기기"가 아니라
# "쳐서 밀어내기"라 실제로 부딪혀야 한다. position_correction/
# torque_reactive는 원래 Stabilizer 역할(contype=0, 쥔 물체는 weld가
# 고정하므로 EE 자체가 충돌할 필요 없음)을 그대로 쓴다.
_VIRTUAL_EE_BLOCK_COLLIDABLE = """
    <body name="stabilizer_ee" pos="0 0 0">
      <joint name="stab_x" type="slide" axis="1 0 0" damping="90"/>
      <joint name="stab_y" type="slide" axis="0 1 0" damping="90"/>
      <joint name="stab_z" type="slide" axis="0 0 1" damping="90"/>
      <inertial pos="0 0 0" mass="0.1" diaginertia="1e-5 1e-5 1e-5"/>
      <geom name="stabilizer_ee_geom" type="sphere" size="0.015" condim="3" rgba="0.2 0.8 0.3 1"/>
      <site name="stabilizer_ee_site" pos="0 0 0" size="0.005"/>
    </body>
"""


def _pascal_case(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\-]", name) if part)


def _base_xml(option_extra: str = "") -> tuple[str, str, str]:
    """(asset_block, worldbody_block, option_line)을 반환 -- 모든 패턴이
    공유하는 Part 1 장식 + implicitfast(lint 체크 #1, Part 3-3 참고)."""
    option_line = f'<option timestep="0.002" integrator="implicitfast" {option_extra}/>'
    return _VISUAL_ASSET_BLOCK, _VISUAL_WORLDBODY_BLOCK, option_line


def build_position_correction_xml(object_size: tuple[float, float, float], object_mass: float) -> str:
    # gravcomp="1": assets/peg_in_hole_bimanual_openarm.xml의 peg/hole_socket과
    # 같은 이유(그 파일 바디 주석 참고) -- freejoint 자유 바디라 바닥이
    # 없으면 중력으로 떨어진다. moving_object는 잡히기 전 잠깐(approach
    # 단계, 최대 60스텝)만 자유 상태라 효과가 작지만, impact_recoil의
    # workpiece(아예 안 붙잡힘)는 이게 없으면 에피소드 내내 자유낙하해서
    # displacement가 수십 m까지 치솟는 버그가 났다(실측 확인, scaffold_
    # task.py 개발 중 발견).
    asset_block, worldbody_block, option_line = _base_xml()
    mo_half = " ".join(str(x) for x in object_size)
    return f"""<mujoco model="scaffolded_position_correction">
  {option_line}
  <asset>{asset_block}  </asset>
  <worldbody>{worldbody_block}
    <body name="moving_object" pos="0.0 0.0 0.25" gravcomp="1">
      <freejoint name="moving_object_free"/>
      <geom name="moving_object_geom" type="box" size="{mo_half}" mass="{object_mass}" rgba="0.95 0.35 0.05 1"/>
    </body>
    <site name="target_site" pos="0.0 0.0 0.1" size="0.005" rgba="1 0 0 0.5"/>
{_VIRTUAL_EE_BLOCK}  </worldbody>
  <equality>
    <weld name="stabilizer_weld" body1="stabilizer_ee" body2="moving_object"
          solref="0.002 1" solimp="0.95 0.99 0.0001 0.5 2" active="false"/>
  </equality>
  <actuator>{_VIRTUAL_EE_ACTUATORS}  </actuator>
</mujoco>
"""


def build_torque_reactive_xml(object_size: tuple[float, float, float], object_mass: float) -> str:
    asset_block, worldbody_block, option_line = _base_xml()
    substructure = torque_reactive.mjcf_substructure({
        "hinge_body_name": "main_hinge_body",
        "hinge_joint_name": "main_hinge",
        "actuator_kp": 8.0,
        "actuator_forcerange": (-1.2, 1.2),
        "body_half_size": object_size,
        "body_mass": object_mass,
    })
    # mjcf_substructure()는 <actuator>/<sensor> 블록까지 자체적으로
    # 닫아서 돌려준다(patterns/torque_reactive.py 참고) -- worldbody
    # 안에는 <body> 부분만, 그 바깥에는 <actuator>/<sensor> 부분만 넣는다.
    body_part, rest = substructure.split("<actuator>", 1)
    actuator_part, sensor_part = rest.split("<sensor>", 1)
    actuator_part = actuator_part.replace("</actuator>", "").strip()
    sensor_part = sensor_part.replace("</sensor>", "").strip()
    return f"""<mujoco model="scaffolded_torque_reactive">
  {option_line}
  <asset>{asset_block}  </asset>
  <worldbody>{worldbody_block}
    {body_part.strip()}
  </worldbody>
  <actuator>
    {actuator_part}
  </actuator>
  <sensor>
    {sensor_part}
  </sensor>
</mujoco>
"""


def build_impact_recoil_xml(object_size: tuple[float, float, float], object_mass: float) -> str:
    asset_block, worldbody_block, option_line = _base_xml()
    wp_half = " ".join(str(x) for x in object_size)
    return f"""<mujoco model="scaffolded_impact_recoil">
  {option_line}
  <asset>{asset_block}  </asset>
  <worldbody>{worldbody_block}
    <body name="workpiece" pos="0.0 0.0 0.1" gravcomp="1">
      <freejoint name="workpiece_free"/>
      <geom name="workpiece_geom" type="box" size="{wp_half}" mass="{object_mass}" rgba="0.3 0.4 0.9 1"/>
    </body>
{_VIRTUAL_EE_BLOCK_COLLIDABLE}  </worldbody>
  <actuator>{_VIRTUAL_EE_ACTUATORS}  </actuator>
</mujoco>
"""


_BUILDERS = {
    "position_correction": build_position_correction_xml,
    "torque_reactive": build_torque_reactive_xml,
    "impact_recoil": build_impact_recoil_xml,
}

# position_correction의 Kd_pos 상한(0.005)은 실측으로 찾은 안정 영역이다
# (scaffold_task.py 개발 중 demo_peg로 스윕: Kp=0.3 기준 Kd<=0.006까지는
# 수렴, Kd>=0.007부터 D항이 매 제어 틱의 오차 부호 반전을 증폭시켜
# final_distance가 급격히 발산했다 -- peg_in_hole의 실제 Kp_xy/Kd_xy도
# "너무 작아도 너무 커도 나쁜" 벼랑형 지형이라는 것과 같은 성질, 그
# 벼랑에서 충분히 떨어진 값을 기본으로 둔다).
_GAIN_DEFAULTS = {
    "position_correction": {"names": ["Kp_pos", "Kd_pos"], "bounds": {"Kp_pos": [0.05, 0.6], "Kd_pos": [0.0, 0.005]}, "x0": [0.2, 0.001], "sigma0": [0.1, 0.002]},
    "torque_reactive": {"names": ["Kp_tau"], "bounds": {"Kp_tau": [0.05, 5.0]}, "x0": [0.3], "sigma0": [0.2]},
    "impact_recoil": {"names": ["Kp_approach"], "bounds": {"Kp_approach": [0.02, 3.0]}, "x0": [0.3], "sigma0": [0.2]},
}

_ENV_MODULE_TEMPLATES = {
    "position_correction": (
        "ScaffoldedPositionCorrectionEnv",
        """def default_scene_config() -> dict[str, Any]:
    return {{"start_offset": (0.02, 0.0, 0.0), "grasp_offset": (0.0, 0.0, 0.0), "success_radius_m": 0.01}}


def sample_scene_config(rng=None) -> dict[str, Any]:
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    dx, dy = rng.uniform(-0.03, 0.03, size=2)
    return {{"start_offset": (float(dx), float(dy), 0.0), "grasp_offset": (0.0, 0.0, 0.0), "success_radius_m": 0.01}}


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(shared_cfg)
""",
    ),
    "torque_reactive": (
        "ScaffoldedTorqueReactiveEnv",
        """def default_scene_config() -> dict[str, Any]:
    return {{"resistance_torque": 0.3, "target_rotation": 1.5}}


def sample_scene_config(rng=None) -> dict[str, Any]:
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    resistance = float(rng.uniform(0.1, 0.8))
    return {{"resistance_torque": resistance, "target_rotation": 1.5}}


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(shared_cfg)
""",
    ),
    "impact_recoil": (
        "ScaffoldedImpactRecoilEnv",
        """def default_scene_config() -> dict[str, Any]:
    return {{"ee_start_pos": (-0.12, 0.0, 0.1), "strike_distance_m": {strike_distance}, "success_displacement_m": 0.006}}


def sample_scene_config(rng=None) -> dict[str, Any]:
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    dy = float(rng.uniform(-0.02, 0.02))
    return {{"ee_start_pos": (-0.12, dy, 0.1), "strike_distance_m": {strike_distance}, "success_displacement_m": 0.006}}


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(shared_cfg)
""",
    ),
}

_CONDITION_FIELDS = {
    "position_correction": ["start_offset"],
    "torque_reactive": ["resistance_torque"],
    "impact_recoil": ["ee_start_pos"],
}


def generate(name: str, pattern: str, object_shape: str, object_size: tuple[float, float, float], object_mass: float) -> None:
    if pattern not in _BUILDERS:
        raise ValueError(f"unknown pattern {pattern!r} -- choices: {sorted(_BUILDERS)}")
    if object_shape != "box":
        # Level 1 범위: box만 지원(치수 3개가 그대로 geom size가 된다).
        # 다른 모양은 "자유형 작성"에 가까워지므로 지금은 지원하지 않는다.
        raise ValueError(f"scaffold_task.py는 지금 object_shape='box'만 지원한다 (got {object_shape!r})")

    xml_text = _BUILDERS[pattern](object_size, object_mass)
    xml_path = os.path.join(_ASSETS_DIR, f"{name}.xml")
    with open(xml_path, "w") as f:
        f.write(xml_text)

    # impact_recoil의 strike_distance_m: 가상 EE(반지름 0.015m, 이 파일의
    # _VIRTUAL_EE_BLOCK_COLLIDABLE) + 물체의 가장 큰 반치수가 실제로
    # 맞닿는 거리보다 넉넉히 커야 한다 -- 작으면 "충돌로 막혀서 그
    # 거리까지 못 들어가는" 상태가 계속되는데 fired 판정은 영원히 False로
    # 남는다(실측으로 발견: 기본값 0.02로는 거의 항상 실패했다).
    strike_distance = round(0.015 + max(object_size) + 0.01, 4)

    class_name = f"{_pascal_case(name)}Env"
    generic_class, scene_fns = _ENV_MODULE_TEMPLATES[pattern]
    scene_fns = scene_fns.format(strike_distance=strike_distance)
    env_module_text = f'''"""scaffold_task.py가 생성함 (RoboTwin 2.0 이식 Part 3-2, pattern={pattern!r}).

이 파일은 "채워 넣기"만 한다 -- 실제 control-loop/reward/success 로직은
전부 sim/generic_pattern_env.py:{generic_class}(패턴 공용 엔진)와
patterns/{pattern}.py(reward/success 공식)에 있다. 이 파일은 그 엔진을
가리키는 1줄 서브클래스 + scene_config 함수 3개(sim/base_task_env.py
모듈 docstring의 "태스크 모듈이 추가로 제공해야 하는 것들" 계약)만
담는다 -- scaffold_task.py가 매 태스크마다 새로 작성하는 유일한
Python 코드이고, 숫자 몇 개 말고는 자유형 로직이 없다.
"""
from __future__ import annotations

import os
from typing import Any

from sim.generic_pattern_env import {generic_class}

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "{name}.xml")


class {class_name}({generic_class}):
    def __init__(self, xml_path: str | None = None):
        super().__init__(xml_path or _DEFAULT_XML)


{scene_fns}'''
    env_path = os.path.join(_SIM_DIR, f"{name}_env.py")
    with open(env_path, "w") as f:
        f.write(env_module_text)

    gains = _GAIN_DEFAULTS[pattern]
    yaml_text = f"""# scaffold_task.py가 생성함 (RoboTwin 2.0 이식 Part 3-2, pattern={pattern}).
name: {name}

gains:
  names: {gains['names']}
  bounds:
{os.linesep.join(f"    {k}: {v}" for k, v in gains['bounds'].items())}
  x0: {gains['x0']}
  sigma0: {gains['sigma0']}

eval_scenarios:
  - {{}}

condition_fields: {_CONDITION_FIELDS[pattern]}
"""
    yaml_path = os.path.join(_TASKS_DIR, f"{name}.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_text)

    _register_task(name, f"sim.{name}_env:{class_name}")
    print(f"[scaffold_task] 생성 완료: {xml_path}, {env_path}, {yaml_path}")
    print(f"[scaffold_task] TASK_REGISTRY['{name}'] = 'sim.{name}_env:{class_name}'")

    _run_lint(name)


def _register_task(name: str, target: str) -> None:
    with open(_REGISTRY_PATH) as f:
        text = f.read()
    marker = "TASK_REGISTRY: dict[str, str] = {"
    if f'"{name}":' in text:
        print(f"[scaffold_task] TASK_REGISTRY에 '{name}'이 이미 있음 -- 등록 생략")
        return
    idx = text.index(marker) + len(marker)
    insertion = f'\n    "{name}": "{target}",  # scaffold_task.py가 생성'
    text = text[:idx] + insertion + text[idx:]
    with open(_REGISTRY_PATH, "w") as f:
        f.write(text)


def _run_lint(name: str) -> None:
    lint_script = os.path.join(os.path.dirname(__file__), "lint_task.py")
    if not os.path.exists(lint_script):
        print("[scaffold_task] lint_task.py가 아직 없음 -- lint 건너뜀(Part 3-3에서 추가됨)")
        return
    result = subprocess.run([sys.executable, lint_script, "--task", name], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(f"[scaffold_task] 경고: '{name}' lint 실패 -- 생성된 에셋을 확인할 것", file=sys.stderr)
    else:
        print(f"[scaffold_task] lint 통과: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pattern", required=True, choices=sorted(_BUILDERS))
    parser.add_argument("--object-shape", default="box", choices=["box"])
    parser.add_argument("--object-size", type=float, nargs=3, required=True, metavar=("HX", "HY", "HZ"))
    parser.add_argument("--object-mass", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.name, args.pattern, args.object_shape, tuple(args.object_size), args.object_mass)


if __name__ == "__main__":
    main()
