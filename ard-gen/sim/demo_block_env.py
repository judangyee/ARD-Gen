"""scaffold_task.py가 생성함 (RoboTwin 2.0 이식 Part 3-2, pattern='torque_reactive').

이 파일은 "채워 넣기"만 한다 -- 실제 control-loop/reward/success 로직은
전부 sim/generic_pattern_env.py:ScaffoldedTorqueReactiveEnv(패턴 공용 엔진)와
patterns/torque_reactive.py(reward/success 공식)에 있다. 이 파일은 그 엔진을
가리키는 1줄 서브클래스 + scene_config 함수 3개(sim/base_task_env.py
모듈 docstring의 "태스크 모듈이 추가로 제공해야 하는 것들" 계약)만
담는다 -- scaffold_task.py가 매 태스크마다 새로 작성하는 유일한
Python 코드이고, 숫자 몇 개 말고는 자유형 로직이 없다.
"""
from __future__ import annotations

import os
from typing import Any

from sim.generic_pattern_env import ScaffoldedTorqueReactiveEnv

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "demo_block.xml")


class DemoBlockEnv(ScaffoldedTorqueReactiveEnv):
    def __init__(self, xml_path: str | None = None):
        super().__init__(xml_path or _DEFAULT_XML)


def default_scene_config() -> dict[str, Any]:
    return {"resistance_torque": 0.3, "target_rotation": 1.5}


def sample_scene_config(rng=None) -> dict[str, Any]:
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    resistance = float(rng.uniform(0.1, 0.8))
    return {"resistance_torque": resistance, "target_rotation": 1.5}


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(shared_cfg)
