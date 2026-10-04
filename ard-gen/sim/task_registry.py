"""태스크 이름 -> Env 클래스/설정 매핑.

각 태스크는 tasks/{name}.yaml 설정 파일 하나 + sim/ 아래 BaseTaskEnv를
상속하는 Env 클래스가 있는 모듈 하나로 등록된다(sim/cap_twist_env.py +
tasks/cap_twist.yaml이 "새 태스크를 config+Env만으로 추가하는" 실제 예시).
TASK_REGISTRY에 이름 -> "module.path:ClassName" 한 줄만 추가하면 --task
인자를 받는 모든 파이프라인 스크립트가 그 태스크를 쓸 수 있다.

사용법:
    from sim.task_registry import load_task_config
    task = load_task_config("peg_in_hole")
    env = task.make_env()
    result = env.run_episode(task.gains_from_vector([kp, kd]), task.default_scene_config())
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

_TASKS_DIR = os.path.join(os.path.dirname(__file__), "..", "tasks")

# 태스크 이름 -> Env 클래스 경로("module.path:ClassName"). 세부 설정(게인
# 이름/범위, 씬 무작위화, 성공 기준, 언어 라벨링 어휘)은 tasks/{name}.yaml에
# 있다 -- 여기는 "이름 -> 코드" 매핑만 담당한다.
TASK_REGISTRY: dict[str, str] = {
    # OpenArm 양팔 버전으로 교체함(VX300s 버전은 sim/peg_in_hole_env.py:PegInHoleEnv
    # 로 여전히 존재하지만 더 이상 등록돼 있지 않다 -- 사용자 요청으로 전환,
    # sim/peg_in_hole_openarm_env.py 상단 docstring 참고).
    "peg_in_hole": "sim.peg_in_hole_openarm_env:PegInHoleOpenArmEnv",
    "cap_twist": "sim.cap_twist_env:CapTwistEnv",
    # 일반 Stabilizer(가상 EE) 버전(sim.tacker_env:TackerEnv)은 peg_in_hole과
    # 같은 이유로 OpenArm 양팔 버전으로 교체했다(사용자 요청) -- 구버전
    # 코드/자산(assets/tacker.xml, sim/tacker_env.py)은 참고용으로 남겨뒀지만
    # 더 이상 등록돼 있지 않다.
    "tacker": "sim.tacker_openarm_env:TackerOpenArmEnv",
    # 레거시 VX300s 단일팔 버전(sim/screw_driving_sim.py:ScrewDrivingSim,
    # optimize/screw_driving_cma_search.py)은 OpenArm 양팔 버전을 최적화하지
    # 못하고 있었던 공백을 메우려고 등록을 전환했다(별도 세션 진단 + 사용자
    # 요청, sim/screw_driving_openarm_env.py 상단 docstring 참고) -- 구버전
    # 코드/자산(assets/screw_driving.xml, sim/screw_driving_sim.py,
    # optimize/screw_driving_cma_search.py)은 참고용으로 남겨뒀지만 더 이상
    # 등록돼 있지 않다.
    "screw_driving": "sim.screw_driving_openarm_env:ScrewDrivingOpenArmEnv",
}


def list_tasks() -> list[str]:
    return sorted(TASK_REGISTRY.keys())


def _task_yaml_path(task_name: str) -> str:
    return os.path.join(_TASKS_DIR, f"{task_name}.yaml")


def _resolve(task_name: str) -> str:
    if task_name not in TASK_REGISTRY:
        raise KeyError(f"unknown task {task_name!r} -- available: {list_tasks()}")
    return TASK_REGISTRY[task_name]


@dataclass
class TaskConfig:
    """tasks/{name}.yaml 하나 + 그 태스크의 Env 모듈을 감싸는 핸들.
    파이프라인 스크립트는 이 객체를 통해서만 태스크에 접근하고, 특정
    task의 sim/Env 모듈을 직접 import하지 않는다."""

    name: str
    module: Any  # Env 클래스가 정의된 모듈 (default_scene_config/sample_scene_config/... 도 여기 있음)
    env_class: type
    gain_names: list[str]
    gain_bounds: dict[str, tuple[float, float]]
    x0: list[float]
    sigma0: list[float]
    eval_scenarios: list[dict[str, Any]]
    condition_fields: list[str] = field(default_factory=list)
    success_reward_threshold: float = float("inf")
    language: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    # -- Env 생성 ---------------------------------------------------------
    def make_env(self, **kwargs) -> Any:
        """tasks/{name}.yaml에 최상위 `stabilizer` 절이 있으면(3단계, 왼팔
        지원 태스크), Env 생성자에 stabilizer_config로 자동 전달한다 --
        호출자(cma_search.py 등)가 그 존재를 몰라도 된다. `stabilizer` 절이
        없는 태스크(향후 단일팔 전용 태스크)의 Env는 이 kwarg를 아예 안
        받으므로 영향이 없다. 호출자가 명시적으로 stabilizer_config나
        use_stabilizer를 kwargs로 넘기면 그걸 그대로 쓴다(setdefault)."""
        stabilizer_cfg = self.raw.get("stabilizer")
        if stabilizer_cfg is not None:
            kwargs.setdefault("stabilizer_config", stabilizer_cfg)
        return self.env_class(**kwargs)

    # -- 게인 벡터 <-> dict -------------------------------------------------
    def gains_from_vector(self, values: list[float]) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.gain_names, values)}

    def gains_to_vector(self, gains: dict[str, float]) -> list[float]:
        return [float(gains[name]) for name in self.gain_names]

    def bounds_lo_hi(self) -> tuple[list[float], list[float]]:
        lo = [self.gain_bounds[n][0] for n in self.gain_names]
        hi = [self.gain_bounds[n][1] for n in self.gain_names]
        return lo, hi

    def clip_gains(self, gains: dict[str, float]) -> dict[str, float]:
        return {
            name: float(min(max(val, self.gain_bounds[name][0]), self.gain_bounds[name][1]))
            for name, val in gains.items()
        }

    # -- 씬 설정 (모듈에 위임) ------------------------------------------------
    def default_scene_config(self) -> dict[str, Any]:
        return self.module.default_scene_config()

    def sample_scene_config(self, rng) -> dict[str, Any]:
        return self.module.sample_scene_config(rng)

    def to_sim_scene_config(self, shared_cfg: dict[str, Any]) -> dict[str, Any]:
        return self.module.to_sim_scene_config(shared_cfg)

    # -- diffusion 조건 벡터 -------------------------------------------------
    def condition_from_scene_config(self, cfg: dict[str, Any]):
        import numpy as np

        parts: list[float] = []
        for name in self.condition_fields:
            v = cfg[name]
            if isinstance(v, (tuple, list)):
                parts.extend(float(x) for x in v)
            else:
                parts.append(float(v))
        return np.array(parts, dtype=np.float32)

    @property
    def condition_dim(self) -> int:
        import numpy as np

        # condition_fields는 (default_scene_config()가 아니라) 1단계 공유
        # scene_config 스키마의 필드 이름이다 -- sample_scene_config()로
        # 하나 뽑아서 차원을 잰다.
        sample = self.sample_scene_config(np.random.default_rng(0))
        return len(self.condition_from_scene_config(sample))


def load_task_config(task_name: str) -> TaskConfig:
    module_path, class_name = _resolve(task_name).split(":")
    module = importlib.import_module(module_path)
    env_class = getattr(module, class_name)

    path = _task_yaml_path(task_name)
    with open(path) as f:
        raw = yaml.safe_load(f)

    gains_cfg = raw["gains"]
    gain_names = list(gains_cfg["names"])
    gain_bounds = {n: tuple(gains_cfg["bounds"][n]) for n in gain_names}

    return TaskConfig(
        name=raw.get("name", task_name),
        module=module,
        env_class=env_class,
        gain_names=gain_names,
        gain_bounds=gain_bounds,
        x0=list(gains_cfg["x0"]),
        sigma0=list(gains_cfg["sigma0"]),
        eval_scenarios=raw.get("eval_scenarios") or [{}],
        condition_fields=list(raw.get("condition_fields", [])),
        success_reward_threshold=float(raw.get("success", {}).get("reward_threshold", float("inf"))),
        language=raw.get("language", {}),
        raw=raw,
    )
