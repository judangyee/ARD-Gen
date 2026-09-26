"""공통 태스크 Env 인터페이스.

파이프라인 각 단계(게인 탐색 optimize/cma_search.py, 부트스트래핑
pipeline/bootstrap.py, diffusion 학습/샘플링 pipeline/diffusion_gains.py,
필터링 pipeline/filter_episodes.py, 언어 라벨링 pipeline/language_labeling.py)
가 특정 태스크의 sim 모듈을 직접 import하는 대신, 이 인터페이스 +
sim/task_registry.py의 TASK_REGISTRY를 통해 --task 인자로 태스크를 동적으로
바꿔 낄 수 있게 한다. tasks/README.md(또는 tasks/cap_twist.yaml +
sim/cap_twist_env.py)가 "새 태스크는 config 하나 + Env 클래스 하나로
추가된다"는 걸 보여주는 실제 예시다.

## reset/step/compute_reward/is_success 4개만 필수(추상)로 강제하는 이유

실제 제어 법칙(게인을 받아서 매 스텝 action을 만드는 방식)은 태스크마다
너무 달라서(예: peg-in-hole의 admittance force-PD vs 나사 조이기의 토크
리미터 turn/rewind 상태 기계) 억지로 하나의 제어 루프 형태로 통일하면
오히려 각 태스크 구현이 부자연스러워진다. 대신 run_episode()는 태스크
구현체가 자유롭게 오버라이드하되, 그 안에서 reward/success 계산만큼은
반드시 compute_reward()/is_success()를 거치도록 강제해서(공식을 루프
안에 직접 인라인하지 않도록) 그 두 조각만큼은 파이프라인이나 테스트가
독립적으로 재사용/검증할 수 있게 한다.

## 태스크 모듈이 추가로 제공해야 하는 것들 (모듈 레벨 함수/상수)

Env 클래스 자체는 이 4개 메서드 + run_episode()만 있으면 되지만, 파이프라인
전체(씬 샘플링, diffusion 조건 벡터, npz 스키마)가 제대로 동작하려면
Env가 정의된 모듈이 아래도 함께 제공해야 한다(peg_in_hole_env.py,
cap_twist_env.py가 실제 예시):

    def default_scene_config() -> dict: ...
    def sample_scene_config(rng) -> dict: ...   # 1단계(공유 씬 샘플링)용
    def to_sim_scene_config(shared_cfg: dict) -> dict: ...  # 1단계 스키마 -> env.reset()/run_episode() 스키마
    CONDITION_FIELDS: list[str]  # diffusion 조건 벡터에 쓸 scene_config 필드 이름들(순서 고정)

sim/task_registry.py의 TaskConfig가 이 함수/상수들을 모듈에서 찾아 감싸서
돌려준다 -- 파이프라인 스크립트는 TaskConfig를 통해서만 접근하고, 모듈을
직접 import하지 않는다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTaskEnv(ABC):
    @abstractmethod
    def reset(self, scene_config: dict[str, Any]) -> Any:
        """씬을 scene_config로 초기화한다. 반환값은 태스크마다 다를 수 있다
        (예: peg-in-hole은 hole 반경) -- 필요 없는 호출자는 무시하면 된다."""

    @abstractmethod
    def step(self, action: Any) -> Any:
        """제어 틱 하나를 진행한다. action의 형태는 태스크마다 다르다."""

    @abstractmethod
    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        """episode_result(에피소드 종료 시점 정보, 또는 조기 종료 판정에
        필요한 부분집합)로부터 스칼라 리워드를 계산한다. 제어 루프 안에
        리워드 공식을 직접 쓰지 말고 항상 이 메서드를 거친다."""

    @abstractmethod
    def is_success(self, episode_result: dict[str, Any]) -> bool:
        """이 결과(또는 그 시점까지의 부분 결과)가 성공 조건을 만족하는지
        판정한다. 스텝 루프 중간의 조기 종료 판정과 최종 판정에 모두
        쓸 수 있도록, episode_result는 필요한 필드만 담은 dict여도 된다."""

    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        """게인으로 에피소드 하나를 처음부터 끝까지 실행한다. 태스크별 제어
        루프(관찰->액션 매핑, 조기 종료 조건)가 다 다르므로, 이 기본
        구현은 없고 서브클래스가 반드시 오버라이드해야 한다(위 docstring
        "reset/step/... 4개만 필수로 강제" 참고) -- 다만 내부에서
        self.compute_reward()/self.is_success()를 호출해야 한다."""
        raise NotImplementedError(f"{type(self).__name__} must implement run_episode()")
