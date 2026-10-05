"""RoboTwin 2.0 이식, Part 1: 5축 시각(visual) domain randomization.

배경 텍스처/조명/clutter(장애물)/테이블 높이 4가지를 다룬다(5번째 "축"은
재질 전체 톤을 함께 바꾸는 "팔레트 선택" -- 아래 PALETTES 참고, 요청의
"배경 텍스처"와 "조명"을 교차시키는 축으로 묶었다).

## 물리 파라미터 무작위화와 완전히 독립적인 이유 (구조적으로, 코드 리뷰로만 보장하는 게 아니라)

1. 이 모듈이 건드리는 MjModel 필드는 전부 렌더링에만 쓰이고 mj_step()의
   동역학 계산에 들어가지 않는다: mat_rgba/mat_specular/mat_shininess,
   geom_matid/geom_rgba(시각 전용 geom만, 아래 참고), geom_pos(clutter --
   contype=0/conaffinity=0이라 어디 둬도 접촉이 생기지 않음), light_*,
   그리고 table_* 바디/geom(애초에 설계부터 contype=0 "순수 시각 요소",
   assets/*.xml의 table 바디 주석 참고 -- 이 모듈은 그 설계를 그대로
   재사용할 뿐 새로 만들지 않는다).
2. 이 모듈은 scene_config(물리 1단계 공유 씬 설정)를 전혀 읽거나 쓰지
   않는다 -- apply_visual_config(model, visual_cfg)는 model만 받고,
   sample_visual_config(rng)은 호출자가 scene_config용 rng와 **분리된**
   별도 np.random.Generator를 넘기게 강제한다(공유 전역 rng를 쓰지
   않음 -- pipeline/filter_episodes.py, pipeline/bootstrap.py의 호출부
   참고, 두 rng 스트림이 완전히 분리돼 있다).
3. tests/test_visual_randomization.py의 test_independent_of_physics가
   이걸 "코드를 읽어서 그럴 것"이 아니라 직접 실측으로 증명한다: 같은
   물리 scene_config/게인으로 서로 다른 visual_cfg를 적용한 두 에피소드가
   reward/success/step_count/trajectory까지 완전히 bit-identical한지
   비교한다.

## 왜 3개 태스크에 공통 함수 하나로 적용되는가

peg_in_hole_bimanual_openarm.xml과 tacker_openarm.xml은 이미 동일한
이름 규칙(ground, table/table_top/table_leg_1..4, light 2개)을 쓰고
있었다(둘 다 bimanual_openarm.xml 계열에서 파생). cap_twist.xml은
원래 이 규칙이 없어서(단순 모델이라 ground/장식 table이 없었음) 이번에
같은 이름 규칙을 맞춰 추가했다(assets/cap_twist.xml 변경 이력 참고,
물리에 쓰이던 "table" plane은 floor_contact로 이름만 바꾸고 그 자체는
손대지 않았다) -- 그 결과 apply_visual_config() 하나가 이름만 보고
3개 XML 모두에 똑같이 동작한다(없는 이름은 조용히 건너뜀, 아래
_try_geom/_try_body 참고 -- 향후 태스크가 일부 장식 요소를 생략해도
에러 없이 동작).

## 외부 생성 모델을 쓰지 않는 이유

이번 축(배경 텍스처/조명/clutter/테이블)은 전부 "프로시저럴"이다 --
MuJoCo의 builtin checker/gradient 텍스처 + rgba 수치 변화만 쓰고, 실제
이미지 생성 API는 호출하지 않는다(요청 "Part 2"에서 그 자리만
인터페이스로 남겨둔다, pipeline/texture_library.py 참고).
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

# 장식용 재질 팔레트 변형 개수(바닥/테이블 각각) -- assets/*.xml의
# <asset>에 ground_mat_0..N-1 / table_mat_0..N-1 이 이 개수만큼 미리
# 구워져 있어야 한다(각 XML의 <asset> 블록 참고).
N_MATERIAL_VARIANTS = 3

# clutter(장애물) 슬롯 개수 -- 각 XML에 clutter_0..N-1 바디가 미리
# 구워져 있다(전부 contype=0 conaffinity=0, "충돌 없이 배치" 요청 그대로).
N_CLUTTER_SLOTS = 3

# 테이블 높이 무작위화 범위(m) -- 위/아래로 비대칭인 이유: mj_geomDistance로
# 직접 스캔해보니(peg_in_hole 기준, tests/test_visual_randomization.py::
# test_table_height_range_safe_for_openarm_tasks가 그대로 재현) home
# 자세에서 table_top과 hole_socket(왼팔이 쥔 hole_floor)의 실제 최소
# 거리가 offset=0에서 20.6mm, +10mm에서 17mm, +20mm에서 7mm, +30mm에서
# -2.9mm(겹침)로 테이블을 올릴수록 빠르게 줄어든다(반대로 내리는 쪽은
# -50mm까지도 18mm 이상 여유가 그대로 유지됨 -- hole이 테이블보다 훨씬
# 위에 떠 있어서 내리는 방향엔 애초에 아무것도 없기 때문). 그래서 위로는
# 15mm(최소 실측 clearance 약 12mm 확보), 아래로는 30mm까지 허용한다.
TABLE_HEIGHT_RANGE_DOWN_M = 0.03
TABLE_HEIGHT_RANGE_UP_M = 0.015

# 조명 밝기/색온도 무작위화 폭.
LIGHT_INTENSITY_RANGE = (0.6, 1.4)  # 기준 diffuse/specular/ambient에 곱할 배율
LIGHT_COLOR_TEMP_RANGE = (-1.0, 1.0)  # -1(따뜻함, R 강조) ~ +1(차가움, B 강조)

# clutter를 작업 공간 밖(로봇 팔이 닿지 않는 바닥 구석)에 두기 위한
# 후보 영역 -- 태스크마다 작업공간 위치가 달라서(peg_in_hole/tacker는
# x>0.1 쪽에 로봇이, cap_twist는 원점 부근에 bottle이 있음) 한 세트로
# 통일하지 않고, "로봇 받침대/바텀 반대쪽 바닥 구석" 4곳을 공통으로
# 쓴다 -- ground 자체가 거의 모든 태스크에서 0.8x0.8m라 이 네 구석은
# 실측상 로봇/작업물 AABB와 거리가 멀다(위 range-safety 테스트가 함께
# 확인).
_CLUTTER_CORNERS_XY = [(-0.55, -0.55), (-0.55, 0.55), (0.55, -0.55)]
_CLUTTER_SHAPES = ("box", "sphere", "cylinder")


def sample_visual_config(rng: np.random.Generator | None = None) -> dict[str, Any]:
    """시각 무작위화 설정 하나를 뽑는다. rng는 물리 scene_config용과
    **반드시 분리된** 스트림을 넘겨야 한다(모듈 docstring "완전히
    독립적인 이유" 2번 참고) -- 호출자가 섞어 쓰면 이 함수는 그걸
    감지할 방법이 없으므로, 호출부 규약으로 강제한다."""
    if rng is None:
        rng = np.random.default_rng()

    ground_variant = int(rng.integers(0, N_MATERIAL_VARIANTS))
    table_variant = int(rng.integers(0, N_MATERIAL_VARIANTS))
    ground_tint = rng.uniform(0.85, 1.15, size=3)
    table_tint = rng.uniform(0.85, 1.15, size=3)

    table_height_offset = float(rng.uniform(-TABLE_HEIGHT_RANGE_DOWN_M, TABLE_HEIGHT_RANGE_UP_M))

    light_intensity_scale = float(rng.uniform(*LIGHT_INTENSITY_RANGE))
    light_color_temp = float(rng.uniform(*LIGHT_COLOR_TEMP_RANGE))

    clutter = []
    for i in range(N_CLUTTER_SLOTS):
        active = bool(rng.uniform() < 0.7)
        corner = _CLUTTER_CORNERS_XY[i % len(_CLUTTER_CORNERS_XY)]
        jitter_xy = rng.uniform(-0.08, 0.08, size=2)
        clutter.append({
            "active": active,
            "pos_xy": (float(corner[0] + jitter_xy[0]), float(corner[1] + jitter_xy[1])),
            "size": float(rng.uniform(0.02, 0.045)),
            "rgba": tuple(float(x) for x in rng.uniform(0.2, 0.9, size=3)) + (1.0,),
            "shape": _CLUTTER_SHAPES[int(rng.integers(0, len(_CLUTTER_SHAPES)))],
        })

    return {
        "ground_variant": ground_variant,
        "ground_tint": tuple(float(x) for x in ground_tint),
        "table_variant": table_variant,
        "table_tint": tuple(float(x) for x in table_tint),
        "table_height_offset": table_height_offset,
        "light_intensity_scale": light_intensity_scale,
        "light_color_temp": light_color_temp,
        "clutter": clutter,
    }


def default_visual_config() -> dict[str, Any]:
    """무작위화를 전혀 하지 않은("기존 동작 그대로") 기준 설정 --
    apply_visual_config(model, default_visual_config())을 불러도
    외형이 XML 원본과 같아야 한다(회귀 테스트가 확인)."""
    return {
        "ground_variant": 0,
        "ground_tint": (1.0, 1.0, 1.0),
        "table_variant": 0,
        "table_tint": (1.0, 1.0, 1.0),
        "table_height_offset": 0.0,
        "light_intensity_scale": 1.0,
        "light_color_temp": 0.0,
        "clutter": [{"active": False, "pos_xy": (0.0, 0.0), "size": 0.03, "rgba": (0.5, 0.5, 0.5, 1.0), "shape": "box"} for _ in range(N_CLUTTER_SLOTS)],
    }


def _try_geom_id(model: mujoco.MjModel, name: str) -> int | None:
    try:
        return model.geom(name).id
    except KeyError:
        return None


def _try_body_id(model: mujoco.MjModel, name: str) -> int | None:
    try:
        return model.body(name).id
    except KeyError:
        return None


def _color_temp_rgb_scale(color_temp: float) -> np.ndarray:
    """-1(따뜻함)~+1(차가움)을 R/G/B 채널별 배율로 변환한다 -- 실제
    색온도(켈빈) 물리 모델이 아니라, "따뜻할수록 R 강조/B 약화, 차가울수록
    반대"의 단순 선형 틴트(렌더링 다양성 목적, 물리적 색온도 계산 불필요)."""
    t = float(np.clip(color_temp, -1.0, 1.0))
    return np.array([1.0 + 0.15 * t * -1, 1.0, 1.0 + 0.15 * t], dtype=np.float64)


def _apply_material_variant(
    model: mujoco.MjModel,
    geom_name: str,
    mat_prefix: str,
    variant: int,
    tint: tuple[float, float, float],
) -> None:
    geom_id = _try_geom_id(model, geom_name)
    if geom_id is None:
        return
    mat_name = f"{mat_prefix}_{variant}"
    try:
        mat_id = model.material(mat_name).id
    except KeyError:
        return
    model.geom_matid[geom_id] = mat_id
    base_rgba = model.mat_rgba[mat_id].copy()
    base_rgba[:3] = np.clip(base_rgba[:3] * np.asarray(tint, dtype=np.float64), 0.0, 1.0)
    model.geom_rgba[geom_id] = base_rgba


def _apply_table_height(model: mujoco.MjModel, body_name: str, offset: float) -> None:
    body_id = _try_body_id(model, body_name)
    if body_id is None:
        return
    pos = model.body_pos[body_id].copy()
    pos[2] = float(pos[2] + offset)
    model.body_pos[body_id] = pos


def _apply_lights(model: mujoco.MjModel, intensity_scale: float, color_temp: float) -> None:
    if model.nlight == 0:
        return
    rgb_scale = _color_temp_rgb_scale(color_temp)
    for i in range(model.nlight):
        for arr_name in ("light_diffuse", "light_ambient", "light_specular"):
            arr = getattr(model, arr_name)
            base = arr[i].copy()
            arr[i] = np.clip(base * intensity_scale * rgb_scale, 0.0, 1.0)


def _apply_clutter(model: mujoco.MjModel, clutter_cfg: list[dict[str, Any]]) -> None:
    for i, slot in enumerate(clutter_cfg):
        body_name = f"clutter_{i}"
        geom_name = f"clutter_{i}_geom"
        body_id = _try_body_id(model, body_name)
        geom_id = _try_geom_id(model, geom_name)
        if body_id is None or geom_id is None:
            continue
        if not slot.get("active", False):
            # 멀리/바닥 아래로 치워서 "꺼짐"을 표현한다(바디/geom 자체를
            # 지우는 문법은 MuJoCo에 없음 -- contype=0이라 어차피 어디
            # 있어도 물리에는 영향 없음, 모듈 docstring 참고).
            pos = model.body_pos[body_id].copy()
            pos[:] = [0.0, 0.0, -5.0]
            model.body_pos[body_id] = pos
            continue
        x, y = slot["pos_xy"]
        model.body_pos[body_id] = np.array([x, y, 0.0], dtype=np.float64)
        size = float(slot.get("size", 0.03))
        current_size = model.geom_size[geom_id].copy()
        current_size[:] = size  # box/sphere/cylinder 전부 첫 size 성분이 "반경류" 스케일
        model.geom_size[geom_id] = current_size
        model.geom_rgba[geom_id] = np.asarray(slot.get("rgba", (0.5, 0.5, 0.5, 1.0)), dtype=np.float64)


def apply_visual_config(model: mujoco.MjModel, visual_cfg: dict[str, Any] | None) -> None:
    """visual_cfg(sample_visual_config() 또는 default_visual_config()의
    반환값)를 model(mujoco.MjModel)에 적용한다. visual_cfg가 None이면
    아무것도 하지 않는다(하위호환 -- 기존 호출자는 이 함수를 아예 모른다).

    이름이 없는 장식 요소(ground/table/clutter_i/light)는 조용히
    건너뛴다 -- 태스크 XML마다 일부가 없을 수 있어서(모듈 docstring
    "왜 3개 태스크에 공통 함수 하나로" 참고), 이게 에러가 되면 공통
    함수로 쓸 수가 없다."""
    if visual_cfg is None:
        return

    _apply_material_variant(model, "ground", "ground_mat", visual_cfg["ground_variant"], visual_cfg["ground_tint"])
    _apply_material_variant(model, "table_top", "table_mat", visual_cfg["table_variant"], visual_cfg["table_tint"])

    for leg in ("table_leg_1", "table_leg_2", "table_leg_3", "table_leg_4"):
        geom_id = _try_geom_id(model, leg)
        if geom_id is not None:
            model.geom_rgba[geom_id, :3] = np.clip(
                model.geom_rgba[geom_id, :3] * np.asarray(visual_cfg["table_tint"], dtype=np.float64), 0.0, 1.0
            )

    _apply_table_height(model, "table", visual_cfg["table_height_offset"])
    _apply_lights(model, visual_cfg["light_intensity_scale"], visual_cfg["light_color_temp"])
    _apply_clutter(model, visual_cfg["clutter"])
