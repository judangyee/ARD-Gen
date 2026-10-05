"""RoboTwin 2.0 이식 Part 2: 텍스처/배경 에셋 라이브러리.

RoboTwin 2.0의 전체 생성형 텍스처 파이프라인(실사 이미지를 생성 모델로
뽑아서 텍스처로 쓰는 것)을 지금 다 만들지는 않는다 -- 이번에 하는 건
"구조"뿐이다:

1. assets/textures/manifest.yaml: 텍스처 에셋의 메타데이터(설명/카테고리)
   스키마. 지금은 Part 1(sim/visual_randomization.py)이 쓰는 프로시저럴
   MuJoCo builtin 텍스처(checker/gradient) 6개만 등록돼 있다 -- 전부
   source: procedural, 실제 이미지 파일은 하나도 없다.
2. request_external_texture(): 나중에 외부 이미지 생성 API를 붙일 자리
   (함수 시그니처만) -- 실제 호출은 구현하지 않는다(요청 범위 밖).

## scaffold_task.py(Part 3)가 이 모듈을 쓰는 방식

새 태스크를 만들 때 "바닥/테이블에 어떤 텍스처를 쓸지"를 이 모듈의
list_by_category("floor"/"table")에서 고르게 하면, 매번 체커 패턴
rgb/texrepeat 숫자를 손으로 다시 안 적어도 된다(get_texture()가 그
숫자를 그대로 돌려준다) -- scaffold_task.py가 생성하는 MJCF의 <asset>
블록을 이 manifest에서 그대로 채워 넣는다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "textures", "manifest.yaml")


@dataclass
class TextureAsset:
    id: str
    category: str
    description: str
    source: str  # "procedural" | "external"
    mujoco: dict[str, Any] = field(default_factory=dict)
    file: str | None = None


def load_manifest(path: str | None = None) -> list[TextureAsset]:
    with open(path or _MANIFEST_PATH) as f:
        raw = yaml.safe_load(f)
    return [TextureAsset(**entry) for entry in raw["textures"]]


def get_texture(texture_id: str, path: str | None = None) -> TextureAsset:
    for tex in load_manifest(path):
        if tex.id == texture_id:
            return tex
    raise KeyError(f"unknown texture id {texture_id!r} -- assets/textures/manifest.yaml 참고")


def list_by_category(category: str, path: str | None = None) -> list[TextureAsset]:
    return [tex for tex in load_manifest(path) if tex.category == category]


def texture_to_mjcf(tex: TextureAsset, texture_name: str, material_name: str) -> str:
    """TextureAsset(source=="procedural")을 <texture>/<material> MJCF
    조각 문자열로 변환한다 -- scaffold_task.py가 새 태스크 XML의
    <asset> 블록에 그대로 끼워 넣을 수 있는 형태(assets/*.xml의
    ground_mat_*/table_mat_* 정의와 동일한 스키마)."""
    if tex.source != "procedural":
        raise ValueError(f"texture_to_mjcf()는 procedural 텍스처만 지원한다 (got source={tex.source!r})")
    m = tex.mujoco
    rgb1 = " ".join(str(x) for x in m["rgb1"])
    rgb2 = " ".join(str(x) for x in m["rgb2"])
    texture_line = (
        f'<texture name="{texture_name}" type="{m["type"]}" builtin="{m["builtin"]}" '
        f'rgb1="{rgb1}" rgb2="{rgb2}" width="{m["width"]}" height="{m["height"]}"/>'
    )
    mat_attrs = [f'name="{material_name}"', f'texture="{texture_name}"']
    if "texrepeat" in m:
        mat_attrs.append(f'texrepeat="{" ".join(str(x) for x in m["texrepeat"])}"')
    if "reflectance" in m:
        mat_attrs.append(f'reflectance="{m["reflectance"]}"')
    material_line = f"<material {' '.join(mat_attrs)}/>"
    return f"{texture_line}\n{material_line}"


def request_external_texture(description: str, category: str) -> TextureAsset:
    """향후 외부 이미지 생성 API(예: 텍스트 설명 -> 텍스처 이미지 모델)를
    붙일 지점. 지금은 인터페이스(함수 시그니처: 자연어 설명 + 카테고리를
    받아서 TextureAsset을 돌려준다)만 정의하고, 실제 네트워크 호출은
    구현하지 않는다(요청 "실제 외부 API 호출은 구현하지 말 것") --
    조용히 가짜 값을 돌려주면 나중에 실제로 연동할 때 호출부를 찾기
    어려워지므로, 명시적으로 NotImplementedError를 낸다."""
    raise NotImplementedError(
        "외부 텍스처 생성 API 연동은 아직 구현하지 않았다(RoboTwin 2.0 이식 "
        "Part 2 범위 밖). 지금은 assets/textures/manifest.yaml에 수록된 "
        "procedural 텍스처만 쓸 수 있다 -- list_by_category()/get_texture() 참고. "
        f"(요청된 description={description!r}, category={category!r})"
    )
