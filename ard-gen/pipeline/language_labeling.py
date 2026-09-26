"""ARD-Gen 5단계: force/torque profile로 자연어 지시문을 생성한다 (태스크 무관).

## 템플릿 기반으로 구현한 이유 (Claude API 대신)

지금은 파이프라인이 0->1->2->4->5까지 끝까지 도는지를 확인하는 단계다 --
"지시문이 얼마나 자연스러운가"가 아니라 "language 필드가 실제로 채워져서
끝까지 나오는가"가 검증 목표라서, 템플릿으로도 목적에 충분하다. 게다가:

1. 오프라인/재현 가능해야 한다 -- API 키/네트워크/비용 없이 몇 번을
   돌려도 항상 같은 결과가 나와야 지금 이 검증 단계에 맞다.
2. PIPELINE.md의 설계 원칙 2("계산 비용이 싼 방법부터 쓴다")와 같은 맥락.
3. 태스크가 여러 개(peg-in-hole, cap_twist 데모 등)가 됐어도, 각 태스크가
   표현할 정보가 여전히 단순한 조합(강도/방향/수량 중 일부)이면 템플릿
   으로 충분하다 -- 조합이 훨씬 다양해지는 태스크가 생기면 그때 Claude
   API로 바꾸는 게 맞다(기존 판단 그대로 유지).

## --task로 어휘를 교체하는 방식 (리팩토링 핵심)

템플릿 문장, 강도어(살짝/보통/세게), 방향어(시계방향/반시계방향 등),
수량어(N바퀴 등)를 전부 이 파일이 아니라 tasks/{task}.yaml의 `language`
절에서 읽는다 -- 이 파일은 "force_max 분포로 강도를 3단계로 나누고,
episode에 있는 값들로 템플릿의 빈칸을 채운다"는 절차만 안다. peg-in-hole은
강도어만 쓰고(direction_words/quantity_words가 비어있음), 회전 태스크는
direction/quantity 필드(pipeline/filter_episodes.py가 env.run_episode()
결과에서 그대로 실어옴, pipeline/episode_io.py의 extra_* 참고)까지 채운다.

사용 예:
    python pipeline/language_labeling.py --task peg_in_hole --episodes-dir ./data/episodes
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from pipeline.episode_io import load_episode, save_language
from sim.task_registry import list_tasks, load_task_config


def _intensity_bucket(force_max: float, low_thresh: float, high_thresh: float) -> str:
    if force_max <= low_thresh:
        return "gentle"
    if force_max <= high_thresh:
        return "normal"
    return "firm"


def make_language(
    episode: dict,
    language_cfg: dict,
    low_thresh: float,
    high_thresh: float,
    template_idx: int,
) -> str:
    """episode(pipeline/episode_io.py 스키마)와 태스크의 language 설정으로
    템플릿 빈칸을 채운다. 템플릿이 참조하지 않는 필드는 계산하지 않아도
    그만이므로, 태스크마다 다른 부분집합(강도만/강도+방향+수량 등)을 써도
    안전하다."""
    templates = language_cfg["templates"]
    # 각 *_words가 yaml에 아예 없으면(키 자체가 없으면) 그 필드를 안 쓰는
    # 태스크라는 뜻이고, {}(빈 dict)면 "필드는 쓰지만 값별 매핑은 없다 --
    # 값을 그대로 문자열로 넣어라"라는 뜻이다(tasks/cap_twist.yaml의
    # quantity_words 참고) -- 그래서 진위(truthy)가 아니라 키 존재로 검사한다.
    intensity_words = language_cfg.get("intensity_words")
    direction_words = language_cfg.get("direction_words")
    quantity_words = language_cfg.get("quantity_words")

    fields: dict[str, str] = {}
    if intensity_words is not None and episode.get("force_max") is not None:
        bucket = _intensity_bucket(episode["force_max"], low_thresh, high_thresh)
        fields["intensity"] = intensity_words.get(bucket, "")
    if direction_words is not None and episode.get("direction") is not None:
        fields["direction"] = direction_words.get(str(episode["direction"]), str(episode["direction"]))
    if quantity_words is not None and episode.get("quantity") is not None:
        fields["quantity"] = quantity_words.get(str(episode["quantity"]), str(episode["quantity"]))

    template = templates[template_idx % len(templates)]
    return template.format(**fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="peg_in_hole", choices=list_tasks())
    parser.add_argument("--episodes-dir", type=str, default="./data/episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_task_config(args.task)
    language_cfg = task.language
    if not language_cfg.get("templates"):
        print(f"[language_labeling] tasks/{args.task}.yaml에 language.templates가 없음 -- 라벨링 불가.")
        return

    paths = sorted(glob.glob(os.path.join(args.episodes_dir, "episode_*.npz")))
    if not paths:
        print(f"[language_labeling] {args.episodes_dir}에 episode_*.npz가 없음 -- 4단계를 먼저 실행하세요.")
        return

    episodes = [load_episode(p) for p in paths]
    force_maxes = np.array([e.get("force_max", 0.0) for e in episodes])
    low_thresh = float(np.percentile(force_maxes, 33))
    high_thresh = float(np.percentile(force_maxes, 66))
    print(
        f"[language_labeling] task={task.name} {len(paths)}개 에피소드, force_max 범위 "
        f"[{force_maxes.min():.1f}, {force_maxes.max():.1f}]N, "
        f"강도 경계(33/66 백분위): {low_thresh:.1f}N / {high_thresh:.1f}N"
    )

    bucket_counts = {"gentle": 0, "normal": 0, "firm": 0}
    for i, (path, episode) in enumerate(zip(paths, episodes)):
        language = make_language(episode, language_cfg, low_thresh, high_thresh, template_idx=i)
        save_language(path, language)
        if episode.get("force_max") is not None:
            bucket_counts[_intensity_bucket(episode["force_max"], low_thresh, high_thresh)] += 1

    print(f"[language_labeling] 강도 분포: {bucket_counts}")
    print(f"[language_labeling] {len(paths)}개 에피소드에 language 필드 추가 완료")


if __name__ == "__main__":
    main()
