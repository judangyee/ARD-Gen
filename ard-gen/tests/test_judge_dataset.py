"""Judge 데이터셋 생성(`pipeline/generate_judge_dataset.py`) 검증 --
시드 결정론, 분할 간 비중복, 판정 샘플 인덱스 스키마, 이어서 생성(청크
스킵)이 실제로 동작하는지. 전부 `/tmp` 아래 scratch 디렉터리에서만
작업하고 실제 데이터 브랜치(`data/closed-loop-v1`)는 안 건드린다
(`--no-push`만 쓴다).

pytest 없이 plain assert로 짰다 -- 이 저장소의 다른 스크립트들과 같은 스타일.

사용법 (osmesa 필요 -- 실제 에피소드를 렌더링하므로):
    MUJOCO_GL=osmesa python tests/test_judge_dataset.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

from pipeline.generate_closed_loop_episodes import load_disturbance_config
from pipeline.generate_judge_dataset import (
    episode_seed,
    episode_split,
    generate_one_episode,
    instruction_text,
)

_TMP_DIR = os.path.join(os.path.dirname(__file__), "_tmp_judge_dataset")
_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "pipeline", "generate_judge_dataset.py")


def test_seed_determinism_and_no_overlap_across_sources() -> None:
    """같은 (source, episode_idx)는 항상 같은 시드 -- 그리고 출처별
    시드 공간이 절대 안 겹친다(사용자 지시 "어느 워커가 돌려도 같은
    결과" + "분할 간 시드·씬이 겹치지 않음을 테스트로 확인")."""
    assert episode_seed("A", 5) == episode_seed("A", 5)
    assert episode_seed("A", 0) != episode_seed("B", 0) != episode_seed("C", 0)
    # 출처당 최대 에피소드 수(여유있게 10만개까지) 안에서는 절대 안 겹친다.
    max_episodes = 100_000
    a_range = {episode_seed("A", i) for i in range(max_episodes)}
    b_range = {episode_seed("B", i) for i in range(max_episodes)}
    c_range = {episode_seed("C", i) for i in range(max_episodes)}
    assert a_range.isdisjoint(b_range) and b_range.isdisjoint(c_range) and a_range.isdisjoint(c_range)
    print("[OK] 시드 결정론 + 출처 간 비중복(각 10만개 범위)")


def test_split_assignment_no_overlap() -> None:
    """B의 train/val/test_B 경계가 정확히 분리되는지, A/C는 전부
    test_A/test_C인지."""
    b_train, b_val = 600, 100
    assert episode_split("A", 0, b_train, b_val) == "test_A"
    assert episode_split("A", 299, b_train, b_val) == "test_A"
    assert episode_split("C", 0, b_train, b_val) == "test_C"
    assert episode_split("B", 0, b_train, b_val) == "train"
    assert episode_split("B", b_train - 1, b_train, b_val) == "train"
    assert episode_split("B", b_train, b_train, b_val) == "val"
    assert episode_split("B", b_train + b_val - 1, b_train, b_val) == "val"
    assert episode_split("B", b_train + b_val, b_train, b_val) == "test_B"
    assert episode_split("B", b_train + b_val + 99, b_train, b_val) == "test_B"
    print("[OK] 분할 경계(B: train/val/test_B, A/C: test_A/test_C) 비중복 확인")


def test_episode_determinism_physics_and_images() -> None:
    """같은 에피소드 번호를 두 번 생성하면 물리(action)·이미지가 완전히
    같은지(사용자 지시 "같은 에피소드 번호를 다시 생성하면 같은 결과")."""
    cfg = load_disturbance_config()
    ep1 = generate_one_episode("B", 11, 600, 100, cfg, None)
    ep2 = generate_one_episode("B", 11, 600, 100, cfg, None)
    assert np.array_equal(ep1["right_arm"]["action"], ep2["right_arm"]["action"])
    assert ep1["split"] == ep2["split"]
    assert len(ep1["subtask_boundaries"]) == len(ep2["subtask_boundaries"])
    assert set(ep1["boundary_images"]) == set(ep2["boundary_images"])
    for key in ep1["boundary_images"]:
        assert np.array_equal(ep1["boundary_images"][key], ep2["boundary_images"][key])
    print(f"[OK] 에피소드 재생성 결정론(B seed={episode_seed('B', 11)}): 물리+이미지 완전히 동일")


def test_instruction_text_format() -> None:
    text = instruction_text("align", False)
    assert "align" in text and "completed: False" in text and "current subtask:" in text
    text2 = instruction_text("insert", True)
    assert "completed: True" in text2
    print(f"[OK] instruction_text 포맷: {text!r}")


def test_resume_skips_completed_chunks() -> None:
    """실제 스크립트를 subprocess로 두 번 돌려서(--no-push), 두 번째
    실행이 이미 완료된 청크를 건너뛰는지 확인한다(사용자 지시 "같은
    명령을 다시 실행하면 이미 푸시된 청크는 건너뛰고 이어서")."""
    if os.path.exists(_TMP_DIR):
        shutil.rmtree(_TMP_DIR)
    os.makedirs(_TMP_DIR)

    def run() -> str:
        result = subprocess.run(
            [
                sys.executable, _SCRIPT, "--data-dir", _TMP_DIR, "--chunk-size", "2",
                "--n-a", "2", "--n-b", "0", "--n-c", "0", "--workers", "2", "--no-push",
            ],
            capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "osmesa"},
        )
        assert result.returncode == 0, f"스크립트 실패: {result.stdout}\n{result.stderr}"
        return result.stdout

    out1 = run()
    assert "[run] A_000000_000002" in out1, out1
    manifest_path = os.path.join(_TMP_DIR, "manifest.json")
    with open(manifest_path) as f:
        manifest_after_first = json.load(f)
    assert "A_000000_000002" in manifest_after_first["completed_chunks"]

    out2 = run()
    assert "[skip] A_000000_000002" in out2, out2
    assert "[run]" not in out2, "두 번째 실행인데 다시 돌렸다(이어서 생성 실패)"
    with open(manifest_path) as f:
        manifest_after_second = json.load(f)
    assert manifest_after_first == manifest_after_second, "스킵했는데 manifest가 바뀌었다"
    print("[OK] 이어서 생성: 두 번째 실행이 완료된 청크를 건너뛰고 manifest도 그대로")
    shutil.rmtree(_TMP_DIR)


def test_judge_sample_schema() -> None:
    """judge_samples.jsonl 한 줄이 필요한 필드를 전부 갖추는지(사용자
    지시 필드 목록 그대로)."""
    if os.path.exists(_TMP_DIR):
        shutil.rmtree(_TMP_DIR)
    os.makedirs(_TMP_DIR)
    result = subprocess.run(
        [
            sys.executable, _SCRIPT, "--data-dir", _TMP_DIR, "--chunk-size", "2",
            "--n-a", "0", "--n-b", "2", "--n-c", "0", "--workers", "1", "--no-push",
        ],
        capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "osmesa"},
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    jsonl_path = os.path.join(_TMP_DIR, "chunks", "B_000000_000002", "judge_samples.jsonl")
    required_keys = {
        "sample_id", "episode_id", "chunk_id", "split", "source", "disturbance_type", "loop_mode",
        "attempt", "applied_recovery", "subtask", "instruction_start", "instruction_end", "images",
        "robot_state_16dof_start", "robot_state_16dof_end", "success", "failure_type", "failure_type_idx",
        "progress", "slipped_but_succeeded",
    }
    n = 0
    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            missing = required_keys - set(sample.keys())
            assert not missing, f"필드 누락: {missing}"
            assert set(sample["images"].keys()) == {"top_cam_start", "wide_cam_start", "top_cam_end", "wide_cam_end"}
            assert len(sample["robot_state_16dof_start"]) == 16
            n += 1
    assert n > 0
    print(f"[OK] judge_samples.jsonl 스키마 확인 ({n}개 샘플, 필드 {len(required_keys)}개 전부 있음)")
    shutil.rmtree(_TMP_DIR)


if __name__ == "__main__":
    test_seed_determinism_and_no_overlap_across_sources()
    test_split_assignment_no_overlap()
    test_episode_determinism_physics_and_images()
    test_instruction_text_format()
    test_resume_skips_completed_chunks()
    test_judge_sample_schema()
    print()
    print("ALL JUDGE DATASET TESTS PASSED")
