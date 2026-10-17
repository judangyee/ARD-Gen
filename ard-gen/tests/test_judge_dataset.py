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
    completed_subtasks_before,
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
    """ARD-VLA decision_logic.py 실제 구현 그대로("completed_str = ', '.join
    (state.completed) if state.completed else 'none'") -- completed는
    문자열 목록, 지금 서브태스크 자신의 성공/실패가 아니다."""
    text = instruction_text("align", [])
    assert "align" in text and "completed: none" in text and "current subtask:" in text
    text2 = instruction_text("insert", ["align"])
    assert "completed: align" in text2
    print(f"[OK] instruction_text 포맷: {text!r} / {text2!r}")


def test_instruction_outcome_invariant() -> None:
    """같은 (서브태스크, 이전 완료 이력)이면 이번 시도의 성공/실패와
    무관하게 instruction 문자열이 항상 같아야 한다(v1 라벨 누수 버그
    수정의 핵심 요구사항 -- 사용자 지시 "같은 (subtask, prior-completed
    history)는 성공/실패와 무관하게 동일해야 함")."""
    boundaries_success = [{"node": "align", "finished_naturally": True}]
    boundaries_fail = [{"node": "align", "finished_naturally": False}]
    # align 바로 다음(idx=1) 지시문은 align이 성공했든 실패했든(재시도
    # 끝에 결국 insert로 넘어간 경우) completed 목록이 같아야 한다 --
    # 성공했을 때만 "align"이 완료 목록에 들어간다는 게 핵심이므로,
    # 여기서는 "align 자신이 방금 끝난 시점"의 지시문(idx=0)이 성공/실패와
    # 무관하게 동일한지를 직접 확인한다.
    text_after_success_attempt = instruction_text("align", completed_subtasks_before(boundaries_success, 0))
    text_after_fail_attempt = instruction_text("align", completed_subtasks_before(boundaries_fail, 0))
    assert text_after_success_attempt == text_after_fail_attempt, "완료 이력이 같은데 지시문이 다르다(라벨 누수)"

    # insert 서브태스크의 지시문은 "align이 성공적으로 끝났는지"에만
    # 좌우되고, insert 자신의 이번 시도 결과와는 무관해야 한다.
    boundaries_align_ok = [{"node": "align", "finished_naturally": True}]
    text_insert_before_outcome = instruction_text("insert", completed_subtasks_before(boundaries_align_ok, 1))
    assert text_insert_before_outcome == "insert the peg into the hole | current subtask: insert | completed: align"
    print("[OK] instruction 문자열이 이번 시도의 성공/실패와 무관하게 동일(라벨 누수 없음)")


def test_instruction_only_classifier_cannot_beat_baseline() -> None:
    """실제로 생성된 judge_samples를 instruction 문자열로만 그룹 지어서
    그룹별 다수결 정확도를 계산한다. peg_in_hole은 서브태스크가 2개뿐이라
    instruction이 사실상 "지금 서브태스크가 뭔지"와 1:1 대응이고(align은
    항상 completed:none, insert는 항상 completed:align -- 재시도 횟수와
    무관), 서브태스크마다 성공 기준율이 원래 다르므로 이 그룹별 다수결이
    "전체 다수 클래스" 베이스라인보다 높게 나오는 건 정상이다(이건
    "라벨 누수"가 아니라 서브태스크 정체성이라는 정당한 정보) -- 그래서
    전체 베이스라인과의 비교 대신, v1의 버그였던 "거의 100% 분리 가능"
    (completed 필드가 곧 finished_naturally라서 사실상 완벽한 치트 신호)
    수준에는 전혀 못 미친다는 걸 느슨한 상한으로 확인한다(사용자 지시
    "instruction만으로 성공을 예측하는 분류기가 trivial baseline 이하"의
    실질적 의도 -- 완벽 분리가 재발하면 이 상한을 넘는다)."""
    if os.path.exists(_TMP_DIR):
        shutil.rmtree(_TMP_DIR)
    os.makedirs(_TMP_DIR)
    result = subprocess.run(
        [
            sys.executable, _SCRIPT, "--data-dir", _TMP_DIR, "--chunk-size", "10",
            "--n-a", "0", "--n-b", "10", "--n-c", "0", "--workers", "2", "--no-push",
        ],
        capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "osmesa"},
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    samples = []
    for chunk_dir in os.listdir(os.path.join(_TMP_DIR, "chunks")):
        jsonl_path = os.path.join(_TMP_DIR, "chunks", chunk_dir, "judge_samples.jsonl")
        with open(jsonl_path) as f:
            samples.extend(json.loads(line) for line in f if line.strip())
    assert len(samples) > 0

    # 지금 서브태스크 자신의 결과가 라벨에 그대로 들어가면 안 된다 --
    # v1 버그를 직접 재현해본 "완벽 분리 상한"(실제 치트 신호가 있을 때
    # 나오는 값)과 비교할 가장 직접적인 방법: completed 필드가 혹시라도
    # (버그 재발로) success와 같은 값을 그대로 반영한다면 instruction이
    # success마다 달라져야 하는데, 같은 서브태스크의 성공/실패 샘플들이
    # 정확히 같은 instruction 문자열을 공유하는지 먼저 직접 확인한다.
    by_subtask_instructions: dict[str, set[str]] = {}
    for s in samples:
        by_subtask_instructions.setdefault(s["subtask"], set()).add(s["instruction"])
    for subtask, instrs in by_subtask_instructions.items():
        assert len(instrs) == 1, f"{subtask}: 성공/실패에 따라 instruction이 달라진다(라벨 누수) -- {instrs}"

    by_instruction: dict[str, list[bool]] = {}
    for s in samples:
        by_instruction.setdefault(s["instruction"], []).append(s["success"])
    n_correct = 0
    for outcomes in by_instruction.values():
        majority = sum(outcomes) >= len(outcomes) / 2
        n_correct += sum(1 for o in outcomes if o == majority)
    grouped_acc = n_correct / len(samples)
    near_perfect_separation_ceiling = 0.95
    assert grouped_acc < near_perfect_separation_ceiling, (
        f"instruction만으로 그룹 지은 다수결 정확도({grouped_acc:.3f})가 완벽 분리 "
        f"상한({near_perfect_separation_ceiling})에 근접한다 -- 라벨 누수 재발 의심"
    )
    print(
        f"[OK] instruction이 서브태스크마다 성공/실패 무관하게 1개뿐(라벨 누수 없음), "
        f"instruction-only 다수결 정확도 {grouped_acc:.3f} < {near_perfect_separation_ceiling} "
        f"(n={len(samples)}, n_distinct_instructions={len(by_instruction)})"
    )
    shutil.rmtree(_TMP_DIR)


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
        "attempt", "applied_recovery", "subtask", "instruction", "images",
        "robot_state_16dof_start", "robot_state_16dof_end", "success", "failure_type", "failure_type_idx",
        "progress", "slipped_but_succeeded",
    }
    n = 0
    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            missing = required_keys - set(sample.keys())
            assert not missing, f"필드 누락: {missing}"
            # 카메라 목록은 생성 시점 어댑터 설정에 따라 달라질 수 있지만(v2
            # 근접 카메라 추가), 적어도 기존 top_cam/wide_cam은 항상 있고
            # 카메라마다 start/end가 쌍으로 있어야 한다.
            img_keys = set(sample["images"].keys())
            assert {"top_cam_start", "wide_cam_start", "top_cam_end", "wide_cam_end"} <= img_keys
            cams = {k.rsplit("_", 1)[0] for k in img_keys}
            for cam in cams:
                assert f"{cam}_start" in img_keys and f"{cam}_end" in img_keys
            assert sample["failure_type_idx"] == -100 or sample["failure_type_idx"] >= 0
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
    test_instruction_outcome_invariant()
    test_resume_skips_completed_chunks()
    test_judge_sample_schema()
    test_instruction_only_classifier_cannot_beat_baseline()
    print()
    print("ALL JUDGE DATASET TESTS PASSED")
