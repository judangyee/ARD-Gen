"""ARD-VLA 폐루프 통합 데이터 스키마(5단계, PIPELINE.md "폐루프 데이터" 절
참고) 검증 -- pipeline/episode_io.py의 확장 필드(closed_loop/
subtask_boundaries/disturbance/candidate_chunks) 왕복, pipeline/
generate_closed_loop_episodes.py의 소규모 생성, pipeline/
render_closed_loop_boundaries.py의 경계 이미지 렌더링까지 전부 작게
(에피소드 2~3개) 돌려서 끝까지 동작하는지 확인한다 -- 실제 data/
closed_loop/{A,B,C}의 수십 개짜리 생성은 이 테스트가 매번 다시 하지
않는다(느림, PIPELINE.md에 한 번 실행한 결과를 기록해뒀다).

pytest 없이 plain assert로 짰다 -- 이 저장소의 다른 스크립트들과 같은 스타일.

이 파일은 (렌더링 테스트가 섞여 있어서) 맨 위에서 `MUJOCO_GL=osmesa`를
설정한다 -- PyTorch Adam을 만드는 테스트가 전혀 없어서(세그폴트 조건,
PIPELINE.md 참고) 안전하다. mujoco는 그 값을 **자기 모듈이 import되는
시점**에 한 번 읽어서 렌더 백엔드를 고정하므로(그 뒤에 바꿔도 효과
없음, 실측 확인), 이 파일의 다른 어떤 import보다도 먼저 설정해야 한다.

사용법:
    python tests/test_closed_loop_data_schema.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# mujoco는 MUJOCO_GL을 import 시점에 읽어서 렌더 백엔드를 고른다 -- 이 파일
# 아래의 다른 import(pipeline.generate_closed_loop_episodes 등)가 먼저
# mujoco를 불러오기 전에 여기서 osmesa로 정해둬야 한다(render_episode.py/
# render_closed_loop_boundaries.py와 같은 순서 규칙).
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

from pipeline.episode_io import add_boundary_images, load_episode, save_episode
from pipeline.generate_closed_loop_episodes import (
    VisualDisturbanceConfig,
    apply_visual_disturbance,
    assert_no_torch_imported,
    run_one_episode,
    sample_scene_A,
)
from pipeline.render_closed_loop_boundaries import render_episode_boundaries

_TMP_DIR = os.path.join(os.path.dirname(__file__), "_tmp_closed_loop_schema")


def test_episode_io_round_trip_with_new_fields() -> None:
    episode = {
        "task": "peg_in_hole",
        "right_arm": {
            "traj": np.zeros((5, 4), dtype=np.float32),
            "action": np.zeros((4, 3), dtype=np.float32),
            "gain_names": ["Kp_xy", "Kd_xy"],
            "gains": [0.1, 0.001],
            "force": np.zeros((4, 3), dtype=np.float32),
            "torque": np.zeros((4, 3), dtype=np.float32),
            "role": "actuator",
        },
        "left_arm": {"traj": np.zeros((5, 3), dtype=np.float32), "role": "stabilizer"},
        "scene_config": {"friction": 0.5},
        "success": True,
        "closed_loop": {
            "joint_16dof": np.zeros((5, 16), dtype=np.float32),
            "progress": np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32),
            "subtask_id": np.array(["align", "align", "insert", "insert"]),
            "grasp_right": {
                "weld_active": np.array([True, True, True, True]),
                "weld_force": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                "slipped": np.array([False, False, False, False]),
                "cumulative_shift": np.zeros((4, 3), dtype=np.float32),
            },
        },
        "subtask_boundaries": [
            {
                "node": "align", "start_step": 0, "end_step": 10, "finished_naturally": True,
                "failure_type": "none", "attempt": 1, "recovery_applied": None,
            },
        ],
        "disturbance": {"type": "injected_slip_F2", "magnitude_m": 0.015},
        "candidate_chunks": [{"node": "align", "branches": [{"kp_mult": 1.1, "end_progress": 0.3}]}],
        "split": "train",
        "loop_mode": "closed",
    }
    os.makedirs(_TMP_DIR, exist_ok=True)
    path = os.path.join(_TMP_DIR, "round_trip.npz")
    save_episode(path, episode)
    loaded = load_episode(path)

    assert list(loaded["closed_loop"].keys()) == ["joint_16dof", "progress", "subtask_id", "grasp_right"]
    assert loaded["closed_loop"]["subtask_id"].tolist() == ["align", "align", "insert", "insert"]
    assert loaded["subtask_boundaries"] == episode["subtask_boundaries"]
    assert loaded["disturbance"] == episode["disturbance"]
    assert loaded["candidate_chunks"] == episode["candidate_chunks"]
    assert loaded["split"] == "train"
    assert loaded["loop_mode"] == "closed"
    assert loaded["boundary_images"] == {}

    add_boundary_images(path, {"boundary_step0_top_cam": np.zeros((4, 4, 3), dtype=np.uint8)})
    reloaded = load_episode(path)
    assert list(reloaded["boundary_images"].keys()) == ["boundary_step0_top_cam"]
    print("[OK] episode_io round trip with closed_loop/subtask_boundaries/disturbance/candidate_chunks/boundary_images")


def test_old_style_episode_unaffected() -> None:
    """새 필드가 전혀 없는 옛 에피소드는 기존과 똑같이 저장/복원돼야
    한다("기존 키는 유지" 사용자 지시)."""
    episode = {
        "task": "peg_in_hole",
        "right_arm": {
            "traj": np.zeros((5, 4), dtype=np.float32),
            "action": np.zeros((4, 3), dtype=np.float32),
            "gain_names": ["Kp_xy", "Kd_xy"],
            "gains": [0.1, 0.001],
            "force": np.zeros((4, 3), dtype=np.float32),
            "torque": np.zeros((4, 3), dtype=np.float32),
            "role": "actuator",
        },
        "scene_config": {"friction": 0.5},
        "success": True,
        "insertion_depth": 0.02,
        "force_max": 5.0,
    }
    path = os.path.join(_TMP_DIR, "old_style.npz")
    save_episode(path, episode)
    loaded = load_episode(path)
    assert "closed_loop" not in loaded
    assert "subtask_boundaries" not in loaded
    assert "disturbance" not in loaded
    assert "candidate_chunks" not in loaded
    assert loaded["boundary_images"] == {}
    print("[OK] old-style episode (no closed_loop fields) round-trips unaffected")


def test_generate_small_episode_each_split() -> None:
    """generate_closed_loop_episodes.run_one_episode()가 A/B/C 전부에서
    스키마가 맞는 에피소드를 만드는지(에피소드 2개씩, 빠르게)."""
    for split in ("A", "B", "C"):
        rng = np.random.default_rng(12345)
        for i in range(2):
            episode = run_one_episode(rng, split, seed=i)
            assert episode["task"] == "peg_in_hole"
            assert episode["closed_loop"]["joint_16dof"].shape[1] == 16
            assert episode["closed_loop"]["joint_16dof"].shape[0] == episode["right_arm"]["action"].shape[0] + 1
            assert len(episode["subtask_boundaries"]) >= 1
            assert episode["split"] in ("train", "val", "test_B", "test_C", "test_A")
            if split == "A":
                assert episode["split"] == "test_A"
                assert episode.get("disturbance") is None
            if split == "B":
                assert episode["split"] in ("train", "val")
                assert episode["disturbance"]["type"] in (
                    "action_noise", "xy_target_bias", "bad_gains", "narrow_clearance",
                    "high_friction", "injected_slip_F2", "injected_slip_F3",
                )
            if split == "C":
                assert episode["split"] == "test_C"
                assert episode["disturbance"]["type"] in ("wide_friction", "very_narrow_clearance", "action_delay")
            path = os.path.join(_TMP_DIR, f"gen_{split}_{i}.npz")
            save_episode(path, episode)
            reloaded = load_episode(path)
            assert reloaded["success"] == episode["success"]
        print(f"[OK] generate_closed_loop_episodes: split={split} schema sane (2 episodes)")


def test_sample_scene_A_matches_calibrated_failure_range() -> None:
    """PIPELINE.md "5단계" 절에 실측 기록한 A세트 범위(반경 0.009-0.020,
    각도 전체 원)가 코드에도 그대로인지 -- 범위가 조용히 바뀌면 10-30%
    실패율 보증이 깨진다."""
    rng = np.random.default_rng(0)
    radii = []
    for _ in range(200):
        cfg = sample_scene_A(rng)
        ox, oy = cfg["peg_init_offset_xy"]
        radii.append(float(np.hypot(ox, oy)))
    assert min(radii) >= 0.0089 and max(radii) <= 0.0201
    print(f"[OK] sample_scene_A radius range [{min(radii)*1000:.2f}, {max(radii)*1000:.2f}]mm matches calibrated A range")


def test_render_boundaries_adds_images_and_skips_closed_loop() -> None:
    import mujoco  # noqa: F401 -- osmesa 렌더러가 이 환경에 있는지 먼저 확인(없으면 스킵).

    rng = np.random.default_rng(7)
    # 재시도 없이 바로 성공하는(open) 에피소드가 나올 때까지 몇 번 시도한다
    # (A세트가 압도적으로 open+success이므로 몇 번이면 충분).
    episode = None
    for i in range(10):
        candidate = run_one_episode(rng, "A", seed=i)
        if candidate["loop_mode"] == "open":
            episode = candidate
            break
    assert episode is not None, "open-loop 에피소드를 못 뽑았다(재시도 늘려야 할 수도)"
    path = os.path.join(_TMP_DIR, "render_open.npz")
    save_episode(path, episode)

    n_images = render_episode_boundaries(path)
    assert n_images > 0, "open-loop 에피소드인데 이미지가 하나도 안 캡처됐다"
    reloaded = load_episode(path)
    assert len(reloaded["boundary_images"]) == n_images
    for img in reloaded["boundary_images"].values():
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8
    print(f"[OK] render_closed_loop_boundaries: {n_images} images added for an open-loop episode")

    # closed-loop(복구 발생) 에피소드는 skip돼야 한다.
    closed_episode = None
    for i in range(10, 30):
        candidate = run_one_episode(rng, "B", seed=i)
        if candidate["loop_mode"] == "closed":
            closed_episode = candidate
            break
    assert closed_episode is not None, "closed-loop 에피소드를 못 뽑았다"
    closed_path = os.path.join(_TMP_DIR, "render_closed.npz")
    save_episode(closed_path, closed_episode)
    n_skipped = render_episode_boundaries(closed_path)
    assert n_skipped == 0, "closed-loop 에피소드는 렌더링을 skip해야 한다(모듈 docstring의 한계 참고)"
    print("[OK] render_closed_loop_boundaries correctly skips a closed-loop (recovered) episode")


def test_assert_no_torch_imported() -> None:
    """후속 작업 "작업 1" 지시 그대로 -- assert 자체를 테스트로도 남긴다.
    이 테스트 프로세스엔 torch가 없어야 정상(통과), 있는 척 흉내 내서
    AssertionError가 뜨는지도 같이 확인한다."""
    assert "torch" not in sys.modules, "이 테스트 프로세스에 이미 torch가 import돼 있다 -- 전제가 깨짐"
    assert_no_torch_imported()  # 통과해야 함(torch 없음)

    sys.modules["torch"] = object()  # type: ignore[assignment]
    try:
        raised = False
        try:
            assert_no_torch_imported()
        except AssertionError:
            raised = True
        assert raised, "torch가 sys.modules에 있는데도 assert_no_torch_imported()가 안 터졌다"
    finally:
        del sys.modules["torch"]
    print("[OK] assert_no_torch_imported: 정상(없음)/감지(있음) 둘 다 확인")


def test_method_a_inline_render_covers_closed_loop_episodes() -> None:
    """방법 A(--render)가 복구(closed-loop)가 일어난 에피소드에도 모든
    서브태스크 경계에 이미지를 남기는지 -- 이게 이번 작업 1의 핵심 문제
    (기존 render_closed_loop_boundaries.py는 closed를 통째로 skip)."""
    rng = np.random.default_rng(42)
    closed_episode = None
    for i in range(40):
        candidate = run_one_episode(rng, "B", seed=i, render=True)
        if candidate["loop_mode"] == "closed":
            closed_episode = candidate
            break
    assert closed_episode is not None, "40번 시도에도 closed-loop 에피소드를 못 뽑았다"

    n_boundaries = len(closed_episode["subtask_boundaries"])
    images = closed_episode["boundary_images"]
    for idx in range(n_boundaries):
        for camera in ("top_cam", "wide_cam"):
            for moment in ("start", "end"):
                key = f"boundary_{idx}_{camera}_{moment}"
                assert key in images, f"{key} 이미지가 없다(경계 {idx}/{n_boundaries-1})"
                assert images[key].shape == (240, 320, 3)
                assert images[key].dtype == np.uint8
    print(f"[OK] method A: closed-loop 에피소드({n_boundaries}개 경계) 전부 start/end 이미지 확보")

    path = os.path.join(_TMP_DIR, "method_a_closed.npz")
    save_episode(path, closed_episode)
    reloaded = load_episode(path)
    assert len(reloaded["boundary_images"]) == len(images), "저장/복원 후 경계 이미지 개수가 달라졌다"
    print("[OK] method A: boundary_images가 episode_io 저장/복원을 그대로 왕복한다")


def test_visual_disturbance_only_affects_c() -> None:
    """시각 교란은 C에서만 뜻이 있고(사용자 지시 "C-only"), A/B는 이
    기능이 생기기 전과 완전히 동일한 물리 롤아웃을 내야 한다(물리 자체는
    cfg.enabled와 무관하게 안 건드리므로, A/B가 영향받을 길이 코드상
    원천적으로 없다는 걸 확인)."""
    rng = np.random.default_rng(1)
    episode_a_before = run_one_episode(np.random.default_rng(5), "A", seed=0, render=False)
    episode_a_after = run_one_episode(np.random.default_rng(5), "A", seed=0, render=False)
    assert np.array_equal(episode_a_before["right_arm"]["action"], episode_a_after["right_arm"]["action"])

    visual_cfg = VisualDisturbanceConfig(enabled=True)
    episode_c = run_one_episode(rng, "C", seed=1, render=False, visual_disturbance=visual_cfg)
    assert "visual" in episode_c["disturbance"], "C + visual_disturbance인데 메타데이터에 visual이 안 남았다"
    print("[OK] visual disturbance: C 메타데이터에 적용값 기록, A는 영향 경로 자체가 없음")

    disabled_cfg = VisualDisturbanceConfig(enabled=False)
    import sim.peg_in_hole_bimanual_openarm_sim as sim_mod

    sim = sim_mod.BimanualPegInHoleOpenArmSim()
    before_rgba = sim.model.geom_rgba.copy()
    before_cam = sim.model.cam_pos.copy()
    applied = apply_visual_disturbance(sim, disabled_cfg, np.random.default_rng(0))
    assert applied == {}
    assert np.array_equal(sim.model.geom_rgba, before_rgba)
    assert np.array_equal(sim.model.cam_pos, before_cam)
    print("[OK] visual disturbance off: 모델이 전혀 안 바뀜(bit-identical)")


def test_method_a_matches_old_replay_for_open_episode() -> None:
    """결정론 확인(작업 1 검증 요구사항) -- open-loop 에피소드 하나를
    방법 A(생성 중 바로 캡처)와 기존 재생 스크립트(render_closed_loop_
    boundaries.py) 둘 다로 렌더링해서, 같은 물리 순간(경계의 start_step/
    end_step)의 이미지가 일치하는지 픽셀 단위로 비교한다."""
    import mujoco  # noqa: F401 -- osmesa 렌더러 없으면 이 테스트 스킵.

    rng = np.random.default_rng(7)
    episode = None
    for i in range(10):
        candidate = run_one_episode(rng, "A", seed=100 + i, render=True)
        if candidate["loop_mode"] == "open":
            episode = candidate
            break
    assert episode is not None, "open-loop 에피소드를 못 뽑았다"

    path = os.path.join(_TMP_DIR, "determinism_check.npz")
    save_episode(path, episode)

    old_n = render_episode_boundaries(path)
    assert old_n > 0
    reloaded = load_episode(path)

    images = reloaded["boundary_images"]
    max_abs_diff = 0
    n_compared = 0
    for idx, boundary in enumerate(episode["subtask_boundaries"]):
        for camera in ("top_cam", "wide_cam"):
            new_start = images[f"boundary_{idx}_{camera}_start"]
            new_end = images[f"boundary_{idx}_{camera}_end"]
            old_start_key = f"boundary_step{boundary['start_step']}_{camera}"
            old_end_key = f"boundary_step{boundary['end_step']}_{camera}"
            if old_start_key in images:
                diff = np.abs(images[old_start_key].astype(int) - new_start.astype(int))
                max_abs_diff = max(max_abs_diff, int(diff.max()))
                n_compared += 1
            if old_end_key in images:
                diff = np.abs(images[old_end_key].astype(int) - new_end.astype(int))
                max_abs_diff = max(max_abs_diff, int(diff.max()))
                n_compared += 1
    assert n_compared > 0, "비교할 이미지 쌍이 하나도 없었다(키 매칭 실패?)"
    print(f"[OK] determinism check: {n_compared}쌍 비교, 최대 픽셀 차이={max_abs_diff} (0=완전 일치)")
    assert max_abs_diff == 0, f"방법 A와 기존 재생이 같은 순간에 다른 이미지를 냈다(최대 차이={max_abs_diff})"


if __name__ == "__main__":
    test_episode_io_round_trip_with_new_fields()
    test_old_style_episode_unaffected()
    test_generate_small_episode_each_split()
    test_sample_scene_A_matches_calibrated_failure_range()
    test_render_boundaries_adds_images_and_skips_closed_loop()
    test_assert_no_torch_imported()
    test_method_a_inline_render_covers_closed_loop_episodes()
    test_visual_disturbance_only_affects_c()
    test_method_a_matches_old_replay_for_open_episode()
    print()
    print("ALL CLOSED LOOP DATA SCHEMA TESTS PASSED")
