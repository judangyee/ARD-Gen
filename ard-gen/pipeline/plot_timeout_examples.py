"""후속 작업("timeout 진단") 작업1 산출물 4/5 -- 유형별 대표 사례의
시계열 그림(png)과 판정 시점(top_cam) 이미지를 뽑는다. 결과물은
analysis/timeout_diagnosis/ 아래에 저장(이 리포에 커밋해서 PIPELINE.md
에서 상대경로로 참조). 일회성 분석 스크립트라 재실행 편의보다 결과
재현성을 우선했다(고정 시드).

사용법 (osmesa 필요 -- top_cam 캡처 때문에):
    MUJOCO_GL=osmesa python pipeline/plot_timeout_examples.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipeline.diagnose_timeout_boundaries as diag
from closedloop.peg_in_hole_adapter import PegInHoleClosedLoopEnv, ScriptedOraclePolicy
from pipeline.generate_closed_loop_episodes import _NOMINAL_GAINS, _load_graph, load_disturbance_config
from sim.grasp_slip import ArmSlipConfig, GraspSlipConfig
from sim.peg_in_hole_bimanual_openarm_sim import BimanualPegInHoleOpenArmSim

_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "analysis", "timeout_diagnosis")

# 유형 이름(한글)을 그림 제목/파일명에 쓸 영문 라벨로 바꾼다 -- 이
# 컨테이너에 한글 폰트가 없어서(fc-list 확인) matplotlib이 글리프를
# 못 그린다(실측: missing glyph 경고). 코드/문서는 한글 그대로 두고
# 이 그림에서만 영문 라벨을 쓴다.
_LABEL_EN = {
    "거의성공(insert, hold 직전 예산소진)": "near_success",
    "걸림(jam)": "jam",
    "정렬_미수렴(insert 중 재정렬 실패)": "misaligned",
    "진동_발산": "oscillating",
}


def _find_examples() -> dict[str, dict]:
    """각 유형(걸림/정렬_미수렴/진동_발산/거의성공) 대표 사례 하나씩을
    찾는다(diagnose_timeout_boundaries.py와 같은 시드/순서라 재현됨)."""
    rng = np.random.default_rng(31415)
    disturbance_cfg = load_disturbance_config()
    found: dict[str, dict] = {}
    wanted = {"걸림(jam)", "정렬_미수렴(insert 중 재정렬 실패)", "진동_발산", "거의성공(insert, hold 직전 예산소진)"}
    for split, n in {"A": 100, "B": 150, "C": 150}.items():
        for i in range(n):
            ep = diag._run_episode_instrumented(rng, split, seed=i, disturbance_config=disturbance_cfg if split == "B" else None)
            for b in ep["subtask_boundaries"]:
                if b["failure_type"] != "timeout":
                    continue
                feats = diag.compute_features(ep["diag"], b["start_step"], b["end_step"], b["node"], ep["outer_half"], ep["target_depth"])
                category = diag.classify_timeout(feats, ep["outer_half"], ep["target_depth"])
                if category in wanted and category not in found:
                    found[category] = {
                        "split": split, "seed": i, "boundary": b, "scene_config": ep["scene_config"],
                        "gains": ep["gains"], "disturbance": ep["disturbance"],
                    }
            if wanted <= set(found):
                return found
    return found


def _rerun_with_render(example: dict) -> tuple[np.ndarray, np.ndarray]:
    """같은 씬/교란/시드로 다시 돌려서(결정론) 시계열 전체(diag) + 경계
    끝 시점 top_cam 이미지를 얻는다."""
    disturbance = example["disturbance"]
    grasp_slip_config = None
    if disturbance and disturbance.get("type") in ("injected_slip_F2", "injected_slip_F3"):
        offset = np.array(disturbance["offset_m"])
        arm_cfg = ArmSlipConfig(mode="injected", injected_at_step=disturbance["step"], injected_offset_local=offset)
        grasp_slip_config = GraspSlipConfig(right=arm_cfg) if disturbance["side"] == "right" else GraspSlipConfig(left=arm_cfg)

    sim = BimanualPegInHoleOpenArmSim(grasp_slip_config=grasp_slip_config)
    env = PegInHoleClosedLoopEnv(
        sim, _load_graph(), example["scene_config"], example["gains"], render=True, legacy_failure_rule=True
    )
    env.reset_episode("align")
    policy = ScriptedOraclePolicy(env)

    b = example["boundary"]
    target_steps = b["end_step"]
    diag_hist = {"dx": [], "dy": [], "raw_depth": []}
    total_steps = 0
    image = None
    while total_steps < target_steps:
        actions, _ = policy.predict_action_chunk_with_decisions(None)
        env.execute(actions[0, 0])
        total_steps += 1
        dx, dy = env.current_dx_dy()
        hole_center = sim.get_hole_center_pos()
        peg_tip = sim.get_peg_tip_pos()
        diag_hist["dx"].append(dx)
        diag_hist["dy"].append(dy)
        diag_hist["raw_depth"].append(float(hole_center[2] - peg_tip[2]))
        if env.current_node != b["node"]:
            # align이 먼저 끝나고 insert로 넘어간 경우(목표가 insert 경계일 때) -- set_subtask 호출.
            if env.subtask_finished():
                env.set_subtask("insert")
    image = env.render_frames()["top_cam"]
    return np.asarray(diag_hist["dx"][b["start_step"]:b["end_step"]]), np.asarray(diag_hist["raw_depth"][b["start_step"]:b["end_step"]]), image


def main() -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)
    examples = _find_examples()
    print(f"찾은 대표 사례: {list(examples.keys())}")

    fig, axes = plt.subplots(len(examples), 1, figsize=(8, 3 * len(examples)))
    if len(examples) == 1:
        axes = [axes]
    for ax, (category, ex) in zip(axes, examples.items()):
        b = ex["boundary"]
        label_en = _LABEL_EN.get(category, category)
        dx, raw_depth, image = _rerun_with_render(ex)
        if b["node"] == "align":
            ax.plot(np.hypot(dx, 0) * 1000, label="|dx| (mm)")
            ax.set_ylabel("xy err proxy (mm)")
        else:
            ax.plot(raw_depth * 1000, label="raw_depth (mm)")
            ax.set_ylabel("depth (mm)")
        ax.set_title(
            f"{label_en} ({ex['split']} seed={ex['seed']} node={b['node']} "
            f"disturbance={(ex['disturbance'] or {}).get('type')})"
        )
        ax.set_xlabel("step (within boundary)")
        ax.legend()
        img_path = os.path.join(_OUT_DIR, f"{label_en}_boundary_end.png")
        plt.imsave(img_path, image)
        print(f"[OK] {category}: 시계열 {len(dx)}스텝, 판정시점 이미지 -> {img_path}")

    fig.tight_layout()
    fig_path = os.path.join(_OUT_DIR, "timeout_category_timeseries.png")
    fig.savefig(fig_path, dpi=100)
    print(f"[OK] 시계열 그림 -> {fig_path}")


if __name__ == "__main__":
    main()
