"""생성한 LeRobot v3.0 데이터셋(pipeline/to_lerobot_v3.py 산출물, 또는 같은
스키마를 따르는 어떤 데이터셋이든)을 RoCo 공식 시뮬 데이터 통계와
비교해서 통과/실패를 출력한다(사용자 지시 "허용 오차를 명시한 검사로
구현").

## 왜 `lerobot` 패키지 없이 pyarrow만으로 읽는가

이 스크립트가 쓰는 값(observation.state/action/episode_index/
frame_index)은 전부 `data/chunk-*/file-*.parquet`에 평범한 숫자 컬럼으로
있다(비디오 프레임 디코딩이 전혀 필요 없음, pipeline/to_lerobot_v3.py
작성 중 실제 parquet 스키마를 pyarrow로 직접 확인함). `lerobot.datasets.
lerobot_dataset.LeRobotDataset`로 읽으면 `__getitem__`이 비디오까지
디코딩하려 들어서 torch/torchvision/torchcodec 전부 깔려 있어야 하고
훨씬 느리다 -- 검증기는 숫자만 보면 되므로 pyarrow로 직접 읽는 게 더
가볍고 빠르고, lerobot이 아예 없는 환경에서도(pyarrow만 있으면) 쓸 수
있다.

## 공식 RoCo 통계(사용자 지시 그대로, 출처는 사용자가 제공한 수치)

- 200 에피소드, 총 121,454프레임(평균 약 607.27프레임 = 10fps 기준 약
  60.7초)
- 왼팔 EE 회전(action의 left_rx/ry/rz, 연속 Euler): rx 0.48~4.10,
  ry -1.26~1.53, rz -2.95~7.00 -- 이 범위 "안"에 있는가와 별개로, 그
  중 실제로 몰려 있는 "중심 구간"(p1~p99 격)은 rx 2.75~3.15, ry 0~0.83,
  rz -0.001~0.93 (check_left_rotation_distribution, 사용자 지시 2/3-b).
- 왼팔 EE 위치(observation.state의 left_ee_x/y/z): x -0.01~0.82,
  y -0.14~0.76, z 1.07~1.80(roco_interface.ROCO_OFFICIAL_LEFT_EE_RANGE와
  동일, 사용자 지시 3-a).
- 왼팔 그리퍼("열림 비율"): 0.060~0.301
- 오른팔은 사실상 정지: 위치 변화 y 4cm·z 3cm 이내, 그리퍼 약 0.2256
  고정, 회전 rx 약 -pi
- 연속 프레임 간 각도 점프 0건(에피소드 경계 안에서)
- 에피소드 초반 약 1초(10프레임, 10fps 기준)는 초기 자세 유지 후 첫
  웨이포인트로 이동

## 이 저장소의 peg_in_hole(OpenArm) 데모로 돌리면 대부분 FAIL이 정상

sim/roco_interface.py 모듈 docstring/pipeline/to_lerobot_v3.py 모듈
docstring에 이미 기록했듯, peg_in_hole은 오른팔=Actuator(움직임)/
왼팔=Stabilizer(고정)라서 공식 관례(왼팔=액티브)와 정반대다 -- 이
스크립트로 그 데모 데이터를 검사하면 "왼팔 정지/오른팔 액티브"로
나와서 C/D 항목이 FAIL하는 게 **정상**이다(검증기 버그가 아니라 두
관례가 다르다는 사실 자체, "태스크 자체는 만들지 않는다"는 사용자
지시와 일치). 이 스크립트 자체가 올바른지는 tests/test_roco_interface.py
의 합성(synthetic) 데이터 검사로 따로 확인한다.

사용 예:
    python scripts/validate_against_roco.py --dataset-root ./data/lerobot_v3/demo
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pyarrow.parquet as pq

import sim.roco_interface as roco

# ---------------------------------------------------------------------------
# 공식 RoCo 통계 + 허용 오차 (사용자 지시 수치 그대로, 오차는 이 스크립트가
# 명시적으로 고른 값 -- "데모 규모(에피소드 몇 개)로도 돌 수 있어야 한다"는
# 실용적 이유로 느슨하게 잡은 것도 있고("에피소드 길이 평균" 등, 작은
# 샘플에서는 분산이 크다), "넘으면 바로 물리적으로 이상하다"는 이유로
# 빡빡하게 잡은 것도 있다(각도 점프, 왼팔 그리퍼 범위) -- 항목마다 아래
# 주석에 근거를 남긴다.
# ---------------------------------------------------------------------------

FPS = roco.CAMERA_FPS  # 10

OFFICIAL_N_EPISODES = 200
OFFICIAL_TOTAL_FRAMES = 121_454
OFFICIAL_AVG_FRAMES = OFFICIAL_TOTAL_FRAMES / OFFICIAL_N_EPISODES  # ~607.27
OFFICIAL_AVG_DURATION_S = OFFICIAL_AVG_FRAMES / FPS  # ~60.73

OFFICIAL_LEFT_RX_RANGE = (0.48, 4.10)
OFFICIAL_LEFT_RY_RANGE = (-1.26, 1.53)
OFFICIAL_LEFT_RZ_RANGE = (-2.95, 7.00)
OFFICIAL_LEFT_GRIPPER_RANGE = (0.060, 0.301)

# 왼팔 EE 위치 범위(사용자 지시 3-a, roco_interface.ROCO_OFFICIAL_LEFT_EE_RANGE
# 와 동일한 공식 수치 -- 그 상수를 그대로 재사용해서 두 곳에서 따로
# 어긋날 일이 없게 한다).
OFFICIAL_LEFT_POS_RANGE = {
    "x": roco.ROCO_OFFICIAL_LEFT_EE_RANGE["x"],
    "y": roco.ROCO_OFFICIAL_LEFT_EE_RANGE["y"],
    "z": roco.ROCO_OFFICIAL_LEFT_EE_RANGE["z"],
}

# 왼팔 회전의 "중심 구간"(사용자 지시 2/3-b, p1~p99에 해당하는 수치로
# 제공됨) -- 전체 범위(OFFICIAL_LEFT_R*_RANGE)보다 훨씬 좁다. 분포 검사
# (check_left_rotation_distribution)는 "관측 프레임 중 몇 %가 이 구간
# 안에 드는가"를 본다 -- 범위 검사(바깥쪽으로 안 벗어나는가)와는 다른
# 질문(안쪽 중심에 실제로 몰려 있는가)이라 둘 다 둔다.
OFFICIAL_LEFT_RX_P1P99 = (2.75, 3.15)
OFFICIAL_LEFT_RY_P1P99 = (0.0, 0.83)
OFFICIAL_LEFT_RZ_P1P99 = (-0.001, 0.93)
DISTRIBUTION_MIN_FRACTION = 0.60  # 기본 기준(사용자 지시 3-b) -- 조정 가능

OFFICIAL_RIGHT_POS_CHANGE_MAX_Y = 0.04
OFFICIAL_RIGHT_POS_CHANGE_MAX_Z = 0.03
OFFICIAL_RIGHT_GRIPPER_FIXED = 0.2256
OFFICIAL_RIGHT_RX_FIXED = -np.pi

# 평균 에피소드 길이는 "비슷한 규모"를 요구하는 게 아니라 "같은 자릿수
# (수백 프레임)인가"를 보는 느슨한 체크다 -- 공식 200개 평균 607프레임
# 자체도 꽤 퍼져 있을 테니, ±40%를 기준으로 삼는다(이 수치의 엄밀한
# 근거는 없다 -- 정직하게 기록, 더 타이트하게 하려면 공식 분산/표준편차가
# 필요한데 사용자 지시에 없었다).
TOL_AVG_FRAMES_RATIO = 0.40

# 범위 체크(왼팔 rx/ry/rz, 왼팔 그리퍼): "관측된 [min,max]가 공식 범위
# 안에 들어오는가"가 아니라 "공식 범위를 완전히 벗어나지 않고 상당히
# 겹치는가"를 본다 -- 작은 데모 에피소드 몇 개로는 공식 범위 전체를
# 못 채울 게 당연하므로(관측 범위가 공식 범위의 **부분집합**이면 통과),
# 반대로 공식 범위를 song벗어나는 값이 나오면(좌표계/단위 버그를 의미)
# 실패해야 한다 -- 공식 범위의 5%만큼 바깥쪽 여유를 준다.
RANGE_PAD_RATIO = 0.05

# 오른팔 "정지" 체크는 공식 수치(4cm/3cm/고정 그리퍼/rx=-pi)에 작은 여유만
# 준다 -- 느슨하게 잡을 이유가 없다(정의 자체가 이미 "거의 안 움직인다"는
# 관용이 들어가 있는 수치라서).
TOL_RIGHT_POS_CHANGE_RATIO = 0.25  # 4cm -> 5cm까지, 3cm -> 3.75cm까지 허용
TOL_RIGHT_GRIPPER_ABS = 0.02
TOL_RIGHT_RX_ABS = 0.15  # rad(~8.6도)

# 각도 연속성: 사용자 지시 "연속 프레임 간 π 이상 점프 0건" 그대로 --
# 허용 오차 없음(엄격).
MAX_ALLOWED_FRAME_JUMP = np.pi

# 초반 유지 구간: 10fps 기준 10프레임(~1초) 동안 왼팔 EE가 거의 안
# 움직이다가, 그 이후 어느 시점에는 뚜렷이 움직여야 한다.
HOLD_FRAMES = round(1.0 * FPS)
HOLD_POS_TOL_M = 0.005  # 5mm -- 노이즈/수치오차 감안
MOVE_AFTER_HOLD_MIN_M = 0.01  # 1cm -- "첫 웨이포인트로 이동"이 실제로 있었는지


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    tolerance_note: str


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, tolerance_note: str) -> None:
        self.checks.append(CheckResult(name, passed, detail, tolerance_note))

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def print_report(self) -> None:
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            print(f"[{status}] {c.name}")
            print(f"       {c.detail}")
            print(f"       (허용 오차: {c.tolerance_note})")
        n_pass = sum(c.passed for c in self.checks)
        print(f"\n{n_pass}/{len(self.checks)} 항목 통과")


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------


def load_dataset_table(dataset_root: str):
    """data/chunk-*/file-*.parquet 전부를 하나로 이어붙인다."""
    paths = sorted(glob.glob(os.path.join(dataset_root, "data", "chunk-*", "file-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"{dataset_root}/data/chunk-*/file-*.parquet 없음 -- LeRobot v3.0 데이터셋이 맞는지 확인")
    tables = [pq.read_table(p) for p in paths]
    import pyarrow as pa

    return pa.concat_tables(tables)


def _col_index(name: str) -> int:
    return roco.ACTION_NAMES.index(name)


def _state_col_index(name: str) -> int:
    return roco.STATE_NAMES.index(name)


# ---------------------------------------------------------------------------
# 개별 검사
# ---------------------------------------------------------------------------


def check_episode_length(table, report: ValidationReport) -> None:
    episode_index = table.column("episode_index").to_numpy()
    n_episodes = len(set(episode_index.tolist()))
    total_frames = len(episode_index)
    avg_frames = total_frames / n_episodes if n_episodes else 0.0
    avg_duration = avg_frames / FPS

    lo = OFFICIAL_AVG_FRAMES * (1 - TOL_AVG_FRAMES_RATIO)
    hi = OFFICIAL_AVG_FRAMES * (1 + TOL_AVG_FRAMES_RATIO)
    passed = lo <= avg_frames <= hi
    report.add(
        "에피소드 수 / 총 프레임 / 평균 길이",
        passed,
        f"관측: {n_episodes}개 에피소드, 총 {total_frames}프레임, 평균 {avg_frames:.1f}프레임(~{avg_duration:.1f}초) "
        f"| 공식: {OFFICIAL_N_EPISODES}개, 총 {OFFICIAL_TOTAL_FRAMES}프레임, 평균 {OFFICIAL_AVG_FRAMES:.1f}프레임"
        f"(~{OFFICIAL_AVG_DURATION_S:.1f}초)",
        f"평균 프레임 수가 공식 평균의 ±{TOL_AVG_FRAMES_RATIO:.0%} 안({lo:.1f}~{hi:.1f})",
    )


def _range_check(report: ValidationReport, name: str, observed_min: float, observed_max: float, official_range: tuple[float, float]) -> None:
    """이름에 "공식 범위 안인가"를 명시한다(사용자 지시 3-c) -- 이 검사가
    "분포가 중심에 몰려 있는가"(check_left_rotation_distribution, 별도)와
    다른 질문이라는 걸 리포트에서도 구분되게 하려고."""
    lo, hi = official_range
    pad = (hi - lo) * RANGE_PAD_RATIO
    passed = observed_min >= lo - pad and observed_max <= hi + pad
    report.add(
        f"{name}(공식 범위 안인가)",
        passed,
        f"관측: [{observed_min:.4f}, {observed_max:.4f}] | 공식: [{lo:.4f}, {hi:.4f}]",
        f"공식 범위에서 양쪽으로 {RANGE_PAD_RATIO:.0%}({pad:.4f}) 이상 벗어나지 않음",
    )


def check_left_rotation_ranges(action: np.ndarray, report: ValidationReport) -> None:
    for name, official_range in (
        ("left_rx", OFFICIAL_LEFT_RX_RANGE),
        ("left_ry", OFFICIAL_LEFT_RY_RANGE),
        ("left_rz", OFFICIAL_LEFT_RZ_RANGE),
    ):
        col = action[:, _col_index(name)]
        _range_check(report, f"왼팔 {name} 범위", float(col.min()), float(col.max()), official_range)


def check_left_gripper_range(action: np.ndarray, report: ValidationReport) -> None:
    col = action[:, _col_index("left_gripper")]
    _range_check(report, "왼팔 그리퍼 범위", float(col.min()), float(col.max()), OFFICIAL_LEFT_GRIPPER_RANGE)


def check_left_position_ranges(state: np.ndarray, report: ValidationReport) -> None:
    """사용자 지시 3-a: 왼팔 EE 위치(x/y/z)가 공식 범위 안인가."""
    for axis in ("x", "y", "z"):
        col = state[:, _state_col_index(f"left_ee_{axis}")]
        _range_check(report, f"왼팔 위치 {axis} 범위", float(col.min()), float(col.max()), OFFICIAL_LEFT_POS_RANGE[axis])


def check_left_rotation_distribution(action: np.ndarray, report: ValidationReport) -> None:
    """사용자 지시 3-b: 범위 검사와는 별개로, 왼팔 rx/ry/rz 값이 공식
    데이터의 "중심 구간"(p1~p99에 해당, 모듈 상단 상수 참고) 안에 실제로
    몰려 있는지를 프레임 비율로 본다 -- 전체 범위를 안 벗어나는 것과
    "분포 모양이 비슷한가"는 다른 질문이라 분리했다. 기본 기준은 60%
    (사용자 지시, DISTRIBUTION_MIN_FRACTION 상수로 조정 가능)."""
    for name, p1p99 in (
        ("left_rx", OFFICIAL_LEFT_RX_P1P99),
        ("left_ry", OFFICIAL_LEFT_RY_P1P99),
        ("left_rz", OFFICIAL_LEFT_RZ_P1P99),
    ):
        col = action[:, _col_index(name)]
        lo, hi = p1p99
        frac = float(np.mean((col >= lo) & (col <= hi)))
        passed = frac >= DISTRIBUTION_MIN_FRACTION
        report.add(
            f"{name} 분포(공식 중심구간 p1~p99 안 비율)",
            passed,
            f"관측: {frac*100:.1f}%의 프레임이 공식 중심구간 [{lo:.4f}, {hi:.4f}] 안에 있음 "
            f"(관측 범위 [{float(col.min()):.4f}, {float(col.max()):.4f}])",
            f"공식 중심구간 안 비율 >= {DISTRIBUTION_MIN_FRACTION:.0%}(초기 기준, 조정 가능)",
        )


def check_right_arm_static(action: np.ndarray, state: np.ndarray, episode_index: np.ndarray, report: ValidationReport) -> None:
    # 위치 변화: 에피소드별로 y/z range를 재고, 그 중 최댓값을 공식 한도와 비교.
    right_x_i, right_y_i, right_z_i = (_state_col_index(f"right_ee_{a}") for a in ("x", "y", "z"))
    max_y_range = 0.0
    max_z_range = 0.0
    for ep in sorted(set(episode_index.tolist())):
        mask = episode_index == ep
        y = state[mask, right_y_i]
        z = state[mask, right_z_i]
        max_y_range = max(max_y_range, float(y.max() - y.min()))
        max_z_range = max(max_z_range, float(z.max() - z.min()))

    y_limit = OFFICIAL_RIGHT_POS_CHANGE_MAX_Y * (1 + TOL_RIGHT_POS_CHANGE_RATIO)
    z_limit = OFFICIAL_RIGHT_POS_CHANGE_MAX_Z * (1 + TOL_RIGHT_POS_CHANGE_RATIO)
    passed = max_y_range <= y_limit and max_z_range <= z_limit
    report.add(
        "오른팔 위치 변화(정지 상태)",
        passed,
        f"관측: 에피소드별 최대 y 변화 {max_y_range*1000:.1f}mm, z 변화 {max_z_range*1000:.1f}mm | "
        f"공식 한도: y {OFFICIAL_RIGHT_POS_CHANGE_MAX_Y*1000:.0f}mm, z {OFFICIAL_RIGHT_POS_CHANGE_MAX_Z*1000:.0f}mm",
        f"공식 한도의 {TOL_RIGHT_POS_CHANGE_RATIO:.0%}까지 추가 허용(y<={y_limit*1000:.1f}mm, z<={z_limit*1000:.1f}mm)",
    )

    right_gripper = action[:, _col_index("right_gripper")]
    grip_dev = float(np.max(np.abs(right_gripper - OFFICIAL_RIGHT_GRIPPER_FIXED)))
    passed = grip_dev <= TOL_RIGHT_GRIPPER_ABS
    report.add(
        "오른팔 그리퍼 고정",
        passed,
        f"관측: 공식값({OFFICIAL_RIGHT_GRIPPER_FIXED})과의 최대 편차 {grip_dev:.4f} "
        f"(관측 범위 [{right_gripper.min():.4f}, {right_gripper.max():.4f}])",
        f"공식값 ±{TOL_RIGHT_GRIPPER_ABS}",
    )

    right_rx = action[:, _col_index("right_rx")]
    rx_dev = float(np.max(np.abs(right_rx - OFFICIAL_RIGHT_RX_FIXED)))
    passed = rx_dev <= TOL_RIGHT_RX_ABS
    report.add(
        "오른팔 회전(rx) 고정",
        passed,
        f"관측: -pi와의 최대 편차 {rx_dev:.4f}rad (관측 범위 [{right_rx.min():.4f}, {right_rx.max():.4f}])",
        f"-pi ±{TOL_RIGHT_RX_ABS}rad",
    )


def check_angle_continuity(action: np.ndarray, episode_index: np.ndarray, report: ValidationReport) -> None:
    """에피소드 경계를 넘지 않고(섞이지 않게) 프레임간 점프를 검사한다."""
    rot_cols = [_col_index(f"{s}_{a}") for s in ("left", "right") for a in ("rx", "ry", "rz")]
    max_jump = 0.0
    n_violations = 0
    for ep in sorted(set(episode_index.tolist())):
        mask = episode_index == ep
        ep_action = action[mask][:, rot_cols]
        if len(ep_action) < 2:
            continue
        diffs = np.abs(np.diff(ep_action, axis=0))
        max_jump = max(max_jump, float(diffs.max()))
        n_violations += int((diffs >= MAX_ALLOWED_FRAME_JUMP).sum())
    passed = n_violations == 0
    report.add(
        "연속 프레임 간 각도 점프",
        passed,
        f"관측: 최대 프레임간 점프 {max_jump:.4f}rad, pi 이상 점프 {n_violations}건",
        f"pi({MAX_ALLOWED_FRAME_JUMP:.4f}rad) 이상 점프 0건(엄격, 오차 없음)",
    )


def check_initial_hold(state: np.ndarray, episode_index: np.ndarray, report: ValidationReport) -> None:
    left_x_i, left_y_i, left_z_i = (_state_col_index(f"left_ee_{a}") for a in ("x", "y", "z"))
    n_ok = 0
    n_total = 0
    details = []
    for ep in sorted(set(episode_index.tolist())):
        mask = episode_index == ep
        pos = state[mask][:, [left_x_i, left_y_i, left_z_i]]
        if len(pos) <= HOLD_FRAMES:
            continue
        n_total += 1
        hold_dev = float(np.max(np.linalg.norm(pos[:HOLD_FRAMES] - pos[0], axis=1)))
        later_idx = min(len(pos) - 1, HOLD_FRAMES * 3)
        move_after = float(np.linalg.norm(pos[later_idx] - pos[0]))
        ok = hold_dev <= HOLD_POS_TOL_M and move_after >= MOVE_AFTER_HOLD_MIN_M
        n_ok += int(ok)
        details.append(f"ep{ep}: hold_dev={hold_dev*1000:.1f}mm move_after={move_after*1000:.1f}mm {'OK' if ok else 'FAIL'}")
    passed = n_total > 0 and n_ok == n_total
    report.add(
        "에피소드 초반 유지 -> 첫 웨이포인트 이동",
        passed,
        f"관측: {n_ok}/{n_total}개 에피소드 통과 | " + "; ".join(details[:5]) + (" ..." if len(details) > 5 else ""),
        f"처음 {HOLD_FRAMES}프레임(~1초) 동안 왼팔 EE 이동 <= {HOLD_POS_TOL_M*1000:.0f}mm, "
        f"그 후 {HOLD_FRAMES*3}프레임째까지 >= {MOVE_AFTER_HOLD_MIN_M*1000:.0f}mm 이동",
    )


def validate(dataset_root: str) -> ValidationReport:
    table = load_dataset_table(dataset_root)
    action = np.stack(table.column("action").to_numpy())
    state = np.stack(table.column("observation.state").to_numpy())
    episode_index = table.column("episode_index").to_numpy()

    report = ValidationReport()
    check_episode_length(table, report)
    check_left_rotation_ranges(action, report)
    check_left_position_ranges(state, report)
    check_left_gripper_range(action, report)
    check_left_rotation_distribution(action, report)
    check_right_arm_static(action, state, episode_index, report)
    check_angle_continuity(action, episode_index, report)
    check_initial_hold(state, episode_index, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[validate_against_roco] {args.dataset_root} 검사 중...\n")
    report = validate(args.dataset_root)
    report.print_report()
    if not report.all_passed:
        print(
            "\n주의: peg_in_hole(OpenArm) 데모 데이터로 돌렸다면 일부 FAIL은 "
            "정상이다(오른팔=Actuator/왼팔=Stabilizer가 공식 관례와 반대 -- "
            "이 파일 모듈 docstring 참고)."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
