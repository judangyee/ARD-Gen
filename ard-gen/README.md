# ard-gen

MuJoCo 기반 peg-in-hole 태스크에서 admittance controller 게인을 CMA-ES로 찾는
파이프라인. **ARD-Gen 파이프라인의 0단계(Seed 확보)**에 해당한다 — 여기서 찾은
"성공하는 게인 + 그 궤적"(`seed_trajectory.npz`)이 다음 단계(2단계, seed를 이용한
부트스트래핑)의 입력이 된다.

팔은 실제 **ALOHA 팔로워 팔인 Trossen ViperX 300 6DOF(VX300s)** 를 그대로 쓴다
(mujoco_menagerie의 공식 MJCF, 메쉬 포함). 그 과정에서 진짜 로봇 팔(회전 조인트 6개, 관성/게인/
마찰이 실측값)을 쓸 때만 드러나는 문제 3가지를 실제로 겪고 고쳤다. 아래
"팔을 ALOHA(VX300s)로 교체하며 겪은 것들" 섹션 참고.

## 디렉토리 구조

```
ard-gen/
├── assets/
│   ├── peg_in_hole.xml          # MuJoCo 모델 (VX300s + hole + peg)
│   ├── screw_driving.xml        # MuJoCo 모델 (VX300s + 드라이버 + 볼트/블록) -- 검증 태스크 2
│   └── vx300s/                  # mujoco_menagerie의 VX300s 메쉬/라이선스
├── sim/
│   ├── peg_in_hole_sim.py       # 시뮬레이션 실행 + admittance controller (태스크 1)
│   └── screw_driving_sim.py     # 시뮬레이션 실행 + 토크 리미터 컨트롤러 (태스크 2)
├── optimize/
│   ├── cma_search.py                  # CMA-ES 게인 탐색 (태스크 1)
│   └── screw_driving_cma_search.py    # CMA-ES 게인 탐색 (태스크 2)
├── render_result.py             # seed_trajectory.npz의 게인으로 롤아웃을 mp4로 렌더링 (태스크 1)
├── render_screw_driving.py      # 픽업+토크 리미터 조이기 시퀀스를 mp4로 렌더링 (태스크 2)
├── requirements.txt
└── README.md
```

이 문서 대부분은 **검증 태스크 1(peg-in-hole)** 을 다룬다. **검증 태스크
2(나사 조이기, screw driving)** 는 맨 아래 별도 섹션에 정리했다 — 모델·
그립·컨트롤러 자체는 완전히 다른 내용이지만, "직접 돌려보고 실측으로
버그를 찾는다"는 이 저장소 전체의 작업 방식은 똑같이 적용했다.

## 모델 (assets/peg_in_hole.xml, assets/vx300s/)

- **팔**: 실제 ALOHA 팔로워 팔(VX300s, 6개 회전 조인트: waist, shoulder, elbow,
  forearm_roll, wrist_angle, wrist_rotate + 손가락 2개). 메쉬/관성/조인트
  범위/게인은 [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  의 `trossen_vx300s` 그대로이며, `assets/vx300s/LICENSE`(BSD-3-Clause,
  Trossen Robotics)를 함께 포함한다.
- **홈 자세**: `waist=0, shoulder=-0.8924, elbow=1.0534, forearm_roll=0,
  wrist_angle=1.4093, wrist_rotate=0` — 그리퍼(peg 방향)가 정확히 수직 아래를
  향하면서 **base(월드 원점) 기준 수평 거리 10cm** 지점에 오도록(world
  (0,0,-1)과 오차 0.0005 이내) 그리드 서치로 찾았다.
- **peg**: `gripper_link`의 자식이지만 자체 조인트가 없음 → 그리퍼에 강체로
  고정된 것으로 취급(grasp 자체는 다루지 않음). `pinch` site(VX300s 기본
  제공 TCP)보다 0.07m 더 뻗어나가며, **tip이 구형**(반지름 = peg half-width
  10mm) — 이유는 아래 "검증하며 알아낸 것들" 참고.
- **hole**: 4개 벽 세그먼트 + 바닥으로 이루어진 소켓, world (0.10, 0, 0.0438)
  위치(base에서 수평 거리 10cm, 홈 자세 peg tip 기준 호버 간격 3cm). nominal
  clearance 3mm(편측 1.5mm)지만, `scene_config['clearance_m']`에 따라 `sim`이
  런타임에 벽의 `geom_pos`/`geom_size`를 직접 덮어써서 바꾼다.
- **카메라 2개** (`render_result.py --cameras`로 선택):
  - `wrist_cam`: `gripper_link`에 **고정 장착**(`mode="fixed"`), 그리퍼의
    작동(삽입) 방향인 로컬 +x축에 **수직**으로 시선을 둔다 — peg가 hole로
    들어가는 과정을 옆에서 보는 프로파일 뷰. 그리퍼가 움직여도 이 상대
    각도는 안 바뀐다(진짜 손목에 붙은 카메라처럼 co-move).
  - `top_cam`: **로봇 정면(+x, 팔이 뻗어나가는 방향)에서 로봇 쪽을
    마주보는** 오블릭(비스듬한) 뷰(world 고정, `mode="targetbody"`로 hole을
    자동으로 바라봄). base-hole 거리가 10cm로 짧다 보니 접힌 upper_arm/
    forearm 링크(각각 30cm/20cm)가 hole 바로 위 공간을 차지해서, 순수 수직
    top view로는 hole이 거의 안 보인다(실측 확인 — 카메라 버그가 아니라 이
    reach에서 이 팔이 실제로 저렇게 접히는 것). 그래서 로봇 앞쪽에서
    마주보는 각도로 배치했다.
- **F/T 센서**: `peg_tip_site`에 3축 force + 3축 torque.
- 목표 삽입 깊이는 0.04m.
- fingers(그리퍼 손가락)는 grasp를 다루지 않으므로 키프레임 기본값 근처로
  고정만 해둔다(peg는 gripper_link에 용접되듯 붙어 있어서 실제로 손가락이
  peg를 쥐는 물리는 없음).

## 시뮬레이션 + admittance controller (sim/peg_in_hole_sim.py)

```
F_error_xy = F_ext_xy - F_desired_xy        (F_desired_xy = 0, world-frame)
dF_error_xy/dt ≈ (F_error_xy - F_error_xy_prev) / dt
Δx_xy = -Kp_xy * F_error_xy - Kd_xy * dF_error_xy/dt   (world-frame)
```

VX300s는 6개 조인트가 서로 결합돼 있어서(단순 슬라이드처럼 x/y/z가 독립
조인트가 아님), "action의 xy = 조인트에 직접 더하기"가 안 통한다. 대신 매
스텝 peg tip의 위치 Jacobian(`mujoco.mj_jacSite`)을 구해서, world-frame 위치
델타 `(Δx, Δy, -Z_RATE)`를 만들어내는 조인트각 델타를 damped least-squares로
역산하는 **리졸브드-레이트(resolved-rate) 오퍼레이셔널 스페이스 제어**를
쓴다. wrist_angle/wrist_rotate/forearm_roll(peg 자세)은 능동 보정하지 않고
초기값을 유지한다.

- **함수 시그니처**: `run_episode(gains: dict, scene_config: dict) -> dict`
  - `gains`: `{"Kp_xy": float, "Kd_xy": float}`
  - `scene_config`: `hole_pos_xy`, `friction`, `clearance_m`,
    `peg_init_offset_xy`, `peg_init_wrist`, `target_insertion_depth`
  - 반환: `ee_poses (T+1,4)`(x,y,z,wrist_rotate), `actions (T,3)`(world-frame
    위치 델타), `forces/force_profile (T,3)`, `torques/torque_profile (T,3)`,
    `insertion_depth`, `success`, `reward`, `max_force`, `step_count`
- 리워드는 **에피소드가 끝난 뒤 한 번** 계산되는 스칼라다 (CMA-ES 적합도로
  바로 쓰기 위함 — 매 스텝 누적하는 RL식 리워드가 아니다):
  ```
  reward = -2.0*dist + 20.0*insertion_depth - 0.001*max(0, max_force-5.0) - 0.01*step_count
  (+ 50.0 if insertion_depth >= target_insertion_depth)
  ```
  `dist`는 에피소드 종료 시점의 3D 거리, `max_force`는 에피소드 전체에서
  관측된 힘 크기의 **최댓값**(피크), `step_count`는 종료까지 걸린 스텝 수.
- `insertion_depth`는 z 깊이만 보지 않고, peg tip이 hole 벽 구조물의 실제
  xy footprint 안에 있을 때만 인정한다(rldc-lite에서 겪은 "정렬 없이 허공으로
  내리꽂아도 성공으로 인정되는" 리워드 해킹을 처음부터 막기 위함).

## 검증하며 알아낸 것들 (단순 팔 버전에서, 실제로 돌려보고 고친 것)

1. **평평한 peg 바닥은 admittance control이 아예 안 됨**: peg 바닥과 벽 윗면이
   둘 다 수평면이면 접촉 normal이 항상 수직이라(기하학적으로 당연함)
   `F_ext_xy`가 **항상 정확히 0**이라 Kp_xy를 얼마로 줘도 결과가 완전히
   똑같았다. 실제 커넥터/핀에 흔한 라운드팁과 같은 이유로 peg tip을 구형으로
   바꿔서 오프셋 상태의 접촉에 xy 성분이 생기게 했다.
2. **부호가 반대였음**: `Δx = +Kp*F_error`로 처음 구현했더니 오히려 안
   좋아졌다. 부호를 뒤집어(`Δx = -Kp*F_error`) 다시 테스트하니 최대 접촉력이
   60.5N(게인 0) → 6.8N(좋은 게인)까지 줄고 안정적으로 성공했다.
3. **오프셋 난이도 구간 확인**: 2~6mm는 peg tip의 곡률만으로 게인 없이도
   자가정렬되고, 12mm 이상은 게인을 아무리 키워도 안 됨. "게인이 있어야만
   성공"하는 구간을 찾아 대표 시나리오로 썼다.

## 팔을 ALOHA(VX300s)로 교체하며 겪은 것들 (진짜 로봇 팔이라 새로 드러난 문제)

플레이스홀더 팔(단순 슬라이드)에서는 안 보이던 문제 3개가, 실제 6-DOF 팔의
실측 관성/게인/마찰을 쓰자마자 나타났다. 전부 실제로 시뮬레이션을 돌려보고
찾았다.

1. **F/T 센서가 peg의 로컬 프레임 기준**: 단순 팔은 우연히 peg 로컬 프레임이
   world와 정렬돼 있어서 문제가 안 됐는데, VX300s는 팔 자체가 기울어 있어서
   peg 로컬 x,y,z가 world x,y,z와 다르다 — 정지 상태에서도 로컬 xy에 중력
   성분이 섞여 나오는 것으로 확인했다. `get_force_torque()`에서
   `site_xmat`으로 world 프레임으로 회전시켜서 고쳤다. (admittance
   controller는 world-frame lateral(x,y) 힘을 가정하므로 이 변환이 없으면
   애초에 말이 안 됨.)
2. **gravcomp 없이는 팔이 처짐**: VX300s의 실제 게인(예: shoulder kp=76)은
   원래 keyframe 근처 자세에서는 안정적이지만, 우리가 쓰는(그리퍼가 수직
   아래를 향하는) 완전히 뻗은 자세에서는 순수 P 제어로 중력을 못 버텨서
   200스텝 만에 6.6cm나 처졌다(실측 확인). 이 프로젝트의 다른 모델들과
   같은 방식으로 팔 바디마다 `gravcomp="1"`을 줘서 해결했다.
3. **elbow의 정지마찰(frictionloss) 데드존**: admittance controller 한 스텝의
   목표 변화량이 아주 작은데(수 mm 스케일), elbow의 실제 정지마찰
   (frictionloss=1.74)보다 그 스텝의 유효 토크가 작아서 **조인트가 아예 안
   움직였다**(qpos가 1e-6 단위로만 변함). 컨트롤 목표를 "이전 목표 +
   델타"로 계속 누적하는 방식(실제 오퍼레이셔널 스페이스 컨트롤러가 하는
   방식)으로 두고, elbow/shoulder의 frictionloss만 0으로 뺐다 — 반대로
   "현재 실제 qpos + 델타"로 매번 다시 고정하는 방식도 시도해봤는데, 이건
   목표가 항상 조인트 바로 앞에만 있어서 정지마찰을 절대 못 넘고 아예
   정지해버렸다(둘 다 실측으로 비교해서 확인).

이 세 가지를 고친 뒤 다시 스크립트로 오프셋을 훑어서, "게인이 있어야만
성공하는" 구간을 다시 찾았다 — 팔이 바뀌면서 정확한 mm 값은 이전(단순 팔,
6~10mm)과 달라졌다(5.5~8mm). peg-hole 접촉 물리 자체(구형 tip, clearance)는
그대로지만 팔의 실제 조인트 특성이 다르므로 재보정이 필요했다.

**base-hole 거리를 10cm로 좁힌 뒤 또 재보정**: 팔을 훨씬 접힌 자세로 바꾸자
(reach 57cm → 10cm) 난이도 구간이 또 바뀌었다 — 이번엔 peg tip의 구형
곡률에 의한 수동 자가정렬이 훨씬 잘 먹혀서 오프셋 8mm까지는 게인 없이도
대부분 성공하고, **게인을 오히려 세게 주면 더 불안정해지는**(반경 6~9mm
근방에서 Kp_xy가 클수록 실패) 정반대 경향까지 나타났다. 반경 14mm(±45°
방향)로 오프셋을 키우자 다시 "게인 있어야 성공, 게인 0이면 실패"하는 구간이
나왔고, 그마저도 Kp_xy가 대략 0.0002~0.001인 좁은 구간에서만 성공했다(그
이상이면 다시 실패). 대표 시나리오와 CMA-ES 탐색 범위를 이 값으로 다시
맞췄다.

## CMA-ES 최적화 (optimize/cma_search.py)

- `cma` 패키지로 `[Kp_xy, Kd_xy]` 2차원을 탐색한다. 스케일이 크게 다른 두
  파라미터라 `CMA_stds` 옵션으로 좌표별 초기 스텝 크기를 따로 준다.
- 매 세대, 실측으로 확인한 대표 시나리오 3개(오프셋 (14,0), (9.9,9.9),
  (9.9,-9.9)mm — "게인 0이면 실패, Kp_xy=0.0005면 성공"을 확인한 조합) 각각에
  대해 `run_episode()`를 돌려 리워드를 평균낸다.
- 종료 조건: 평균 리워드가 `--threshold`에 도달 **하거나** `--max-generations`
  (기본 100)에 도달하면 멈춘다.
- 종료 시: 최종 게인으로 대표 시나리오(첫 번째)를 다시 실행해 그 궤적을
  `seed_trajectory.npz`로 저장하고, 세대별 최고 리워드로 수렴 곡선
  (`convergence.png`)을 그린다.

### seed_trajectory.npz 데이터 구조 (2단계 부트스트래핑 입력)

| 키 | shape | 설명 |
|---|---|---|
| `ee_poses` | `(T+1, 4)` | (x, y, z, wrist_rotate) 시퀀스, 초기 상태 포함 |
| `actions` | `(T, 3)` | 매 스텝 실제로 적용한 world-frame 위치 델타 (Δx, Δy, Δz) |
| `forces` | `(T, 3)` | F/T 센서 force (world frame) |
| `torques` | `(T, 3)` | F/T 센서 torque (world frame) |
| `gains` | `(2,)` | `[Kp_xy, Kd_xy]` |
| `scene_config` | 스칼라(문자열) | `json.dumps()`된 scene_config — `json.loads(str(arr))`로 복원 |
| `success` | 스칼라(bool) | 목표 삽입 깊이 도달 여부 |
| `insertion_depth` | 스칼라(float) | 최종 삽입 깊이 |
| `reward` | 스칼라(float) | 최종 리워드 |
| `best_reward_per_gen` | `(G,)` | CMA-ES 수렴 곡선 원본 데이터 |

## 로컬 실행 (Colab 불필요, CPU면 충분)

```bash
pip install -r requirements.txt

python optimize/cma_search.py \
    --max-generations 100 \
    --popsize 10 \
    --threshold 49.0 \
    --out-path ./seed_trajectory.npz \
    --curve-path ./convergence.png

# 결과 영상 확인 (헤드리스 환경은 MUJOCO_GL=osmesa 필요, 카메라 2개 각각 mp4로 저장)
MUJOCO_GL=osmesa python render_result.py \
    --seed-path ./seed_trajectory.npz --out-dir . --cameras wrist_cam,top_cam
```

## 실제 실행 결과 (base-hole 거리 10cm, 위 버그 3개 수정 후)

- **수렴 여부**: 수렴함. 일부러 나쁜 초기 게인(`Kp_xy=0.00003`)에서 시작해서
  1세대 -4.09 → 2세대 -4.09 → 3세대 12.89 → 4세대 47.29 → 5~9세대 47.4 안팎
  정체 → **10세대 49.18**로 threshold(49.0)에 도달해 조기 종료.
- **최종 게인**: `Kp_xy ≈ 0.000515`, `Kd_xy ≈ 2.4e-05`
- **성공 여부**: 최종 게인으로 메인 시나리오(오프셋 14mm/0mm) 재실행 결과
  `success=True`, `insertion_depth=0.0405m`(목표 0.04m 초과 달성), `reward=49.36`
- `render_result.py`로 131스텝 만에 peg가 실제로 hole에 삽입되는 것을
  wrist_cam/top_cam 두 시점 모두에서 영상으로 확인함.

### 일반화 검증 (`evaluate_generalization.py`)

CMA-ES는 고정된 대표 시나리오 3개의 평균 리워드만 보고 게인을 골랐다 — 그
3개에서 잘 되는 게 "일반적으로 잘 된다"는 뜻은 아니므로, 찾은 게인을
`sim.sample_scene_config()`로 매번 새로 뽑은 무작위 시나리오(오프셋 반경
9~14mm 전체, 각도 ±45° 전체, 마찰 0.2~0.8, clearance 2.5~3.5mm 전부 랜덤)
N개에 실행해서 실제 성공률을 쟀다. 게인=0(피드백 없이 그냥 수직 하강만 하는
경우) 베이스라인과 비교해서, 게인이 실제로 기여하는지도 같이 확인했다.

```bash
python evaluate_generalization.py --n-trials 200 --seed 42
python evaluate_generalization.py --n-trials 300 --seed 123
```

| 시드, N | CMA-ES 게인 성공률 | 게인=0 베이스라인 성공률 |
|---|---|---|
| 42, 200 | **100.0%** (200/200) | 70.5% (141/200) |
| 123, 300 | **99.3%** (298/300) | 74.7% (224/300) |

두 시드 모두 게인이 있을 때 성공률이 25~30%p 더 높게 나와서, 게인이 우연이
아니라 실제로 기여한다는 게 확인됐다. 다만 게인=0에서도 70% 넘게 성공하는
걸 보면, peg 끝 구형 캡의 수동 자기정렬 효과 자체가 상당히 크고
admittance controller는 그 위에 "부족한 부분을 보완"하는 정도라는 것도
같이 드러났다.

**실패 케이스(시드 123, 2건)를 뜯어본 결과**: 둘 다 힘 부족이 아니라 벽에
낀 채 400스텝(최대 스텝)을 다 채우는 잼밍(jamming) 실패였다(`max_force`
492N·702N — 정상 성공 케이스의 평균 ~50N 대비 10배 이상). 둘 다 마찰이
샘플 범위 상단(0.69~0.73)이었고, friction>0.7 구간(48건)의 성공률은
97.9%, friction≤0.7 구간(252건)은 99.6%로 — 표본이 작아 단정하긴 이르지만
방향은 일치한다. **높은 마찰에서 잼밍 위험이 남아있다는 게 정직한 한계다.**

## 부트스트래핑 (bootstrap/) — ARD-Gen 2단계

0단계에서 확보한 seed(admittance 게인 + 궤적)를 이용해, "힘 상태 -> 행동"을
직접 흉내내는 신경망 정책을 시연 데이터로부터 학습(행동 복제, behavior
cloning)한다. 3개 스크립트로 구성:

```bash
# 1) 검증된 게인으로 무작위 시나리오를 돌려 성공한 에피소드의 (상태,행동) 쌍을 모은다
python bootstrap/collect_demonstrations.py --seed-path ./seed_trajectory.npz \
    --n-episodes 300 --seed 7 --out-path ./bootstrap/demonstrations.npz

# 2) 그 데이터로 작은 MLP를 지도학습시킨다
python bootstrap/train_policy.py --data-path ./bootstrap/demonstrations.npz \
    --out-path ./bootstrap/policy.pt --epochs 200

# 3) evaluate_generalization.py와 동일한 시드/분포로 학습된 정책의 성공률을 잰다
python bootstrap/evaluate_policy.py --policy-path ./bootstrap/policy.pt --n-trials 200 --seed 42
```

state = `[Fx, Fy, dFx/dt, dFy/dt]` (admittance controller가 실제로 보는 것과
동일), action = `[Δx, Δy]`. z축은 이전과 동일하게 게인과 무관하게 일정
속도로 내려간다(학습 대상이 아님).

### 실제로 겪은 문제: 학습 지표는 완벽한데 배포하면 완전히 실패함

처음 버전(평범한 MLP, bias 있는 Linear + 평균-중심화 정규화)은
`demonstrations.npz`(300 에피소드, 39284 스텝, 성공률 99.3%)로 학습했을 때
검증 MSE가 `1.7e-7`로 사실상 완벽했다. 그런데 이 정책을
`evaluate_generalization.py`와 똑같은 조건(무작위 시나리오, 시드 42/123)에
그대로 넣어 굴려보니 **성공률이 0.5%, 0.3%로 완전히 붕괴**했다.

원인을 실제로 파고들어 보니: 힘이 정확히 0일 때(정지 상태) admittance
공식의 정답은 `(0,0)`인데, 학습된 신경망은 이 지점에서 `(0.00024, -0.00006)`
정도의 아주 작은 편향(bias)을 출력하고 있었다. 전체 39284 스텝에 대한
residual(예측-정답) 평균도 x축에서 `+0.00022`로, action의 표준편차 대비
4.6%밖에 안 되는 작은 값이라 학습 중엔 MSE에 거의 안 보였다.

문제는 이 컨트롤러가 매 제어 스텝마다 `ctrl`에 행동을 그냥 더해서
"누적"시키는 구조라는 것 — 즉 정지-상태 편향은 속도 명령의 offset처럼
작용해서, 130~400스텝에 걸쳐 그대로 적분(누적)된다. `0.00022 x 400 ≈
0.088m`, hole clearance(3mm)나 성공 오프셋 범위(9~14mm)보다 훨씬 큰
drift라서 peg가 아예 딴 곳으로 밀려나 버린 것이다. **검증 데이터에서의
낮은 MSE가 실제 폐루프(closed-loop) 배포 성능을 보장하지 않는다**는 걸
직접 겪은 셈이다.

**고친 방법**: `AdmittancePolicy`의 모든 `Linear`에 `bias=False`를 주고
활성화를 홀함수인 `Tanh`(tanh(0)=0)만 쓰면, 네트워크 구조상
`f(0)=(0,0)`이 항상 정확히 보장된다. 정규화도 평균을 빼지 않고
표준편차로 스케일만 해서, "힘=0"이 정규화된 입력 공간에서도 여전히
정확히 0이 되도록 맞췄다. 재학습 후:

| | 성공률 |
|---|---|
| admittance controller (원본 공식, 시드42/123) | 100.0% / 99.3% |
| 학습된 정책 (수정 전, bias 있는 MLP) | 0.5% / 0.3% |
| 학습된 정책 (수정 후, bias-free + f(0)=0) | **100.0% / 100.0%** |

수정 후에는 손으로 짠 admittance 공식과 사실상 동일한 성능까지
따라잡았다 — 즉 "시연 데이터만으로 admittance 법칙을 신경망에 재현시키는
것"은 구조적 제약(f(0)=0)만 지켜지면 실제로 된다는 걸 확인했다.

**한계**: 지금 이 정책은 손으로 짠 공식을 재현한 것 이상은 아니다(입력이
force 상태뿐이라 admittance controller가 가진 정보 이상을 배울 수 없다).
진짜 가치는 여기서부터 시작된다 — 이 정책을 RL(PPO 등) 파인튜닝의 초기
정책으로 써서, force 상태보다 더 풍부한 관측(예: peg 자세, wrist 정렬
오차)까지 반영하는 더 일반적인 정책으로 확장하는 게 다음 단계다.

## 지금 임시로 되어있는/한계인 부분

- **wrist(회전) misalignment는 보정하지 않음**: admittance controller가 xy
  위치만 보정하고 wrist_angle/wrist_rotate/forearm_roll은 초기값 유지다.
  peg 초기 자세가 이미 정렬돼 있다고 가정한 것 — 실제로는 회전 오차도
  힘/토크 피드백으로 보정해야 할 수 있다.
- **오프셋 9~14mm(±45° 방향) 범위에서만 검증됨**: 이 범위/게인 구간을 벗어나면
  (너무 작은 오프셋에서 게인을 세게 주거나, 너무 큰 오프셋) 다시 실패한다.
  이건 버그가 아니라 "순수 힘 피드백 admittance control"의 실제 한계이자,
  base-hole 거리를 10cm로 좁혀 팔이 접힌 자세일 때의 특성이다(reach가 바뀌면
  또 재보정이 필요할 것).
- **top_cam은 순수 수직 top view가 아님**: base-hole 거리 10cm에서는 접힌
  팔 링크가 hole 바로 위를 가려서 순수 top view가 거의 안 보였다. 옆에서
  비스듬히 내려다보는 오블릭 뷰로 타협했다 — reach를 늘리면(예: 15~20cm)
  순수 top view도 다시 가능할 수 있다.
- **hole clearance/actuator 게인/질량은 대략적인 값**: hole 위치·clearance·
  peg 치수는 이 태스크가 풀리는 선에서 임의로 잡았다(팔 자체의 관성/게인은
  VX300s 실측값을 그대로 씀). gravcomp와 shoulder/elbow의 frictionloss 제거는
  admittance controller를 순수하게 테스트하기 위한 단순화이며, 실제 로봇에는
  둘 다 존재한다(실제 컨트롤러는 보통 별도의 중력보상/마찰보상 항으로 이를
  다룬다).
- **CMA-ES 탐색 자체는 대표 시나리오 3개로만 평균냄**: 매 세대 무작위로
  다시 뽑지 않고 고정된 3개 시나리오로 게인을 고른다. (사후에
  `evaluate_generalization.py`로 무작위 500건에 대해 99%대 성공률을
  확인하긴 했지만, 이건 "탐색 후 검증"이지 "탐색 자체가 다양한 조건에
  노출된 것"은 아니다 — 매 세대 `sim.sample_scene_config()`로 새로
  샘플링하도록 바꾸면 더 견고한 게인을 찾을 수도 있다, 이미 구현은 돼있고
  `cma_search.py`에서 아직 안 쓰고 있을 뿐이다.)
- **높은 마찰(0.7 이상)에서 잼밍 실패가 남아있음**: 위 일반화 검증에서 발견.
  힘이 과도하게(400~700N) 쌓이는데도 admittance controller가 빠져나오지
  못하는 경우가 있다 — 게인을 마찰에 따라 적응시키거나, 힘이 임계치를
  넘으면 후퇴(retract) 후 재시도하는 로직이 없어서다.

---

# 검증 태스크 2: 나사 조이기 (screw driving)

peg-in-hole과 같은 VX300s 팔/홈 자세를 재사용하되, 그리퍼 끝에 드라이버를
달고 나사(볼트+블록)를 완전히 조여 넣는 태스크. peg-in-hole이 "위치 보정"
admittance를 다뤘다면, 이쪽은 "회전/토크"를 다룬다 — `PIPELINE.md`의
"검증 태스크" 절에서 원래 계획했던 "뚜껑돌리기(토크제어)" 대신 이 태스크로
구현했다.

- 모델: `assets/screw_driving.xml`
- 시뮬레이션 + 컨트롤러: `sim/screw_driving_sim.py`
- 게인 탐색: `optimize/screw_driving_cma_search.py`
- 데모 렌더링: `render_screw_driving.py`

## 모델 (assets/screw_driving.xml)

- **나사산을 실제 지오메트리 없이 구현**: bolt body에 slide(전진)+hinge(회전)
  조인트를 같은 축에 겹쳐서 달고, "회전량 → 전진량"(피치 2mm/rev)을
  강제한다. 처음엔 `<equality><joint>`로 이 관계를 걸었는데(그리고
  wrist_rotate↔bolt_hinge 결합도 같은 방식으로), **실측해보니 이
  모델 규모에서 `<equality><joint>`가 외력(중력)에 전혀 저항하지
  못했다** — 회전을 하나도 안 시키고 중력만 준 상태에서 볼트가 혼자
  완전 삽입 깊이까지 미끄러져 내려갔다. polycoef를 키우거나 solref/solimp를
  훨씬 강하게 줘도, 두 조인트를 다른 body로 분리해도 동일했다. 두 결합
  모두 같은 해법으로 대체했다: **전용 position actuator + 컨트롤 코드가
  매 스텝 목표값을 계산해서 대입** — 이 모델 규모에서는 이 패턴이
  `<equality><joint>`보다 신뢰할 만하다는 게 두 번 반복 확인된 경험칙이다.
- **드라이버를 바닥(픽업 스탠드)에서 집는 것부터 시작**: 원래(peg-in-hole의
  peg와 같은 패턴) 드라이버는 그리퍼에 강체로 고정돼 있었는데, "집는 것부터
  시작하자"는 요청으로 자유 바디(freejoint)로 바꿨다. 실제 손가락
  접촉/마찰로 쥐는 물리는 다루지 않고(이 저장소 전체의 grasp 단순화
  방향과 동일), 팔이 픽업 자세에 도달해 손가락을 닫는 순간 코드에서
  weld equality를 켜서 "쥐었다"를 흉내낸다. weld의 relpose가 원래 고정
  오프셋과 정확히 같은 값이라 스냅(튕김) 없이 잡힌다.
- **wrist_rotate는 진짜 ±180˚로 막혀 있다**: 실제 VX300s 하드웨어를 그대로
  가져온 값인지 의심이 들어서 mujoco_menagerie 원본(`assets/vx300s/vx300s.xml`,
  이 저장소가 손댄 적 없는 파일)을 확인했는데 똑같이
  `range="-3.14158 3.14158"`였다 — 우리가 임의로 좁힌 게 아니라 실제
  하드웨어 스펙. 그래서 피치 2mm/rev로 34mm를 박으려면 17~20바퀴가
  필요한데 한 번에 반 바퀴 이상 못 돈다는 실제 물리적 제약이 있다.

## 다회전(turn/rewind) 시퀀스

한 번에 못 돌리니 "한계까지 돌리기(engaged) → 손목만 되감기(disengaged,
볼트는 그 자리에 고정) → 다시 물고 이어서 돌리기"를 반복한다. 초기 버전은
"ctrl을 목표 극단값으로 한 번에 던지고 500스텝 정착"시키는 방식이었는데
(wrist_rotate 액추에이터가 즉각적인 큰 램프를 못 따라간다는 걸 실측으로
확인하고 고른 임시방편), 이후 매 컨트롤 틱마다 `NOMINAL_RATE*dt`만큼만
전진시키는 진짜 속도 제어로 정리했다.

## 드라이버가 볼트를 계속 따라가야 하는 이유 (z-추종)

볼트가 조여지는 동안 34mm 내려가는데 팔이 안 움직이면 드라이버와 볼트 머리
사이 간격이 계속 벌어진다. 매 스텝 Jacobian 기반으로 팔(waist~wrist_angle
5개 관절)을 보정해서 드라이버가 볼트를 따라 내려가게 한다. 드라이버가
gripper_link의 자식이 아니라 weld로 붙은 자유 바디가 된 뒤로 실측으로
잡은 버그: `driver_tip_site` 자체로 Jacobian을 구하면 팔 관절 쪽이 전부
0이 나온다(weld 같은 등호 제약은 mj_jacSite가 보는 강체 트리 구조에 안
잡힌다) — gripper_link 위에 참조 사이트(`gripper_tip_ref`, 원래 자식이었을
때의 오프셋과 동일)를 따로 두고 Jacobian은 거기서, 오차는 여전히 진짜
tip/head 사이트로 계산해서 해결했다.

## 컨트롤러: 속도 조절 admittance → 토크 리미터로 재설계

처음엔 peg-in-hole처럼 "저항 토크가 크면 회전 속도를 늦추는" PD형
admittance(`kp_torque`, `kd_torque`)를 만들었다. `optimize/screw_driving_cma_search.py`로
실제 게인 탐색을 돌려보니 **CMA-ES가 두 게인을 전부 0으로 수렴시켰다** —
실측 추적 결과, 이 모델의 저항 토크는 회전 *속도*가 아니라 *사이클(=삽입
깊이)*에 달려 있어서 느리게 돈다고 그 순간의 저항이 줄지 않았다(오히려
느리게 돌면 저항이 큰 뒷부분 사이클에 더 오래 머물러서 평균 저항이 살짝
올라가기까지 했다).

그래서 실제 전동 드라이버의 **토크 리미터(클러치)** 방식으로 다시 짰다:
속도는 항상 일정하게 돌리고, 저항 토크가 `torque_limit`을 넘으면 그
사이클의 돌리기를 즉시 멈추고 되감기로 넘어간다. 이번엔 실제로
`max_torque`를 낮출 수 있었다(0.437/0.658/1.063/1.380 — limit
0.3/0.6/1.0/무제한). 하지만 뚜렷한 **문턱값**도 함께 발견했다: 이 씬의
자연 저항 최대치(~1.4, 마찰이 더 큰 씬에서는 ~1.5) *미만*으로 설정하면
매 사이클이 항상 조기 종료돼서 **삽입이 영원히 멈춘다**(스텝 예산을
15000→40000으로 늘려도 깊이가 그대로 — 느린 게 아니라 진짜 데드락, 실제
클러치를 너무 낮게 설정했을 때와 같은 고장 모드). 문턱값 이상이면 리미터가
사실상 안 걸려서 "무제한"과 완전히 동일하게 끝난다.

완료 판정은 (peg-in-hole과 마찬가지로) 토크가 아니라 기하학(`bolt_slide`
깊이)으로 한다 — 저항 토크가 목표 깊이의 1/3도 안 됐을 때부터 이미 크게
뛰고 사이클마다 계속 커져서(seat에 눌리며 생기는 누적 효과로 보임), 고정
임계값으로 "덜 조여짐"과 "다 조여짐"을 구분할 수 없었다.

## CMA-ES 게인 탐색 (optimize/screw_driving_cma_search.py)

`torque_limit` 하나를 탐색한다. 평가 시나리오 2개(기본 마찰, 5배 마찰)에
대한 평균 리워드로 적합도를 매긴다. 이 과정에서 실제로 부딪힌 버그 두 개:

1. **리워드가 실패를 보상하는 역설**: 처음엔 성공 여부와 무관하게 항상
   `-3.0*max_torque`를 뺐는데, 그러면 데드락(실패) 구간에서 `torque_limit`을
   낮출수록 `max_torque`도 같이 낮아지니 "무조건 낮게 설정하는 게
   유리해 보이는" 가짜 경사가 생긴다 — 이게 실제 성공 신호(깊이, +50
   보너스)보다 강해서 CMA-ES가 `torque_limit≈0.15`(데드락 영역)로
   수렴해버렸다(reward −15.5, 무제한 베이스라인 +36.9보다 훨씬 나쁨).
   **토크 페널티를 성공한 경우에만 적용**하도록 고쳐서 해결.
2. **`cma` 패키지 자체 버그**: 1차원 파라미터 + `bounds` 옵션 조합으로
   돌리면 몇 세대 뒤 `es.tell()` 내부에서 `"not yet initialized (dimension
   needed)"` 오류로 죽는다(최소 재현 스크립트로 우리 코드/리워드와 무관함을
   확인). `bounds`를 라이브러리에 안 맡기고 `evaluate_gains()`에서 직접
   클리핑하는 방식으로 우회.
3. 시작점(`x0`)을 문턱값보다 훨씬 낮게 잡았더니, 초기 세대 후보가 전부
   "실패 평지" 안에 몰려서(리워드 분산이 거의 0) CMA-ES가 조기 수렴
   조건(`tolfun`)에 걸려 일찍 멈추는 것도 확인했다 — `x0`을 문턱값 근처로,
   `sigma0`을 넉넉히 잡아서 매 세대 성공/실패 후보가 섞이게 해서 해결.

**최종 결과**: `torque_limit ≈ 1.79`, reward 36.80, `success=True`,
`max_torque=1.380` — 무제한 베이스라인과 완전히 동일한 성능(성공 구간
안에서는 리미터가 걸리지 않는 한 어떤 값이든 리워드가 같은 평지이고,
CMA-ES는 그 평지 안의 한 지점에 안착했다). 실제 클러치 설정값으로는
문턱값보다 안전 마진이 있는 합리적인 값이다.

## 실행 방법

```bash
# 모델 검증(단독, 게인 하나 고정)
python sim/screw_driving_sim.py

# 게인 탐색
python optimize/screw_driving_cma_search.py --max-generations 10 --popsize 6

# 픽업 -> 운반 -> 토크 리미터 조이기 전체 시퀀스를 3개 카메라로 렌더링
MUJOCO_GL=osmesa python render_screw_driving.py --out-dir .
```

## 한계 / 아직 안 된 것

- **ARD-Gen 파이프라인(0→1→2-A→2-B→4→5단계)에 편입되지 않았음**:
  peg-in-hole과 달리 모델+컨트롤러+게인 탐색까지만 있고, 씬 랜덤화
  (`pipeline/scene_sampler.py` 대응)·diffusion 기반 게인 생성·에피소드
  생성/필터링·언어 라벨링 단계는 아직 없다.
- **`seat_friction_scale`은 효과가 없어서 뺐다**: block_wall_* 접촉의
  마찰 계수를 0.02~100배로 흔들어도 저항 토크가 전혀 안 변하는 걸 실측으로
  확인했다(sliding 성분만이든 [sliding,torsional,rolling] 세 성분을 다
  스케일하든 동일) — 이 모델의 저항은 거의 전적으로 `bolt_hinge`의
  frictionloss에서 나온다.
  `hinge_friction_scale`은 실제로 유의미한 영향을 준다(scene_config에 남김).
- **CMA-ES 탐색이 아직 소규모**: popsize 6, 세대 10, 시나리오 2개짜리
  검증 수준이다. peg-in-hole의 `evaluate_generalization.py`처럼 무작위
  시나리오 대량(N=200~300)에 대한 사후 검증은 아직 안 해봤다.
- **토크 센서 기반 완료 판정은 포기했다**: 저항 토크가 사이클마다
  누적돼서 커지는 패턴이라(원인 미확정 — seat friction은 아님을 확인,
  bolt_hinge frictionloss가 유력 후보) 고정 임계값으로 "다 조여짐"을
  판정할 수 없었다. 지금은 `bolt_slide` 깊이(기하학적 신호)로만 판정한다
  — 실제 로봇이라면 이 센서가 없으니, 나중에 진짜 토크 센서만으로 완료를
  판정해야 한다면 이 사이클별 누적 원인부터 규명해야 한다.
