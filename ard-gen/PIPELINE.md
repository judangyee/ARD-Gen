# ARD-Gen 파이프라인 설계 문서

> 이 문서는 설계 전체를 미리 정리해둔 참고 자료다. **아직 구현되지 않은
> 단계가 대부분이며, 실제 구현은 이 문서를 보고 그때그때 요청받은 단계만
> 진행한다.** 각 단계 구현 후에는 이 문서의 해당 단계 상태를 갱신한다.
>
> **현재 상태: 3단계(Stabilizer, 왼팔)까지 완료했다.** hole_socket/bottle을
> freejoint(자유물체)로 바꾸고, 왼팔을 IK 없는 가상 엔드이펙터 + weld로
> 구현해서, peg_in_hole/cap_twist 둘 다 이제 `left_arm`+`right_arm`이
> 모두 있는 진짜 양팔 에피소드를 만든다. `sim.task_registry.TaskConfig.
> make_env()`가 tasks/*.yaml의 `stabilizer` 절을 보고 자동으로 Stabilizer를
> 붙이므로, 0→1→2-A→2-B→4→5단계 스크립트 어느 것도 코드를 고칠 필요가
> 없었다(--task만 그대로 쓰면 양팔로 동작한다). 자세한 내용/실측 수치는
> 아래 3단계 절 참고.

## 배경

**ARD-VLA**는 양팔 로봇의 두 팔에 서로 다른 역할을 부여하는 아키텍처다.

- **오른팔 (Actuator)**: 도구를 쥐고 미세조작을 수행한다.
- **왼팔 (Stabilizer)**: 물체를 고정하는 역할을 한다.

이 모델을 학습시킬 데이터가 필요하지만, 텔레오퍼레이션 장비도 GPU 학습
자원도 없는 환경(군 복무 중, Colab + Claude Code만 사용 가능)이라
**시뮬레이션만으로 데이터를 생성하는 파이프라인**을 설계했다 — 그게
ARD-Gen이다.

## 핵심 설계 원칙

1. **두 팔은 증강 방식이 다르다.** Stabilizer는 기하학적 변환(MimicGen
   스타일)으로, Actuator는 물리 기반 방식(diffusion)으로 증강한다.
2. **계산 비용이 싼 방법부터 쓴다.** RL처럼 정책을 처음부터 학습시키는
   무거운 방법 대신, 이미 설계된 admittance controller의 파라미터만
   탐색하는 가벼운 방법을 우선한다.
3. **두 팔의 데이터는 물리적으로 일관돼야 한다.** Actuator의 반작용이
   Stabilizer에도 전달되므로, 공유된 씬 조건 위에서 Actuator를 먼저 생성하고
   그 결과를 참고해 Stabilizer를 생성한다.

## 전체 파이프라인 (6단계)

### 0단계 — Seed 확보 ✅ 완료

CMA-ES로 admittance controller의 게인(`Kp_xy`, `Kd_xy`)을 탐색해서,
peg-in-hole 태스크를 최초로 성공시킨 궤적을 확보했다.

- 구현 위치: `optimize/cma_search.py`, `sim/peg_in_hole_sim.py`
- 산출물: `seed_trajectory.npz` (게인 + 궤적 + force/torque)
- 검증: `evaluate_generalization.py`로 무작위 시나리오 500건+ 대상 성공률
  99~100% 확인 (게인=0 베이스라인은 70~75%)
- 부가 작업(스코프 밖이었지만 먼저 해봄): `bootstrap/`에 이 seed를
  행동 복제(behavior cloning)로 신경망 정책에 재현하는 실험 — 구조적
  수정(bias-free, f(0)=0 보장) 후 성공률 100% 확인. **2-A 구현 결과,
  이 정책은 2단계와 목적이 달라 재사용하지 않기로 함** (아래 2-A 참고).

### 1단계 — 공유 씬 설정 ✅ 완료

매 에피소드마다 다음을 무작위로 샘플링한다:
- hole 위치/자세: `hole_pose = (dx, dy, theta)`, dx/dy는 ±1cm, theta는
  ±0.1rad(현재 sim은 아직 회전을 지원하지 않아 소비되지 않음 — 3단계용
  선반영)
- 마찰계수: 기준값(0.5)의 0.7~1.3배
- clearance: 2mm~5mm (friction보다 넓게 — margin 무작위화가 더 안정적인
  신호라는 원칙)
- peg 초기 오프셋: ±1.5cm

- 구현 위치: `pipeline/scene_sampler.py` (`sample_scene_config()`,
  `to_sim_scene_config()`)
- `sim/peg_in_hole_sim.py`의 기존 `sample_scene_config()`/
  `_default_scene_config()`는 CMA-ES 전용으로 남겨두고 건드리지 않음 —
  파이프라인 공식 스키마는 `pipeline/scene_sampler.py` 쪽. `run_episode()`에
  넘기려면 `to_sim_scene_config()`로 변환.

### 2단계 — Actuator(오른팔) 증강: Diffusion

seed 하나만으로는 diffusion을 학습시킬 수 없으므로 두 단계로 나눈다.

- **2-A 부트스트래핑 ✅ 완료**: seed 게인 주변에 넓은 랜덤 노이즈를 주고
  시뮬레이션을 반복 실행해서 `(씬 조건, 게인, 성공여부, force_profile)`
  기록을 대량으로 쌓는다.
  - 구현 위치: `pipeline/bootstrap.py`
  - 방식: `pipeline.scene_sampler.sample_scene_config()`로 씬 샘플링,
    seed 게인(Kp_xy, Kd_xy) 각각에 독립적으로 [0.5, 2.0]배 균등분포
    노이즈를 곱해서 `sim/peg_in_hole_sim.py`의 `run_episode()` 그대로 실행.
    성공/실패 모두 저장한다 — 2-B 구현 결과 diffusion 자체는 성공
    샘플만으로 학습시켰지만(아래 참고), 실패 샘플도 조건(scene_config)
    정규화 통계 계산에는 쓰인다.
  - 산출물: `data/bootstrap/bootstrap_dataset.npz` (1000건 기본)
  - **실제 실행 결과(N=1000, seed=0)**: 전체 성공률 76.0%. 성공 게인
    분포는 `Kp_xy` 평균 0.000649 (seed 0.000515보다 약간 큼), `Kd_xy`
    평균 3.19e-05. clearance로 4분위 나눠 보면 2.0~2.76mm 구간
    성공률 64.4% → 4.22~5.00mm 구간 83.2%로 뚜렷하게 clearance가
    넓을수록 성공률이 오른다. offset 크기도 성공군 평균 10.6mm vs
    실패군 평균 14.7mm로 명확히 갈리는 반면, friction은 성공군 0.498 vs
    실패군 0.507로 거의 차이가 없다 — 이 태스크에서 friction보다
    clearance/offset이 성공/실패를 가르는 훨씬 강한 신호라는 게 실측으로
    확인됨(설계 원칙 "margin 무작위화가 더 안정적인 신호"와 일치).
- **2-B diffusion 가동 ✅ 완료**: 축적된 기록으로 diffusion 모델을
  학습시켜서, 새로운 씬 조건이 주어지면 랜덤 샘플링 대신 diffusion이
  성공 확률 높은 게인 조합을 직접 생성하도록 전환한다.
  - 구현 위치: `pipeline/diffusion_gains.py`
  - 모델: 조건(7차원: hole_pose 3 + friction 1 + clearance_m 1 +
    peg_init_offset 2) + timestep을 받아 노이즈를 예측하는 작은 MLP
    (`GainDiffusionNet`, hidden=128) + 표준 DDPM 스케줄
    (`GaussianDiffusion`, T=100).
  - **학습 데이터는 성공 샘플만 사용**(1000건 중 760건). 조건 정규화
    통계(평균/표준편차)만 전체 1000건으로 계산 — 성공 샘플만으로
    계산하면 "쉬운 씬" 쪽으로 치우친 통계가 나와서다. 실패까지 포함해서
    diffusion을 학습시키는 classifier-guidance 방식도 검토했지만, 조건
    7차원/출력 2차원짜리 저차원 문제에는 과한 복잡도라 판단해
    "성공 사례만 보고 그 분포를 재현"하는 단순한 방식을 택함(자세한 이유는
    스크립트 docstring 참고).
  - 생성된 게인은 `optimize/cma_search.py`와 동일한 탐색 범위
    (`Kp_xy∈[2e-5,3e-3]`, `Kd_xy∈[0,5e-4]`)로 clip해서 물리적으로 말이
    안 되는 값(음수 등)을 방지.
  - **실제 검증 결과**: 새 무작위 씬 100개(2-A 데이터 생성에 쓰지 않은
    시드)에 대해 diffusion이 생성한 게인으로 직접 시뮬레이션 실행.
    - 시드 999: **86.0%** (86/100)
    - 시드 1234: **84.0%** (84/100)
    - 시드 7777: **84.0%** (84/100)
    - 2-A 부트스트래핑(무작위 노이즈) 베이스라인 **76.0%** 대비 세 시드
      모두에서 **+8~10%p 일관되게 개선** — 우연이 아님을 확인.
    - diffusion이 생성한 게인 분포는 `Kp_xy≈0.00064±0.00024`,
      `Kd_xy≈3.2e-05±1.1e-05`로, 2-A의 성공 게인 분포(`Kp_xy` 평균
      0.000649)와 비슷한 영역이지만 씬 조건에 따라 값을 조절해서 뽑는다는
      점이 다르다(무작위 노이즈는 조건과 무관하게 넓게 뿌리기만 함).

> `bootstrap/`(구 디렉토리, 행동 복제 MLP 정책)와 `pipeline/bootstrap.py`
> (2-A)는 이름은 비슷하지만 다른 것이다 — 전자는 "force 상태 → 행동"을
> 흉내내는 정책이고, 후자는 "씬 조건 → 게인이 성공하는지"를 기록해 2-B
> diffusion의 학습 데이터를 만드는 것. 실제 구현해보니 서로 재사용할
> 부분이 없어 별도로 유지한다.

### 3단계 — Stabilizer(왼팔) 증강: 기하 변환 ✅ 완료

#### 1. 물체를 자유물체로 전환

`assets/peg_in_hole.xml`의 `hole_socket`과 `assets/cap_twist.xml`의
`bottle`을 world-고정 body에서 `<freejoint>`가 있는 자유물체로 바꿨다.
둘 다 이전엔 "떠받쳐줄 것"이 필요 없었으므로(world에 그냥 고정) 실제로
충돌하는 테이블면도 이번에 처음 추가했다(각각 `hole_table`/`table` 지오m,
기존 고정 위치와 정확히 같은 높이에 둬서 초기 낙하 없이 시작한다).
`hole_socket`/`bottle`이 이제 `<inertial>`을 명시적으로 갖는다. 또한
"강한 위치 액추에이터 + weld" 조합에서 긴 에피소드 동안 서서히 처지는
버그(screw_driving 태스크에서 이미 겪음)를 선제적으로 피하려고 두 모델
다 `integrator="implicitfast"`로 바꿨다.

**전환 직후(Stabilizer 아직 없음) 실측 성공률 변화** (동일 게인, N=200
무작위 씬, 95% Wilson 신뢰구간 포함 — 처음엔 N=40으로 봤는데 표본이
작아 우연일 가능성을 배제하려고 N=200으로 재확인함):

| 태스크 | 전환 전(고정 물체) | 전환 후(자유물체, Stabilizer 없음) |
|---|---|---|
| peg_in_hole (Kp_xy=0.000515, Kd_xy=2.4e-05) | **71.0%** (142/200, CI 64.4-76.8%) | **38.0%** (76/200, CI 31.6-44.9%) |
| cap_twist (Kp_tau=0.9) | **100%** (200/200) | **100%** (200/200) |

신뢰구간이 겹치지 않아 peg_in_hole의 하락은 우연이 아니다 — Stabilizer가
실제로 필요하다는 직접적인 증거다. cap_twist는 변화가 없었다: bottle이
테이블에 얹힌 접촉 자체만으로 이미 현재 저항 토크 범위(0.2~1.0N·m)에서
밀려나지 않는다(정직하게 밝힘: 정확히 어떤 접촉 메커니즘이 이걸 만드는지
더 깊게 파진 않았다 — friction 계수를 낮춰봐도 차이가 없었는데, 이는
이번 3단계 스코프를 넘는 별도 조사가 필요해 보류했다). 즉 cap_twist에서
Stabilizer는 "필수"는 아니지만, 아래처럼 붙여도 성공률을 해치지 않는다.

#### 2. Stabilizer 스크립트 — `sim/stabilizer.py`

실제 팔(Jacobian IK)을 왼팔로 하나 더 만드는 대신(이전
`assets/peg_in_hole_bimanual.xml` 시도가 겪은 문제: hole 위치마다 왼팔
홈 자세를 손으로 다시 풀어야 해서 위치 무작위화를 포기했었음), **IK가
필요 없는 3-슬라이드 조인트 가상 엔드이펙터**로 구현했다 — 직렬 링크가
없어 역기구학이 항등함수라, world 목표 좌표를 그대로 ctrl에 넣으면
끝난다. cap_twist는 애초에 오른팔조차 실제 기구학을 모델링하지 않는
태스크라(손목 회전=cap 힌지 각도 직접 구동), 왼팔만 실제 팔로 만드는
것도 일관성이 없었다 — 이 단순화로 **peg_in_hole/cap_twist 둘 다 같은
`sim/stabilizer.py` 코드 하나**로 다룬다(태스크별 차이는 tasks/*.yaml의
`stabilizer: {object_body, grasp_offset}` 두 필드뿐).

- **접근(approach)**: reset() 시점(Actuator가 아직 아무것도 안 건드린
  때) 물체의 world 위치 + grasp_offset을 목표로 EE 위치 액추에이터에
  즉시 명령한다.
- **weld(고정)**: EE가 목표에 도달하거나(2mm 이내) 최대 대기 스텝(60)을
  넘기면, **그 순간의 실제 상대 포즈**를 MuJoCo weld equality의
  relpose로 굳혀서 활성화한다.
- **유지(hold)**: weld가 물체를 EE에 강체로 고정하고, EE 자신의 위치
  액추에이터도 계속 같은 목표를 명령해 이중으로 버틴다.
- **실측으로 잡은 버그 3건**(정직하게 기록):
  1. weld의 `eq_data` 레이아웃을 처음에 잘못 알아서(relpos를 0번 슬롯에
     썼는데 실제로는 anchor(3)+relpos(3)+relquat(4)+torquescale(1) 순서라
     3번부터 시작함) 활성화 즉시 물체가 엉뚱한 곳으로 튕겨나갔다 — 컴파일된
     모델의 eq_data 기본값을 직접 찍어보고 실제 레이아웃을 알아냈다.
  2. Actuator 제어 루프 "안"에서 매 스텝 Stabilizer.tick()을 같이 부르면,
     Stabilizer가 붙잡기도 전에(최대 60스텝) Actuator가 이미 삽입을
     시작해버려 반작용력이 무구속 물체를 먼저 밀어낸다 — 이걸로 측정한
     성공률이 42.5%→57.5%로만 올랐다(N=40 기준). **run_approach_phase()**
     로 "먼저 확실히 쥔 뒤에" Actuator를 시작하도록 순서를 강제해서 고쳤다.
  3. **(심층 검증에서 발견) EE 위치 액추에이터가 압도적으로 물러 터졌다.**
     처음엔 kp=400으로 두고 "peg가 벽에 걸려 100N+까지 힘이 쌓이는 경계선
     씬 몇 건이 남는 건 물체가 완전 강체가 아니게 되며 생기는 정당한
     물리"라고 결론 내렸는데(kp를 1500으로 4배 올려도 그 씬들이 똑같이
     실패하는 걸 보고 내린 판단), 이건 성급한 결론이었다. 아래 "물리적
     타당성 점검" 절 참고 — kp=400에서 100N 정적 하중에 EE가 254.77mm나
     밀린다는 걸 직접 측정하고서야 진짜 원인(weld가 아니라 EE 자신의
     스프링 강성 부족)을 찾았다. kp=20000으로 올려 해결.

#### 2-b. 물리적 타당성 심층 검증

**통계적 신뢰도(N=200)**: peg_in_hole은 자유물체+Stabilizer(kp=20000)로
**64.5%→69.0%**로 개선(N=200, 95% CI 62.0-75.4%) — 강체 baseline
71.0%(CI 64.4-76.8%)와 신뢰구간이 겹칠 만큼 근접했다. 원래 성공하던
142개 중 131개(92.3%)가 그대로 성공, 11개만 새로 실패, 7개는 오히려
새로 성공(강체보다 컴플라이언트한 편이 우연히 더 잘 맞은 케이스 — 의외지만
드물지 않은 결과). cap_twist는 N=200에서도 100%/100% 그대로.

**실패 11건이 "정당한 물리적 한계"인지 "씬 샘플링 범위 문제"인지**: 11건
전부 |peg_init_offset| 10.8~20.2mm(범위 상한 ±15mm/성분, 대각선 최대
~21.2mm)에 몰려 있고 max_force가 105~111N으로 일정하다 — "peg가 벽
위에 걸려 못 들어가고 Z_RATE가 계속 눌러서 힘만 쌓이는" 동일 패턴.
이게 "말이 안 되는 씬"인지 확인하려고 **같은 오프셋들을 원래(강체) 물리로
다시 돌려봤다** — 11건 전부 강체에서도 이미 54~365N의 큰 힘을 필요로
하는 아슬아슬한 성공이었다(예: 강체에서도 365N까지 필요했던 씬이 있음).
추가로 |offset| 3mm 구간별 강체 성공률을 재봤더니 100%(0-3mm)→93%(6-9mm)
→75%(12-15mm)→33%(18-21mm)로 **매끄럽게 감소**하는 난이도 곡선이었다 —
절벽(cliff)이 아니라 원래부터 설계된 연속적인 난이도 스펙트럼이라는 뜻.
**결론: 씬 샘플링 범위(offset ±15mm/성분, clearance 2-5mm) 문제가
아니다 — 애초에 강체 기준으로도 "게인이 있어야만 겨우 성공하는" 경계
지대였던 씬들이, 완전 강체가 아닌 조건에서 몇 개 더 넘어간 것뿐이다.
범위를 좁히라고 제안하지 않는다** — 좁히면 오히려 게인 탐색을 의미
있게 만드는 난이도 다양성 자체가 사라진다(tasks/peg_in_hole.yaml에 이미
이 범위가 "게인이 있어야만 성공하는 난이도"로 의도적으로 잡혀 있다고
문서화돼 있음).

**Stabilizer weld/EE 물리 정합성** (N=30~40 계측 에피소드, kp=20000):
- **hold 구간 drift**: 잡은 순간부터 에피소드 끝까지 hole 위치 변화량
  평균 1.35mm, 95th 백분위 1.55mm, 최대 2.01mm — 삽입 공차(2~5mm)보다
  작아 대부분 문제 없는 수준.
- **액추에이터 saturation**: forcerange(±3000N) 대비 실사용 최대
  0.62% — 전혀 포화되지 않는다(현재 부하 수준에서 kp=20000 선택에
  여유가 충분함을 보여줌).
- **접촉력-반작용력 상관관계**: peg-wall 직접 접촉력(mj_contactForce로
  실측)과 weld가 실제로 전달하는 힘(qfrc_constraint)의 상관계수
  **0.959**(n=6343 스텝) — 물리적으로 일관됨. 참고로 이 상관계수는
  kp=400일 때는 0.179였다 — 강성이 낮아 물체가 크게 밀리면서 접촉
  기하 자체가 계속 바뀌어(밀린 벽을 따라 peg가 미끄러짐) 힘-반작용
  관계가 흐트러졌던 것으로 보인다. **주의**: peg 끝단의 force 센서
  판독값(최대 111N) 자체는 접촉력이 아니라 peg 질량의 관성/중력 성분이
  섞인 값이다(정지 상태에서도 0이 아님, sim/peg_in_hole_sim.py 기존
  주석 참고) — 그래서 "접촉력"은 센서값이 아니라
  `mj_contactForce()`로 peg-wall 접촉만 직접 뽑아서 비교했다.

**타이밍 엣지케이스**: N=200 전체 씬에서 `run_approach_phase()`가
실제로 쓴 스텝 수는 **전부 2스텝**(APPROACH_MAX_STEPS=60의 30분의 1
수준), grasp 시점 잔차 오차도 평균 1.17mm/최대 1.29mm로 여유롭다 —
60스텝 타임아웃은 한 번도 발동하지 않았다. kp=20000으로 EE가 매우
빨리 수렴하게 되면서(자연주파수가 높아짐) approach가 사실상 즉시
끝나버려, 원래 의도했던 "여러 스텝에 걸친 접근 동작"은 시각적으로는
거의 안 보인다(아래 시각 확인 이미지 참고) — 물리적으로는 문제 없지만,
"접근하는 모습"을 보여주는 게 목적이라면 kp를 낮추고 REACH_TOL을
동적으로 조절하는 등 트레이드오프가 있다는 점은 밝혀둔다.

**시각 확인**: peg_in_hole/cap_twist 각각 성공 1건 + 실패 1건에 대해
grasp 직후/중간/종료 시점 3프레임씩 오프스크린 렌더링으로 저장해서
직접 확인했다 — 초록 구체(Stabilizer EE)가 물체 옆에 붙어 함께 움직이는
것, peg_in_hole 실패 케이스에서 peg(주황)가 hole 블록(회색) 위에
얹힌 채 안 들어가는 모습을 육안으로 확인했다. cap_twist는 cap이 자기
축으로만 회전해서 카메라 각도상 성공/실패가 육안으로는 거의 구분 안 됨
(원통 대칭이라 회전이 안 보이는 것뿐, 버그 아님) — 정직하게 밝혀둔다.

#### 3. MimicGen 스타일 기하 증강 — `pipeline/stabilizer_augment.py`

seed 에피소드의 왼팔 궤적(world-frame)을 물체 기준 좌표계로 저장해두고,
새 씬의 물체 포즈로 SE(3) 변환(순수 좌표 계산, 물리 재시뮬레이션 없음)해서
새 궤적을 만든다. **실측 검증**: 변환된 궤적의 최종(유지) 위치와, 그
씬에서 Stabilizer를 직접 다시 돌린("라이브 재계산") 최종 위치를 비교.

- peg_in_hole(물체 위치가 씬마다 실제로 다름): 5개 무작위 씬에서 최대
  오차 **1.95mm** — weld의 잔류 컴플라이언스 수준(앞서 측정한 반작용력
  드리프트와 같은 크기)과 일치, SE(3) 변환 자체는 정확함을 확인.
- cap_twist(물체 위치가 현재 씬 무작위화 대상이 아님, 1단계 스키마 참고):
  변환이 사실상 항등이라 오차 **0.47mm 이하**(측정 잡음 수준) — 이건
  버그가 아니라 이번 3단계 스코프에서 bottle 위치 자체를 무작위화하지
  않았다는 사실의 정직한 반영이다.

#### 4. 양팔 통합

Actuator(2단계 diffusion 결과)와 Stabilizer가 **같은 물리 시뮬레이션
안에서 동시에** 실행된다(`run_approach_phase()`로 먼저 쥔 뒤,
`env.run_episode()`의 기존 루프 안에서 Stabilizer는 그냥 고정된 채
따라가고 Actuator만 능동 제어한다) — 별도의 "동시 실행 모드"를 새로
만들 필요가 없었다, 애초에 하나의 MjModel/MjData를 공유하기 때문이다.
`sim.task_registry.TaskConfig.make_env()`가 tasks/*.yaml의 `stabilizer`
절을 보고 자동으로 Stabilizer를 붙이므로, **0/2-A/2-B/4단계 스크립트
전부 코드를 한 줄도 안 고쳤다**. `pipeline/episode_io.py`/
`pipeline/filter_episodes.py`만 `left_arm` 필드를 새로 다루도록 확장했다
(있으면만 저장 — stabilizer 절이 없는 미래 태스크와 하위호환).

**실측: 0→1→2-A→2-B→4→5단계 전체를 처음부터 다시 돌린 결과** (--task만
바꿔서, 코드 수정 없이):

| | peg_in_hole | cap_twist |
|---|---|---|
| 0단계 CMA-ES | gen 4에 수렴, Kp_xy≈0.00126, Kd_xy≈3.4e-05 | gen 2에 수렴, Kp_tau≈0.70 |
| 2-A 부트스트랩(N=300) | 79.0% | 98.7% |
| 2-B diffusion(N=60, 새 시드) | 85.0% (+6.0%p) | 100.0% (+1.3%p) |
| 4단계 filter_episodes(양팔 동시 실행) | **78.3%** (47/60) | **100.0%** (30/30) |
| 5단계 언어 라벨링 | 47/47 완료 | 30/30 완료 |

(위 peg_in_hole 수치는 EE kp=20000으로 교정한 뒤 재실행한 것 — 2-b절의
kp 버그 수정 참고. 게인 탐색을 씬 60개로 늘려 돌린 값이라 이전에 40개로
봤던 85.0%와 표본이 달라 직접 비교는 어렵지만, 3단계 이전(단일팔) 84.0%와
같은 수준이다.) 두 태스크 다 저장된 episode npz에 `left_arm`(Stabilizer
궤적)과 `right_arm`(Actuator 궤적+게인+force/torque)이 함께 들어있는 걸
실측 확인했다 — 게인 탐색(0/2단계)이 이제 처음부터 "Stabilizer가 붙어
있는" 물리로 이뤄지므로, 3단계가 성공률 자체를 깎지 않고 오히려 이전과
동등한 수준을 유지하면서 진짜 양팔 데이터를 만들어낸다는 뜻이다.

> 회귀 테스트(`tests/test_regression_peg_in_hole.py`) 관련 정직한 참고:
> 그 파일은 여전히 통과하지만, 3단계로 물리 자체(hole freejoint화,
> integrator 교체)가 바뀌었으므로 "3단계 이전과 결과가 같다"는 뜻이
> 아니다 — "2단계 리팩토링 결과물이 그 사이 추가로 안 깨졌다"만 보증한다
> (자세한 설명은 그 파일 docstring에 추가해뒀다).

### 4단계 — 동시 실행 & 필터링 ✅ 완료 (규모를 키워 정식 데이터셋 생성까지 완료)

설계대로 "Actuator + Stabilizer 궤적을 같은 시뮬레이션에서 동시에 재생"한다
(3단계 완료로 실현됨, 위 3단계의 "4. 양팔 통합" 절 참고). 이번에 200개+
규모로 다시 돌려서 실제로 쓸 만한 학습 데이터셋을 만들었다.

- 구현 위치: `pipeline/filter_episodes.py`
- 흐름: 1단계로 씬 샘플링 → 2-B diffusion으로 그 씬의 Actuator 게인 생성
  → `env.run_episode()`가 같은 물리에서 Stabilizer(3단계)와 Actuator를
  동시 실행 → **Actuator 자체의 성공 판정 하나로 필터링**(Stabilizer
  전용 판정은 따로 두지 않는다 -- 두 팔이 같은 MjModel/MjData를 공유해서
  물리적으로 이미 하나로 얽혀 있으므로, Stabilizer가 못 버티면 그 결과가
  곧바로 Actuator의 성공 판정에 반영된다는 걸 3단계에서 실측으로 이미
  확인했다) → 성공한 것만 저장.
- **태스크별 디렉터리 분리**: `--out-dir`를 생략하면 이제
  `./data/episodes/{task}/`에 저장한다(이전엔 `./data/episodes/`
  플랫 구조라 여러 태스크를 순서대로 돌리면 서로 지워졌다).
- **cap_twist의 실제 스텝별 궤적이 비어 있던 문제를 이번에 고쳤다**:
  `sim/cap_twist_env.py`의 `run_episode()`가 스칼라 요약값만 반환해서,
  `right_arm.traj/action/torque`가 전부 빈 배열로 저장되고 있었다(오른팔
  학습 데이터가 사실상 없는 셈이었음) -- peg_in_hole과 같은 키 이름
  (`ee_poses`/`actions`/`torques`)으로 매 스텝 cap 각도/각속도 명령/토크를
  기록하도록 고쳤다. `pipeline/filter_episodes.py`/`episode_io.py`도
  `right_arm.action`(매 스텝 실제 제어 명령, LeRobotDataset의
  `action`에 대응)을 새로 저장/복원하도록 확장했다 -- 이것도 이전엔
  아예 저장되지 않고 있었다.
- 저장 포맷은 여전히 `pipeline/episode_io.py`가 정의한 npz 구조다:
  `{"task", "left_arm": {"traj","role":"stabilizer"} (있으면만),
  "right_arm": {"traj","action","gains","force","torque","role":"actuator"},
  "scene_config", "success", "language": {...} (5단계에서 채움)}`.
- 실패 에피소드는 2-A/2-B와 달리 **보관하지 않는다**(완성 데이터로 못
  쓰므로).

**실제 실행 결과** (`--scene-seed 42 --sample-seed 0`, 재현성 고정):

| | peg_in_hole | cap_twist |
|---|---|---|
| 시도한 씬 | 280 | 210 |
| 성공(=최종 에피소드 수) | **232개** (82.9%) | **209개** (99.5%) |
| 저장 위치 | `data/episodes/peg_in_hole/` | `data/episodes/cap_twist/` |

### 5단계 — LLM 기반 언어 라벨링 ✅ 완료 (템플릿 기반, role_labels 포함으로 확장)

원래 설계는 "성공한 궤적의 force/torque profile을 LLM에 넘겨 자연어
지시문으로 변환"이지만, 지금도 **템플릿 기반**으로 구현돼 있다(이유는
아래). 이번에 `episode["language"]`를 문자열이 아니라 dict로 확장했다
-- ARD-VLA의 role classifier(어느 팔이 Actuator/Stabilizer인지 구분하는
모델) 학습에 `role_labels`가 필요해서다:

```python
episode["language"] = {
    "task_instruction": "오른손으로 peg를 구멍에 살짝 삽입하라.",
    "role_labels": {"right_arm": "actuator", "left_arm": "stabilizer"},
    "quantity_target": 2.0,   # cap_twist만(회전수), peg_in_hole은 None
    "direction": "cw",         # cap_twist만, peg_in_hole은 None
}
```

`role_labels`는 `episode`에 `left_arm`이 있는지(3단계 Stabilizer 지원
태스크인지)로 자동 결정되므로 태스크마다 따로 지정할 필요가 없다.
`pipeline/episode_io.py`는 이 dict를 JSON 문자열로 직렬화해서 저장한다
(scene_config와 같은 패턴).

- 구현 위치: `pipeline/language_labeling.py`
- **Claude API 대신 템플릿을 쓴 이유**(유지): 두 태스크가 표현할 정보
  조합(강도 하나, 또는 방향+수량)이 여전히 단순해서 템플릿으로 충분하고,
  오프라인/재현 가능해야 하기 때문 -- 조합이 훨씬 다양해지는 태스크가
  생기면 그때 바꾸는 게 맞다고 본다.
- 강도어(살짝/적당한 힘으로/힘있게)는 **그때그때 episodes의 force_max
  33/66 백분위수**로 정한다.

**실제 실행 결과**:

| | peg_in_hole (232개) | cap_twist (209개) |
|---|---|---|
| force_max 범위 | [0.5, 148.7]N | (없음 -- 이 태스크는 강도어를 안 씀) |
| 강도 경계(33/66 백분위) | 22.1N / 53.1N | — |
| 강도/방향 분포 | gentle 77 / normal 76 / firm 79 | direction/quantity 조합별로 분산 |
| 샘플 | "오른손으로 peg를 구멍에 힘있게 삽입하라." | "뚜껑을 시계방향으로 2.0바퀴 돌려라." |
| role_labels | `{"right_arm":"actuator","left_arm":"stabilizer"}` (전체 232개 동일) | 〃 (전체 209개 동일) |

두 태스크 다 **232/232, 209/209 전부 언어 라벨링까지 완성**됐다(5단계
에서는 탈락이 없다 -- 이미 성공한 에피소드만 들어오므로 당연함).

### LeRobotDataset 변환 — `pipeline/to_lerobot.py` (신규)

완성된 에피소드를 LeRobotDataset 호환 포맷으로 변환한다. **"호환"이지
"동일"은 아니다**(정직하게 밝힘) -- 이 환경엔 `lerobot`/`pandas`/`pyarrow`
가 없어서, 실제 parquet 대신 **같은 컬럼/메타데이터 스키마를 가진
npz**로 프레임 데이터를 저장한다. 디렉터리 레이아웃(`meta/info.json`,
`meta/episodes.jsonl`, `meta/tasks.jsonl`, `data/chunk-000/episode_*.npz`)
과 필드 이름(`observation.state`, `action`, `episode_index`,
`frame_index`, `timestamp`, `task_index`)은 LeRobotDataset v2.x를
최대한 따랐다 -- pyarrow가 있는 환경으로 옮기면 이 npz들을 데이터프레임화
해서 parquet으로 다시 쓰기만 하면 된다.

- **태스크별 별도 데이터셋**: peg_in_hole/cap_twist는 관측/행동 차원
  자체가 달라서(전자 7/3차원, 후자 4/1차원) `data/lerobot/{task}/`에
  독립 데이터셋을 만든다.
- **state/action 구성**: state = 오른팔 관측(right_arm.traj) + 왼팔
  관측(left_arm.traj, 있으면) 이어붙인 벡터. 왼팔 궤적은 접근 단계가 있어
  더 기므로, 오른팔과 동시에 기록된 **뒤쪽 구간만 잘라서** 길이를 맞춘다.
  action = right_arm.action(매 스텝 실제 제어 명령).
- **fps=100**: 두 태스크 다 제어 주기 DT=N_SUBSTEPS(5)×timestep(0.002s)
  =0.01s로 동일(실측 확인).

**실제 변환/검증 결과**:

| | peg_in_hole | cap_twist |
|---|---|---|
| 총 에피소드 | 232 | 209 |
| 총 프레임 | 30,842 | 82,758 |
| state_dim (names) | 7 (`right_ee_x/y/z`, `right_wrist_rotate`, `left_ee_x/y/z`) | 4 (`cap_angle_rad`, `left_ee_x/y/z`) |
| action_dim (names) | 3 (`delta_x/y/z`) | 1 (`omega_command_rad_s`) |
| 고유 task_instruction 수 | 9 | 16 |

episode 0/중간/마지막을 `load_lerobot_episode()`로 직접 로드해서 shape이
`(length, state_dim)`/`(length, action_dim)`으로, `meta/episodes.jsonl`의
`length`와 정확히 일치하는 것까지 실측 확인했다.

### 파이프라인 오케스트레이션: `run_pipeline.py`

0→1→2-A→2-B→4→5단계를 한 번에 실행하는 드라이버. 0/2단계 산출물(seed,
bootstrap dataset, diffusion 체크포인트)은 이미 있으면 재사용하고
`--force-seed`/`--force-bootstrap`/`--force-diffusion`로만 다시 만든다.
`pipeline/to_lerobot.py`는 아직 이 드라이버에 편입하지 않았다(4/5단계
출력을 그대로 입력받는 후처리 단계라 필요할 때 따로 돌리면 된다).

**최종 산출물**: peg_in_hole **232개**, cap_twist **209개**, 총 441개의
`left_arm`+`right_arm`+`language`(role_labels 포함)가 모두 채워진 완성
에피소드 (`data/episodes/{task}/episode_*.npz`), 그리고 각각의
LeRobotDataset 호환 변환본(`data/lerobot/{task}/`).

## 검증 태스크

- **1차**: peg-in-hole(위치보정) ✅, 나사 조이기(토크제어) ⚠️ — 두 단순
  태스크로 파이프라인 자체가 작동하는지 검증. 애초 계획은 "뚜껑돌리기"였는데
  브레인스토밍 끝에 ARD-VLA의 "Actuator가 도구를 쥔다"는 프레이밍에 더
  맞는 나사 조이기로 정했다(peg-in-hole과 같은 VX300s 팔을 재사용,
  드라이버를 픽업 스탠드에서 집는 것부터 시작해서 토크 리미터 컨트롤러로
  완전히 조여 넣는다). 모델·컨트롤러·게인 탐색(CMA-ES)까지는 검증
  완료했지만, peg-in-hole처럼 0→1→2-A→2-B→4→5단계 파이프라인에 편입하는
  작업은 아직 안 했다 — 자세한 내용은 `ard-gen/README.md`의 "검증 태스크
  2: 나사 조이기" 절 참고.
- **추후 확장**: GPU/실물 로봇 확보 시 0단계를 RL(RLDC 스타일)로 교체해서
  드릴 조작 같은 복잡한 다단계 태스크로 확장

## 현재 저장소 상태와의 매핑

파이프라인은 `sim/task_registry.py`의 `TASK_REGISTRY`로 태스크 무관화돼
있어서(모든 스크립트가 `--task`를 받음), 아래는 peg_in_hole/cap_twist
둘 다에 코드 변경 없이 적용된다.

| 파이프라인 단계 | 관련 코드 | 상태 |
|---|---|---|
| 0단계 (Seed 확보) | `optimize/cma_search.py`, `sim/{peg_in_hole,cap_twist}_env.py` | ✅ 완료 |
| (스코프 외) 부트스트랩 정책 실험 | `bootstrap/` | ✅ 완료 (2단계와는 무관, 별도 유지) |
| 1단계 (공유 씬 설정) | `sim/{peg_in_hole,cap_twist}_env.py`의 `sample_scene_config`/`to_sim_scene_config` | ✅ 완료 |
| 2-A (게인 부트스트래핑) | `pipeline/bootstrap.py`, `data/bootstrap/` | ✅ 완료 |
| 2-B (diffusion) | `pipeline/diffusion_gains.py`, `data/bootstrap/diffusion_gains.pt` | ✅ 완료 |
| 3단계 (Stabilizer 기하 변환) | `sim/stabilizer.py`, `pipeline/stabilizer_augment.py` | ✅ 완료 |
| 4단계 (동시 실행 & 필터링) | `pipeline/filter_episodes.py`, `pipeline/episode_io.py`, `data/episodes/{task}/` | ✅ 완료 (양팔 동시 실행, 232+209개 정식 데이터셋) |
| 5단계 (언어 라벨링) | `pipeline/language_labeling.py` | ✅ 완료 (템플릿 기반 + role_labels, LLM 전환은 보류 -- 이유는 해당 절 참고) |
| LeRobotDataset 변환 | `pipeline/to_lerobot.py`, `data/lerobot/{task}/` | ✅ 완료 (npz 기반 호환 스키마 -- 정확한 범위는 해당 절 참고) |
| 파이프라인 오케스트레이션 | `run_pipeline.py` | ✅ 완료 (--task로 태스크 선택, to_lerobot은 미편입) |

지금 저장소(`ard-gen/`)는 peg_in_hole/cap_twist 두 태스크 모두 **왼팔
(Stabilizer)+오른팔(Actuator)이 같은 물리 시뮬레이션에서 동시에 실행되는
0→1→2-A→2-B→4→5단계 + LeRobotDataset 변환까지** 정식 규모(232/209개
에피소드)로 끝까지 도는 것을 검증 완료한 상태다. 남은 것: 5단계를 실제
LLM 기반으로 바꾸는 것(현재는 표현할 정보 조합이 단순해 템플릿으로
충분하다고 판단해 보류 중), 진짜 `lerobot`/`pandas`/`pyarrow` 패키지가
있는 환경에서의 parquet 변환 검증(지금은 스키마만 호환, 위 LeRobotDataset
절 참고), 그리고 peg_in_hole 3단계에서 발견된 "고부하 경계선 케이스
11/142건" 잔여 실패의 근본 개선(현재는 정직하게 실패로 남겨둠, 위 3단계
절 참고).

## TASK_REGISTRY["peg_in_hole"]을 VX300s에서 OpenArm 양팔로 교체

위 232개 생산 데이터셋은 **VX300s 단일팔(Actuator)+가상 EE Stabilizer**
(`assets/peg_in_hole.xml`, `sim/peg_in_hole_env.py:PegInHoleEnv`) 기준이었다.
이 파일 자체와 검증 결과는 여전히 유효하고 코드도 그대로 남아있지만("아니
로봇이 왜 다시 바뀐거야" 피드백으로 드러난 것처럼, 이 VX300s 라인과 별도로
진행되던 `assets/peg_in_hole_bimanual_openarm.xml`/`sim/
peg_in_hole_bimanual_openarm_sim.py`(OpenArm 양팔, commit 573291e부터) 두
라인이 한 번도 합쳐진 적이 없었다), **사용자가 명시적으로 OpenArm 라인으로
교체를 요청**해서 `sim/task_registry.py`의 `TASK_REGISTRY["peg_in_hole"]`을
`sim.peg_in_hole_openarm_env:PegInHoleOpenArmEnv`로 바꿨다. 아래는 그
전환과 재생성 기록이다.

### 근본적 차이 (VX300s -> OpenArm)

- **팔 자유도**: 6-DOF(VX300s) -> 7-DOF(OpenArm), 두 팔 받침대 간격 6.2cm로
  훨씬 좁음.
- **왼팔(Stabilizer)이 generic 클래스가 아니다**: VX300s는 `sim/
  stabilizer.py`의 가상 EE(3-슬라이드 조인트, IK 불필요)라 어떤 태스크든
  `object_body`/`grasp_offset` 두 필드만 주면 재사용된다. OpenArm은 왼팔이
  **진짜 7-DOF 팔**이고 hole_socket을 쥔 채 강화된 위치 게인으로 태스크
  내내 고정돼 있다 -- 이 고정이 XML/sim 모듈에 이미 구워져 있어서
  `tasks/peg_in_hole.yaml`에 `stabilizer` 절이 없다(`sim/
  peg_in_hole_openarm_env.py` 모듈 docstring 참고). 대신 매 스텝 왼팔 EE
  위치를 기록해서 스키마 일관성(episode의 `left_arm`, role classifier용)은
  그대로 유지한다.
- **오른팔 제어**: admittance(접촉힘 PD) -> 역동역학(computed-torque, 매
  스텝 `mj_fullM`으로 실제 관성 반영) + 위치 오차 기반 xy 타겟팅. 접촉힘이
  거의 항상 0이라 admittance 자체가 신호를 못 받았기 때문(`sim/
  peg_in_hole_bimanual_openarm_sim.py` 모듈 docstring에 실패한 시도들까지
  전부 기록해뒀다 -- gravcomp 누락, 널스페이스 표류, 관절 한계 눌어붙음,
  타이밍 불일치 가설 등).
- **게인의 의미 자체가 다르다**: `Kp_xy`/`Kd_xy` 이름은 같지만 VX300s는
  접촉힘(N)에, OpenArm은 위치 오차(m)에 곱하는 게인이라 스케일이 전혀
  다르다(`tasks/peg_in_hole.yaml`의 bounds가 0.00002~0.003 -> 0.001~0.5로
  바뀐 이유).
- **condition_fields 4차원(7 -> 4)**: OpenArm의 hole 위치는 왼팔의 고정
  자세로 결정되고 scene_config로 옮길 자유도가 아니라서 `hole_pose`(3차원)
  가 조건 벡터에서 빠졌다.

### 재생성 실행 기록 (0→1→2-A→2-B→4→5→LeRobotDataset)

기존 `data/bootstrap/peg_in_hole_*`, `data/episodes/peg_in_hole/`,
`data/lerobot/peg_in_hole/`는 전부 VX300s 기준이라 그대로 두면 새 Env와
안 맞는다(둘 다 `.gitignore`돼 있어 커밋 오염은 없었지만, 로컬에 남아있으면
헷갈리므로) -- 아래 순서로 지우고 새로 생성했다:

1. `optimize/cma_search.py --task peg_in_hole` -- **1세대 만에 조기 종료**
   (reward=48.91 >= threshold 45.0). 이미 알려진 좋은 게인(아래 참고) 근방
   탐색 공간이 매우 관대하다는 뜻.
2. `pipeline/bootstrap.py --task peg_in_hole --n-trials 1000` -- seed 게인
   주변 무작위 노이즈(0.5~2.0배)로 1000회 실행, **성공률 99.1%**(VX300s는
   기록에 따르면 훨씬 낮았다 -- OpenArm 버전이 실측으로 훨씬 로버스트함).
3. `pipeline/diffusion_gains.py --task peg_in_hole` -- 조건부 diffusion
   학습(500 epoch), 검증(새 무작위 씬 100개): **성공률 100%**(부트스트랩
   대비 +0.9%p).
4. `pipeline/filter_episodes.py --task peg_in_hole --n-scenes 250` -- **249/250
   성공(99.6%)** -> `data/episodes/peg_in_hole/`.
5. `pipeline/language_labeling.py --task peg_in_hole` -- 249개 전부 언어
   라벨 완성.
6. `pipeline/to_lerobot.py --task peg_in_hole` -- **249개 에피소드, 38,784
   프레임**, state_dim=7/action_dim=3(VX300s와 동일 차원 -- 우연이 아니라
   `get_ee_pose()`가 같은 `[x,y,z,wrist_rotate]` 형태를 반환하도록 맞췄기
   때문), task_instruction 9종.

### 정직하게 밝히는 한계: force_max에 실질적 변화가 없다

5단계 로그에 `force_max 범위 [0.5, 0.5]N`이 그대로 찍힌다 -- 249개 에피소드
전부 최대 접촉힘이 사실상 peg 자체 무게(0.4905N)에 고정돼 있고, "삽입
강도"(gentle/normal/firm) 라벨은 33/66 백분위 경계가 둘 다 0.5N이라
**진짜 힘 차이가 아니라 부동소수점 수준의 임의 분할**이다. 원인은 위
"근본적 차이" 절의 제어 방식 전환 자체다 -- 오른팔이 이제 hole의 실제
좌표를 직접 타겟팅해서 벽에 세게 부딪히기 전에 정렬을 마치므로, VX300s
버전(admittance가 접촉힘 피드백으로 동작해서 힘 프로파일에 실제 편차가
있었음, force_max 범위 [0.5, 148.7]N)과 달리 강한 접촉 자체가 거의
일어나지 않는다. 그래서 **OpenArm peg_in_hole 데이터셋의 "삽입 강도"
언어 라벨은 통계적으로 의미가 없다**(같은 "적당한 힘으로"라는 문구가
힘과 무관하게 붙는다) -- 다음에 이어서 개선할 사람에게: 강도를 의미 있게
만들려면 (a) 제어 자체에 의도적인 힘 변주를 넣거나(admittance 성분을
다시 살리거나), (b) 강도 어휘를 force 대신 다른 신호(예: 접근 속도,
Kp_xy 크기)로 바꾸는 것을 고려할 만하다.

### 환경 함정: `MUJOCO_GL=osmesa` + PyTorch `Adam` 세그폴트

`torch.optim.Adam`을 처음 생성할 때 PyTorch가 내부적으로 `triton` 임포트를
시도하는데(`torch/_dynamo`), 이 컨테이너에서 **`MUJOCO_GL=osmesa`를 설정한
상태로 `mujoco`를 먼저 임포트한 뒤** Adam을 생성하면 세그폴트가 난다(추정
원인: OSMesa의 소프트웨어 래스터라이저와 triton이 서로 다른 버전의
LLVM 공유 라이브러리를 동시에 로드해서 생기는 ABI 충돌 -- 확실친 않음,
재현만 확인). 렌더링을 안 하는 스크립트(`optimize/cma_search.py`,
`pipeline/bootstrap.py`, `pipeline/diffusion_gains.py`,
`pipeline/filter_episodes.py`)는 애초에 `MUJOCO_GL`을 설정할 필요가 없으므로
(오프스크린 렌더러를 안 만듦), **`render_episode.py`처럼 실제로 프레임을
캡처하는 스크립트에서만 `MUJOCO_GL=osmesa`를 쓰고 나머지는 안 쓰는 것**이
가장 간단한 회피책이다(실측: 두 조합 다 정상 동작 확인).

## [레거시] tacker(타카) 최초 버전 -- 일반 Stabilizer(가상 EE) 기반

> **이 절 전체가 대체됐다.** peg_in_hole이 VX300s에서 OpenArm으로 교체된 것과
> 같은 이유로, tacker도 사용자 요청에 따라 아래 "tacker(타카), OpenArm 양팔
> 버전" 절의 구조로 교체했다 -- `TASK_REGISTRY["tacker"]`는 더 이상 여기 설명된
> `sim.tacker_env:TackerEnv`(3-슬라이드 가상 Stabilizer, `sim/stabilizer.py`
> 재사용)를 가리키지 않는다. 코드/자산(`assets/tacker.xml`, `sim/tacker_env.py`,
> `tests/test_tacker_task.py`의 옛 버전)은 참고용으로만 저장소에 남겨뒀다 --
> 아래 절은 설계 배경 기록으로서 그대로 둔다.

## 신규 태스크: tacker(타카) -- "위치 정확도 + 1회성 발사"

peg_in_hole/cap_twist는 둘 다 Actuator가 에피소드 내내 연속적으로 힘/토크를
조절해야 하는 태스크였다(admittance PD 루프). tacker는 의도적으로 그
반대다 -- Actuator는 목표 지점까지 정확히 접근하기만 하면 되고, 도달하는
순간 자동으로 발사되는 **이산적 이벤트**다. 그래서 이 태스크가 검증하는
물리적 핵심이 Actuator가 아니라 **Stabilizer(발사 반동 흡수)**로 넘어간다
-- 구현 세부는 `assets/tacker.xml`, `sim/tacker_env.py`, `tasks/tacker.yaml`
상단 docstring 참고, 여기서는 설계 배경과 실측 결과만 정리한다.

### 설계 요약

- **workpiece**: peg_in_hole의 hole_socket과 같은 패턴(freejoint, 테이블 위에
  얹힌 채 시작). **nail_site**: workpiece에 강체로 붙은 순수 목표점(로컬
  오프셋을 reset()이 무작위화). **tacker_ee**: Actuator의 가상 엔드이펙터
  (stabilizer_ee와 자매 설계, 같은 kp=20000 -- 이 강성 자체는 탐색 대상이
  아니다).
- **게인 1개(Kp_approach)**: 접근 오차에 곱하는 비례 게인 하나뿐이다 --
  peg_in_hole/cap_twist의 게인이 "매 스텝 관측되는 물리 피드백에 반응"했던
  것과 달리, 접근 중에는 반응할 물리 신호가 없어서(발사 전까지 접촉/저항
  없음) 순수 기하학적 접근 속도 프로파일만 정한다.
- **발사 트리거**: 사람이 "발사" 액션을 주지 않는다 -- EE와 nail_site 거리가
  FIRE_TOLERANCE_M(3mm) 이내로 들어오는 스텝에서 자동 발사(에피소드당 정확히
  1회).
- **반동**: workpiece의 freejoint 선속도에 직접 velocity kick을 가한다
  (impulse의 이상화). 수평 성분이 주고 수직은 작게 섞었다 -- 순수 수직
  킥은 중력+테이블 접촉만으로 금방 멈춰서 Stabilizer 유무가 거의 안
  갈리기 때문(이 태스크를 만든 목적 자체가 "Stabilizer 유무로 성공률이
  갈리는가"라서, 그게 실제로 갈리는 방향으로 설계해야 의미가 있다).
- **성공 조건**: 발사됐고(fired) + 발사 후 workpiece 변위가
  SUCCESS_DISPLACEMENT_M(6mm) 이내.

### 실측 버그 발견: freejoint 바디의 XML `<sensor>` force/torque는 항상 0이다

처음엔 peg_in_hole의 peg_tip_site와 같은 패턴으로 nail_site에
`<sensor><force/><torque/></sensor>`를 붙였는데, 실측해보니(`xfrc_applied`류
직접 검증) workpiece가 실제로 밀리고 있어도 센서 값이 수치 잡음 수준(1e-16
이하)으로 항상 0이었다. 원인: MuJoCo의 site force/torque 센서는 "그 바디를
부모(world)에 연결하는 조인트를 통해 전달되는 구속력"을 재는데, freejoint는
6DOF가 전부 자유(구속 없음)라서 정의상 그 조인트를 통해 전달되는 구속력이
항상 0이다 -- peg_in_hole.xml의 peg처럼 조인트 없이 부모에 강체로 고정된
자식 바디였다면 그 "고정" 자체가 암묵적 구속이라 센서가 정상 작동하지만,
freejoint 바디에서는 구조적으로 작동할 수 없다.

**중요한 파생 발견**: OpenArm peg_in_hole의 `peg_force` 센서도 `peg`가
freejoint 바디(`peg_free`)라 같은 구조적 결함을 겪고 있을 가능성이 높다 --
이전 절("정직하게 밝히는 한계: force_max에 실질적 변화가 없다")에서 그
원인을 "제어 방식이 접촉 자체를 거의 안 만들어서"라고 설명했는데, 이제 보니
그 설명은 불완전했을 수 있다(제어 방식 문제가 실제로 있더라도, 설령 강한
접촉이 있었어도 애초에 그 센서로는 안 잡혔을 것이라는 뜻) -- 두 원인을
분리하지 않았다(범위 밖이라 이번엔 손대지 않았다), 다음에 그 태스크의
force_max를 의미 있게 만들려면 이 센서 자체부터 OpenArm peg처럼 freejoint인
바디에서 qfrc_constraint 방식으로 바꿔야 할 가능성을 먼저 확인할 것.

tacker에서는 XML 센서를 아예 없애고, `sim/tacker_env.py`가
`data.qfrc_constraint`(그 바디의 자유도에 실제로 작용하는 모든 구속력의
합 -- 접촉+equality 전부 포함, 이 프로젝트가 Stabilizer kp=20000을 검증할 때
이미 썼던 것과 같은 종류의 진단량)를 매 스텝 직접 읽어서 로깅한다. 이걸로
바꾸니 실측값이 물리적으로 올바르게 나왔다(정지 상태 ≈0.15kg*9.81≈1.47N,
발사 순간 스파이크 최대 수 N).

### 검증 1: CMA-ES가 게인 1개로도 정상 수렴하는가 -- 수렴은 하지만 landscape가 매우 평탄함

1세대 만에 목표 리워드(65.0)에 도달했다(`Kp_approach≈0.50`). Kp_approach를
0.02(하한)부터 3.0(상한)까지 수동으로 스윕해봐도 **전 구간에서 성공** --
차이는 step_count(162 -> 8)와 리워드(68.37 -> 69.91, 0.5 부근이 근소하게
최고)뿐이었다. peg_in_hole/cap_twist의 "너무 작아도 커도 확실히 실패하는"
뚜렷한 이중 실패 모드와 달리 이 게인은 얕은 최적점만 가진 평탄한 문제다 --
이건 버그가 아니라 설계 의도와 일치하는 결과다(이 태스크의 물리적 핵심이
애초에 Actuator 게인이 아니라 Stabilizer이므로, Actuator 쪽 탐색이 쉬운 게
당연하다). 부트스트랩(1000 trial, seed 게인 ±0.5~2배 노이즈) 성공률
100.0%, diffusion 검증(새 무작위 씬 100개) 성공률 100.0%(둘 다 Stabilizer
있는 상태 기준).

### 검증 2: Stabilizer 유무 성공률 차이 (이 태스크를 만든 목적 자체에 대한 검증)

같은 씬 시퀀스(N=150, 시드 고정) + 같은 고정 게인(Kp_approach=0.5014, CMA-ES
seed 값)으로 Stabilizer 있음/없음만 바꿔 비교했다:

| | Stabilizer 있음 | Stabilizer 없음 |
|---|---|---|
| 성공률 | **100.0%** (150/150) | **37.3%** (56/150, 95% CI 대략 ±7.7%p) |
| 발사 후 변위(중앙값) | 0.008mm | 8.857mm |
| 발사 후 변위(90백분위) | 0.015mm | 14.891mm |
| 발사 후 변위(최대) | 0.024mm | 18.618mm |

**차이가 확실하게 난다** -- Stabilizer가 있으면 weld(kp=20000)가 반동을
거의 완전히 흡수해서(변위가 0.01mm 스케일) 사실상 항상 성공하고, 없으면
workpiece가 테이블 마찰만으로 버티다 보니 중앙값 기준 1000배 가까이 더
밀려서(수 mm~2cm) 60% 이상이 실패한다. 이 태스크를 추가한 목적(Actuator가
아니라 Stabilizer가 물리적 핵심이라는 걸 데이터로 보여주는 것)이 실제로
성립함을 확인했다 -- `tests/test_tacker_task.py`에 더 작은 N(40)으로 같은
검증을 회귀 테스트로 남겼다.

### 파이프라인 실행 기록

`optimize/cma_search.py` → `pipeline/bootstrap.py`(1000 trial) →
`pipeline/diffusion_gains.py`(500 epoch) → `pipeline/filter_episodes.py`
(250 scenes) → `pipeline/language_labeling.py` → `pipeline/to_lerobot.py`
순서로 전부 실행했다:

- **필터링**: 250/250 성공(100.0%, `data/episodes/tacker/`).
- **언어 라벨링**: force_max 범위 [1.9, 20.4]N(qfrc_constraint 수정 이후 --
  peg_in_hole OpenArm과 달리 실제 편차가 있는 값이라 강도 라벨이 의미
  있다), 강도 분포 gentle 83 / normal 82 / firm 85. 샘플: "표시된 지점을
  조준해서 힘있게 발사하라." role_labels 250개 전부
  `{"right_arm":"actuator","left_arm":"stabilizer"}` 동일.
- **LeRobotDataset 변환**: 250개 에피소드, 9,601프레임, state_dim=6(오른팔
  3 + 왼팔 3 -- wrist 회전 없음, 아래 참고), action_dim=3, task_instruction
  9종. `data/lerobot/tacker/`.

### wrist 회전을 안 둔 이유 (3차원 action으로 충분한지 검토한 결론)

이 태스크는 타카를 표면에 수직으로 대고 누르는 동작이라, peg_in_hole(삽입축
정렬)이나 cap_twist(회전 진행도)처럼 "각도가 성공 조건에 들어가는" 요소가
전혀 없다 -- 성공은 순수하게 위치(발사 지점 도달) + 발사 후 변위로만
정의된다. 그래서 action/state 모두 3차원 위치만으로 충분하다고 판단했고,
실제로 4차원을 추가할 이유가 하나도 나오지 않았다(위 검증 결과가 이미
3차원만으로 100% 성공률을 보여준다).

### render_episode.py 일반화

`--task` choices를 하드코딩된 2개 목록 대신 `sim.task_registry.list_tasks()`
로 바꿨다(태스크가 늘어날 때마다 이 파일을 고칠 필요가 없어짐) --
`_camera_for()`도 이름 -> 카메라 dict로 바꿔서 tacker의 `top_cam`을 추가로
등록했다.

## tacker(타카), OpenArm 양팔 버전 -- 왼팔이 진짜 팔로 workpiece를 쥐도록 재설계

peg_in_hole이 VX300s에서 OpenArm 양팔로 교체된 뒤(위 절 참고), 사용자가
"tacker도 일반 Stabilizer(가상 EE) 패턴을 다시 쓰지 말고 peg_in_hole의
OpenArm 구조를 그대로 이어서 쓰라"고 명시적으로 요청했다 -- `TASK_REGISTRY["tacker"]`
를 `sim.tacker_openarm_env:TackerOpenArmEnv`로 교체했다. 설계 철학(오른팔=
위치 접근+자동 발사, 왼팔=반동 흡수가 물리적 핵심)은 이전 버전과 동일하고,
바뀐 건 **왼팔/오른팔을 구현하는 방식**뿐이다.

### 설계: grasp anchor/home 자세를 새로 찾지 않고 그대로 재사용

작업 전에 사용자에게 확인 질문을 했다 -- "왼팔이 workpiece를 쥐는 위치,
오른팔이 tacker_tool을 쥐는 위치, 양팔의 home 자세를 `assets/
peg_in_hole_bimanual_openarm.xml`에서 이미 검증된 값(hole_socket/peg의 grasp
anchor·home pose)을 그대로 재사용할지, tacker 전용으로 새로 찾을지" -- 사용자가
**재사용**을 선택했다. 근거: 그 값들은 "왼팔이 어떤 각도로 뭔가를 쥔 채 오른팔이
그 위 6cm(`_HOVER_GAP_M`)에서 접근한다"는 순수 기하학적 관계만 인코딩하고
있어서, 쥐는 대상의 형상(hole vs 평평한 workpiece)이나 오른팔이 쥔 도구의
용도(peg 삽입 vs 타카 발사)와 무관하게 유효하다. 그래서 `assets/
tacker_openarm.xml`은 `hole_grasp`/`peg_grasp` weld의 relpose, `_HOME_QPOS`/
`_LEFT_ARM_HOME_QPOS`, gravcomp 오버라이드, vendor 액추에이터 무력화, 관절별
강화 게인(pos_left_j*/pos_right_j*)을 **숫자 그대로** 물려받았다 -- 바뀐 건
hole_socket(벽+바닥)을 평평한 workpiece 블록으로, peg를 tacker_tool로 재해석한
것뿐이다(같은 body pos/quat, 같은 지오metry 재사용). 이 선택 덕분에 원래
peg_in_hole_bimanual_openarm.xml이 5~6차례 반복 보정했던 grasp anchor
재조정/home 자세 CMA-ES 재탐색을 전혀 다시 할 필요가 없었다.

오른팔 제어는 peg_in_hole_bimanual_openarm_sim.py의 resolved-rate Jacobian
IK(널스페이스 투영 + 관절 한계 회피) + computed-torque(매 스텝 `mj_fullM`)
구조를 코드 그대로 재사용하되, 그 위에 얹는 태스크 루프만 admittance+
z-rate 스케줄에서 순수 위치오차 비례 제어(`Kp_approach`, 일반 Stabilizer
버전 tacker와 같은 설계 철학)로 바꿨다.

### 발견 1(실측 버그): tacker_tool과 workpiece의 실제 충돌을 꺼야 했다

peg의 지오metry(샤프트+구형 tip, `contype=1`)를 그대로 재사용했는데, 이건
peg_in_hole에서는 hole 벽과의 접촉 자체가 물리의 핵심이라 당연히 켜져
있어야 하지만, tacker는 "발사"가 순수 스크립트 이벤트(velocity kick)라서
tacker_tool이 workpiece에 실제로 부딪히면 **의도치 않은 접촉 충격력**이
추가로 생긴다. 처음 돌려봤을 때 `recoil_strength=0`(반동 없음)에서도
workpiece가 40~60mm씩 튕겨나가서 recoil 계산 자체가 잘못됐나 의심했는데,
recoil_strength를 0부터 스윕해도 displacement가 거의 안 바뀌는 걸 보고
(반동과 무관한 원인이라는 뜻) 실제 원인을 찾았다 -- `<contact><exclude
body1="tacker_tool" body2="workpiece"/></contact>`로 껐다.

### 발견 2(더 중요한 실측 버그): `integrator="implicitfast"` 누락으로 정지
### 홀드만으로도 20mm+ 드리프트 -- 원본 OpenArm peg_in_hole도 같은 결함일 가능성

위 접촉 버그를 고치고도 `recoil_strength=0`에서 여전히 약 9~11mm의
displacement가 남았다. 오른팔을 전혀 움직이지 않고 **왼팔만 고정 자세로
300ms(60스텝) 정지 홀드**시키는 격리 테스트를 해보니, workpiece가 그것만으로
20mm 넘게 드리프트했다 -- 반동/오른팔과 완전히 무관한, 왼팔의 정지 홀드
자체의 문제였다. 원인을 찾다가 `assets/peg_in_hole_bimanual_openarm.xml`의
`<option>`에 `cone="elliptic" impratio="10"`만 있고 **integrator가 지정돼
있지 않다**(기본값=세미암시적 오일러)는 걸 발견했다 -- 이건 이 저장소가
`assets/peg_in_hole.xml`(VX300s)과 `screw_driving_bimanual_openarm.xml`에서
**이미 겪고 고쳤던, 문서화까지 해둔** 바로 그 버그다: "강한 위치 액추에이터
+ 항상 활성인 weld" 조합에서 기본 오일러 적분기가 서서히 에너지를 새게 해서
드리프트를 만든다. `assets/tacker_openarm.xml`에 `integrator="implicitfast"`를
추가하니 같은 정지 홀드 테스트에서 드리프트가 **20mm대 -> 5μm 수준**으로
완전히 사라졌다(왼팔 강화 게인/vendor 게인 둘 다 동일하게 해결됨).

**파생 시사점(정직하게 밝힘)**: 현재 `TASK_REGISTRY["peg_in_hole"]`이 쓰는
`assets/peg_in_hole_bimanual_openarm.xml`도 이 옵션이 그대로 빠져 있다 --
그 태스크 개발 당시 겪었던 극심한 드리프트 논의("2000스텝까지 34~48mm,
4000스텝에 272mm", "45~75mm 범위에서 진동")가 전부 "관절 한계 여유 부족"과
"결합 동역학"으로만 설명됐는데, 이 더 단순한(그리고 이미 두 번이나 겪었던)
원인이 최소한 일부는 기여했을 가능성이 있다. 이번 작업 범위 밖이라 그
파일은 손대지 않았지만, 다음에 그 태스크의 드리프트/정밀도를 더 개선하고
싶은 사람은 이 옵션 추가부터 실측해볼 것을 권한다.

### recoil_strength 스케일 재보정

일반 Stabilizer 버전(순수 3-슬라이드 가상 EE, 관성 0.1kg, kp=20000)은
recoil_strength 0.15~0.45(m/s 스케일 velocity kick)만으로도 뚜렷한 성공/실패
차이가 났지만, 진짜 7-DOF 팔 전체의 분산된 관성/댐핑이 훨씬 크게 개입하는
이 버전에서는 같은 스케일의 kick이 거의 무해했다(위 두 버그를 고친 뒤
실측 스윕: 왼팔 강화 게인 기준 `recoil_strength<=15`는 변위<4mm로 항상
성공, 25 이상부터 threshold(6mm)를 넘기 시작해 130에서 238mm까지 커진다).
그래서 `sample_scene_config()`의 반동 강도 범위를 5~35로 재보정했다 --
"왼팔 강화 게인으로도 성공/실패가 실제로 갈리는" 구간이다.

### 검증 1: CMA-ES가 게인 1개로도 정상 수렴하는가

일반 Stabilizer 버전과 같은 패턴 -- 1세대 만에 목표 리워드(65.0)에 도달했다
(`Kp_approach≈0.174`). 이 게인 자체의 landscape는 여전히 평탄하다(0.02~3.0
전 구간에서 성공, 반동 강도가 고정이라면) -- 이 태스크의 물리적 핵심이
애초에 이 게인이 아니라 왼팔이라는 설계 의도와 일치한다. 부트스트랩(1000
trial, seed 게인 ±0.5~2배 노이즈) 성공률 **72.5%**(725/1000) -- 재보정된
반동 강도 범위 덕분에 실제로 실패도 섞여 있는, 이전 버전(항상 100%)보다
훨씬 의미 있는 난이도 분포다. diffusion 검증(새 무작위 씬 100개) 성공률
**77.0%**(+4.5%p 개선).

### 검증 2: 왼팔 강화 게인 유무 (이 태스크를 만든 목적 자체에 대한 검증)

이 버전은 grasp weld가 컴파일 시점부터 항상 활성(왼팔이 처음부터 쥐고
있음)이라, 일반 Stabilizer 버전처럼 "붙였다/안 붙였다"로 비교할 수 없다 --
대신 **왼팔의 강화 위치 게인(`pos_left_j*`, kp=2273.98/87.90/411.75) vs
OpenArm vendor 기본 게인(`assets/openarm/openarm_bimanual.xml`의
`left_joint*_ctrl`, kp=230/190/30, motor_DM8009/DM4340/DM4310)**으로 비교했다.
같은 씬 시퀀스(N=150, 시드 고정) + 같은 고정 게인(Kp_approach=0.1739)으로:

| | 왼팔 강화 게인 | 왼팔 vendor(약한) 게인 |
|---|---|---|
| 성공률 | **73.3%** (110/150, 95% CI 대략 ±7.1%p) | **46.7%** (70/150, 95% CI 대략 ±8.0%p) |
| 발사 후 변위(평균) | 4.387mm | 6.685mm |
| 발사 후 변위(중앙값) | 2.882mm | 6.245mm |

두 신뢰구간이 겹치지 않는다(66.2%~ vs ~54.7%) -- 통계적으로 유의미한 차이다.
일반 Stabilizer 버전(100% vs 37.3%, 변위 1000배 차이)만큼 극적이진 않지만
(그 버전은 순수 스프링 vs 아예 없음이라는 이진 비교였고, 이건 "강한 실제
팔" vs "약한 실제 팔"이라는 더 현실적인 비교라서 차이가 더 점진적이다),
방향과 유의성은 명확하다 -- `tests/test_tacker_task.py`에 더 작은 N(40)으로
같은 검증을 회귀 테스트로 남겼다.

### 파이프라인 실행 기록 (0→1→2-A→2-B→4→5, 3단계 없음)

사용자 요청대로 3단계(Stabilizer) 별도 스텝을 건너뛰었다 -- 왼팔이
`peg_in_hole_openarm`처럼 Env/XML 안에 이미 통합돼 있어서 실행할 별도
단계 자체가 없다.

- **필터링**: 350개 씬 중 252개 성공(72.0%, `data/episodes/tacker/`) --
  아래 "후퇴(retract) 단계 추가" 절에서 재실행한 뒤에도 완전히 같은 수치다.
- **언어 라벨링**: force_max 범위(retract 추가 이전 실측) [5.9, 22.6]N(실제
  편차 있음, `qfrc_constraint` 기반 로깅이 이번에도 유효했다 -- workpiece가
  freejoint라 XML `<sensor>` 대신 이 방식을 처음부터 채택), 강도 분포
  gentle 83 / normal 83 / firm 86.
- **LeRobotDataset 변환**(retract 추가 이전): 252개 에피소드, 10,916프레임,
  state_dim=6(오른팔 3 + 왼팔 3, wrist 없음), action_dim=3, task_instruction
  9종 -- 프레임 수는 아래 retract 절에서 갱신됨.

### render_episode.py 추가 수정

`task.make_env(use_stabilizer=True)`를 강제로 호출하던 부분을 `task.make_env()`
로 바꿨다 -- `TackerOpenArmEnv`는 왼팔이 항상 붙어있는 실제 팔이라 애초에
`use_stabilizer` kwarg를 안 받아서, 그 kwarg를 강제로 넘기면 `TypeError`가
났다(peg_in_hole/cap_twist는 여전히 그 kwarg를 받으므로 기존 동작에 영향 없음,
둘 다 재렌더링해서 확인).

## tacker에 후퇴(retract) 단계 추가

사용자 피드백: "지금은 발사 성공 + 워크피스 변위 체크에서 에피소드가
끝나는데, 실제 타카 작업처럼 접근→압착→발사→후퇴까지 전체 사이클을
포함하도록 수정해달라." 압착은 이 태스크에서 발사 트리거 자체(tolerance
진입)와 같은 순간이라 별도 단계로 안 나눴고, 접근 뒤에 **정착(settle) ->
후퇴(retract) -> 최종 정착(final settle)** 세 단계를 추가했다(`sim/
tacker_openarm_env.py`).

### 설계

- **후퇴 제어는 접근과 완전히 같은 메커니즘을 재사용**했다(검토 결과
  재사용 가능함을 확인) -- `delta = clip(Kp_approach * (목표 - EE위치),
  MAX_APPROACH_STEP_M)`을 그대로 쓰고, 목표점만 `nail_site`에서
  `nail_site + [0,0,RETRACT_SAFE_DISTANCE_M]`(위로 5cm)로 바뀐다. 새 게인을
  따로 안 만들었다 -- 접근/후퇴 둘 다 "정확한 위치오차 비례 제어"라는
  같은 성질의 문제라서.
- **정착 단계도 raw `mj_step` 대신 `step()`(computed-torque)을 delta=0으로
  계속 호출하도록 바꿨다** -- 예전엔 발사 직후 ctrl이 마지막 값에 그대로
  얼어붙어 있었는데, "도구를 그 자리에 눌러 유지한다"는 실제 동작에 더
  맞게 고쳤다(부작용 없음, 실측 확인).
- **성공 조건 확장**: `is_success()`가 이제 "발사 성공 + (발사 정착/후퇴
  중/최종 정착) 3개 구간 각각의 workpiece 변위가 전부 허용치 이내 + 후퇴가
  예산 안에 완료됨"을 전부 요구한다. 리워드도 3개 구간을 따로 페널티화해서
  (`compute_reward()`) 어느 구간에서 밀렸는지 신호가 섞이지 않게 했다.

### 왼팔 강화 게인을 언제 놓아도 되는가 -- 세 가지 다 구현해서 N=150 실측 비교

`TackerOpenArmEnv(release_grip_after=...)`로 세 시점을 구현했다: `"none"`
(기본값, 끝까지 유지), `"fire_settle"`(발사 반동이 가라앉자마자, 후퇴
시작 전에 vendor 게인으로 낮춤), `"retract"`(후퇴까지 다 끝난 뒤에 낮춤).
셋 다 게인을 낮춘 뒤 `FINAL_SETTLE_STEPS`(20스텝)만큼 더 관찰해서 그 영향을
측정한다. 같은 씬 시퀀스(N=150, 시드 고정) + 같은 고정 게인으로:

| | `none`(끝까지 유지) | `fire_settle`(정착 직후 해제) | `retract`(후퇴 후 해제) |
|---|---|---|---|
| 성공률 | 73.3% | 72.7% | 73.3% |
| 후퇴 중 변위(`retract_bump`, 중앙값) | 0.118mm | **2.164mm** | 0.118mm |
| 최종 정착 변위(`final_bump`, 중앙값 / 최댓값) | 0.191mm / 1.1mm | 0.204mm / 5.7mm | **2.138mm / 19.0mm** |

**전체 성공률은 세 옵션 다 거의 안 갈린다(72.7~73.3%, 노이즈 수준 차이)** --
하지만 *어느 구간에서* workpiece가 밀리는지는 뚜렷하게 갈린다. 원인을
따져보면 일관된 그림이 나온다: 게인을 낮추면 (반동과 무관하게, 이전
절에서 실측한 "vendor 게인은 중력만으로도 sag가 생긴다"는 것과 같은
현상으로) workpiece가 새로운(더 처진) 평형점으로 서서히 이동하는데, **그
이동량 자체는 언제 게인을 낮추든 비슷하고, 다만 그게 어느 측정 구간에서
잡히느냐가 달라질 뿐이다** -- `fire_settle`은 그 이동이 후퇴 구간에서
일어나(그래서 `retract_bump`가 커짐) 후퇴가 끝날 때쯤엔 이미 새 평형점에
안착해 있어 `final_bump`는 작고, `retract`는 반대로 이동을 전부 최종 정착
구간(20스텝, 후퇴 구간보다 짧다) 안에 몰아넣어서 그 구간의 변위가 더 크고
들쭉날쭉하다(최댓값 19mm는 세 옵션 중 최악).

**결론(정식 데이터 생성에 반영)**: 안전 마진이 가장 큰 건 `"none"`(끝까지
유지)이라 프로덕션 데이터는 이걸로 생성했다. 그립 힘을 실제로 아끼고 싶은
경우라면 `"retract"`보다 `"fire_settle"`이 낫다 -- 평균적인 성공률은
동일하지만 최악의 경우(worst-case final_bump)가 훨씬 덜 나쁘다(5.7mm vs
19.0mm). `tests/test_tacker_task.py`에 이 비교의 핵심 신호(그립을 일찍
놓으면 후퇴 중 밀림이 뚜렷하게 커진다)를 회귀 테스트로 남겼다.

### 재검증 결과: retract 추가로 새로운 실패 모드가 생기지 않았다

같은 파이프라인(0→1→2-A→2-B→4→5)을 retract 포함 버전으로 재실행했다:

- CMA-ES 1세대 만에 수렴(`Kp_approach≈0.217`, reward≈79.2 -- 후퇴 완료
  보너스(+10)가 추가돼 리워드 상한이 올라가서 `success.reward_threshold`도
  65.0 -> 75.0으로 같이 올렸다).
- 부트스트랩 72.6%(726/1000), diffusion 검증 77.0%(+4.4%p) -- retract 추가
  전(72.5%/77.0%)과 사실상 동일하다.
- **필터링: 350개 씬 중 252개 성공(72.0%) -- retract 추가 전과 정확히
  같은 숫자다.** 사용자가 우려했던 "후퇴 단계에서 새로운 실패 모드가
  생기는지"는 기본 설정(`release_grip_after="none"`)에서는 실측상 없었다
  (retract 중 밀림은 항상 발사 반동 자체보다 훨씬 작아서 성공/실패를
  가르는 요인이 되지 않았다).
- **LeRobotDataset 변환: 252개 에피소드, 18,706프레임**(10,916 ->
  18,706, 약 1.71배) -- 에피소드당 평균 프레임이 43.3 -> 74.2로 늘었다
  (후퇴(~30~40스텝) + 최종 정착(20스텝)이 추가된 만큼).
- 언어 라벨링 force_max 범위 [5.9, 20.1]N, 강도 분포 gentle 83/normal
  83/firm 86 -- 이전과 사실상 동일(발사 반동 자체는 안 바뀌었으므로 당연).

## 공식 OpenArm 모델 검증 (로봇을 OpenArm 양팔 + 공식 그리퍼로 확정)

`enactic/openarm_mujoco` v2(HEAD `1c1a2d4`, 이 세션 작업 시점)와
`assets/openarm/` 전체(팔+그리퍼+받침대 MJCF, 모든 visual/collision 메시)를
직접 바이트 단위로 diff했다. **검증됨**: 링크 치수/관절 range/질량/관성/
메시 파일 전부 완전히 동일 (README의 "수정 없이 복사만 했다"는 주장이
정확함), 양팔 베이스 간격도 official `openarm_bimanual.xml`의
`openarm_left_base_link pos="0 0.031 0"` / `openarm_right_base_link
pos="0 -0.031 0"`에서 그대로 유도되는 **6.2cm**가 정확함(기존 기록 그대로).

**발견된 차이 3개, 전부 공식값으로 교체**:

1. `openarm_bimanual.xml`/`openarm_pedestal.xml`/`pedestal.xml`의 `<option>`에서
   `timestep="0.001" integrator="implicitfast"`가 빠져 있었음 -- 복원해서
   세 파일 다 공식 원본과 바이트 단위로 완전히 동일해짐. (단, `<option>`은
   `<attach>`를 안 타고 넘어가므로 -- 공식 파일 자신의 주석이 이미 그렇게
   경고하고 있었다 -- 이 복원 자체는 최종 컴파일된 씬에 아무 영향이 없다.
   실제로 영향 있는 곳은 각 씬 자신의 `<option>`이다, 아래 2번.)
2. 왼쪽 손가락 액추에이터(`left_finger1_ctrl`) `ctrlrange`가 공식
   `-0.4~0.7854`에서 `0~0.7854`로, 오른쪽(`right_finger1_ctrl`)이 공식
   `-0.7854~0.4`에서 `-0.7854~0`으로 좁아져 있었음 -- 공식값으로 복원.
   (조인트 자체의 물리적 range는 원래부터 안 바뀌어 있었음 -- 액추에이터
   명령 가능 범위만 좁았던 것.)
3. **(실제 버그, 실측으로 심각성 확인)** `assets/peg_in_hole_bimanual_openarm.xml`
   자신의 `<option>`에 `integrator="implicitfast"`가 없었다 -- tacker/
   screw_driving에서 이미 발견/수정했던 것과 완전히 같은 버그(강한 position
   액추에이터 + 항상 활성 weld 조합의 semi-implicit Euler 에너지 누출)가
   peg_in_hole에도 있었던 것으로, 이전 세션에서 "의심되지만 미확인"으로
   남겨뒀던 항목이다. **직접 측정**: 왼팔이 hole_socket을 쥔 채 오른팔
   입력 없이 2000스텝(20초) 정지 유지만 시켰을 때 — 수정 전 **69.08mm**
   드리프트, 수정 후 **0.0000mm**. `assets/bimanual_openarm.xml`(Python
   코드에서 안 쓰이는 범용 데모 씬)에도 같은 이유로 동일하게 적용.

**의도적으로 공식값을 안 따른 것 1개**: 위 세 씬(`peg_in_hole_bimanual_openarm.xml`,
`bimanual_openarm.xml`) + 기존 `tacker_openarm.xml`/
`screw_driving_bimanual_openarm.xml`의 `timestep`은 공식값(0.001)이 아니라
**0.002를 그대로 유지**한다 — 네 파일의 sim 모듈 전부 `N_SUBSTEPS=5`(제어
주기 dt=0.01s)와 그걸 전제로 튜닝된 모든 게인(Kp_xy/Kd_xy, NOMINAL_RATE,
Z_RATE, OMEGA_N 등)이 timestep=0.002 가정이라, 0.001로 바꾸면 네 태스크
전부 게인을 처음부터 다시 탐색해야 한다(이번 "공식 모델 검증"의 범위를
크게 벗어남). 네 파일 모두 `timestep="0.002"`를 명시해 이 선택이 의도적임을
기록해뒀다(안 적어도 기본값이 0.002라 동작은 동일하지만, 공식 서브모델이
이제 `timestep="0.001"`을 갖게 되면서 `<attach>` 시 "parent 값 유지" 경고가
뜨게 됨 -- 경고 자체는 무해하고 예상된 것).

**그리퍼(공식 구조, 이미 ARD-Gen 코드가 정확히 따르고 있었음)**: 손가락당
물리 조인트 2개(`finger_joint1`/`finger_joint2`), `<equality><joint>` mimic
제약(`polycoef="0 1 0 0 0"`, joint2 = joint1)으로 **1개 actuated DOF만**
구동 — 즉 그리퍼 자체는 손당 1-DOF 평행 그리퍼다. ARD-Gen의 OpenArm sim
모듈(`tacker_openarm_env.py`, `screw_driving_bimanual_openarm_sim.py` 등)은
이미 이 구조를 정확히 따르고 있다(두 조인트 qpos를 함께 설정, 액추에이터는
`*_finger1_ctrl` 하나만 사용). 공식 모델에서 직접 FK로 측정한 스트로크
(MuJoCo 충돌 메시 기준, 공식 데이터시트 수치 아님 -- **미검증**): 조인트
range(0~45°) 전체에서 핑거팁 콜리전 메시 간격 약 49mm(닫힘측 극단) ~
84mm(열림측 극단).

**손목 카메라(이미 공식 위치로 존재, 아직 파이프라인에 연결 안 됨)**:
`openarm_bimanual.xml`에 `camera_wrist_left`/`camera_wrist_right`가 공식
pos/euler/fovy/resolution 그대로 이미 정의돼 있다. 이 세션 이전까지 어떤
Python 코드도 참조하지 않았다 -- 다음 Phase(손목 카메라 연결)에서는 "추가"가
아니라 "이미 있는 공식 위치의 카메라를 observation에 연결"만 하면 된다.

## 공식 그리퍼 반영 (액션 공간을 팔당 7관절 + 그리퍼로 확장)

공식 모델 검증에서 확인한 대로 그리퍼는 이미 ARD-Gen 코드가 올바르게
구동하고 있었다(손가락당 물리 조인트 2개 + equality mimic, 1 actuated
DOF) -- 이번 작업은 "그리퍼를 고치는" 게 아니라, 지금까지 **기록되지
않고 있던** 조인트 공간 state/action을 데이터셋 스키마에 노출시키는
것이다.

**설계**: 기존 Cartesian state(ee_pos+wrist_rotate)/action(delta_xyz)은
전혀 안 건드렸다(이미 검증된 admittance 컨트롤러의 입출력이라 바꾸면
제어 자체가 깨짐) -- 대신 그 뒤에 이어붙이는 방식("확장")으로:

- `sim/peg_in_hole_bimanual_openarm_sim.py`에 순수 조회 메서드 5개 추가
  (`get_right_joint_pos/action`, `get_left_joint_pos`,
  `get_right/left_gripper_ctrl` -- 물리/제어에 전혀 영향 없음).
- `sim/peg_in_hole_openarm_env.py`의 `run_episode()`가 매 스텝 이 값들을
  기록해서 `right_joint_pos`/`left_joint_pos`(관측, 팔당 7)/
  `right_joint_action`(명령 목표 qpos, 7)/`right_gripper_action`/
  `left_gripper_action`(그리퍼 명령값, 팔당 1)을 결과에 추가.
- `pipeline/episode_io.py`: `right_arm`/`left_arm` dict에 이 필드들을
  **있으면만** 저장/복원하도록 확장(없는 태스크, 예: cap_twist는 전혀
  영향 없음 -- `.get()`이 None을 돌려주고 조용히 생략됨).
- `pipeline/filter_episodes.py`: 이 필드들을 `_KNOWN_RESULT_FIELDS`에
  추가하고(배열이라 스칼라 전용 extra_* 자동통과 루프에 걸리면
  TypeError가 남) episode dict로 실어 나름.
- `pipeline/to_lerobot.py`: `build_frames()`가 있으면 `observation.state`/
  `action` 뒤에 이어붙인다(`_align_trailing()`로 일반화 -- 기존
  `_align_left_arm()`을 임의 차원에 쓸 수 있게 한 것). `_STATE_NAMES`/
  `_ACTION_NAMES["peg_in_hole"]`도 그만큼 늘렸다.

**검증 결과**: `observation.state` 7(기존 Cartesian) + 7(오른팔 7관절) +
7(왼팔 7관절) = **21차원**, `action` 3(기존 delta_xyz) + 7(오른팔 7관절
명령) + 1(오른팔 그리퍼) + 1(왼팔 그리퍼) = **12차원**. 0→1→2-A→2-B→4→
LeRobotDataset 변환 전체를 작게(popsize 6/3세대, bootstrap 20건, 10 에피소드)
다시 돌려서 확인: CMA-ES 수렴(reward 49.16, 기존과 동일 범위), 부트스트랩
100%, diffusion 100%, 필터링 100%(10/10), `to_lerobot.py`가
state_dim=21/action_dim=12로 정확히 변환 + meta/info.json의 feature
names가 21/12개와 정확히 일치. 회귀 테스트 7개 전부 통과(기존 Cartesian
필드는 전혀 안 바뀌어서 그대로 통과).

## 관절 토크 센서 + 노이즈/제어 지연 옵션

`assets/peg_in_hole_bimanual_openarm.xml`의 `<sensor>`에 `jointactuatorfrc`
14개(팔당 7) 추가 -- 기존 `peg_force`/`peg_torque`(freejoint 바디의 site
force/torque 센서)와 종류가 다른 센서라 그 버그(아래 참고)에 안 걸린다.
`get_right/left_joint_torque()`가 이 센서값을 읽어서 `right_joint_torque`/
`left_joint_torque`(관측, (T+1,7))로 기록되고, `to_lerobot.py`의
`observation.state`에도 (조인트 공간 state 뒤에) 이어붙었다("observation에
joint_torque 필드 포함" 요청) -- state_dim 21 -> **35**(팔당 7 토크 x 2).

**실측 확인**(default 설정 그대로, 노이즈/지연 0): 토크값이 스텝마다
실제로 변함(예: 오른팔 joint1이 1~2스텝째 -0.62 -> -0.91N*m로 변화) --
freejoint site 센서와 달리 상수가 아니다.

**실기 모사 옵션(기본 꺼짐, "설정값으로" 요청)**: `BimanualPegInHoleOpenArmSim`/
`PegInHoleOpenArmEnv` 생성자에 `torque_noise_std`(N*m, 토크 관측값에만
가우시안 노이즈 -- 제어 루프 자체에는 영향 없음)와 `control_delay_steps`
(제어 틱 단위 FIFO 지연, `step()`에 들어오는 delta 명령을 그만큼 늦춤)
추가. 둘 다 기본값 0이면 이전 동작과 완전히 동일(실측 확인: 같은 게인으로
reward 49.29, 변화 없음) -- 기존 파이프라인(`task.make_env()`가 kwarg 없이
생성)은 전혀 영향 없다. 직접 켜서 확인: `torque_noise_std=0.5` -> 관측값이
눈에 띄게 흔들림(노이즈가 관측에만 적용되는 것 확인), `control_delay_steps=5`
-> 같은 씬에서 성공은 유지되지만 step_count 121->126, reward 49.29->49.24로
살짝 느려짐(지연의 정성적으로 타당한 효과).

**중요 발견(아직 안 고침, 다음 Phase에서 다룸)**: `get_force_torque()`가
읽는 `peg_force`/`peg_torque` 센서는 **peg가 freejoint 바디라서 접촉력을
전혀 못 읽고 peg 자신의 무게(질량*g)만 고정 반환한다** -- 실측: z축 힘이
에피소드 내내 0.4905N, 표준편차 6.6e-7(완전한 상수). `qfrc_constraint`
기반으로 바꾸면 실제 접촉 동역학(0.0000~0.253N, 표준편차 0.059)이 보인다.
tacker/screw_driving에서 이미 겪은 것과 같은 freejoint-site-sensor 버그
계열이고, **"force_max가 0.5N에 고정되던 문제"(다음 Phase 참고)의 근본
원인으로 보인다.**

## 손목 카메라 연결

공식 모델 검증(위 절)에서 이미 확인한 대로 `camera_wrist_left`/
`camera_wrist_right`는 공식 enactic/openarm_mujoco v2 MJCF에 공식
위치/각도로 이미 정의돼 있었다 -- 이번 작업은 "추가"가 아니라 렌더
파이프라인에 **연결**하는 것이다.

`render_peg_in_hole_bimanual_openarm.py`의 `--cameras` 기본값을
`wide_cam,top_cam`에서 `wide_cam,top_cam,camera_wrist_right,camera_wrist_left`
로 확장했다(기존 시점 유지 + 추가 채널, 요청 그대로) -- 이 스크립트는
이미 임의의 카메라 이름 리스트를 받아 각각 별도 mp4로 저장하는 범용
멀티카메라 구조라 코드 변경은 default 값 하나뿐이다.

**검증됨**: `mj_ray`로 두 손목 카메라의 광축을 직접 쐈을 때(렌더 없이도
되는 순수 기하 질의) 그리퍼 자신의 손가락(`finger_inner_*`/`finger_outer_*`
visual mesh, 27~61mm 거리)을 향하고 있음을 확인 -- "그리퍼 하우징 안에서
잡은 물체를 보는" 손목 카메라로서 기하학적으로 타당한 배치다(collision
그룹까지 포함한 첫 ray-cast는 자기 자신의 collision proxy를 6mm에서
맞혀 혼란을 줬는데, visual 그룹(group=2)만으로 다시 쏴서 진짜 광학
경로를 확인했다).

**미검증**: 이 컨테이너에는 OSMesa도 EGL도 설치돼 있지 않아서(`libOSMesa`/
`libEGL` 자체가 없음 -- `python -c "import mujoco"`가 `MUJOCO_GL=osmesa`든
`egl`든 즉시 import 단계에서 죽는다, 기존 wide_cam/top_cam 렌더링도 이미
이 환경에서는 똑같이 불가능했다) 실제 픽셀 렌더링(mp4 프레임 내용)은
이번 세션에서 확인하지 못했다 -- 렌더가 가능한 환경에서 직접 돌려서
눈으로 확인 필요.

## peg_in_hole 재생성 (force_max 0.5N 고정 버그 수정 + 전체 파이프라인 재실행)

### force_max가 0.5N에 고정되던 진짜 원인: 컨트롤러 문제가 아니라 센서 버그

사용자가 제시한 두 선택지(①컨트롤러에서 삽입 강도 의도적 다양화, ②토크
신호 기반 intensity 재설계) 둘 다 틀린 방향이었다 -- 실측으로 직접 확인한
진짜 원인은 `get_force_torque()`가 읽는 `peg_force`/`peg_torque` site
센서였다. `peg`가 freejoint 바디라서 이 site force/torque 센서는 peg-hole
접촉력을 전혀 못 읽고 peg 자신의 무게(질량*g)만 고정 반환했다(실측:
z축 힘이 에피소드 내내 0.4905N, 표준편차 6.6e-7 -- 완전한 상수). tacker/
screw_driving에서 이미 겪은 것과 같은 freejoint-site-sensor 버그 계열.

**수정(사용자 승인)**: `get_force_torque()`를 `qfrc_constraint`(peg의
freejoint 6-DOF에 실제로 걸리는 구속력) 기반으로 바꿨다 -- 컨트롤러는
전혀 안 건드렸다. 이제 dead code가 된 `_force_slice`/`_torque_slice`
(더 이상 안 쓰는 site 센서 조회)도 같이 제거했다. freejoint의 선형/각
DOF는 둘 다 월드 프레임이라 기존 함수처럼 site 회전행렬을 곱할 필요가
없다(기존 함수의 반환값도 사실은 이미 월드 프레임이었다 -- 자세한
설명은 그 메서드 docstring 참고).

### 재생성 실행 기록 (0→1→2-A→2-B→4→5→LeRobotDataset)

- **0단계(CMA-ES)**: 1세대 만에 수렴(`Kp_xy=0.0681, Kd_xy=0.00006`,
  reward=49.25) -- 수정 전과 거의 동일한 범위(reward 변화 없음, 수정이
  컨트롤러/보상에 영향 안 줬다는 뜻). **재실행 max_force=0.53**(수정 전
  고정값 0.49~0.5와 겉보기엔 비슷해 보이지만, 아래에서 보듯 이제 씬마다
  실제로 다르다).
- **여러 오프셋 직접 비교**(같은 게인, 수정 후): offset=(0,0) ->
  max_force=0.253, (0.014,0) -> 0.501, (0.0099,0.0099) -> 0.569,
  (0.0099,-0.0099) -> **1.329**, (-0.012,0.005) -> 0.367 -- 씬마다
  최대 ~5배까지 차이가 나는 실제 접촉 신호로 바뀌었다(수정 전엔 전부
  정확히 0.4905였다).
- **2-A단계(부트스트랩)**: 300트라이얼, **성공률 100%(300/300)**,
  **force_max 분포: [0.24, 0.73]N, 평균 0.356, 표준편차 0.092, 반올림
  3자리 기준 고유값 173개**(수정 전엔 사실상 1개 값이었다) -- "force_max가
  0.5N에 고정되던 문제"가 완전히 사라졌다.
- **2-B단계(diffusion)**: 500 epoch 학습, 새 무작위 씬 100개 검증 --
  부트스트랩 100% -> diffusion **100%(100/100, 변화 없음, 이미 최대치)**.
- **4단계(필터링)**: 150개 씬 중 **150개 성공(100%)**.
- **5단계(언어 라벨링)**: 150개 에피소드, **force_max 범위 [0.3, 0.7]N**,
  강도 경계(33/66 백분위) 0.3N/0.4N, **강도 분포 gentle 50/normal 49/
  firm 51** -- 수정 전(이전 세션 기록: 전부 "gentle"로 쏠림)과 완전히
  달라졌다. 수정 하나로 의도한 결과(사용자가 제안한 ①②를 적용하지 않고도
  intensity 라벨이 자연스럽게 균등하게 갈림)를 그대로 얻었다.
- **LeRobotDataset 변환**: 150개 에피소드, 18,604프레임, state_dim=35
  (Phase 2/3에서 늘린 조인트+그리퍼+토크 공간 포함), action_dim=12,
  정상 변환 확인.
- 회귀 테스트 7개 전부 통과(이 수정으로 어떤 기존 테스트도 안 깨짐 --
  force_max를 구체적 숫자로 검증하는 테스트가 원래 없었다).

### 검증됨 vs 미검증 (이번 5개 Phase 전체 요약)

**검증됨** (실측 확인):
- 공식 OpenArm 모델(링크/관절/질량/관성/베이스 간격 6.2cm/그리퍼 구조)과
  현재 자산의 일치, 발견된 3개 차이의 수정.
- peg_in_hole의 누락된 `integrator="implicitfast"` 버그와 그 효과(69.08mm
  -> 0.0000mm 드리프트).
- 조인트 공간 state/action(팔당 7+그리퍼) 기록/저장/변환 전체 경로.
- 관절 토크 센서(jointactuatorfrc) 값이 실제로 변하는 것, 노이즈/지연
  옵션이 기본값(0)에서 이전 동작과 동일한 것.
- 손목 카메라의 광축이 그리퍼 핑거를 향하는 기하학적 타당성(ray-cast).
- force_max 버그의 원인(freejoint site 센서)과 수정 결과(0.24~0.73N
  분포, intensity 균등 분포).

**미검증** (이번 세션에서 확인 못 함, 명시적으로 남김):
- 손목 카메라의 실제 픽셀 렌더링 내용(이 컨테이너에 OSMesa/EGL 둘 다
  없어서 렌더 자체가 불가능했음).
- 관절 토크 센서의 절대적 정확도(실제 OpenArm 하드웨어의 토크 센서
  스펙과의 비교 -- 시뮬레이션 내부의 jointactuatorfrc 값이 물리적으로
  합당한 크기인지는 확인했지만 실기 데이터와 직접 비교하지는 않았다).
- 그리퍼 스트로크 수치(49~84mm, MuJoCo 충돌 메시 기준 직접 측정 -- 공식
  데이터시트 수치와는 비교 안 함).
- torque_noise_std/control_delay_steps를 0이 아닌 값으로 켠 상태에서의
  CMA-ES/부트스트랩/diffusion 전체 파이프라인 재검증(옵션 자체의 동작은
  확인했지만, "그 옵션을 켠 상태로 전체 데이터셋을 생성해도 안전한가"는
  별도 검증이 필요 -- 이번 요청 범위 밖).
