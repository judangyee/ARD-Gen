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

### 4단계 — 동시 실행 & 필터링 ✅ 완료 (3단계 이후 Actuator+Stabilizer 동시 실행)

설계대로 "Actuator + Stabilizer 궤적을 같은 시뮬레이션에서 동시에 재생"한다
(3단계 완료로 실현됨, 위 3단계의 "4. 양팔 통합" 절 참고). 아래는 3단계
이전(Actuator 단독) 시절 최초 실행 기록으로, 재현성 이슈 발견/수정
경위 등은 여전히 유효해 남겨둔다.

- 구현 위치: `pipeline/filter_episodes.py`
- 흐름: 1단계로 씬 샘플링 → 2-B diffusion으로 그 씬의 게인 생성 →
  `sim/peg_in_hole_sim.py`의 `run_episode()`로 Actuator 실행 →
  `insertion_depth` 기준 성공 여부(`run_episode()`가 이미 판정)로 필터링
  → 성공한 것만 저장.
- 저장 포맷: `LeRobotDataset`이 아니라 지금도 `pipeline/episode_io.py`가
  정의한 단순 npz 포맷(`data/episodes/episode_NNNN.npz`)이다 — 3단계
  완료로 이제 `{"left_arm": {"traj","role":"stabilizer"}, "right_arm":
  {"traj","gains","force","torque","role":"actuator"}, "scene_config",
  "success", "insertion_depth", "force_max"}` 구조(왼팔은 stabilizer 절이
  있는 태스크만)를 평탄화해서 저장한다. 진짜 LeRobotDataset 변환 자체는
  여전히 별도 작업으로 남아있다(이번 3단계 요청 범위 밖).
- 실패 에피소드는 2-A/2-B와 달리 **보관하지 않는다**(완성 데이터로 못
  쓰므로).
- **재현성 이슈 발견 및 수정**: 처음 실행에서 같은 `--scene-seed`인데도
  실행할 때마다 성공률이 81%/85%로 달라지는 걸 발견했다 — 원인은
  diffusion의 reverse sampling이 torch의 전역 RNG를 시드 고정 없이
  써서, 씬은 같아도 매번 다른 게인이 뽑혔기 때문. `--sample-seed`
  인자를 추가해 `torch.manual_seed()`로 고정한 뒤 동일 시드로 두 번
  실행해 84%/84%로 재현되는 걸 확인했다.
- **실제 실행 결과** (`--n-scenes 100 --scene-seed 42 --sample-seed 0`):
  **84/100 성공 (84.0%)** — 2-B 검증 때의 84~86%와 일치하는 범위.

### 5단계 — LLM 기반 언어 라벨링 ⚠️ 부분 구현 (템플릿 기반)

원래 설계는 "성공한 궤적의 force/torque profile을 LLM에 넘겨 자연어
지시문으로 변환(토크 부호→회전 방향, 회전수→수량, 최대 접촉력→강도)"이지만,
지금은 **템플릿 기반**으로 구현했다. 강도(intensity)/방향(direction)/
수량(quantity) 세 어휘 축 모두 지원하지만, 실제로 어느 축을 쓰는지는
`tasks/{name}.yaml`의 `language` 절이 정한다 -- peg_in_hole은 강도만
(direction_words/quantity_words가 비어 있음), cap_twist는 방향+수량만
(intensity_words가 비어 있음, force/torque가 아니라 목표 회전각 부호/
크기로 정해지므로) 쓴다.

- 구현 위치: `pipeline/language_labeling.py`
- **Claude API 대신 템플릿을 쓴 이유**:
  1. 지금은 "지시문이 자연스러운가"가 아니라 "language 필드가 파이프라인
     끝까지 채워져서 나오는가"를 확인하는 단계라 템플릿으로 충분하다.
  2. 오프라인/재현 가능해야 하는데(설계 원칙 2: 계산 비용이 싼 방법부터),
     지금 단계에서 API 호출은 과한 비용이다.
  3. 지금 두 태스크가 표현할 정보 조합(강도 하나, 또는 방향+수량)이
     여전히 단순하다 -- 조합이 훨씬 다양해지는 태스크가 생기면 그때
     Claude API로 바꾸는 게 맞다고 본다.
- 강도어(살짝/적당한 힘으로/힘있게)는 **그때그때 episodes의 force_max
  33/66 백분위수**로 정한다(하드코딩된 절대값이 아님) — 씬/게인 분포가
  바뀌면 force_max 스케일 자체가 달라지므로.
- **peg_in_hole 실제 실행 결과**(단일팔 시절, 84개 에피소드): force_max 범위 [0.5, 480.4]N,
  강도 경계 44.2N/86.7N, 강도 분포 gentle 28 / normal 27 / firm 29건
  (거의 균등 — 백분위수 기반이라 당연한 결과). 생성된 문장 예:
  "오른손으로 peg를 구멍에 살짝 삽입하라.", "오른팔을 이용해 페그를
  구멍 안으로 힘있게 밀어 넣어라."

최종 산출물(3단계 완료 후 현재 버전): **왼팔(left_arm, Stabilizer 궤적) +
오른팔(right_arm 궤적/게인) + force/torque + 언어**가 모두 포함된 양팔
에피소드(`data/episodes/episode_*.npz`) — 위 실측 수치는 아래 3단계
절의 최신(양팔) 결과를 참고. 이 문단의 84개는 3단계 이전(단일팔) 시절
기록으로 남겨둔다.

### 파이프라인 오케스트레이션: `run_pipeline.py`

0→1→2-A→2-B→4→5단계를 한 번에 실행하는 드라이버. 0/2단계 산출물(seed,
bootstrap dataset, diffusion 체크포인트)은 이미 있으면 재사용하고
`--force-seed`/`--force-bootstrap`/`--force-diffusion`로만 다시 만든다
(비용이 드는 단계라서). 4/5단계(episode 생성 + 라벨링)는 지금 실제로
검증하려는 부분이라 매번 새로 실행한다. 마지막에 최종 에피소드 수와
샘플 몇 개를 자세히 출력한다.

**실제 실행 결과**: 씬 100개 시도 → 84개 성공(4단계) → 84개 전부
언어 라벨링까지 완성(5단계에서 탈락 없음, 이미 성공한 것만 넘기므로
당연함) → **최종 84개의 완성된 에피소드**.

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
| 4단계 (동시 실행 & 필터링) | `pipeline/filter_episodes.py`, `pipeline/episode_io.py`, `data/episodes/` | ✅ 완료 (양팔 동시 실행, `left_arm`+`right_arm`) |
| 5단계 (언어 라벨링) | `pipeline/language_labeling.py` | ⚠️ 부분 구현 (템플릿 기반, LLM 아님 -- 이유는 해당 절 참고) |
| 파이프라인 오케스트레이션 | `run_pipeline.py` | ✅ 완료 (--task로 태스크 선택) |

지금 저장소(`ard-gen/`)는 peg_in_hole/cap_twist 두 태스크 모두 **왼팔
(Stabilizer)+오른팔(Actuator)이 같은 물리 시뮬레이션에서 동시에 실행되는
0→1→2-A→2-B→4→5단계 전체**가 끝까지 도는 것까지 검증 완료한 상태다.
남은 것: 5단계를 실제 LLM 기반으로 바꾸는 것(현재는 표현할 정보 조합이
단순해 템플릿으로 충분하다고 판단해 보류 중), `LeRobotDataset` 형식으로의
최종 변환, 그리고 peg_in_hole 3단계에서 발견된 "고부하 경계선 케이스
4/28건" 잔여 실패의 근본 개선(현재는 정직하게 실패로 남겨둠, 위 3단계
절 참고).
