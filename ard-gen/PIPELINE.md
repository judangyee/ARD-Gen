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

## 폐루프 데이터 (closed-loop-data 브랜치)

ARD-VLA(judangyee/ARD-VLA, 브랜치 closed-loop-structure)의 Judge/Selector
폐루프 구조를 학습·실험하는 데 필요한 시뮬·데이터 인프라를 ARD-Gen 쪽에
만든다. Judge 학습/SmolVLM2 특징 추출/실험 비교는 범위 밖 -- 이 절은
시뮬·데이터 인프라만 다룬다.

### 0단계 — 추적 (코드 수정 없음) ✅ 완료

**브랜치 관계**: `main`(d02575e, OpenArm peg_in_hole·tacker 등록된 최신
라인)에서 `closed-loop-data`를 땄다. `claude/queue-dyy0ir`의
screw_driving, `ard-gen-v2`, `ard-gen-insertion`/`ard-gen-insertion-horizontal`,
`feature/official-openarm-model`은 이번 작업에서 병합하지 않는다 --
전부 main과 별도로 갈라진 독립 라인이고, 이번 폐루프 인프라는 그 중
어느 것에도 의존하지 않는다(peg_in_hole_openarm_env.py/
peg_in_hole_bimanual_openarm_sim.py만 건드린다, main에 이미 있는 그대로).

1. **controller_step 경계**: `sim/peg_in_hole_openarm_env.py:PegInHoleOpenArmEnv.run_episode()`
   의 for 루프 안에서, "관측(`sim.get_peg_tip_pos()`/`get_hole_center_pos()`)
   → 오차(`cur_dx/cur_dy`, 이전 스텝 값 `prev_dx/prev_dy`와의 차분) →
   admittance(`delta_xy`) + adaptive_z_rate(`z_rate`) → 액션(`delta`)"
   계산 블록(해당 파일 148~159행)이 순수 함수로 뽑을 수 있는 경계다 --
   force/torque 기록, `self.step(delta)` 호출, 성공 판정은 전부 이
   블록의 바깥(호출부)에 남는다. 서명: `controller_step(sim, gains,
   prev_dx, prev_dy, outer_half, target_depth) -> (delta, new_prev_dx,
   new_prev_dy)` (1단계에서 그대로 구현).

2. **weld 런타임 조작 (실측, mujoco 3.14.0)**:
   - `data.eq_active[eq_id] = False`로 끄면 다음 `mj_step`부터 그 weld가
     바로 비활성화된다(실측: `peg_grasp`를 끄고 20스텝 돌려도 peg_tip이
     한 치도 안 움직임 -- **이건 버그가 아니라 peg 바디 자체에
     `gravcomp="1"`이 XML에 박혀 있어서 weld를 꺼도 중력이 상쇄돼 그냥
     그 자리에 뜬 채로 남기 때문이다**. "놓쳐서 떨어진다"처럼 보이게
     하려면 release 순간 `model.body_gravcomp[peg_body_id] = 0`도 같이
     꺼야 한다 -- 2단계 release 모드에 반영.
   - `model.eq_data[eq_id]`(relpose, WELD 타입은 `[anchor(3), relpose_pos(3),
     relpose_quat(4), torquescale(1)]` 11원소)를 런타임에 바꾸면 **다음
     `mj_step`부터 바로 반영된다**(`mj_forward`로 먼저 갱신 확인 가능,
     `mj_step`도 매 스텝 모델을 다시 읽으므로 추가 조치 불필요). 실측:
     relpose x를 +1cm 튕겨놓고 30스텝 보면 weld 구속력이 27.0(튕긴 직후)
     → 0.04(20스텝 후)로 감쇠하면서 peg_tip이 새 목표를 향해 **점진적으로**
     (텔레포트 아님) 수렴한다 -- `force_limited`/`injected` 미끄러짐
     모델이 쓸 바로 그 메커니즘.
   - weld 구속력은 `data.efc_force`에서 `data.efc_type[i] ==
     mujoco.mjtConstraint.mjCNSTR_EQUALITY`이고 `data.efc_id[i] == eq_id`인
     행(최대 6개: 위치3+회전3)의 노름으로 읽을 수 있다(실측 확인, API만
     있고 별도 센서 불필요). 참고로 이 모델은 equality가 4개 있다
     (`peg_grasp`=0, `hole_grasp`=1 둘 다 WELD, 나머지 2개는 양팔 그리퍼
     finger_joint2 mimic용 JOINT equality -- 무관).

3. **명목 조건 weld 구속력 분포** (기본 게인 Kp_xy=0.124630/Kd_xy=0.001125,
   peg_in_hole의 기존 eval_scenarios 3개 실측, 2단계 임계값 근거):

   | weld | 역할 | mean | max | p95 |
   |---|---|---|---|---|
   | `peg_grasp` (오른팔) | Actuator 파지(F2) | 0.71 | 11.86 | 7.00 |
   | `hole_grasp` (왼팔) | Stabilizer 미끄러짐(F3) | 10.04 | 18.55 | 15.82 |

   `hole_grasp`는 baseline 자체가 이미 높다(왼팔이 강화 게인으로 중력/
   미세 교란에 계속 저항 중이라는 뜻, tacker PIPELINE 절의 "vendor
   게인은 중력만으로도 sag가 생긴다"와 같은 현상) -- 그래서 2단계
   `force_limited` 임계값은 두 weld에 **따로** 둬야 한다(공통 스칼라
   하나로는 `hole_grasp`가 항상 걸리거나 `peg_grasp`가 전혀 안 걸림).
   (주의: 3개 eval_scenarios만 돌렸다 -- `sample_scene_config`의 전체
   무작위 분포는 아직 안 돌려봄, 분산이 더 클 수 있다.)

4. **tacker 단계 구분**: `sim/tacker_openarm_env.py`의 `run_episode()`
   안에 approach → settle(발사 포함) → retract → final_settle 네 단계가
   **주석으로만 구분된 순차 코드 블록**으로 있다(별도 함수/명시적 `stage`
   필드 없음, `release_grip_after` 설정값도 "`fire_settle`"/"`retract`"
   문자열로 각주를 달 뿐 스텝별 기록은 안 남긴다). 3단계 그래프 yaml에는
   네 단계를 노드로 옮겨 적지만(문서/메타데이터), tacker의 실행 루프
   자체를 controller_step 패턴으로 다시 쪼개는 건 이번 작업 범위에 넣지
   않았다 -- 2/4단계(미끄러짐 모델, 폐루프 어댑터)는 명시적으로 이름이
   나온 `peg_grasp`/`hole_grasp`(peg_in_hole)에만 구현한다. tacker의
   `tacker_grasp`/`workpiece_grasp` weld도 같은 패턴이라 메커니즘 자체는
   옮겨 쓸 수 있을 것으로 보이지만(미검증), 150+ 에피소드짜리 기존 회귀
   테스트가 있는 안정된 태스크라 이번 범위에서는 안 건드렸다.

5. **npz 스키마 vs ARD-VLA 16 DoF**: 현재 `pipeline/episode_io.py` 스키마는
   `right_arm.traj`(ee_poses, 4차원: xyz+joint7), `right_arm.action`(델타
   xyz, 3차원), `left_arm.traj`(월드 EE 위치, 3차원)만 있다 -- **관절
   공간(16 DoF = 팔당 7관절+그리퍼1) 기록은 지금 전혀 없다.** 다만
   `sim/openarm_bimanual_base.py:OpenArmBimanualBase`(main에 없음 --
   이 저장소의 다른 브랜치 `ard-gen-insertion-horizontal`에서 만든 공통
   베이스, 이번 브랜치는 그걸 병합하지 않으므로 아래는 "가능하다는 사실"
   참고용 기록이고 실제 구현은 이번 브랜치의 sim 클래스에 **직접** 작은
   getter로 추가한다)가 이미 `right_arm_qpos()`/`left_arm_qpos()`/
   `*_gripper_qpos()` 패턴의 getter를 증명한 적이 있다 -- `self.data.qpos`
   에서 그냥 읽는 것뿐이라 **관절 공간 기록은 기술적으로 아무 장애 없이
   가능하다**(5단계에서 `BimanualPegInHoleOpenArmSim`에 같은 getter를
   작게 추가해서 매 스텝 기록한다).

6. **렌더링 비용** (`MUJOCO_GL=osmesa`, 640x480, 실측): `top_cam`
   324ms/frame, `wide_cam` 441ms/frame (둘 다 소프트웨어 렌더러라 2~3fps
   수준 -- 느리다). `render_episode.py`의 `env.step`을 감싸서 캡처하는
   몽키패치 패턴은 그대로 재사용 가능(판정 시점에만 호출하면 매 스텝이
   아니라 서브태스크 경계 수만큼만 들어서 비용이 줄어듦) -- 5단계는 이
   비용 때문에 "별도 렌더링 프로세스에서 같은 시드로 재생"(지시사항
   원문)을 그대로 따른다(물리 롤아웃 자체는 렌더링 없이 빠르게 돌리고,
   경계 이미지만 osmesa 프로세스에서 재생해서 뽑는다).

7. **명목 개루프 성공률 기준선** (시드 고정 `np.random.default_rng(12345)`,
   `sample_scene_config`로 N=200 무작위 씬, 알려진 좋은 게인
   Kp_xy=0.124630/Kd_xy=0.001125, 실측): **200/200 = 100.0%**
   (insertion_depth 평균 23.37mm, 최대 25.71mm). 200 에피소드에 66.9초
   (334.6ms/episode). 이게 2단계 게이트("기준선보다 5%p 초과 하락하면
   보고")의 기준선이다. 명목 분포에서 이미 100%라 5단계의 "자연 실패
   10~30%" 범위(A)는 이 분포보다 훨씬 좁은 clearance/훨씬 큰 오프셋
   반경으로 따로 찾아야 한다(5단계에서 수행).

### 1단계 — controller_step 분리 ✅ 완료

0단계 1번에서 찾은 경계 그대로 `sim/peg_in_hole_bimanual_openarm_sim.py`에
`controller_step(sim, gains, prev_dx, prev_dy, outer_half, target_depth) ->
(delta, new_prev_dx, new_prev_dy)`를 추가했다(물리 전진은 안 함, 호출부가
`sim.step(delta)`를 따로 부른다). `_run_episode_with_sim()`(같은 파일)과
`PegInHoleOpenArmEnv.run_episode()`(`sim/peg_in_hole_openarm_env.py`) 둘 다
이 함수 하나를 쓰도록 바꿨다 -- 제어 로직이 더 이상 두 곳에 복붙돼
있지 않다. 연산 순서는 리팩토링 전과 완전히 동일(그냥 블록을 함수로
옮긴 것뿐).

**회귀 테스트(비트 단위 동일성)**: `tests/test_regression_peg_in_hole.py`
9케이스 전부 리워드 값까지 정확히 동일(예: `kp=0.0003 kd=5e-05
offset=(0.014,0.0): reward=49.4675` -- 리팩토링 전 기록과 일치), scene
sampling 5개 시드 일치, `task_registry wiring` 통과. `tests/
test_tacker_task.py`도 전부 통과(tacker는 이 함수를 안 쓰지만 import
그래프가 안 깨졌는지 확인). 2단계(미끄러짐 모델)/4단계(폐루프 어댑터)는
이제 이 함수만 재사용한다 -- 제어 로직을 또 복사하지 않는다.

### 2단계 — 파지 미끄러짐 모델 ✅ 완료 (게이트 통과)

`sim/grasp_slip.py`(신규, `BimanualPegInHoleOpenArmSim`만 의존 -- tacker는
이번 범위에 안 넣음)에 weld를 유지한 채 런타임에 조작하는 미끄러짐 모델을
만들었다. **진짜 손가락 접촉 물리가 아니다** -- grasp anchor 재조정에
6차례가 걸렸던 걸 감안해 의도적으로 피했다(파일 상단 docstring에 명시).
`ArmSlipConfig(mode=...)`로 오른팔(`peg_grasp`, Actuator 파지 실패=F2)/
왼팔(`hole_grasp`, Stabilizer 미끄러짐=F3) 각각 독립 설정:

| mode | 동작 |
|---|---|
| `none`(기본) | 아무 개입 없음 -- `grasp_slip_config=None`이면 트래커 자체를 안 만든다 |
| `force_limited` | weld 구속력(`efc_force` 노름)이 `force_threshold`를 넘으면, 그 스텝의 relpose를 "실제 위치 쪽으로 `yield_fraction`만큼" 양보시킨다 |
| `injected` | 지정 스텝에 지정 오프셋(ee 로컬, m)을 relpose에 한 번 더한다(결정론적 교란 이벤트) |
| `release` | 지정 스텝에 weld 비활성(`eq_active=False`) + 그 바디 `body_gravcomp=0`(0단계 2번 실측 -- gravcomp를 안 내리면 안 떨어짐) |

매 스텝 `GraspState`(weld_active, weld_force, slipped_this_step,
cumulative_shift_pos, cumulative_shift_rot_deg=0.0 고정)를 기록한다 --
회전 미끄러짐은 이번 범위에서 다루지 않음(파일 docstring에 확장 방법
명시). `force_limited`의 미끄러짐 "방향"은 진짜 힘 벡터가 아니라 "물체의
실제 위치 쪽으로 양보"하는 근사다(efc_force를 world 벡터로 복원하려면
weld의 제약 Jacobian 투영이 필요해서 단순화했다 -- 파일 docstring에
한계와 확장 방법 명시).

**비트 단위 동일성**: `grasp_slip_config=None`(기본값)이면
`tests/test_regression_peg_in_hole.py` 9케이스 리워드 값이 전과 정확히
동일(49.4675 등 -- 1단계와 같은 수치), `tests/test_tacker_task.py`도
전부 통과. `tests/test_grasp_slip.py`(신규)의
`test_default_off_matches_no_config`로 grasp_slip 쪽 관점에서도 재확인.

**게이트 결과 (기준선: 0단계 7번, N=200 무작위 씬, 100.0% 성공)**:
같은 N=200/시드로 `force_limited`를 다른 세 임계값에서 실측했다(두 weld
독립, 0단계 3번 분포 기준):

| 설정 | peg_grasp 임계값 | hole_grasp 임계값 | 성공률 | 미끄러짐 발생 에피소드 |
|---|---|---|---|---|
| 1.3x 명목 최대 | 15.42 | 24.12 | 200/200 (100.0%) | 0/200 |
| 1.05x 명목 최대(타이트) | 12.46 | 19.48 | 200/200 (100.0%) | 18/200 |
| 명목 p95(매우 타이트) | 7.00 | 15.82 | 200/200 (100.0%) | 175/200 |

**게이트 통과(기준선 대비 하락 0%p, 5%p 한도 안쪽)** -- 심지어 아주
타이트한 임계값(명목 p95, 즉 명목 분포의 95번째 백분위를 넘기만 하면
바로 미끄러짐)에서도 175/200 에피소드가 실제로 미끄러짐을 겪으면서도
전부 성공했다. `yield_fraction=0.1`(기본, 과부하 한 스텝당 실제-목표
오차의 10%만 반영)이 충분히 작아서, admittance 피드백이 실시간으로
미끄러짐을 보상하는 것으로 보인다(실측, 원인을 더 깊이 파진 않음).

**더 강하게 실측(별도 스윕, 진짜 실패를 만들 수 있는지 확인)**:
- `yield_fraction=0.8`(한 스텝에 오차의 80%, 공격적) + p95 임계값: 그래도
  100/100 성공 -- force_limited류(점진적, 피드백 루프 안에서 발생)는
  이 태스크에서 실패를 만들기 매우 어렵다.
- `injected`(한 번에 왕창): 고정 씬(offset=(0.014,0.0)) 기준 +5/10mm는
  항상 성공, **+15mm는 46/50(92%) 실패**, 그런데 +20/25/30mm는 오히려
  다시 대부분 성공(33/50, 결과가 magnitude에 비단조적) -- 안 믿기지만
  실측 그대로 기록한다. 원인은 깊이 파지 않았다(추정: 큰 교란 뒤
  admittance가 우연히 비슷하게 좋은 경로로 재수렴하는 경우가 있는 것
  같다).
- `release`(완전히 놓침): **release_at_step=10/30/50/80/100 전부(고정
  씬) 성공했다** -- peg_in_hole이 수직 낙하 삽입이라, 놓치는 시점에
  이미 xy가 거의 맞춰진 채 떠 있으면 중력(release 시 gravcomp를
  껐으므로 실제로 작용)이 삽입을 대신 끝내버린다. **이건 이 모델의
  중요한 한계다** -- release 메커니즘(weld 끄기 + gravcomp 끄기) 자체는
  올바로 동작하지만, 이 특정 태스크(수직 낙하)에서는 F2(Actuator 파지
  실패) 학습 데이터로 쓸 "진짜 실패"를 안정적으로 못 만든다. 5단계
  B세트(F2/F3 주입 실패)는 `release`가 아니라 **xy를 확실히 흩트리는
  `injected`(예: +15mm류, 실측으로 실패 확인된 값) 또는 큰
  `yield_fraction`의 `force_limited`+훨씬 좁은 clearance 조합**을 쓴다.

**새 테스트**: `tests/test_grasp_slip.py`(5개) -- 기본 꺼짐 동일성,
force_limited 타이트해도 명목 성공 유지, injected가 relpose를 실제로
옮기고(이 고정 씬에서는) 실패시킴, release가 weld/gravcomp를 정확히
끄는지(결과 success는 단언 안 함, 위 한계 참고), reset()이 다음
에피소드 전에 relpose/eq_active를 원래대로 복원하는지.

### 3단계 — 서브태스크 분할과 완료 조건 ✅ 완료

`tasks/peg_in_hole.graph.yaml`(신규, align→insert 두 노드)과
`tasks/tacker.graph.yaml`(신규, approach→settle→retract→final_settle
네 노드, 0단계 4번 조사 결과를 그대로 옮긴 메타데이터 -- 완료 조건 체크
코드는 아직 없음, `implemented: false`로 명시) 둘 다 ARD-VLA의
`DecisionLogic.load_task_graph()`가 그대로 읽을 수 있는 스키마
(`next`/`recoveries`/`max_retries`)로 작성했고, 실제로 그 함수에 넣어
로드되는지 확인했다(`tests/test_subtask_graph.py`가 아니라 1단계 조사
스크립트로 직접 ARD-VLA 저장소를 import해서 확인 -- 이 테스트 파일
자체는 ARD-VLA 없이도 돌게 `yaml.safe_load`만 쓴다). `max_steps`/
`completion` 등 ARD-Gen 전용 키는 `load_task_graph()`가 그냥 무시한다
(그 함수가 `spec.get()`으로만 읽으므로 안전).

peg_in_hole의 두 노드는 admittance 컨트롤러의 구조(2 피드백축+1 가변
삽입축)를 그대로 반영한다 -- **align**(xy 오차가 hole 반경
`outer_half` 안으로 10스텝 연속 유지, 기존 `is_success()`의
`xy_within_hole_footprint` 판정과 같은 기준)과 **insert**(목표 깊이
이상을 기존 `_SUCCESS_HOLD_STEPS`만큼 유지 -- 사실상 기존 에피소드
전체 성공 판정 그대로). `sim/peg_in_hole_bimanual_openarm_sim.py`에
추가한 함수들:

- `align_progress(cur_dx, cur_dy, initial_xy_err) -> float`: 초기 xy
  오차 대비 감소율(0~1).
- `insert_progress(raw_depth, target_depth) -> float`: 목표 깊이 대비
  현재 원시 깊이(0~1).
- `align_subtask_finished(...)`/`insert_subtask_finished(...)`: 그래프
  yaml의 completion 조건을 코드로 반영(hold-steps 카운터는 호출부
  책임 -- 기존 `is_success()`/`hold_count` 패턴과 동일).
- `classify_failure(success, right_weld_active, right_cumulative_shift_m,
  left_weld_active, left_cumulative_shift_m, peg_touching_hole_wall,
  raw_depth, target_depth, timed_out, slip_distance_threshold_m=0.003) ->
  str`: F2 > F3 > F1 > timeout > none 우선순위(파지 실패가 가장 직접적인
  원인이라고 보고 먼저 검사). `success=True`면 항상 "none"(2단계에서
  발견한 "release해도 중력으로 우연히 성공" 사례를 실패로 잘못 세지
  않기 위함 -- 반대 선택도 가능하다는 걸 함수 docstring에 명시해둠,
  이번 작업 범위에서 결정하지 않음).
- `BimanualPegInHoleOpenArmSim.is_peg_touching_hole_wall()`(새 메서드):
  F1(정렬 실패/걸림) 판정용 -- weld가 아니라 진짜 `<contact>`를
  `data.contact`에서 직접 읽는다(2단계의 `efc_force` 필터링과는 다른
  API, EQUALITY가 아니라 보통 접촉 제약이라서).

**실측 확인** (`tests/test_subtask_graph.py`, 4개): 그래프 yaml
스키마, 알려진 좋은 게인(Kp_xy=0.124630/Kd_xy=0.001125, offset=
(0.014,0.0)) 실제 에피소드에서 align이 스텝10에 끝나고 insert가
스텝157에 끝남(기존 알려진 성공 스텝 수와 정확히 일치) -- align이
insert보다 항상 먼저 끝나는지, 이 시나리오에서 실제로 hole 벽 접촉이
있었는지(F1 메커니즘이 이론만이 아니라 실제로 발동하는 경로인지),
`classify_failure`의 우선순위 5가지 입력 조합, `align_progress`가
30스텝 뒤 0.773까지 올라가는지. 기존 회귀 스위트(`test_regression_
peg_in_hole.py`/`test_tacker_task.py`/`test_grasp_slip.py`) 전부 여전히
통과(이번 단계는 순수 함수/신규 메서드 추가뿐이라 기존 경로에 영향
없음).

### 4단계 — 폐루프 환경 어댑터 + 스크립트 복구 ✅ 완료

`closedloop/peg_in_hole_adapter.py`(신규 패키지): `PegInHoleClosedLoopEnv`
가 ARD-VLA의 `ClosedLoopEnv` duck-typing 프로토콜(`observe`/`execute`/
`subtask_finished`/`settle`/`safe_stop`/`run_recovery`) + 확장 메서드
`set_subtask(node_id)`를 구현한다. `set_subtask()`는 ARD-VLA 러너가
모르는 메서드라서(사용자 지시대로 "ARD-VLA 쪽에는 나중에 연결") 호출부가
`run_closed_loop(..., post_decision=...)` 콜백에서 `ADVANCE` 결정마다
직접 불러준다 -- 통합 테스트가 그 연결 방법을 실제로 보여준다.

**observe()**: 양팔 관절(`right_arm_qpos`/`left_arm_qpos`, 0단계 5번에서
"가능하다"고 확인했던 관절 공간 기록 -- `BimanualPegInHoleOpenArmSim`에
`right_arm_qpos()`/`left_arm_qpos()`/`*_gripper_qpos()`/
`joint_observation_16dof()` getter를 이번 단계에서 추가, 5단계도 이걸
재사용한다), 서브태스크 id, (`render=True`일 때만) top_cam/wide_cam
프레임. 렌더링은 `settle()` 호출 시점에 한 번만 갱신(판정 시점 이미지,
0단계 6번의 렌더링 비용 때문에 매 스텝은 안 함) -- `render=False`(기본)
면 `MUJOCO_GL`을 전혀 안 건드린다("렌더링 안 하는 스크립트에서는
MUJOCO_GL=osmesa 설정 금지" 규칙 유지). `render=True`로 osmesa를 켜는
경우도 이 모듈/통합 테스트 둘 다 PyTorch Adam을 만들지 않으므로(그게
세그폴트 조건, PIPELINE.md 위쪽 절 참고) 안전하다고 보고 적용했다 --
torch import 자체(ARD-VLA의 `closed_loop.py`가 함)는 세그폴트 조건이
아니다.

**오라클(`ScriptedOraclePolicy`)**: `predict_action_chunk_with_decisions
(batch, selector_mask=None) -> (actions, decisions)` + `cancel_pending_
actions()`를 학습 없이 물리값으로 구현. **액션 청크는 항상 길이
1**이다(`actions.shape==(1,1,3)`) -- admittance 컨트롤러가 매 스텝 실제
관측에 의존하는 연속 피드백이라 한 번의 관측으로 여러 스텝을 미리
정확히 예측할 수 없어서(파일 docstring 참고), 매 물리 스텝마다 새로
`controller_step()`을 불러 정확한 명령 하나만 돌려준다. `decisions`의
`success_prob`은 물리값 기반 0/1(사용자 지시 그대로), `failure_probs`는
3단계 `classify_failure()`의 결과를 4종(F1/F2/F3/timeout) 원-핫으로,
`progress`는 `align_progress`/`insert_progress`. 판정은 `subtask_
finished()`를 다시 부르지 않고 그 메서드가 True를 반환한 순간 남긴
`_subtask_done_naturally`/`_subtask_timed_out` 플래그만 읽는다(다시
부르면 hold-steps 카운터가 망가짐 -- 파일 docstring "판정은 방금 끝난
서브태스크를 본다" 절 참고).

**스크립트 복구 (controller_step 재사용, 학습 없음)**:

| ID | 동작 |
|---|---|
| R1 | no-op -- 다음 iteration이 현재 물리 상태에서 그대로 재시도 |
| R2 | `set_subtask()`가 찍어둔 체크포인트(qpos/qvel/`_virtual_qpos`)로 복원 + 재파지 |
| R3 | 게인 상승(`Kp_xy *= 1.3` 기본) + 재파지(물리 상태는 안 건드림) |
| R4 | `safe_stop()` 호출 |

**실측으로 발견한 버그와 수정 (재파지 ≠ 에피소드 리셋)**: 처음엔 R3가
`GraspSlipTracker.reset_episode()`를 그대로 썼는데, 그 메서드가
`injected`/`release`의 "이미 발생함" 플래그까지 같이 풀어버려서 --
`injected_at_step`은 "그 스텝 **이상**이면" 조건이라, 그 스텝을 이미
넘긴 뒤에 플래그를 풀면 **다음 스텝에 같은 교란이 또 발동**했다(실측:
R3를 3번 연속 골라도 매번 똑같이 F2로 실패, max_retries 초과로 STOP).
`sim/grasp_slip.py`에 `regrasp()`(물리적 그립만 복원, 플래그는 안
건드림)를 새로 추가해서 R3/R2 둘 다 이걸 쓰도록 고쳤다 -- `reset_
episode()`는 에피소드 전체를 다시 시작할 때만 쓴다.

**실측으로 발견한 한계 (R3만으로는 부족, R2가 필요했다)**: injected
+15mm(스텝80, 2단계에서 실패 확인된 값)로 F2를 일으킨 뒤 R3(재파지만,
물리 상태는 그대로)로 복구를 매핑했더니 **그립은 고쳐졌지만(재주입 안
됨) insert가 그래도 600스텝 예산 안에 못 끝났다**(timeout으로 3번
연속 재시도 후 STOP). 추적해보니 xy는 재파지 후에도 금방 안정됐지만
(outer_half 안쪽 1-3mm) **삽입 깊이(z)가 진행하다가 역행하는 패턴**이
있었다 -- 추정 원인(확실친 않음): resolved-rate IK의 "가상(shadow) 상태"
추적점은 고정 오프셋(`_PEG_LOCAL_OFFSET`)을 가정하는데, 교란으로 peg가
그 가정과 다른 실제 위치에 있었던 동안 가상 상태와 실제 상태가 벌어져서,
그립만 고쳐도(relpose는 복원되지만 팔의 현재 qpos/가상 상태는 안 바뀜)
그 벌어진 차이가 안 없어진다는 뜻으로 보인다. **R2(체크포인트로 qpos/
qvel/virtual_qpos까지 전부 복원 + 재파지)로 바꾸니 바로 복구됐다** --
그래서 통합 테스트는 F1/F2/F3를 전부 R2로 매핑한다(R3는 코드/테스트
모두 있지만, 이 특정 교란 시나리오의 복구용으로는 선택하지 않았다 --
다른 더 약한 교란에는 충분할 수도 있음, 실측 안 함).

**통합 테스트** (`tests/test_closed_loop_adapter.py`, `ARD_VLA_REPO`
환경변수로 ARD-VLA 저장소 경로를 주면 실행, 없으면 skip): 실제
`lerobot.policies.smolvla.decision_logic.DecisionLogic`/`closed_loop.
run_closed_loop`를 import해서(둘 다 `third_party/lerobot/src`에 있고,
`decision_logic.py`는 torch 의존성이 없고 `closed_loop.py`는 torch만
의존 -- 전체 lerobot 패키지의 다른 무거운 의존성 없이 단독 import 성공
확인), `injected`(F2, +15mm, 스텝80)로 교란을 주입한 뒤 align→insert→
(F2 실패 감지)→R2 복구→insert 재시도→성공까지 끝까지 돌려서
`termination_reason=="task_graph_end"`이고 `decision_history`에
`('recover','R2','F2')`가 있는지 확인한다(1회 통과 확인). ARD-VLA 없이
도는 어댑터 자체 smoke test 2개(프로토콜 다섯 메서드, R1~R4 네 복구)도
있다. 기존 회귀 스위트 전부 여전히 통과(새 getter/모듈 추가뿐, 기존
경로 비영향).

### 5단계 — 통합 데이터 스키마 + 소규모 생성 ✅ 완료

**스키마 확장** (`pipeline/episode_io.py`, 기존 키 전부 유지): 선택적
최상위 필드 4개 추가 -- `closed_loop`(매 스텝: `joint_16dof`(T+1,16),
`progress`(T,), `subtask_id`(T,), `grasp_right`/`grasp_left`(각각
`weld_active`/`weld_force`/`slipped`/`cumulative_shift`, 2단계
GraspState 그대로)), `subtask_boundaries`(서브태스크 경계마다
node/start_step/end_step/finished_naturally/failure_type/attempt/
recovery_applied를 담은 JSON 리스트, scene_config와 같은 직렬화
패턴), `disturbance`(교란 종류·강도 JSON), `candidate_chunks`(K-분기
결과, 기본 안 씀). `split`/`loop_mode`는 이미 있던 "스칼라 자동 저장"
경로(`extra_*`)를 그대로 탄다. 경계 이미지는 `add_boundary_images()`
(신규)로 **나중에** 별도 npz 키(`boundary_step{N}_{camera}`)로
덧붙인다 -- dict 추상화를 안 타고 npz를 직접 읽고 써서 큰 이미지
배열의 불필요한 복사를 줄였다. `tests/test_closed_loop_data_schema.py`
의 `test_old_style_episode_unaffected`로 옛 에피소드 왕복이 그대로인지
확인.

**16 DoF 레이아웃**: `[왼팔 7, 왼팔 그리퍼 1, 오른팔 7, 오른팔 그리퍼 1]`
(사용자 지시 "앞 8=왼팔, 뒤 8=오른팔" 그대로) -- 4단계에서 추가한
`joint_observation_16dof()`가 애초에 반대 순서(오른팔 먼저)였던 걸
이번 단계에서 바로잡았다(그 getter를 그때까지 아무도 안 썼어서 안전하게
고칠 수 있었다).

**생성 스크립트** (`pipeline/generate_closed_loop_episodes.py`, 신규):
4단계의 `PegInHoleClosedLoopEnv`/`ScriptedOraclePolicy`를 그대로
재사용한다(ARD-VLA의 `DecisionLogic`은 안 씀 -- 이 생성 스크립트는
외부 저장소 없이 독립적으로 돌아야 해서, 실패 감지 시 R2 복구를
최대 1회만 시도하는 아주 단순한 로컬 규칙으로 폐루프를 만든다).
`--k-branch`로 K-분기(기본 꺼짐, 비용 큼)도 지원 -- 서브태스크 시작
체크포인트에서 Kp_xy를 바꾼 K=4개 분기를 짧게(50스텝) 돌려 진행도만
기록하고 복귀한다.

**교란 세트 B** (학습용, 매 에피소드 7종 중 하나 무작위): action_noise/
xy_target_bias/bad_gains(CMA-ES x0)/narrow_clearance/high_friction/
injected_slip_F2/injected_slip_F3. 씬 자체는 명목 분포(오프셋 반경
0.009-0.014m, ±45°) 그대로 두고 교란만 다르게 해서, "교란이 실패
원인"이라는 라벨이 깨끗하게 나오게 했다.

**교란 세트 C** (테스트 전용, 학습에 안 씀 -- B보다 넓은 범위 3종):
wide_friction(1.8-2.2, B의 high_friction 1.0-1.8 바로 위)/
very_narrow_clearance(0.0012-0.0018m, B의 narrow_clearance
0.0012-0.0020m과 겹치지만 더 좁은 쪽 포함)/action_delay(5스텝).
**조명·카메라 자세 변화는 렌더링 단계 항목이라 이 물리 생성
스크립트에는 없다** -- Part 1의 시각 랜덤화 코드는 `ard-gen-v2`
브랜치에 있고 이번 작업은 그 브랜치를 병합하지 않는다(0단계 브랜치
관계 절 참고). 실측(N=20 단일 유형 격리 테스트): friction 1.8-2.2는
1/20, 1.4-1.8(B와 겹침, 참고용)은 7/20, clearance 0.0010-0.0014는
0/20, 0.0014-0.0018은 1/20 성공 -- **C는 의도적으로 거의 항상 실패하는
난이도다**(B 바로 위 경계를 썼더니 이 태스크 동역학이 그 근방에서
급격히 어려워졌다, 실측). 소규모 생성(N=16) 결과 7/16(43.8%) 성공 --
action_delay(5스텝) 비중이 높아서 wide_friction/very_narrow_clearance
단독 실측보다는 성공률이 높게 나왔다.

**교란 세트 A** (자연 실패, 0단계 7번 기준선 대비): 교란 없이 오프셋
반경만 0.009-0.020m(명목 0.009-0.014m보다 넓힘)·각도를 ±45°에서 전체
원(±180°)으로 넓혔다. 실측(명목 clearance/friction 그대로, 알려진
좋은 게인, N=60/시드777 그리고 N=100/시드999 두 번 재확인): **개루프
실패율 정확히 20.0%**(10-30% 목표 범위 안쪽) -- clearance를 같이
좁히거나 friction도 같이 올리면 실패율이 50%를 넘어서(A2/A3 실측,
각각 50.0%/53.3%) 목표 범위를 벗어났다, 그래서 오프셋 반경만 넓히는
걸로 확정. 소규모 생성(N=20, 다른 시드) 결과 17/20(85.0%) 성공 --
20% 목표와 거의 일치(15.0% 실패, 작은 N의 샘플링 변동 안쪽).

**분할**: `split` 필드는 `train`/`val`/`test_B`/`test_C`/`test_A` 중
하나(사용자 지시 그대로) -- B만 80:20으로 train/val을 더 나누고, C/A는
전부 test_*(C는 "학습에 안 씀", A는 자연 실패라 애초에 평가용). 분할은
에피소드(= 독립 시드) 단위라 같은 씬이 두 분할에 걸치는 일이 없다.

**실제 생성 결과** (`data/closed_loop/{A,B,C}/`, `.gitignore`에
`data/closed_loop/` 추가 -- 기존 `data/episodes`, `data/lerobot` 등과
같은 컨벤션, 커밋 안 함):

| 세트 | N | 분할 | 성공:실패 | loop_mode(open:closed) |
|---|---|---|---|---|
| A | 20 | test_A ×20 | 17:3 (85.0%) | 17:3 |
| B | 24 | train ×18, val ×6 | 11:13 (45.8%) | 7:17 |
| C | 16 | test_C ×16 | 7:9 (43.8%) | 7:9 |

서브태스크 경계 전체(세 세트 합산)의 `failure_type` 분포: `none`
94(성공), `timeout` 52, `F2` 1, `F3` 1 -- `injected_slip_F2`/`F3`를
고른 에피소드가 적어서(무작위 7종 중 하나, N이 작음) F2/F3 표본이
적다(소규모 생성의 한계로 그대로 기록 -- 더 많이 보려면 B의
disturbance_type 샘플링에서 injected_slip_* 비중을 올리거나 N을
늘리면 된다).

**렌더링** (`pipeline/render_closed_loop_boundaries.py`, 신규, 별도
프로세스/`MUJOCO_GL=osmesa`): 저장된 `right_arm.action`(매 스텝 실제
실행된 행동, 노이즈/편향/지연까지 이미 반영됨)을 그대로 재생해서
서브태스크 경계(시작/끝) 시점에 top_cam/wide_cam을 캡처한다 -- 제어
로직을 다시 계산하지 않고 액션을 그대로 재실행하므로 결정론적으로
똑같은 물리 경로가 나온다. **한계(정직하게 기록)**: R2 복구는 물리
상태를 체크포인트로 되돌리는 불연속 점프인데 그 시점 qpos/qvel 자체는
저장 용량 때문에 안 남겨둬서, 액션만 재생하는 이 방식으로는 복구가
일어난(`loop_mode=="closed"`) 에피소드를 재현할 수 없다 -- 그래서 이
스크립트는 `loop_mode=="open"`인 에피소드만 렌더링하고 나머지는 skip
한다(로그로 명시). 실제 결과: A 17/20 렌더링(3개 skip), B 7/24(17개
skip -- B는 injected_slip/narrow_clearance/bad_gains 등으로 closed
비중이 높아서), C 7/16(9개 skip). 다음에 이어서 풀 사람에게: 체크포인트
시점 qpos/qvel도 같이 저장하거나(용량이 커지는 트레이드오프),
`subtask_boundaries`에 "jump_to_step" 필드를 추가해서 재생 스크립트가
액션 재생 중간에 그 지점으로 다시 reset+재생하게 만들면 closed-loop
에피소드도 풀린다.

**실측으로 발견한 버그 하나 더(판정 한 박자 늦음, 4단계와 같은
함정)**: 생성 스크립트 초안은 `subtask_finished()`로 서브태스크 종료를
확인한 뒤 **그 직전에 받아둔(한 스텝 전 상태를 반영한) decisions**로
`failure_type`을 기록했다 -- 그래서 align이 방금 자연 성공으로 끝났는데도
`failure_type`이 "F1"로 잘못 찍혔다(실측으로 발견). ARD-VLA의
`closed_loop.py` 모듈 docstring이 경고하는 것과 똑같은 함정이라서, 같은
해법(경계를 감지한 **뒤에** 오라클을 한 번 더 불러 그 결과를 쓴다)으로
고쳤다 -- 수정 후 재확인: 자연 성공 경계는 전부 `failure_type=="none"`.

**새 테스트**: `tests/test_closed_loop_data_schema.py`(5개) -- 확장
필드 왕복, 옛 스타일 에피소드 비영향, A/B/C 각각 소규모 생성 스키마
확인, A세트 오프셋 반경이 실측 기록한 범위와 일치하는지(조용히 바뀌면
10-30% 보증이 깨지므로), 렌더링이 open-loop에 이미지를 추가하고
closed-loop을 skip하는지. 기존 회귀 스위트 전부 여전히 통과.

### 완료 보고 (main 대비 전체 diff 기준)

**변경·추가 파일** (16개, `+2742/-36`, `git diff --stat main..closed-loop-data`):

신규:
- `sim/grasp_slip.py` -- 2단계 파지 미끄러짐 모델
- `closedloop/__init__.py`, `closedloop/peg_in_hole_adapter.py` -- 4단계 폐루프 어댑터+오라클
- `tasks/peg_in_hole.graph.yaml`, `tasks/tacker.graph.yaml` -- 3단계 서브태스크 그래프
- `pipeline/generate_closed_loop_episodes.py`, `pipeline/render_closed_loop_boundaries.py` -- 5단계 생성/렌더링
- `tests/test_grasp_slip.py`, `tests/test_subtask_graph.py`, `tests/test_closed_loop_adapter.py`, `tests/test_closed_loop_data_schema.py` -- 신규 테스트 4개

수정(기존 키/동작 보존):
- `sim/peg_in_hole_bimanual_openarm_sim.py` -- `controller_step()`(1단계), `align_progress`/`insert_progress`/`align_subtask_finished`/`insert_subtask_finished`/`classify_failure`/`is_peg_touching_hole_wall()`(3단계), `grasp_slip_config` 생성자 인자 + `get_grasp_states()`(2단계), `right_arm_qpos()`/`left_arm_qpos()`/`*_gripper_qpos()`/`joint_observation_16dof()`(4/5단계) 추가
- `sim/peg_in_hole_openarm_env.py` -- `run_episode()`가 `controller_step()` 재사용(1단계)
- `pipeline/episode_io.py` -- `closed_loop`/`subtask_boundaries`/`disturbance`/`candidate_chunks` 선택적 필드 + `add_boundary_images()`(5단계)
- `.gitignore` -- `data/closed_loop/`, `tests/_tmp_closed_loop_schema/` 추가
- `PIPELINE.md` -- 이 "폐루프 데이터" 절 전체

**새 설정 플래그 표**:

| 플래그/인자 | 위치 | 기본값 | 의미 |
|---|---|---|---|
| `grasp_slip_config` | `BimanualPegInHoleOpenArmSim(...)` | `None` | 파지 미끄러짐 모델 전체 on/off |
| `ArmSlipConfig.mode` | `sim/grasp_slip.py` | `"none"` | `none`/`force_limited`/`injected`/`release` |
| `render` | `PegInHoleClosedLoopEnv(...)` | `False` | top_cam/wide_cam 캡처(이때만 MUJOCO_GL 건드림) |
| `r3_kp_boost` | `PegInHoleClosedLoopEnv(...)` | `1.3` | R3 복구의 게인 상승 배수 |
| `--k-branch` | `generate_closed_loop_episodes.py` | 꺼짐 | K-분기(후보 청크) 기록 |

전부 기본값이 꺼짐(또는 기존 동작과 동일)이고, `tests/test_regression_peg_in_hole.py`/`tests/test_tacker_task.py`(1~5단계 전부 작업 끝난 지금도 그대로 통과)로 비트 단위 동일성을 확인했다.

**테스트 수**: main 기준 2개 파일/7개 함수(`test_regression_peg_in_hole.py` 3개,
`test_tacker_task.py` 4개) -> 이 브랜치 6개 파일/24개 함수(새 파일 4개에
17개 함수 추가). 전부 통과(`ARD_VLA_REPO` 미설정 시 그 통합 테스트
하나만 skip, 나머지 23개는 항상 실행).

**0단계 실측값**: 명목 개루프 기준선 200/200(100.0%, N=200, 시드
고정); weld 구속력 분포(명목, 3개 eval_scenarios) -- `peg_grasp`(오른팔)
mean 0.71/max 11.86/p95 7.00, `hole_grasp`(왼팔) mean 10.04/max
18.55/p95 15.82; 렌더링 비용(osmesa, 640x480) top_cam 324ms/frame,
wide_cam 441ms/frame.

**2단계 게이트 결과**: force_limited를 명목 p95 임계값(가장 타이트)까지
켜도 N=200에서 성공률 하락 0%p(200/200 유지, 175/200이 실제로
미끄러짐을 겪고도 성공) -- 5%p 한도 안쪽, 게이트 통과.

**실패 유형별 발생 빈도** (소규모 생성 데이터 `data/closed_loop/{A,B,C}`
서브태스크 경계 합산, 세 세트 합쳐 94+52+1+1=148개 경계): `none`(성공)
94, `timeout` 52, `F2` 1, `F3` 1 -- F1은 관측되지 않았다(이 데이터에서는
"깊이 미달+벽 접촉" 조합이 "timeout"보다 먼저 분류될 상황이 없었다는
뜻, 우선순위상 F2/F3가 먼저 걸리거나 벽 접촉 없이 그냥 timeout인
경우가 대부분이었다). F2/F3가 1개씩밖에 없는 건 B의 7종 교란 중
`injected_slip_*`가 무작위로 적게 뽑힌 소규모 생성의 샘플링 한계 --
더 보려면 N을 늘리거나 B의 교란 샘플링에서 `injected_slip_*` 비중을
올리면 된다(코드 변경 없이 `_B_DISTURBANCE_TYPES`에 중복으로 넣는 것만
으로도 가능).

**개루프 실패율 10-30%를 만든 씬 범위(A세트)**: `peg_init_offset_xy`
반경 U(0.009m, 0.020m), 각도 U(-π, π)(전체 원 -- 명목은 U(-π/4, π/4)),
clearance/friction은 명목(U(0.0025,0.0035)m / U(0.2,0.8)) 그대로.
실측 20.0%(N=60, 시드777 / N=100, 시드999 두 번 재확인) -- clearance를
같이 좁히거나 friction도 같이 올리면 50%를 넘어서 범위를 벗어난다
(그래서 오프셋 반경만 넓히는 것으로 확정).

**16 DoF 관절 기록 가능 여부**: **가능, 이번 작업에서 실제로 구현함.**
main의 npz 스키마는 EE pose(4차원)/델타 액션(3차원)만 있고 관절 공간이
전혀 없었다 -- `self.data.qpos`를 기존 `_arm_qposadr`/`_left_arm_qposadr`
딕셔너리로 그냥 읽으면 되는 것뿐이라 기술적 장애가 없었다(4/5단계에서
`right_arm_qpos()`/`left_arm_qpos()`/`*_gripper_qpos()`/
`joint_observation_16dof()` 추가, `closed_loop.joint_16dof`로 실제
저장까지 확인).

**아직 확인하지 못한 항목** (정직하게 기록):
- tacker의 서브태스크 그래프(`tasks/tacker.graph.yaml`)는 메타데이터뿐
  -- 완료 조건 체크 코드/진행도 함수/폐루프 어댑터가 없다(`implemented:
  false`로 명시). peg_in_hole만 전체 파이프라인이 끝까지 연결됨.
- `closed-loop`(복구가 일어난) 에피소드의 경계 이미지 렌더링 -- R2
  체크포인트의 qpos/qvel을 안 저장해서 액션 재생만으로는 재현 불가,
  5단계 절에 다음 단계 제안 기록.
- injected +15mm가 왜 비단조적으로(15mm는 거의 항상 실패, 20-30mm는
  오히려 성공) 거동하는지 근본 원인 -- 2단계에서 실측만 하고 원인 분석은
  안 함.
- R3(재파지만, 물리 상태 안 되돌림)이 F2/F3 복구로 충분한 **다른**
  (더 약한) 교란 시나리오가 있는지 -- 4단계 통합 테스트의 특정 시나리오
  (15mm, 스텝80)에서는 R2가 필요했다는 것만 확인했고, R3가 전혀 못
  쓰는 복구인지는 추가로 안 봤다.
- F1(정렬 실패/걸림) 라벨이 실제로 나오는 시나리오를 소규모 생성에서
  하나도 못 봤다 -- `is_peg_touching_hole_wall()` 자체는
  `tests/test_subtask_graph.py`의 성공 에피소드 안에서도 벽 접촉이
  있었음을 확인했지만(스쳐가는 접촉), "깊이 미달 + 벽 접촉"이 동시에
  성립하는 진짜 실패 사례는 소규모 생성 N에서 안 나왔다.
- Judge/Selector 실제 학습, SmolVLM2 특징 추출, 폐루프 vs 개루프 실험
  비교 -- 전부 범위 밖(사용자 지시), 전혀 안 함.
- 대규모(수백~수천 에피소드) 데이터 생성 -- 범위 밖, 전혀 안 함.

**브랜치**: `closed-loop-data`(main에서 분기, 6개 커밋: 0~5단계 각 1개),
push 완료. `claude/queue-dyy0ir`/`ard-gen-v2`/`ard-gen-insertion`/
`ard-gen-insertion-horizontal`/`feature/official-openarm-model`은
병합하지 않았다(0단계 브랜치 관계 절 그대로).

### 후속 작업 1 — 폐루프 에피소드 경계 이미지 ✅ 완료

**문제**: `render_closed_loop_boundaries.py`(저장된 액션 재생)는 R2 복구의
불연속 점프를 재현 못 해서 `loop_mode=="closed"` 에피소드를 전부 skip했다
-- 5단계 소규모 생성에서 B세트 24개 중 7개만 이미지가 있었고, 하필
그 7개가 전부 복구가 "안" 일어난(즉 Judge 학습에 덜 중요한) 쪽이었다.

**선택한 방법: A (생성 중 바로 캡처)**. B(상태 전체를 저장해 재생
스크립트가 점프까지 재현)는 검토했지만 재구성해야 할 상태가 많다
(qpos/qvel/`_virtual_qpos`/양쪽 weld의 `eq_data`+`eq_active`/
`body_gravcomp`/grasp_slip 내부 플래그/점프한 스텝 번호)는 점과, 어차피
생성 프로세스가 물리 상태를 실시간으로 들고 있으므로 재구성할 필요 자체가
없다는 점에서 A가 더 단순했다. 전제 조건(torch 미사용)은 실측 확인:
`python3 -c "import pipeline.generate_closed_loop_episodes; 'torch' in
sys.modules"` → `False`(이 스크립트 자체도, 재귀적으로 당겨오는 모든
모듈도 torch를 안 쓴다) -- `assert_no_torch_imported()`를 생성 시작 시
호출하게 해두고, 그 assert 자체도 테스트로 남겼다
(`tests/test_closed_loop_data_schema.py::test_assert_no_torch_imported`).

**구현**:
- `closedloop/peg_in_hole_adapter.py`에 `PegInHoleClosedLoopEnv.render_frames()`
  추가(`settle()`의 캐시와 별개로 호출 시점 top_cam/wide_cam을 즉시
  새로 렌더).
- `pipeline/generate_closed_loop_episodes.py`의 `run_one_episode(...,
  render=False)` -- 에피소드 시작 직후 1회, 그리고 매 서브태스크 경계마다
  `render_frames()`로 캡처. 다음 경계의 "start"는 점프가 없으면(자연
  완료 후 다음 노드로 진행) 방금 찍은 "end"를 그대로 재사용하고, 점프가
  있으면(`run_recovery("R2")`) 그 직후 다시 캡처한다(물리 상태가
  바뀌므로). 키 형식은 `episode_io.py`가 이미 문서화해둔 그대로
  `boundary_{idx}_{camera}_{start|end}`(idx=subtask_boundaries 리스트
  인덱스 -- 점프로 전역 스텝 수가 꼬여도 유일함, 기존
  `render_closed_loop_boundaries.py`의 `boundary_step{N}_{camera}`
  스킴과는 의도적으로 다르다).
- `pipeline/episode_io.py::save_episode()`가 `episode["boundary_images"]`
  를 그대로 npz에 병합하도록 확장(`_FIXED_TOP_LEVEL_KEYS`에 추가) --
  `add_boundary_images()`가 나중에 덧붙이는 것과 같은 모양.
- 시각 교란(`VisualDisturbanceConfig`/`apply_visual_disturbance()`,
  C세트 전용·기본 꺼짐): 조명 방향(±0.15rad)/세기(±25%), top_cam/
  wide_cam 위치(±3cm), 구멍/페그 RGB 곱(±15%) -- 컴파일된 모델 인스턴스를
  직접 변형(`_apply_clearance()`와 같은 패턴, 인스턴스 간 간섭 없음).
  물리에는 전혀 영향 없음(geom_rgba/cam_pos/light_*만 건드림). 실제
  적용값은 `episode["disturbance"]["visual"]`에 기록.
- `--render`/`--visual-disturbance` CLI 플래그 추가. `--render`가 없으면
  `MUJOCO_GL`을 절대 안 건드린다(이미 있던 "렌더링 안 하면 osmesa 설정
  금지" 규칙 유지, `sys.argv`를 argparse보다 먼저 직접 검사).

**결정론 확인** (실측, `test_method_a_matches_old_replay_for_open_episode`):
open-loop 에피소드 하나를 방법 A(생성 중 캡처)와 기존
`render_closed_loop_boundaries.py`(저장된 액션 재생) 둘 다로 렌더링해서
같은 순간(경계의 start_step/end_step)의 이미지를 픽셀 단위로 비교 --
**8쌍(경계 3개×카메라 2개 중 일부, 첫 경계는 start==episode 시작이라
0 스텝 하나로 합쳐짐) 전부 최대 절대 차이 0, 완전히 일치**. 같은
시뮬레이션 상태에서 같은 렌더러로 그리므로 당연한 결과지만, 실제로
어긋나는 부분(예: 카운터 꼬임, 캡처 시점 오프바이원)이 없다는 걸 실측으로
확인한 것.

**경계 이미지 커버리지** (실측, A 10개/B 12개/C 10개 소규모 재생성,
`--render`, C는 `--visual-disturbance`도 같이):

| 세트 | 전체 에피소드 중 모든 경계에 이미지 있음 | 그 중 closed-loop만 |
|---|---|---|
| A | 10/10 (100.0%) | 1/1 |
| B | 12/12 (100.0%) | 5/5 |
| C | 10/10 (100.0%) | 6/6 |

이전(재생 방식, closed 전부 skip)과 비교하면 B의 closed-loop 커버리지가
0/7(5단계 당시 실측 "24개 중 7개만 이미지") → 5/5(이번 재생성에서 나온
closed 전부)로 바뀐 셈 -- 표본 수가 달라 직접 비율 비교는 어렵지만,
"복구가 일어난 에피소드는 이미지가 없다"는 구조적 공백 자체가 사라졌다는
점이 핵심.

**렌더링 비용** (실측, osmesa 소프트웨어 렌더러, 이 컨테이너 기준):

| 세트 | 에피소드 수 | 총 시간 | 에피소드당 | 경계 수 | 경계당 |
|---|---|---|---|---|---|
| A | 10 | 50.0s | 5.00s | 21 | 2.38s |
| B | 12 | 73.1s | 6.09s | 29 | 2.52s |
| C | 10 | 66.2s | 6.62s | 26 | 2.55s |

경계당 ~2.4-2.6초(카메라 2개×2(start/end) 렌더 포함) -- osmesa
소프트웨어 렌더러의 비용이 대부분이고 세트별 차이는 경계 수(복구 발생
빈도)에서 온다. **대규모 생성에 이 비용을 그대로 곱하면 비싸진다(예:
B 1000개면 경계 수 비례로 수 시간) -- 그래서 사용자 지시대로 이번에도
대규모 생성은 하지 않았고, 다음에 할 사람은 `--render` 없이 물리만
먼저 다 생성한 뒤 이미지가 필요한 부분집합만 따로 렌더링하는 전략을
고려할 것**(이 스크립트 구조상 `render=False`로 생성→나중에 같은 시드로
`render=True` 재생성해도 결정론이 보장되므로 가능, 위 결정론 확인 절
참고).

**A/B 이미지는 이전과 동일함 확인**: `--render`(및 C에서 `visual_disturbance.
enabled=False`) 자체가 물리를 전혀 안 바꾸는지 실측 -- 같은 시드로
`render=False` vs `render=True`를 돌려 `right_arm.action`/`success`가
완전히 동일(`np.array_equal`)함을 A/B/C 전부에서 확인(`test_visual_disturbance_
only_affects_c` 및 수동 비교 스크립트). 시각 교란이 켜진 건 코드상 split
=="C"일 때만 `apply_visual_disturbance()`를 호출하므로, A/B는 이 기능이
생기기 전과 호출 경로 자체가 다르지 않다(영향받을 길이 없음).

**테스트**: `tests/test_closed_loop_data_schema.py`에 4개 추가
(`test_assert_no_torch_imported`/
`test_method_a_inline_render_covers_closed_loop_episodes`/
`test_visual_disturbance_only_affects_c`/
`test_method_a_matches_old_replay_for_open_episode`) -- 전부 통과.
기존 회귀 테스트(`test_regression_peg_in_hole.py`/`test_grasp_slip.py`/
`test_subtask_graph.py`/`test_closed_loop_adapter.py`)도 전부 그대로
통과(이번 작업이 기존 동작을 안 건드렸다는 확인).

**아직 확인 못 한 것**: 대규모(수백+) 생성에서의 실제 비용/저장 용량
(소규모 N=10-12로만 실측), 조명/카메라/색 지터 범위가 "현실적인 도메인
랜덤화"로 적절한지(임의로 고른 값, 별도 조정 없음), `top_cam`/`wide_cam`이
`mode="targetbody"`(고정 타겟 추적) 상태에서 `cam_pos`를 지터링했을 때의
시각적 결과가 실제로 그럴듯한지(픽셀 diff로 "바뀌었다"는 확인만 했고
사람이 눈으로 본 적은 없음).

### 후속 작업 2 — 교란 분포 재조정 ✅ 완료 (게이트는 통과, 걸리지 않음)

**문제**: `force_limited`는 admittance가 과부하를 실시간으로 보상해서
임계값을 아무리 좁혀도 실패를 못 만든다(2단계에서 이미 확인된 결론,
이번에 다시 스윕하지 않았다 -- 사용자 지시 범위가 `injected`만). 5단계
소규모 생성(B 24개)의 경계 판정 분포는 `timeout 52 / F2 1 / F3 1`로
F2/F3가 거의 없었다 -- `injected`가 고정 15mm·고정 ee-로컬 x축·시드
2개짜리 변동(부호, step randint(60,150))뿐이라 표본이 너무 작고 운에
좌우됐다(이번 작업에서 재현: 같은 코드로 새 시드 60개를 돌려도
`timeout 29.1% / F2 4.7% / F3 2.0%`로 비슷하게 희소함, "아래 B 재생성
표"의 "이전" 열 참고).

#### 2-1. 스윕 (`pipeline/sweep_injected_slip.py`)

무작위 씬(B와 같은 범위, 고정 씬 아님) 30개를 뽑아 각각 교란 없이
한 번 먼저 돌려(dry run) 그 씬의 insert 시작 스텝/길이를 구하고
(실측: 전부 insert_start=10, insert_len 범위 121-152·중앙값
146 -- B 난이도 씬에서는 놀랍도록 일정했다), 그 비율(진행도 0.3/0.6)에
해당하는 절대 스텝에 같은 씬으로 다시 주입해서 magnitude(10/15/20/30mm)
x 방향(+x/-x) x 타이밍(진행도 0.3/0.6) x 대상 팔(right=F2/left=F3)
그리드(32칸, 칸당 N=30)를 돌렸다. 복구 없이 align→insert 한 번만
도는 단순화된 러너를 썼다(복구가 섞이면 "이 교란 자체가 뭘 만드는지"
신호가 흐려짐).

**결과 표** (실패율/경계 판정 분포/"미끄러짐은 났지만 성공"률, 전부
N=30):

| arm | mag(mm) | dir | frac | fail% | F2% | F3% | timeout% | none% | slip-but-ok% |
|---|---|---|---|---|---|---|---|---|---|
| right(F2) | 10 | +x | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| right(F2) | 10 | +x | 0.6 | 16.7 | 16.7 | 0.0 | 0.0 | 83.3 | 83.3 |
| right(F2) | 10 | -x | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| right(F2) | 10 | -x | 0.6 | 86.7 | 86.7 | 0.0 | 0.0 | 13.3 | 13.3 |
| right(F2) | 15 | +x | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| right(F2) | 15 | +x | 0.6 | 96.7 | 96.7 | 0.0 | 0.0 | 3.3 | 3.3 |
| right(F2) | 15 | -x | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| right(F2) | 15 | -x | 0.6 | 73.3 | 73.3 | 0.0 | 0.0 | 26.7 | 26.7 |
| right(F2) | 20 | +x | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| right(F2) | 20 | +x | 0.6 | 70.0 | 70.0 | 0.0 | 0.0 | 30.0 | 30.0 |
| right(F2) | 20 | -x | 0.3 | 3.3 | 3.3 | 0.0 | 0.0 | 96.7 | 96.7 |
| right(F2) | 20 | -x | 0.6 | 90.0 | 90.0 | 0.0 | 0.0 | 10.0 | 10.0 |
| right(F2) | 30 | +x | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| right(F2) | 30 | +x | 0.6 | 80.0 | 80.0 | 0.0 | 0.0 | 20.0 | 20.0 |
| right(F2) | 30 | -x | 0.3 | 30.0 | 30.0 | 0.0 | 0.0 | 70.0 | 70.0 |
| right(F2) | 30 | -x | 0.6 | 93.3 | 93.3 | 0.0 | 0.0 | 6.7 | 6.7 |
| left(F3) | 10 | +x | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| left(F3) | 10 | +x | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| left(F3) | 10 | -x | 0.3 | 16.7 | 0.0 | 16.7 | 0.0 | 83.3 | 83.3 |
| left(F3) | 10 | -x | 0.6 | 33.3 | 0.0 | 33.3 | 0.0 | 66.7 | 66.7 |
| left(F3) | 15 | +x | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| left(F3) | 15 | +x | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| left(F3) | 15 | -x | 0.3 | 3.3 | 0.0 | 3.3 | 0.0 | 96.7 | 96.7 |
| left(F3) | 15 | -x | 0.6 | 50.0 | 0.0 | 50.0 | 0.0 | 50.0 | 50.0 |
| left(F3) | 20 | +x | 0.3 | 6.7 | 0.0 | 6.7 | 0.0 | 93.3 | 93.3 |
| left(F3) | 20 | +x | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| left(F3) | 20 | -x | 0.3 | 13.3 | 0.0 | 13.3 | 0.0 | 86.7 | 86.7 |
| left(F3) | 20 | -x | 0.6 | 83.3 | 0.0 | 83.3 | 0.0 | 16.7 | 16.7 |
| left(F3) | 30 | +x | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| left(F3) | 30 | +x | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| left(F3) | 30 | -x | 0.3 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| left(F3) | 30 | -x | 0.6 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |

(timeout 열이 모든 칸에서 0.0%인 건 이 스윕 전용 러너가 복구 없이
한 번만 돌기 때문 -- `_MAX_ITERATIONS=2000`이 충분히 커서 "실패"가
전부 insert 서브태스크의 `classify_failure()`가 F2/F3로 판정될 만큼
명확했다는 뜻이고, 실제 생성 파이프라인(복구 있음)에서는 timeout도
나온다, 아래 B 재생성 표 참고.)

**게이트 결과: 통과(걸리지 않음)**. F3(왼팔, Stabilizer)도 -x 방향
30mm에서 N=30 전부(100%) 실패를 만들어냈다 -- "hole_grasp 주입이 어떤
범위에서도 실패를 거의 못 만든다"는 게이트 조건이 성립하지 않았다.
다만 F2와는 성격이 다르다: **방향 의존성이 강하다**(+x는 0-6.7%,
-x는 16.7-100%) -- F2는 두 방향 다 어느 정도 효과가 있는데(+x
16.7-96.7%, -x 73.3-93.3%) F3는 +x가 거의 항상 무의미하다. 그래서
F3는 F2보다 magnitude를 더 크게(20-35mm vs 10-30mm), 방향도 전체
원이 아니라 -x 근방으로 몰아서 뽑아야 한다(아래 2-2 설정 참고).

**비단조성 원인 메모** (사용자 지시대로 깊이 안 팜, trajectory
레벨만): right(F2) 15mm/+x/frac=0.6는 96.7% 실패인데 20mm/+x/frac=0.6는
70%로 더 낮다 -- "더 큰 교란이 덜 실패"하는 비단조 패턴이 실제로
재현됨. 같은 씬(스윕의 첫 번째 씬)에서 15mm와 20mm 둘 다를 추적해보면
(`dx`가 주입 직후 +3~4mm로 튀었다가 약 40스텝에 걸쳐 진동하며
-4~-6mm까지 오버슈트한 뒤 서서히 0으로 수렴), 두 magnitude 모두
비슷한 형태의 감쇠 진동을 보였지만 **그 진동의 위상/타이밍이 미묘하게
달랐다** -- 이 특정 씬에서는 둘 다 결국 실패(=이 씬 하나로는 "15mm가
더 나쁘다"는 집계 추세를 재현 못 함, 집계 추세는 여러 씬 평균의
효과). 정성적 해석: 주입은 relpose에 거리를 한 번 영구히 더하는
"계단 입력"이고, admittance 컨트롤러가 그걸 쫓아가는 과정이 감쇠
진동이다(PD 제어의 전형적인 거동) -- hold_steps 조건(깊이+정렬이
"연속" 몇 스텝 유지돼야 함)이 이 진동의 정확한 위상에 걸리기 때문에,
magnitude가 조금만 달라져도(진동 주기/오버슈트 크기가 바뀌므로) 마침
그 순간에 두 조건이 동시에 맞아떨어지는지 여부가 바뀌어서 비단조적인
실패율로 집계된다 -- "운 좋게 수렴 타이밍이 hold window에 맞는가"의
문제라, 원인을 깊이 파려면(사용자 지시로 범위 밖) 진동의 주파수/감쇠율을
게인·목표깊이에 대해 명시적으로 풀어야 할 것으로 보임.

#### 2-2. 재조정 (`pipeline/disturbance_config.yaml`)

B의 교란 타입 가중치(`b_weights`)와 `injected`의 magnitude/방향/
타이밍 연속분포를 전부 YAML로 뺐다(사용자 지시 "코드 상수로 남기지
않는다") -- `generate_closed_loop_episodes.py --disturbance-config
pipeline/disturbance_config.yaml`로 켠다. **안 주면(기본) 5단계 원래
동작과 bit-identical**(새 기능 기본 꺼짐 규칙, `test_disturbance_
config_off_is_bit_identical`로 확인).

최종 설정(위 스윕 결과 기준, 세 번 정도 반복 조정한 값 -- 아래 "실제
달성 분포" 절 참고):
- `b_weights`: 기존 5타입(action_noise/xy_target_bias/bad_gains/
  narrow_clearance/high_friction)은 1.0, `injected_slip_F2`=9.0,
  `injected_slip_F3`=10.0(13개 중 F2 9/13=36%, F3 10/13 ≈ F2와 비슷,
  나머지 5타입은 각 1/29≈3.4%로 줄어듦 -- F2/F3 표본을 늘리는 대가로
  다른 5타입의 비중이 크게 줄었다는 걸 명시적으로 기록).
- `injected.magnitude_mm`: F2 [10,30], F3 [20,35](F3가 방향 의존성
  때문에 더 큰 magnitude가 필요, 스윕 결과 반영).
- `injected.direction`: F2는 전체 원에서 균등(스윕에서 양방향 다
  효과적이었으므로), F3는 -x(180도) 중심 ±40도로 몰고 30%는 전체
  원에서 뽑음(`off_target_prob`) -- 방향이 안 맞아 실패 안 하는
  경우를 hard negative로도 활용.
- `injected.timing_effective_step_range`=[75,135] /
  `timing_hard_negative_step_range`=[15,55](절대 스텝 -- 2-1 스윕의
  진행도 0.45-0.85/0.05-0.3을 B 난이도 씬의 전형적인 insert_start=10·
  insert_len≈146으로 환산), `hard_negative_prob`=0.25(타이밍을
  일부러 "거의 안 실패하는" 구간에서 뽑을 비율).

**실제 달성 비율 실측** (설정값과 다를 수 있음을 솔직히 기록 --
타이밍 구간만으로는 hard negative가 보장되지 않고 방향/크기도
영향을 준다): 고정 시드 샘플 통계 테스트(N=4000)에서 `b_weights`
비율은 설정값과 3%p 이내로 일치(`test_disturbance_config_follows_
configured_weights`). B 재생성(N=120, 아래)에서 "미끄러짐은 났지만
episode는 성공"(hard negative, success=True & 최종 cumulative_shift
>3mm로 근사 측정 -- R2 복구가 중간에 relpose를 되돌리는 경우까지는
구분 못 함, 한계로 기록) 비율은 injected 타입 episode 95개 중 34개
= **35.8%**(설정한 `hard_negative_prob`=0.25보다 높음 -- 타이밍
구간과 무관하게 "방향이 안 맞아서"/"magnitude가 작아서" 자연히
실패 안 하는 경우도 hard negative로 잡히기 때문).

**B 재생성 경계 판정 분포 (이전 vs 이후)**:

| | 표본 | none | timeout | F2 | F3 | F1 |
|---|---|---|---|---|---|---|
| 이전(5단계, 24개, 사용자 지시에 인용된 실측) | 24ep | - | 52 | 1 | 1 | - |
| 이전 재현(이번 작업, 같은 코드 새 시드 60개) | 60ep/148경계 | 64.2% | 29.1% | 4.7% | 2.0% | 0.0% |
| 이후(재조정, 새 시드 120개) | 120ep/320경계 | 68.8% | 11.9% | 8.4% | 10.3% | 0.6% |

목표(F2/F3/timeout 각 10-15%)는 **F3(10.3%)/timeout(11.9%)는 달성,
F2(8.4%)는 근접했지만 10%에 살짝 못 미침** -- N=60 단위로는 시드마다
F2가 5~10% 사이에서 흔들렸다(아래 참고). 정직하게 기록: 이 가중치
(F2=9/F3=10)는 N=60 두 시드(seed=780: F2 9.0%/F3 10.3%/timeout
11.5%, seed=781: F2 5.1%/F3 13.4%/timeout 8.3%)와 N=120(seed=2200,
위 표의 "이후" 행) 세 번을 보고 고른 값이라 소규모 생성 특성상
표본 변동이 크다 -- F2 가중치를 더 올리면(예: 12) F2는 9.6%까지
오르지만 timeout이 7.0%로 떨어지는 trade-off가 있었다(가중치를 올려
F2/F3 표본을 늘리는 대신 timeout을 만드는 나머지 5타입의 표본이
줄어서). 지금 값(9/10)이 셋을 동시에 비교적 고르게 만족시키는 균형점
이었다.

**F1은 그대로 둠**(사용자 지시 범위 밖) -- 이번 재생성(120ep)에서도
2건(0.6%)만 나왔다, 5단계 때와 비슷하게 희소.

**테스트**: `tests/test_closed_loop_data_schema.py`에 2개 추가
(`test_disturbance_config_off_is_bit_identical`/
`test_disturbance_config_follows_configured_weights`) -- 전부 통과.
`disturbance_config=None`이면 B 샘플링이 5단계 원래 동작과 완전히
동일(action 배열까지 `np.array_equal`)함을 확인했다. 기존 회귀
테스트도 전부 그대로 통과.

**아직 확인 못 한 것**:
- F2 가중치를 조금 더 올리면 timeout/F3와의 균형을 유지하면서 F2도
  10% 이상으로 올라갈지 -- 세 번 반복 조정 후 "충분히 괜찮은" 값에서
  멈췄다(시간 예산상 더 정밀한 그리드서치는 안 함).
- 2-1 스윕은 방향을 +x/-x 둘만 봤다 -- 재조정의 F2 `full_circle`/F3
  `concentrated`(center_deg=180) 설정은 그 두 축만의 실측에서
  일반화한 것이고, 대각선 방향(예 45도)이 실제로 그 사이 거동을
  보이는지는 확인 안 했다.
- hard negative 비율 측정(35.8%)은 "최종 cumulative_shift"만 보는
  근사라서, R2 복구가 중간에 미끄러짐을 지운 뒤 재시도가 성공한
  경우(이것도 2-1 스윕과 무관하게 이미 일어나고 있었다 -- "2-1 스윕"
  절 위의 "B 재생성" 실측값 설명 참고)를 hard negative로 셀지 말지는
  모호하게 남아있다.
- Judge를 실제로 이 데이터로 학습해서 "F2/F3가 10% 안팎이면 충분한지"
  검증 -- 범위 밖(사용자 지시), 전혀 안 함.

