"""ARD-Gen peg-in-hole 태스크, OpenArm 양팔 버전 시뮬레이션 + admittance controller.

assets/peg_in_hole_bimanual_openarm.xml(오른팔=Actuator가 peg를 쥐고 hole에
꽂고, 왼팔=Stabilizer가 hole을 공중에서 붙잡는 모델) 위에서 돈다. 구조는
sim/peg_in_hole_sim.py(VX300s 단일 팔)와 거의 같다 -- resolved-rate
Jacobian IK + xy admittance PD, z는 일정 속도(Z_RATE) -- 하지만 두 가지가
다르다.

1. **7-DOF**: VX300s는 6개 팔 조인트였지만 OpenArm은 팔마다 7개
   (openarm_right_joint1..7). Jacobian 열이 하나 더 많을 뿐 damped
   least-squares 자체는 동일하게 동작한다.

2. **peg/hole이 그리퍼의 XML 자식이 아니라 weld로 붙은 자유 바디**
   (assets/peg_in_hole_bimanual_openarm.xml 상단 docstring 참고 -- OpenArm
   팔은 <attach>로 가져온 것이라 그 바디에 XML 자식을 못 붙임). 이게
   Jacobian 계산에 영향을 준다: mj_jacSite는 site가 강체 부모-자식 트리로
   연결된 경우에만 그 트리를 타고 올라가서 Jacobian을 만든다 -- weld
   equality로만 연결된 site는 (screw_driving.xml에서 이미 겪은 버그와
   똑같이) 팔 관절에 대해 0에 가까운 Jacobian을 내놓는다. 여기서는 별도
   "reference site"를 만드는 대신(그 방법도 안 통함 -- openarm_right_ee_base_link
   자체가 <attach>로 봉인된 바디라 참조용 site조차 못 붙임)
   **mj_jac(point, body)**를 직접 쓴다: peg_tip_site의 "현재 실제 월드
   좌표"를 openarm_right_ee_base_link에 강체로 붙어있다고 가정했을 때의
   Jacobian을 구해준다 -- 별도 site 없이, 그리고 site 트릭보다 더
   정확하다(그 트릭은 어차피 ee 원점 기준이라 실제 peg tip과의 모멘트암
   오차가 있었는데, mj_jac(point=...)는 원하는 지점 자체에 대해 정확히
   계산해준다).

reset() 시 오른팔 초기 자세를 잡는 반복 IK(펼치는 과정, 물리 시뮬레이션
아님)에서는 매 반복마다 peg free body의 qpos를 "그 시점 오른팔 ee pose로
부터 FK로 계산한 값"으로 직접 덮어쓴다(이 프로젝트에서 반복적으로 확인한
패턴: mj_forward만으로는 weld로 연결된 자유 바디가 안 움직이므로, 실제로
그 자리로 옮기려면 FK로 직접 qpos를 써야 한다). 왼팔은 이 태스크 내내
안 움직이므로(peg_in_hole_bimanual_sim.py의 왼팔 처리와 동일한 이유),
hole_socket qpos도 파일의 "home" 키프레임 값을 매번 그대로 복사해서 쓴다
(왼팔 qpos가 안 바뀌니 매번 다시 FK를 계산할 필요가 없음).

admittance 수식은 sim/peg_in_hole_sim.py와 완전히 동일(부호/이유 포함,
그 파일 docstring 참고) -- 팔 기구학만 바뀌었지 peg-hole 접촉 물리/컨트롤
법칙 자체는 이 저장소에서 이미 검증된 걸 그대로 재사용한다.

## 현재 상태: 삽입이 아직 안 됨 (정직하게 기록)

이 파일을 __main__으로 실행하면(대표 시나리오, VX300s가 찾은 기존
게인 Kp_xy=0.000515/Kd_xy=2.4e-05 그대로) **아직 실패한다**
(insertion_depth=0, max_force가 peg 자체 무게(0.49N)에 고정된 채 끝남
-- 즉 hole 벽에 실제로 닿아보지도 못하고 400스텝을 다 씀). 원인은
admittance 게인이 아니라 오른팔이 목표 궤적을 못 쫓아가는 것 -- 실측
경과를 그대로 남긴다(다음에 이어서 고칠 사람을 위해):

1. **vendor 액추에이터가 우리 액추에이터와 싸움**: assets/openarm/openarm_bimanual.xml
   의 <position> 액추에이터(left/right_joint{1..7}_ctrl)가 ctrl=0(기본값)
   그대로 남아있는데, home 자세 자체가 대부분 0이 아니라서 이게 계속
   "0으로 끌어당기는" 토크를 냈다(실측: right_joint2_ctrl이 -40N, 즉
   forcerange 한계까지 포화). __init__에서 이 14개 액추에이터의
   gainprm/biasprm을 0으로 만들어서 무력화했다. **-> 고쳤지만 이것만으론
   부족했다.**
2. **gravcomp 없음**: vendor 파일 어디에도 gravcomp가 없다(VX300s는 이
   프로젝트가 모든 팔 바디에 직접 넣어뒀음). __init__에서 body_gravcomp를
   1.0으로 덮어썼다. **-> 고쳤지만 이것만으론 부족했다.**
3. **레졸브드-레이트 널스페이스 표류**: 7-DOF가 3개(xyz) 목표만 만족하면
   되니 널스페이스가 4차원이다. 처음엔 이게 감쇠 없이 방치돼서 관절이
   0.4~0.8rad/s로 표류하는 걸로 보였다 -- home 자세로 당기는 2차 목표를
   널스페이스에 투영해서 추가했다(_NULLSPACE_GAIN=0.05). **-> 추가했지만
   qvel 표류/진동 패턴 자체는 실측상 별 차이가 없었다(널스페이스가
   원인이 아니었거나, 원인의 일부일 뿐이었다는 뜻).**
4. **여전히 남은 문제**: 위 세 가지를 다 고친 뒤에도 오른팔 관절
   속도(qvel)가 0.4~1.0rad/s 사이에서 스텝마다 부호가 바뀌는 진동을
   보인다(정상적인 매끄러운 추종이라면 이 정도 크기의 진동이 있으면 안
   됨). kp를 vendor 기본값의 최대 26배(6000/4500/1800), kv를 최대
   30배(kv=kp*0.3~0.6)까지 올려봐도 진동 패턴 자체가 거의 안 바뀌었다
   (이 정도로 게인에 둔감한 건 "게인 부족"보다는 "특정 방향으로 감쇠가
   근본적으로 안 맞다" 쪽에 가까운 신호로 보이는데, 정확한 원인은 못
   찾았다). Z_RATE를 4배 늦춰서 같은 거리를 더 오래 걸려 가도 최종
   드리프트가 거의 그대로였다(대역폭 문제가 아니라 이동 거리에 비례하는
   문제라는 뜻 -- 그래서 3번의 널스페이스 가설을 세웠던 것인데, 위처럼
   그것만으론 안 풀렸다). 결과적으로 peg tip이 hole 목표에서 xy로 최대
   ~7cm까지 벗어나서, z 삽입 깊이 자체는 목표(0.04m)에 근접해도
   "xy_within_hole_footprint" 판정에 걸려 성공 판정이 안 난다.

## 이어서: CMA-ES로 kp/kv 체계적 탐색 (VX300s 사례 재적용)

위 "다음 단계" 제안대로 실제로 해봤다. 손으로 추측한 kp/kv 대신, 먼저
mj_fullM으로 실제 질량행렬을 구해서 관절별 유효 관성(대각 성분, 예:
베이스 관절 M_ii≈0.12~0.19, 손목 관절 M_ii≈0.01~0.014)을 실측하고 그걸
CMA-ES 시작점(kv≈2*sqrt(kp*M_ii) 임계감쇠 추정치)으로 삼아, (kp_base,
kv_base, kp_mid, kv_mid, kp_wrist, kv_wrist, 나중엔 _NULLSPACE_GAIN까지)
7개 파라미터를 탐색했다(대표 시나리오 오프셋 14mm/0mm, 평가 길이 700
스텝, 비용 = 최고 삽입 깊이 + 궤적 뒷부분 300스텝의 평균 xy 오차).

**결과: 확실히 나아졌지만 아직 성공은 아니다.** 손으로 고른 값(모든
관절 kv=kp*0.1~0.6)에서는 xy 드리프트가 시간이 지날수록 계속 커지기만
했는데(최대 7cm, 발산), CMA-ES가 찾은 값에서는 드리프트가 **한계 진동
(bounded oscillation)으로 바뀌었다** -- 수렴/발산 없이 계속 진동만 한다.

발견한 것 중 특히 의외였던 점: CMA-ES가 수렴한 kv 값들이 거의 다 그
관절의 kp보다 크거나 비슷하다(특히 중간 관절은 kv가 kp의 몇 배씩) --
이건 처음에 손으로 시도했던 kv=kp*0.1~0.6 범위를 한참 벗어난다. 질량
행렬 대각 성분만으로 추정한 "임계감쇠"(관절당 5~40 정도)와도 완전히
다른데, 이건 대각 성분만으로는 이 정도로 결합(coupled)된 7-DOF 팔의
실제 동역학(비대각 관성/코리올리 항, 관절 간 상호작용)을 전혀 설명 못
한다는 뜻으로 보인다 -- 즉 진짜 원인은 "관절 하나짜리 2차 시스템"으로
단순화할 수 없는 다관절 결합 동역학이었던 것 같다(확실친 않음, 더
깊이 파려면 완전한 역동역학/computed-torque 제어가 필요해 보인다).

## 2차 탐색: optimize/peg_in_hole_openarm_gain_search.py로 정식화, 근접 실패 원인 발견

탐색 스크립트를 optimize/ 디렉터리로 정식 이관하고(재현 가능하게),
이전 결과(베이스 kp=2081.80/kv=1565.01, 중간 kp=145.00/kv=1052.43, 손목
kp=358.48/kv=569.50, _NULLSPACE_GAIN=0.052)에서 이어서 더 오래(18세대,
평가 길이 800스텝) 돌렸다. 결과: **베이스 kp=2273.98/kv=1568.18, 중간
kp=87.90/kv=894.41, 손목 kp=411.75/kv=663.96, _NULLSPACE_GAIN=0.03**
(현재 파일에 반영된 값) -- 1200스텝까지 실측해보니 **실제로 스텝 200
근처에서 dx=2.0mm, dy=4.9mm까지 거의 완벽하게 정렬된다**(hole 반경
19.5mm 안에 여유 있게 들어옴). 그런데 바로 그 순간 z_gap(peg tip이 hole
중심보다 얼마나 위에 있는지)이 아직 0.065m -- 목표(hole 중심보다
0.04m 아래, 즉 z_gap ≤ -0.04)에 한참 못 미친다. xy가 정렬된 채로
유지되는 시간(대략 스텝 150~250)보다 z가 그 안에 다 내려가는 데 걸리는
시간이 훨씬 길어서, "동시에" 조건을 못 채우고 지나간다 -- **원인이
게인 부족이 아니라 xy 정렬 구간과 z 하강 속도의 타이밍이 안 맞는
문제라는 걸 처음으로 명확히 확인했다.**

이 발견에 맞춰 비용 함수도 고쳤다(원인을 몰랐을 때는 "동시에 맞았을
때만 카운트되는 삽입 깊이"만 봐서, best_depth가 항상 0으로 고정돼
CMA-ES에 아무 방향 정보도 못 줬다 -- 지금은 "xy가 가장 잘 맞았던
순간의 판정 없는 원시 삽입 깊이"를 추가로 보상해서 이 타이밍 문제
자체에 그라디언트가 생기게 했다). 이걸로 한 번 더 돌려본 결과(18세대)
는 xy 정렬은 더 좋아졌지만(스텝 300 근처 dx=-0.8mm, dy=-0.5mm) z_gap이
오히려 0.06~0.07에서 안 내려가고 정체돼서(z 추종이 더 나빠짐) 전체적
으로는 개선이 아니었다 -- **xy 정렬과 z 추종이 이 게인 공간에서 서로
트레이드오프 관계라 6~7개 스칼라 파라미터로는 둘 다 동시에 잘 만족
시키기 어려워 보인다**(이 2차 결과는 파일에 반영 안 하고 버렸다 --
그래서 위 "1차 탐색" 값이 지금 파일의 상태다).

## 3차 시도: adaptive_z_rate (xy 정렬 상태에 따라 하강 속도를 가변으로) -- 역시 실패, 그리고 새로 발견한 문제

위 타이밍 불일치 가설대로 `adaptive_z_rate()`를 구현했다(xy 오차가
작을수록 Z_RATE에 가깝게, hole 반경의 `_Z_GATE_RADIUS_MULT`배 이상
벗어나면 `_Z_GATE_MIN_FRACTION`(기본 0.15)까지 선형으로 낮춤 -- 완전히
0으로는 안 낮춘다, 영원히 재정렬을 못 하면 그대로 멈춰버리는 걸 막기
위해). `_run_episode_with_sim`과 render_peg_in_hole_bimanual_openarm.py
양쪽 다 반영.

**실측 결과: 도움이 안 됐고, 오히려 새로운 문제를 발견했다.**
1. 게이트를 켠 상태로 1200스텝을 보면 스텝 280 근처에서 dx=6.4mm,
   dy=1.5mm까지 잘 맞는데(z_rate가 그 순간 0.000566까지 올라감, 최대
   Z_RATE=0.0006에 근접), 그 시점 z_gap은 여전히 0.059m -- 목표까지
   필요한 하강량(0.10m) 중 절반도 못 왔다. 정렬 구간(대략 200스텝)
   동안 번 z 진행량 자체가 필요량에 비해 너무 적어서, 게이트를 아무리
   잘 조절해도 "그 구간 안에" 다 못 내려간다.
2. `_Z_GATE_MIN_FRACTION`을 0.3/0.5/0.7/1.0(=사실상 게이트 없음, 원래
   고정 Z_RATE와 동일)까지 다 실측해봤는데 **1200스텝 안에 전부
   best_depth=0으로 실패**했다 -- 게이트를 아예 꺼도(1.0) 실패한다는
   건, 이 게인 조합 자체가 (게이팅 여부와 무관하게) 1200스텝 예산
   안에서는 원래 안 풀린다는 뜻이다(2차 탐색 검증 때 1200스텝을 본 건
   이번이 처음이었다 -- 그 전엔 700~800스텝만 봤음).
3. 더 심각한 발견: 게이트를 켠 채로 4000스텝까지 늘려봤더니, dy가
   **경계진동이 아니라 계속 커지기만 했다**(step 400: 24mm -> step
   4000: 89mm, 계속 우상향). 이전 절("1차/2차 탐색")에서 "한계 진동
   (bounded oscillation)"이라고 적은 건 1000~1200스텝까지만 본 것에
   근거한 결론이었는데, 더 길게 보면 사실은 아주 느리게 발산하고
   있었을 가능성이 크다 -- **"진동"이라는 진단 자체가 관찰 구간이
   짧아서 나온 착시였을 수 있다.**

이 세 가지를 종합하면: 문제는 "xy 정렬 구간과 z 하강 속도의 타이밍이
안 맞는다"는 표면적 증상보다 더 근본적이다 -- 지금의 (게인, Z_RATE,
resolved-rate 방식) 조합 자체가 이 태스크(7-DOF, 무거운 팔, 좁은 hole
공차)에서 장시간 안정적으로 수렴하는 궤적을 못 만든다. adaptive_z_rate
자체는 코드에 남겨뒀다(방향은 맞다고 보임, 최소한 해는 안 됨 -- 다음
사람이 더 나은 게인/제어 방식과 조합해서 다시 시도해볼 수 있게).

다음에 이어서 할 사람에게: (a) 관절별(6개 클래스가 아니라 14개 전부
따로) 게인을 CMA-ES로 찾되, **평가 길이를 최소 2000~4000스텝으로
늘려서**(위 발견 때문에 700~1200스텝 평가는 "성공처럼 보이는" 거짓
양성을 낼 수 있음) 장시간 안정성까지 같이 보는 것. (b) 아예 완전한
역동역학(computed-torque) 제어로 바꾸는 것. (c) 혹은 애초에 hole
공차(clearance)를 넓히거나 hover 간격을 줄여서 필요 이동거리 자체를
줄이는 태스크 난이도 조정도 고려할 만하다.

## grasp anchor 재조정(1~6차) 이후: home 자세를 다시 풀어야 했던 이유

peg/hole grasp을 "진짜 손끝으로 잡는 것처럼" 보이게 재조정하는 과정
(assets/peg_in_hole_bimanual_openarm.xml의 peg/hole_socket 바디 주석,
_PEG_LOCAL_OFFSET/_RIGHT_GRIPPER_GRASP_CTRL/_LEFT_GRIPPER_GRASP_CTRL
주석 참고)에서 peg tip과 hole 목표점의 ee-로컬 위치 자체가 여러 번
바뀌었는데, home 자세(_HOME_QPOS/_LEFT_ARM_HOME_QPOS, 고정된 14개
관절각)는 그대로 뒀다. 그 결과("어떻게 해결할거야" 질문으로 조사) 오른팔
joint4가 range의 90%, joint7이 94.3%까지 붙어 있었다는 걸 발견했다 --
**이 파일 위쪽에서 이미 겪은 "관절 한계 페널티 없이 찾은 첫 home 자세"와
똑같은 실패 패턴**이고, 실측(1500스텝)으로도 xy 드리프트가 108mm까지
벌어지는 걸로 확인했다.

optimize/peg_in_hole_openarm_home_pose_search.py로 같은 비용 함수(방향/
xy/z/손 간격/작업공간/관절 한계 여유, assets 파일 docstring "2. home
자세" 참고)를 새 로컬 오프셋 기준으로 현재 값 근방에서(완전히 새로
찾지 않고 x0=현재 home 자세) 다시 풀었다 -- 양팔 14관절 전부 margin
15% 이내로 들어왔다(가장 타이트한 것도 joint1 18.1%, joint6 19.4%).
재검증(1500스텝): 관절이 더 이상 한계 근처가 아니고, xy 드리프트가
더 이상 단조 발산하지 않는다(step 300: 62mm -> step 1500: 46mm, 오히려
줄어듦). 4000스텝까지 늘려 재확인(위 "3. 더 심각한 발견"의 교훈 그대로
적용)해도 45~75mm 범위에서 진짜로 진동만 하고 발산하지 않는다 -- 이번엔
"진동"이라는 진단이 착시가 아니라는 뜻이다.

다만 이 진동 범위(45~75mm)가 목표 정렬 허용치(outer_half ≈ 19.5mm)보다
아직 2.5~4배 넓어서, 관절 한계 문제는 해결됐어도 삽입 자체는 여전히
성공 못 한다 -- kp/kv/nullspace 게인(optimize/peg_in_hole_openarm_gain_search.py)과
admittance 게인(Kp_xy, Kd_xy -- 이 파일 맨 위 "아직 검증 안 된 것" 절
참고, 처음부터 VX300s 값을 그대로 썼을 뿐 이 팔 기구학에 맞게 재탐색한
적이 아직 없음)이 새 home 자세/anchor 기준으로는 다시 검증돼야 한다.
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

from sim.openarm_bimanual_base import OpenArmBimanualBase

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "peg_in_hole_bimanual_openarm.xml")

N_SUBSTEPS = 5  # mj_step 호출당 substep 수 (timestep=0.002 -> 제어 주기 dt=0.01s)
DT = N_SUBSTEPS * 0.002

Z_RATE = 0.0006  # m / control step, xy 정렬 상태에 따라 게이팅되는 "최대" 하강 속도
# xy를 접촉힘이 아니라 hole 실제 위치로 직접 targeting하도록 바꾼 뒤
# (_run_episode_with_sim 주석 참고) 실측해보니, 이 위치 게인 스케일에서
# 실제 성공은 대략 500~800스텝 사이에 일어난다 -- 예전 400은 admittance
# 시절 VX300s 기본값을 그대로 물려받은 것이라 이 태스크엔 처음부터
# 너무 짧았다. 여유를 두고 늘렸다.
MAX_STEPS = 1000

# xy 정렬 상태에 따라 Z_RATE를 가변으로 만드는 게이트. 고정 Z_RATE로는
# xy가 잘 맞는 짧은 구간(sim/peg_in_hole_bimanual_openarm_sim.py 모듈
# docstring "2차 탐색" 절 참고 -- 스텝 200 근처에서 dx=2mm,dy=5mm까지
# 맞았는데 z_gap은 아직 0.065m나 남아있었음)을 z가 못 따라잡고 지나쳐서
# 실패했다. xy가 잘 맞을 때만 빠르게 내려가고 안 맞을 때는 느리게 만들면
# (정렬되는 순간을 "기다렸다가" 그때 내려가는 효과) 이 타이밍 불일치를
# 줄일 수 있을 것으로 보고 추가했다 -- __main__ 결과가 최초 실측.
_Z_GATE_MIN_FRACTION = 0.15  # xy가 아무리 안 맞아도 완전히 안 멈추고 이 비율로는 계속 내려감(영원히 못 만나는 상황 방지)
_Z_GATE_RADIUS_MULT = 2.0  # 이 배수*outer_half 밖이면 최소 속도로 클립

# "아까보다는 나아졌네 근데 구멍위치를 잘못파악하고있는거 같은데" 피드백으로
# 조사해서 발견한 버그: _Z_GATE_MIN_FRACTION이 "절대 완전히 안 멈춘다"는
# 뜻이라, xy가 한 번도 안 맞으면 z가 목표 깊이를 한참 지나서도 끝없이
# 계속 내려갔다. 실측(400스텝 예산을 넘겨 4000스텝까지 봄): force 센서가
# 그동안 거의 항상 0(접촉이 한 번도 없었다는 뜻)인데도 raw depth는
# hole 자체 깊이(0.059m)의 2배 넘게 계속 커졌고, xy 드리프트도 같이
# 계속 커졌다 -- 팔이 이미 지나친 깊이까지 계속 아래로 명령받으면서
# Jacobian 의사역행렬 해가 (그 방향이 점점 더 도달하기 어려워지면서)
# xy 쪽으로 새어나간 것으로 보인다. "정렬이 안 맞아도 끝까지 계속
# 내려간다"는 원래 설계 의도(위 _Z_GATE_MIN_FRACTION 주석)는 유지하되,
# 목표 깊이를 한참 지나고도(_OVERSHOOT_DEPTH_MULT배) 정렬이 안 됐으면
# 그건 "곧 맞겠지"가 아니라 "이미 지나쳤다"는 신호로 보고 완전히 멈춘다.
_OVERSHOOT_DEPTH_MULT = 1.5


def adaptive_z_rate(dx: float, dy: float, outer_half: float, raw_depth: float, target_depth: float) -> float:
    """현재 xy 정렬 오차(dx,dy)와 hole 반경(outer_half)으로 이번 스텝의
    하강 속도를 정한다. xy_err=0이면 Z_RATE(최대), xy_err가
    _Z_GATE_RADIUS_MULT*outer_half 이상이면 Z_RATE*_Z_GATE_MIN_FRACTION
    (최소)로 선형 보간. 단, raw_depth(정렬 여부와 무관한 원시 깊이)가
    target_depth의 _OVERSHOOT_DEPTH_MULT배를 넘으면(이미 hole을 지나쳤다는
    뜻) 더 내려가지 않는다(0 반환)."""
    if raw_depth >= _OVERSHOOT_DEPTH_MULT * target_depth:
        return 0.0
    xy_err = (dx**2 + dy**2) ** 0.5
    align_quality = max(0.0, 1.0 - xy_err / (outer_half * _Z_GATE_RADIUS_MULT))
    scale = _Z_GATE_MIN_FRACTION + (1.0 - _Z_GATE_MIN_FRACTION) * align_quality
    return Z_RATE * scale

# "어느정도 들어가면 그만 눌러도 될 것 같은데, hole 길이의 절반이 되는
# 지점을 (peg_tip 기준) 목표로 잡아서 해봐" 피드백으로 0.04 -> hole
# 길이의 절반(0.0275m)으로 낮췄었다. hole 길이(assets/
# peg_in_hole_bimanual_openarm.xml의 hole_wall_* geom: size z=0.0275
# (반높이), pos z=-0.0275, 즉 벽이 hole_center_site(로컬 z=0, 입구)에서
# 바닥 상단(z=-0.055, hole_floor geom과 맞닿는 지점)까지 뻗어 있음)은
# 0.055m.
#
# 실측해보니(절반 기준 실행 트레이스) 목표 깊이를 낮춰도 167mm 킥
# 자체는 안 줄었다 -- 킥은 raw_depth가 목표에 도달하기 훨씬 전, hole
# 입구에서 벽에 걸렸다 풀리는 접촉 이벤트라 목표 깊이와 무관하게
# 일어났기 때문(같은 스텝 483, 같은 167mm로 재현됨).
#
# "그럼 아예 더 얕은 지점에서 멈추자, 기준은 hole 길이의 1/3"이라는
# 후속 피드백으로 다시 낮췄다(0.055/3). 이번엔 다른 이유로 실측이 다르게
# 나온다: 같은 트레이스를 보면 peg는 킥이 나기 훨씬 전인 스텝 300 근방
# 부터 이미 raw_depth가 22mm대(1/3 임계값 18.3mm보다 큼)이고 xy도 정렬돼
# 있어서, `_SUCCESS_HOLD_STEPS`(20스텝 연속 유지) 조건이 킥(스텝 439~448)
# 이 일어나기 전에 이미 만족돼 버린다 -- 즉 깊이를 낮추는 게 킥을
# "막는" 게 아니라, 킥이 일어나기 전에 에피소드 자체를 끝내버려서
# 결과적으로 킥을 안 보게 된다(사용자의 "일정량 들어갔으면 놔도 된다"는
# 의도와 정확히 맞음).
# (scene_config로 덮어쓸 수 있음)
TARGET_INSERTION_DEPTH = 0.055 / 3  # m, hole 길이(0.055m)의 1/3

# 성공 판정 버그(실측으로 발견, optimize/peg_in_hole_openarm_gain_search.py의
# "성공 판정 버그" 절 참고): xy가 안 맞은 채로 hole을 완전히 지나쳐
# 허공에서 raw_depth가 물리적으로 불가능한 값(hole 실측 깊이 0.059m보다
# 훨씬 큼)까지 커진 뒤, 어쩌다 한 스텝 xy가 우연히 허용치 안으로 들어오면
# "성공"으로 잘못 판정됐다(정렬을 유지하며 꽂은 게 아니라 뚫고 지나가다
# 스친 것). raw_depth를 hole 실측 깊이보다 살짝 큰 값으로 clip하고,
# 목표 깊이 이상을 이만큼 연속으로 유지해야만 성공으로 인정한다.
_MAX_SANE_DEPTH_M = 0.065
_SUCCESS_HOLD_STEPS = 20

# reset()에서 IK 목표의 z를 잡는 값 -- peg anchor 재조정 3차(위
# _PEG_LOCAL_OFFSET 참고) 전에는 "home 팔 자세에서 peg tip이 자연히
# 가 있는 z"(home_tip_z)를 그대로 썼는데, anchor가 손목에서 멀어지면서
# (peg가 손끝 쪽으로 더 길게 뻗으면서) 같은 home 자세에서 그 z가
# hole 중심 쪽으로 계속 딸려 들어와 호버 간격이 0을 지나 음수가 될
# 뻔했다. grasp 위치가 어떻게 바뀌든 호버 간격은 독립적으로 유지되도록
# 명시적 상수로 뺐다(값은 재조정 1차 직후 실측했던 자연스러운 호버
# ~39mm에 맞춘 것).
_HOVER_GAP_M = 0.039

# hole 벽 nominal 치수 (assets/peg_in_hole.xml과 동일)
_PEG_HALF_WIDTH = 0.010
_WALL_HALF_THICKNESS = 0.004
_WALL_HALF_HEIGHT = 0.0275
_WALL_CENTER_Z = -0.0275
_FLOOR_HALF_THICKNESS = 0.002
_NOMINAL_CLEARANCE_M = 0.003

# OpenArm 오른팔 7개 관절 (fingers 제외).
_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]

# assets/peg_in_hole_bimanual_openarm.xml 상단 docstring에서 CMA-ES로 구한
# home 자세(오른팔 peg tip이 hole 목표점 6cm 위에서 수직 아래를 향함,
# 관절 한계 여유 페널티까지 넣어 재탐색한 최종 값) -- 였는데, grasp anchor
# 재조정(1~5차, _PEG_LOCAL_OFFSET/hole_grasp relpose 참고)으로 peg tip과
# hole 목표점의 ee-로컬 위치 자체가 바뀌면서 이 home 자세로는 오른팔
# joint4가 range의 90%, joint7이 94.3%까지 붙어버렸다 -- "관절 한계
# 페널티 없이 찾은 첫 home 자세"와 똑같은 패턴(1500스텝 실측: xy 드리프트가
# 108mm까지 벌어짐, 게인 문제가 아니라 관절이 한계에 눌어붙어 힘을 못 냄).
# optimize/peg_in_hole_openarm_home_pose_search.py로 새 peg tip/hole 목표점
# 로컬 오프셋 기준 같은 비용 함수(방향/xy/z/손 간격/작업공간/관절 한계
# 여유)를 현재 값 근방에서 다시 풀었다.
_HOME_QPOS = {
    "openarm_right_joint1": -0.4932859043730308,
    "openarm_right_joint2": 2.7728121142883237,
    "openarm_right_joint3": 0.7779694449047385,
    "openarm_right_joint4": 2.068501962471789,
    "openarm_right_joint5": 0.5568966487876724,
    "openarm_right_joint6": -0.5030469209189363,
    "openarm_right_joint7": 0.973709272141154,
}

# 왼팔(Stabilizer)은 이 태스크 내내 고정 -- 위와 같은 재탐색으로 나온 값.
_LEFT_ARM_HOME_QPOS = {
    "openarm_left_joint1": -0.33119160341930914,
    "openarm_left_joint2": -0.6927695819191171,
    "openarm_left_joint3": 1.0264533366778317,
    "openarm_left_joint4": 1.8918419565784326,
    "openarm_left_joint5": -0.9724271650596741,
    "openarm_left_joint6": -0.11753742523671289,
    "openarm_left_joint7": -0.306424166993611,
}
# 오른팔/왼팔 finger_joint1/2 range 부호가 서로 반대다: 오른팔은
# [-0.7854, 0](0이 닫힘 끝), 왼팔은 [0, 0.7854](0이 역시 닫힘 끝 --
# 두 팔이 거울 대칭이라 range 부호만 반대일 뿐, 둘 다 0이 "닫힘"이다,
# 처음엔 반대로 착각했었음). mj_geomDistance로 직접 재보니 완전히 닫힌
# 각도(0)에서 손끝 패드끼리 거의 닿기만 하고(파고듦 3μm 수준, 무시
# 가능) 예전에 기록했던 "1.5cm 자기충돌" 관찰은 바디 world position을
# 잘못 재던(조인트를 바꿔도 바디 원점 자체는 안 움직이는 걸 못 알아챈)
# 스크립트 버그였던 것으로 보인다.

_RIGHT_GRIPPER_GRASP_CTRL = -0.030  # 오른팔(peg를 쥠). 재조정 1~3차(아래
# _PEG_LOCAL_OFFSET 참고)는 anchor 위치만 옮겼을 뿐 손가락은 완전히
# 닫힘(패드 간격 0에 가까움)이었고, 4차에서 mj_geomDistance로 "표면이
# 실제로 가장 가까운 지점"의 간격이 peg 폭만큼(~21mm) 벌어지는 각도
# (-0.115)를 찾아 열었다. 그런데 그 "표면이 가장 가까운 지점"은 진짜
# 손끝(finger 끝 corner)이 아니라 그보다 안쪽(힌지 쪽)이었다 -- geom_aabb로
# 진짜 손끝 corner의 간격을 각도별로 다시 재보니, 완전히 닫힌 각도(0)에서도
# 이미 13.96mm 벌어져 있고(패드가 곡면이라 표면끼리는 닿아도 끝 corner는
# 안 닿음), 4차에서 쓴 -0.115는 손끝 corner 간격이 42.5mm까지 벌어져
# 있었다(peg 폭 2cm의 두 배 넘게 과하게 벌어짐 -- "그리퍼의 끝으로
# 잡으라고" 피드백은 이걸 가리킨 것). geom_aabb 손끝 corner 간격이 peg
# 폭+살짝 여유(~21mm)가 되는, 훨씬 덜 벌린 각도(-0.030)로 다시 정했다.

_LEFT_GRIPPER_GRASP_CTRL = 0.110  # 왼팔(hole_socket을 쥠). 오른팔과 같은
# 이유로 재조정: geom_aabb 손끝 corner 간격이 hole 폭+여유(~41mm)가 되는
# 각도. mj_geomDistance 표면-최근접점 기준으로 잡았던 이전 각도(0.335,
# 그 기준 간격도 ~41mm였지만 손끝 corner 기준으로는 훨씬 더 벌어져 있었음)
# 보다 훨씬 덜 벌어진다.

# peg free body를 오른팔 ee 프레임으로부터 직접 FK로 세팅할 때 쓰는 로컬
# 오프셋 -- assets/peg_in_hole_bimanual_openarm.xml의 weld relpose와 동일한 값
# (relpose_quat이 항등원이라 orientation은 그냥 ee_quat를 그대로 씀).
#
# 재조정 1차 -- 렌더링 영상에서 "그리퍼로 집은게 아니라 들려있다"는 관찰
# 후: 손가락을 _GRIPPER_CLOSED_CTRL(닫힘)로 뒀을 때 손끝 패드가 실제로
# 맞닿는 지점을 mj_geomDistance로 측정하니 ee_base_link 로컬
# (-0.0259,0,-0.1689)였다 -- 기존 anchor(-0.00143,0,-0.133)는 이 지점보다
# 훨씬 위/옆이라(peg의 shaft 중심이 아니라 tip 끝 쪽이 겨우 그 근방)
# 시각적으로 패드 사이에 물린 게 아니라 그 아래 매달린 것처럼 보였다.
#
# 재조정 2차 -- "손끝으로 잡아야지 중간에 잡고있으니까 안되지" 피드백
# 후: 1차 수정은 접촉점을 shaft "중심에서 1.5cm 아래"(로컬 z=-0.015)에
# 둬서 shaft 길이 대부분(위로 4.5cm)이 손끝 안쪽에 걸쳐 있었다 -- 손가락
# 끝이 아니라 중간을 쥔 것처럼 보인 원인. 접촉점을 peg shaft의 맨 위 끝
# (로컬 z=+0.03)에서 1cm만 떨어진 지점(로컬 z=+0.02)으로 옮겼는데, 그때
# 쓴 "접촉점"이 충돌 메시(mj_geomDistance)가 닿는 지점이었을 뿐, 렌더링에
# 보이는 비주얼 메시(pale_silver 손끝 패드)는 그보다 2.3cm 더 끝까지
# 뻗어 있었다 -- 그래서 여전히 손가락 끝이 아니라 그 조금 안쪽으로 보였다
# ("그리퍼의 끝부분으로 잡으라고" 피드백).
#
# 재조정 3차 -- geom_aabb로 비주얼 메시(finger_inner/outer_right_01)의
# 로컬 경계 코너를 world로 변환해 진짜 손끝 끝점을 다시 쟀다: ee_base_link
# 로컬 (-0.02222,0,-0.19186)(inner/outer 평균). 이 지점에서 peg 로컬
# z=+0.02가 오도록 anchor를 다시 잡았다. anchor가 손목에서 멀어질수록
# (peg가 더 길게 뻗을수록) 같은 home 팔 자세에서 자연히 나오던 호버
# 간격이 줄어들다 못해 음수가 될 뻔했는데, 그건 reset()에서
# home_tip_z 대신 _HOVER_GAP_M을 직접 쓰도록 고쳐서(아래) grasp 위치와
# 무관하게 만들었다.
#
# 재조정 4차 -- "지금 물건을 잡는방식이 그리퍼로 잡는게 아닌거 같은데"
# 피드백 이후 근본 원인 발견: 1~3차 내내 손가락은 완전히 닫힘(패드 간격
# ~0)이었다 -- anchor를 아무리 옮겨도 peg가 들어갈 틈 자체가 없었으니
# "펜치로 집은 것"이 아니라 "닫힌 손가락 끝에 아무렇게나 붙어있는 것"처럼
# 보일 수밖에 없었다. 손가락을 벌려서(패드 표면 간격 ~21mm) 재측정: 실제
# 손끝 접촉쌍(finger_inner_right_00 / finger_outer_right_00 -- 3차에서 쓴
# "01" corner-extreme이 아니라, mj_geomDistance로 두 표면이 실제로 마주보고
# 가장 가까워지는 쌍)의 간격 중점이 ee_base_link 로컬 (-0.0259,0,-0.1604).
# peg 로컬 z=+0.02(샤프트 맨 위에서 1cm)가 여기 오도록 anchor를 다시 잡았다.
#
# 재조정 5차 -- "그리퍼의 끝으로 잡아야지" 피드백 이후 재확인: 4차에서 쓴
# "표면 최근접점"은 진짜 손끝(finger 끝 corner)이 아니었다 -- 패드가
# 곡면이라 닫을 때 먼저 닿는 지점(표면 최근접점)과 실제 finger 끝 corner는
# 다른 위치다. geom_aabb로 진짜 끝 corner를 각도별로 재보니 z가 각도에
# 거의 안 바뀌고(-0.192 부근에 고정, 힌지 축 반경 방향과 거의 나란해서),
# 4차에서 쓴 -0.115는 이 진짜 끝 corner 기준 간격이 42.5mm(peg 폭의 2배
# 이상)까지 벌어져 있었다 -- "표면"은 peg 폭만큼 좁았지만 "끝"은 그보다
# 훨씬 넓게 벌어진 채였다는 뜻. 진짜 끝 corner 간격이 peg 폭+여유가 되는
# 훨씬 덜 벌린 각도(-0.030, 위 _RIGHT_GRIPPER_GRASP_CTRL)에서 그 corner
# 중점(ee_base_link 로컬 (-0.02222,0,-0.19214))을 다시 재서 anchor로 썼다
# (peg 로컬 z=+0.02 마진은 그대로).
_PEG_LOCAL_OFFSET = np.array([-0.02222, 0, -0.21214])

_JAC_DAMPING = 1e-4
_IK_MAX_ITERS = 200
_IK_STEP_SCALE = 0.5
_NULLSPACE_GAIN = 0.03  # 근거: _advance_virtual docstring 참고 (CMA-ES 게인 탐색으로 재조정)
_LIMIT_FREEZE_MARGIN = 0.08  # 근거: _advance_virtual docstring "관절 한계 회피" 참고

# 오른팔 역동역학(computed-torque) 제어용 -- "2번(완전한 역동역학 제어로
# 바꾸기)" 결정 이후 추가(모듈 최상단 docstring "grasp anchor 재조정 이후"
# 절 참고). 관절 kp/kv/nullspace, admittance 게인(Kp_xy/Kd_xy) 재탐색
# 둘 다 cost가 거의 안 움직이는 평평한 landscape였다 -- 게인을 아무리
# 바꿔도 한계가 있다는 뜻이라, 관절별 독립 PD 자체를(그 게인이 뭐든)
# 바꾸기로 했다.
#
# 1차 시도(실패로 폐기): home 자세에서 mj_fullM의 대각 성분(M_ii)만 한
# 번 재서 관절별 고정 Kp_i=M_ii*omega_n^2, Kd_i=2*zeta*omega_n*M_ii를
# __init__에서 미리 계산해뒀었다. 정적 홀드 테스트는 잘 됐지만(500스텝,
# 최대 오차 0.3도) 실제 admittance 루프로 4000스텝 돌려보니 처음
# 2000스텝은 34~47mm로 이전과 비슷했다가 그 뒤로 계속 벌어져
# step 4000에 272mm까지 갔다 -- 팔이 home 자세에서 멀어질수록 실제
# 유효 관성이 그 한 번 잰 M_ii에서 벗어나서, 그 배열에 맞춰 놓은
# 임계감쇠가 다른 자세에서는 과소감쇠가 됐던 것으로 보인다(관절 하나만
# 보는 M_ii는 다른 관절 결합에 의한 관성 변화도 못 잡는다).
#
# 2차 시도(현재 채택): 관절별 Kp/Kd를 미리 계산해두는 대신, "가속도
# 공간"에서 목표를 정하고(qacc_cmd_i = omega_n^2*(q_des_i-q_i) -
# 2*zeta*omega_n*qvel_i -- 관절마다 같은 omega_n/zeta) step()이 매 스텝
# **그 순간의** mj_fullM 7x7 부분행렬을 곱해서 토크로 바꾼다
# (tau = M(q)_rr @ qacc_cmd + qfrc_bias_r - qfrc_passive_r). 이러면
# 관성이 자세에 따라 달라져도(그리고 관절 간 결합도) 매 스텝 다시
# 반영되므로 1차 시도의 문제가 구조적으로 없다 -- 표준 computed-torque
# 정의(가속도 feedforward 없이 PD만 있는 버전)에 더 가깝다.
#
# omega_n=40은 timestep=0.002s 대비 여유 있게 안정적이면서
# (omega_n*timestep=0.08) 500ms 이내에 정착하는 값으로 골랐다(실측 전
# 첫 추정치 -- 아래 __main__/render 결과로 검증).
_TORQUE_OMEGA_N = 40.0
_TORQUE_ZETA = 1.0


def _default_scene_config() -> dict[str, Any]:
    return {
        "friction": 0.5,
        "clearance_m": _NOMINAL_CLEARANCE_M,
        "peg_init_offset_xy": (0.0, 0.0),
        "target_insertion_depth": TARGET_INSERTION_DEPTH,
    }


def sample_scene_config(rng: np.random.Generator) -> dict[str, Any]:
    """CMA-ES 평가/견고성 테스트용 무작위 scene_config 샘플러 (VX300s와 동일한
    오프셋 반경 -- 이 태스크도 호버 갭 이후 자유낙하 없이 바로 접촉하는
    구조라 같은 스케일이 적절할 것으로 보고 그대로 시작, 실측으로 재검토
    가능)."""
    angle = rng.uniform(-np.pi / 4, np.pi / 4)
    radius = rng.uniform(0.009, 0.014)
    return {
        "friction": float(rng.uniform(0.2, 0.8)),
        "clearance_m": float(rng.uniform(0.0025, 0.0035)),
        "peg_init_offset_xy": (radius * np.cos(angle), radius * np.sin(angle)),
        "target_insertion_depth": TARGET_INSERTION_DEPTH,
    }


class BimanualPegInHoleOpenArmSim(OpenArmBimanualBase):
    """역동역학/resolved-rate IK/vendor 액추에이터 무력화/gravcomp 주입 등
    OpenArm 양팔 공통 메커니즘은 sim/openarm_bimanual_base.py:OpenArmBimanualBase로
    옮겼다(sim/tacker_openarm_env.py에도 거의 동일한 코드가 있던 걸 보고 뽑아낸
    공통 부분 -- 이 리팩토링으로 peg_in_hole의 물리/제어 결과는 바이트 단위로
    그대로다, tests/test_regression_peg_in_hole_openarm_base.py 참고). 이 클래스는
    peg/hole geometry, clearance, force 센서, reset 오케스트레이션처럼 이
    태스크에만 있는 부분만 남겼다."""

    def __init__(self, xml_path: str | None = None):
        xml_path = xml_path or _DEFAULT_XML
        super().__init__(
            xml_path=xml_path,
            right_home_qpos=_HOME_QPOS,
            left_home_qpos=_LEFT_ARM_HOME_QPOS,
            # 앵커(그리퍼가 쥔 peg 원점 오프셋)와 추적점(peg tip, hole에 꽂히는
            # 끝)은 서로 다르다 -- base 모듈 docstring "오른팔이 쥔 포인트의 두
            # 오프셋" 절 참고. 기존 _virtual_peg_tip_and_jac()이 anchor에 추가로
            # 더하던 [0,0,-0.04]를 여기서 합쳐 넘긴다(동작은 완전히 동일).
            right_anchor_offset_ee=_PEG_LOCAL_OFFSET,
            right_track_point_offset_ee=_PEG_LOCAL_OFFSET + np.array([0.0, 0.0, -0.04]),
            right_held_free_joint="peg_free",
            nullspace_gain=_NULLSPACE_GAIN,
            limit_freeze_margin=_LIMIT_FREEZE_MARGIN,
            jac_damping=_JAC_DAMPING,
            ik_max_iters=_IK_MAX_ITERS,
            ik_step_scale=_IK_STEP_SCALE,
            torque_omega_n=_TORQUE_OMEGA_N,
            torque_zeta=_TORQUE_ZETA,
            n_substeps=N_SUBSTEPS,
        )

        self._hole_body_id = self.model.body("hole_socket").id
        self._peg_body_id = self.model.body("peg").id
        self._hole_site_id = self.model.site("hole_center_site").id
        self._peg_tip_site_id = self.model.site("peg_tip_site").id
        self._peg_geom_ids = [self.model.geom("peg_shaft").id, self.model.geom("peg_tip_ball").id]

        self._wall_geom_ids = {
            name: self.model.geom(f"hole_wall_{name}").id for name in ("px", "nx", "py", "ny")
        }
        self._floor_geom_id = self.model.geom("hole_floor").id

        force_adr = self.model.sensor("peg_force").adr[0]
        torque_adr = self.model.sensor("peg_torque").adr[0]
        self._force_slice = slice(force_adr, force_adr + 3)
        self._torque_slice = slice(torque_adr, torque_adr + 3)

        # "home" 키프레임에서 hole_socket의 qpos(7) -- 왼팔이 안 움직이므로
        # 매 reset()마다 그대로 복사해서 쓴다 (재계산 불필요).
        self._hole_qposadr = self.model.joint("hole_free").qposadr[0]
        key_id = self.model.key("home").id
        self._hole_home_qpos = self.model.key_qpos[key_id][
            self._hole_qposadr : self._hole_qposadr + 7
        ].copy()

    # ------------------------------------------------------------------
    def _apply_clearance(self, clearance_m: float) -> float:
        inner_half = _PEG_HALF_WIDTH + clearance_m / 2.0
        outer_half = inner_half + _WALL_HALF_THICKNESS * 2.0

        wall_center = inner_half + _WALL_HALF_THICKNESS
        self.model.geom_pos[self._wall_geom_ids["px"]] = [wall_center, 0, _WALL_CENTER_Z]
        self.model.geom_pos[self._wall_geom_ids["nx"]] = [-wall_center, 0, _WALL_CENTER_Z]
        self.model.geom_pos[self._wall_geom_ids["py"]] = [0, wall_center, _WALL_CENTER_Z]
        self.model.geom_pos[self._wall_geom_ids["ny"]] = [0, -wall_center, _WALL_CENTER_Z]

        self.model.geom_size[self._wall_geom_ids["px"]] = [_WALL_HALF_THICKNESS, outer_half, _WALL_HALF_HEIGHT]
        self.model.geom_size[self._wall_geom_ids["nx"]] = [_WALL_HALF_THICKNESS, outer_half, _WALL_HALF_HEIGHT]
        self.model.geom_size[self._wall_geom_ids["py"]] = [outer_half, _WALL_HALF_THICKNESS, _WALL_HALF_HEIGHT]
        self.model.geom_size[self._wall_geom_ids["ny"]] = [outer_half, _WALL_HALF_THICKNESS, _WALL_HALF_HEIGHT]

        self.model.geom_size[self._floor_geom_id] = [inner_half, inner_half, _FLOOR_HALF_THICKNESS]
        return outer_half

    def reset(self, scene_config: dict[str, Any]) -> float:
        mujoco.mj_resetData(self.model, self.data)

        for geom_id in self._peg_geom_ids:
            self.model.geom_friction[geom_id][0] = scene_config["friction"]

        outer_half = self._apply_clearance(scene_config["clearance_m"])

        # 왼팔(고정) + hole_socket: home 키프레임 값 그대로 복사 (base의
        # reset_left_arm_fixed_hold가 왼팔 qpos/ctrl + 그리퍼 + 쥔 물체 qpos를
        # 한 번에 설정 -- 기존 로직과 동일, sim/openarm_bimanual_base.py 참고).
        self.reset_left_arm_fixed_hold(
            _LEFT_GRIPPER_GRASP_CTRL,
            left_held_free_joint="hole_free",
            left_held_home_qpos=self._hole_home_qpos,
        )
        mujoco.mj_forward(self.model, self.data)

        hole_center = self.data.site_xpos[self._hole_site_id].copy()

        # xy는 hole 중심 + scene_config 오프셋, z는 hole 중심 + 고정
        # 호버 간격(_HOVER_GAP_M)으로 IK 목표를 잡는다. VX300s 쪽은 여전히
        # "home 자세에서 peg tip이 자연히 가 있는 z"를 쓰지만(sim/peg_in_hole_sim.py
        # 참고), 여기서는 못 쓴다 -- peg anchor가 손목에서 멀어질 때마다
        # 그 자연스러운 z가 같이 딸려 들어오기 때문(_HOVER_GAP_M 정의 참고).
        target_pos = np.array(
            [
                hole_center[0] + scene_config["peg_init_offset_xy"][0],
                hole_center[1] + scene_config["peg_init_offset_xy"][1],
                hole_center[2] + _HOVER_GAP_M,
            ]
        )
        self.solve_right_initial_pose(target_pos, lambda: self.data.site_xpos[self._peg_tip_site_id])
        # step()의 resolved-rate 계획은 이 시점부터 실제 상태가 아니라 이
        # 값에서 이어간다(base __init__의 self._virtual_qpos 주석 참고).
        self.capture_virtual_qpos_from_data()

        self.set_right_gripper(_RIGHT_GRIPPER_GRASP_CTRL)

        mujoco.mj_forward(self.model, self.data)
        return outer_half

    def get_force_torque(self) -> tuple[np.ndarray, np.ndarray]:
        site_rot = self.data.site_xmat[self._peg_tip_site_id].reshape(3, 3)
        force_local = self.data.sensordata[self._force_slice]
        torque_local = self.data.sensordata[self._torque_slice]
        return site_rot @ force_local, site_rot @ torque_local

    def get_ee_pose(self) -> np.ndarray:
        pos = self.data.site_xpos[self._peg_tip_site_id]
        joint7 = self.data.qpos[self._right_arm_qposadr["openarm_right_joint7"]]
        return np.array([pos[0], pos[1], pos[2], joint7])

    def get_peg_tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._peg_tip_site_id].copy()

    def get_hole_center_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._hole_site_id].copy()

    # get_left_ee_pos()/step()은 sim/openarm_bimanual_base.py:OpenArmBimanualBase에서
    # 그대로 상속한다(동작 동일 -- tests/test_regression_peg_in_hole_openarm_base.py로 확인).


def run_episode(gains: dict[str, float], scene_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _default_scene_config()
    if scene_config:
        cfg.update(scene_config)
    sim = BimanualPegInHoleOpenArmSim()
    return _run_episode_with_sim(sim, gains, cfg)


def _run_episode_with_sim(
    sim: BimanualPegInHoleOpenArmSim, gains: dict[str, float], cfg: dict[str, Any]
) -> dict[str, Any]:
    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]

    kp_xy = float(gains["Kp_xy"])
    kd_xy = float(gains["Kd_xy"])

    ee_poses = [sim.get_ee_pose()]
    actions = []
    forces = []
    torques = []

    prev_dx, prev_dy = 0.0, 0.0
    max_force_mag = 0.0
    insertion_depth = 0.0
    success = False
    step_count = 0
    hold_count = 0

    for step_count in range(1, MAX_STEPS + 1):
        force, torque = sim.get_force_torque()
        force_mag = float(np.linalg.norm(force))
        max_force_mag = max(max_force_mag, force_mag)

        # xy는 접촉힘이 아니라 hole의 실제 위치를 직접 목표로 삼는다 --
        # "어짜피 성공하는 데이터를 모아서 학습에 쓰는 게 목적이니 실제
        # 좌표를 써도 된다"는 결정 이후(위 admittance 설계였을 때는 접촉힘이
        # 거의 항상 0이라 xy 보정 신호 자체가 없었다, adaptive_z_rate
        # 주석의 "성공 판정 버그" 조사 때 실측 확인). Kp_xy/Kd_xy는 이제
        # force가 아니라 **위치 오차(dx,dy)**에 곱하는 게인이다(이름은
        # 그대로 두지만 단위/의미가 다르다 -- force 기반 값(0.000515 등)을
        # 그대로 재사용하면 안 되고 다시 탐색해야 함).
        cur_peg_tip = sim.get_peg_tip_pos()
        cur_hole_center = sim.get_hole_center_pos()
        cur_dx = float(cur_hole_center[0] - cur_peg_tip[0])
        cur_dy = float(cur_hole_center[1] - cur_peg_tip[1])
        d_dx = (cur_dx - prev_dx) / DT
        d_dy = (cur_dy - prev_dy) / DT
        prev_dx, prev_dy = cur_dx, cur_dy
        delta_xy = np.array([kp_xy * cur_dx + kd_xy * d_dx, kp_xy * cur_dy + kd_xy * d_dy])

        # 이번 스텝 시작 시점(직전 스텝 결과)의 정렬 상태로 이번 스텝의
        # 하강 속도를 정한다 -- adaptive_z_rate 정의부 주석 참고.
        cur_raw_depth = max(0.0, float(cur_hole_center[2] - cur_peg_tip[2]))
        z_rate = adaptive_z_rate(cur_dx, cur_dy, outer_half, cur_raw_depth, target_depth)
        delta = np.array([delta_xy[0], delta_xy[1], -z_rate])

        sim.step(delta)

        actions.append(delta.copy())
        forces.append(force)
        torques.append(torque)
        ee_poses.append(sim.get_ee_pose())

        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        xy_within_hole_footprint = abs(dx) < outer_half and abs(dy) < outer_half
        raw_depth = min(_MAX_SANE_DEPTH_M, max(0.0, float(hole_center[2] - peg_tip[2])))
        insertion_depth = raw_depth if xy_within_hole_footprint else 0.0

        hold_count = hold_count + 1 if insertion_depth >= target_depth else 0
        if hold_count >= _SUCCESS_HOLD_STEPS:
            success = True
            break

    final_peg_tip = sim.get_peg_tip_pos()
    final_hole_center = sim.get_hole_center_pos()
    dist = float(np.linalg.norm(final_hole_center - final_peg_tip))

    reward = (
        -2.0 * dist
        + 20.0 * insertion_depth
        - 0.001 * max(0.0, max_force_mag - 5.0)
        - 0.01 * step_count
    )
    if success:
        reward += 50.0

    return {
        "trajectory": {"ee_poses": np.stack(ee_poses).astype(np.float32)},
        "ee_poses": np.stack(ee_poses).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "force_profile": np.stack(forces).astype(np.float32),
        "torque_profile": np.stack(torques).astype(np.float32),
        "forces": np.stack(forces).astype(np.float32),
        "torques": np.stack(torques).astype(np.float32),
        "insertion_depth": float(insertion_depth),
        "final_distance": dist,
        "max_force": max_force_mag,
        "step_count": step_count,
        "success": success,
        "reward": float(reward),
        "gains": dict(gains),
        "scene_config": cfg,
    }


if __name__ == "__main__":
    # xy를 접촉힘이 아니라 hole 실제 위치로 직접 targeting하도록 바꾼 뒤
    # (_run_episode_with_sim 주석 참고) optimize/peg_in_hole_openarm_admittance_gain_search.py로
    # 다시 찾은 값 -- 1500스텝 평가에서 success_step~478로 안정적으로 성공.
    gains = {"Kp_xy": 0.124630, "Kd_xy": 0.001125}
    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = (0.014, 0.0)
    result = run_episode(gains, cfg)
    print(
        f"[peg_in_hole_bimanual_openarm_sim] success={result['success']} "
        f"insertion_depth={result['insertion_depth']:.4f}m "
        f"max_force={result['max_force']:.2f}N "
        f"step_count={result['step_count']} reward={result['reward']:.2f}"
    )
