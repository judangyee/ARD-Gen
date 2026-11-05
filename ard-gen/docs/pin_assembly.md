# pin_assembly: RoCo식 "조립 + 오류 초기 상태 복구"

OpenArm 양팔 기반 새 태스크. 기존 `peg_in_hole`/`tacker`는 건드리지 않았다.
`sim/roco_interface.py`(기존 작업물)를 그대로 재사용한다.

## 관례: 왼팔=Actuator, 오른팔=정지 (기존 태스크와 반대)

`peg_in_hole`/`tacker`는 오른팔이 움직이고 왼팔이 고정이다. 이 태스크는
반대다 -- 왼팔이 5개 부품(planet x3, sun, ring)을 순차로 집어서 끼우고,
오른팔은 공식 RoCo 데이터의 "사실상 정지" 휴지 자세를 유지한다
(`sim/pin_assembly_openarm_sim.py`의 `RIGHT_HOME_QPOS`, `peg_in_hole`의
오른팔 home을 그대로 재사용 -- 이미 "로컬 -z가 world -z와 일치"가 실측
확인된 값). `roco_interface.py`의 스키마는 어느 팔이 움직이는지 가정하지
않으므로, 이 태스크가 이 저장소 최초로 "공식 왼팔-액티브 관례와 일치하는"
데이터를 낸다.

## 장면 (assets/pin_assembly_openarm.xml)

중앙 핀 1개 + 둘레 핀 3개가 있는 캐리어 플레이트, 둘레 핀용 원판
(planet) 3개, 중앙용 원판(sun) 1개, 바깥 링 1개 -- 전부 단순 원판(실제
기어 톱니/구멍 없음, 사용자 지시 1). 각 원판 위에는 작은 사각 핸들(2cm
단면, `peg_in_hole`의 peg 샤프트와 동일 크기)을 얹어서, 원판 테두리
전체를 핀치하는 대신 이미 검증된 2cm 핀치 그립을 재사용한다. 캐리어/
스테이징 플랫폼은 "허공에 뜬 고정 플랫폼"(peg_in_hole의 peg/hole이
테이블과 무관하게 허공에 떠 있던 것과 같은 단순화). 핀은 전부 시각
전용(contype=0) -- 실제 구멍/삽입 접촉을 모델링하지 않는다.

### 홈 자세 재탐색 (실측으로 2차 보정)

왼팔이 "포인팅 다운 + 적당한 reach"로 처음 움직여야 해서 6-DOF 댐핑
최소제곱 IK를 무작위 재시작으로 돌렸다. **1차**(목표점 (0.22,0.05,0.50))는
수렴은 했지만, 그 자세에서 왼팔 손목 링크가 carrier_plate와 9~34mm
침투해서 접촉 제약력이 수백 N급으로 튀어 포지션 서보가 무력화되는 문제를
실측으로 발견했다(ctrl=qpos인데도 20프레임 만에 0.4rad+ 드리프트). **해결**:
작업 공간 전체를 8cm 위로 올리고(플랫폼 표면 z=0.50->0.60) 다시 탐색 --
정확히 수직은 아니지만(dot=0.874) 손목이 플랫폼과 안 겹치고 관절 한계
여유가 있는 해를 찾았다(최소 4.9%). 자세한 과정은
`assets/pin_assembly_openarm.xml`과 `sim/pin_assembly_openarm_sim.py`의
모듈 docstring에 그대로 기록돼 있다.

## 제어: 포지션-전용 resolved-rate IK, "운동학적" 관절 적용 (2차 설계)

`sim/pin_assembly_openarm_sim.py`의 `advance_toward()`가 매 프레임
Jacobian 한 스텝으로 가상 목표를 전진시킨다(사용자 지시 3-2 "처음 10
프레임 유지 -> 11프레임째 큰 점프 -> 감속 수렴"에 맞춘 설계).

**1차 설계**(IK로 웨이포인트의 목표 qpos를 한 번에 구해서 `pos_left_j*`
ctrl에 "점프"시키고 물리(포지션 PD)가 알아서 수렴하게 두는 방식)는 실측
에서 폐기했다 -- 이 작업의 home 근처 관절 구성에서 `peg_in_hole`이 검증한
`kv` 값들이 극단적으로 과감쇄(ζ≈33, 임계감쇄의 30~150배)라는 걸
`mj_fullM`으로 유효 관성을 직접 재서 확인했다(관성은 관절 구성마다 다르므로
`peg_in_hole`의 값이 그대로 맞을 이유가 없었다). 실측: 15프레임(0.15s) 만에
관절 하나가 목표보다 0.8rad 넘게 뒤처지고, 수렴에 1초 이상 걸려서 에피소드
전체가 손끝이 거의 안 움직이는 것처럼 보이는 버그로 나타났다.

**2차**(현재): `kv`를 임계감쇄 근방으로 재조정(assets XML 참고)**했고
그래도 남는 지연을 없애려고, 매 프레임 실제(real) qpos/Jacobian에서
다시 계산한 목표를 `qpos`/`qvel`에 직접 덮어쓴다**(서보가 쫓아올 시간을
기다리지 않음 -- "운동학적" 적용, `ctrl`도 같은 값으로 맞춰서 추가
요동이 없게 함). 디스크 물리(중력/접촉에 의한 자연스러운 안착)는 그대로
`mj_step`으로 돈다 -- 팔만 운동학적이다.

## 그립: weld 대신 "운동학적 들기"

`start_carry()`가 그립 시작 시점의 실제 FK로 손 기준 로컬 오프셋을 한 번
계산해두고, 들고 있는 동안 매 서브스텝 그 오프셋을 그대로 유지하며 물체의
freejoint qpos/qvel을 덮어쓴다. XML에 weld anchor를 미리 맞출 필요가
없고(peg_in_hole/tacker가 겪은 "anchor 재조정 1~6차" 같은 반복이 불필요),
여러 물체를 순차로 들고 놓는 이 태스크에 자연스럽게 맞는다.

## 배치 정확도: 물리 낙하 대신 "스냅" (정직하게 기록한 단순화)

운동학적 팔 제어로도 손끝이 목표에 수 mm~수 cm 이내로 수렴하지만, 그
잔차가 원판 중심 좌표로 그대로 전파되면 seat 판정 허용오차(12mm)를
가끔 넘긴다. **`pick_and_place()`는 놓는 순간 원판을 목표 자리에 정확히
스냅한다**(완전한 물리 기반 낙하/안착 시뮬레이션이 아니다). 추가로,
이미 놓은 원판이 그 뒤 pick-place의 충돌(특히 들고 있는 물체와의 정상
충돌)로 밀려나는 것도 실측으로 발견해서, 매 pick-place가 끝날 때마다
"지금까지 놓은" 원판들을 전부 다시 스냅한다(`run_episode()`의
`_resnap_placed()`). 이 두 가지 모두 **물리적 정밀도보다 데이터셋의
성공/배치 정확도를 보장하는 쪽을 택한 설계 결정**이다 -- 팔의 궤적
자체(위치/방향/타이밍)는 운동학적으로 생성되지만 실제로 그 경로를 따라
움직이고, 원판이 최종적으로 어디 놓이는지만 스냅으로 보정한다.

## 오류 상태 생성기 (사용자 지시 4)

`generate_error_state()`: 무작위로 planet 하나를 고르고 그 위에 작은
위치 오차로 sun을 올린 뒤, **40프레임(0.4s) 실제 물리를 돌려서** sun이
그 자리에서 미끄러져 떨어지지 않고(xy drift<=15mm) 여전히 planet
위에(z가 충분히 높게) 남아 있는지 검사한다 -- 아니면 폐기하고 다시
뽑는다(최대 20회). `tests/test_pin_assembly.py::test_generate_error_state_rejects_unstable`
가 8개 시드 전부 물리적으로 유효한 상태를 반환하는지 확인한다.

## 복구 서브스킬 (사용자 지시 3)

`recover_misplaced_part()`는 분리된 함수다 -- 내부적으로
`pick_and_place()`를 재사용하지만 `phase="recovery"`로 라벨링한다. S3
에피소드는: (1) 오류 상태를 그대로 들고 시작하는 첫 `HOLD_FRAMES`
구간(phase="failure", loss_weight=0 -- "이 행동이 좋은 예"라고 학습시킬
이유가 없는 구간), (2) `recover_misplaced_part()`로 sun을 치움
(phase="recovery", loss_weight=1 -- 복구 자체는 유용한 지도 신호), (3)
치운 자리에서 다시 집어 중앙 핀에 올바르게 끼움(phase="normal"), (4)
ring 배치(phase="normal") 순으로 구성된다.

## 성공 지표

`count_correctly_assembled()`: planet 3개+sun은 각자의 핀에 xy/z
허용오차(12mm/15mm) 이내로 seated인지, ring은 중앙 xy + sun 위에 얹힌
높이(carrier 표면+sun 전체두께+ring 반두께)에 있는지로 5점 만점 카운트.
`episode_success = (n_assembled == 5)`.

## 방향 오프셋 + 분기 선택 (사용자 지시 3-1, roco_interface.py에 추가)

`sim/roco_interface.py`의 `OPENARM_EE_DOWNWARD_OFFSET_MAT`(이번 작업에서
추가) + `continuous_euler_xyz_from_quats()`의 `initial_rx_branch` 파라미터를
`pipeline/to_lerobot_pin_assembly.py`가 실제로 쓰는 첫 호출부다 --
`ee_frame_offset_mat=OPENARM_EE_DOWNWARD_OFFSET_MAT`,
`left_initial_rx_branch="positive_pi"`, `right_initial_rx_branch="negative_pi"`.
자세한 유도/테스트는 `tests/test_roco_interface.py`와 그 모듈 docstring
참고.

## 짐벌락 대응 (사용자 지시 6)

전문가 동작의 목표 자세는 이 태스크에서 특별히 큰 ry 회전을 만들지
않는다(픽-플레이스가 대체로 top-down 접근이라 ry가 자연히 좁은 범위에
머문다) -- 그래도 `roco_interface.py`의 `continuous_euler_xyz_from_quats()`
자체가 범용적으로 짐벌락 근처(ry~pi/2) 분기 전환에 대응하도록 이미
고쳐져 있다(이전 작업, `tests/test_roco_interface.py`의
`test_continuous_euler_gimbal_lock_branch_switch` 참고 -- ry~1.5~1.7인
합성 입력에서 naive 추출은 pi만큼 튀지만 이 함수는 0.006rad 이내로
복원).

## 알려진 미검증/한계 (정직하게 기록)

- **물리적 정밀도**: 위 "배치 정확도" 절 그대로 -- 최종 배치는 스냅
  기반이다. 팔의 궤적 자체는 실제로 그 경로를 따라 움직이지만, "손이
  정확히 물체를 쥐고 자연스럽게 내려놓아 중력/접촉으로 안착한다"는
  수준의 물리적 리얼리즘은 없다.
- **렌더링 성능**: OSMesa 소프트웨어 렌더가 실측 ~1초/프레임(3카메라
  합산) -- 시나리오당 25개(수백 프레임씩, 75개 총 33,898프레임) 전부
  렌더링하면 9시간 이상 걸린다. 그래서 `--no-render`(이미지는 0으로
  채움, parquet 숫자 컬럼은 전부 실제 값 -- validate_against_roco.py는
  이미지를 안 보므로 검증에 영향 없음)로 대량 생성하고, 실제 렌더링은
  소규모 샘플(시나리오당 1개)만 따로 만들어 파이프라인이 끝까지 도는지
  확인했다(완료 보고 참고).

## 실측 결과 (validate_against_roco.py, 75개 에피소드/no-render)

10/10 전부 PASS(평균 길이 452.0프레임(공식 범위의 ±40% 안), 왼팔 rx/ry/rz
범위, 왼팔 그리퍼, 오른팔 정지(y/z 변화 0.0mm -- 운동학적 고정이라 완전
정지), 오른팔 그리퍼 고정, 오른팔 rx 고정, 각도 연속성, 초반 유지->이동
75/75). 오른팔 "정지" 검사가 처음 FAIL(y 56mm/z 57mm, 그리퍼도 [0.06,0.27]
로 벌어짐)했다가 PASS로 바뀐 과정 자체가 교훈적이다 -- 오른팔은 reset()
에서 ctrl만 한 번 설정하고 그 뒤로 전혀 안 건드렸는데도(이 저장소의
기존 "고정 팔" 패턴, peg_in_hole의 Stabilizer와 동일), RIGHT_HOME_QPOS가
"정확히" 중력-평형점은 아니라서 매 스텝 조금씩 처지는 게 누적됐다.
왼팔 쪽에서 먼저 겪은 것과 같은 문제라, 같은 해법(운동학적 고정, 매
스텝 qpos/qvel 직접 덮어쓰기)을 오른팔 관절+그리퍼에도 적용해서 해결
했다(`sim/pin_assembly_openarm_sim.py`의 `_pin_right_arm()`).
- **IK 도달성**: `scripts/check_pin_assembly_reachability.py` 실측 결과는
  완료 보고 참고.
- **대규모 생성 시간 특성**: 수백 개 규모(공식 데이터 200에피소드/
  121,454프레임)로 확장할 때의 메모리/시간 특성은 미확인(이번엔 수십
  개 규모만 생성).
