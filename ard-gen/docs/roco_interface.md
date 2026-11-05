# RoCo 인터페이스 기반 작업

이 문서는 "우리 시뮬이 RoCo 공식 시뮬 데이터와 같은 인터페이스로 데이터를
내도록 하는 기반 작업"의 결과물을 설명한다. **태스크 자체는 이 작업
범위에 없다** -- `sim/roco_interface.py`는 어떤 태스크의 제어 루프도 모르는
범용 스키마/변환 모듈이고, 로봇은 OpenArm(팔당 7관절)을 그대로 쓴다.

관련 파일:
- `sim/roco_interface.py` -- 스키마(이름/순서) + 변환 함수 + 카메라 추가.
- `pipeline/to_lerobot_v3.py` -- 위 스키마로 LeRobot v3.0 parquet+h264 mp4를 쓰는 writer.
- `scripts/validate_against_roco.py` -- 생성 데이터를 공식 통계와 비교.
- `tests/test_roco_interface.py` -- 위 세 가지 전부에 대한 회귀 테스트.

## 1. 액션 (14차원)

절대 EE 목표: `[left x,y,z,rx,ry,rz,gripper, right x,y,z,rx,ry,rz,gripper]`
(`roco.ACTION_NAMES`). 회전은 Euler, 라디안 -- **규약 정정(사용자 지시,
실측으로 확인)**: 처음엔 intrinsic XYZ(R=Rx·Ry·Rz)로 구현했는데, 공식
데이터 카드의 "intrinsic XYZ" 표기 자체가 틀렸다. 공식 규약은 **scipy
기준 extrinsic 'xyz'**, 즉 R = Rz(rz)·Ry(ry)·Rx(rx)(표준 ZYX/항공
요-피치-롤 분해와 동치)다. `sim/roco_interface.py`의 `euler_xyz_from_mat()`/
`euler_xyz_to_mat()`을 이 규약으로 고쳤고, 사용자가 준 golden pair 2개
(쿼터니언<->Euler, 1e-5~1e-6 이내)로 `tests/test_roco_interface.py`가
직접 확인한다. 짐벌락 위치(|ry|=pi/2)와 "이전 프레임에 가장 가까운 해"
분기 선택(아래 "짐벌락 대응" 절)은 이 규약 정정 후에도 그대로 유효함을
재검증했다(대안 분기 공식 `(rx+pi, pi-ry, rz+pi)`가 두 규약 모두에서
동일한 행렬을 줌, 실측 확인).

**연속 시퀀스 전체에 대해 unwrap**되어 있어서(래핑 없음) 프레임마다 다시
계산하면 안 된다 -- `continuous_euler_xyz_from_quats(quats_wxyz)`에 전체
(T,4) 쿼터니언 시퀀스를 한 번에 넘겨야 한다(단일 프레임만 넘기면 unwrap
기준점이 없어서 연속성을 보장할 수 없음). "액션[t] = 상태[t+1]에서
도달해야 하는 EE 목표"(RT-1/Bridge류 절대-다음-목표 관례)는
`roco_interface.py`가 강제하지 않는다 -- 호출자가 시퀀스를 한 스텝 미리
밀어서 넘겨야 한다(`pipeline/to_lerobot_v3.py`가 이렇게 한다).

## 2. State (44차원)

이름/순서는 사용자가 지정한 공식 그대로 재현한다(`roco.STATE_NAMES`):

```python
[f"{s}_ee_{a}" for s in ("left", "right") for a in ("x", "y", "z", "qw", "qx", "qy", "qz")]
+ [f"{s}_jpos_{i}" for s in ("left", "right") for i in range(7)]
+ [f"{s}_jvel_{i}" for s in ("left", "right") for i in range(7)]
+ ["left_gripper", "right_gripper"]
```

EE pose는 `openarm_{side}_ee_base_link` 바디의 월드 pose(쿼터니언은
wxyz, 정규화됨)이고, 관절 위치/속도는 `openarm_{side}_joint{1..7}`에서 읽는다.
그리퍼 값은 `GripperMapping`으로 매핑한다(아래 4절).

## 3. 카메라

`head`, `left_hand`, `right_hand` 3개, 각 320x240(가로x세로), 10fps
(`roco.CAMERA_NAMES`/`CAMERA_WIDTH`/`CAMERA_HEIGHT`/`CAMERA_FPS`).
`build_roco_model(base_xml_path)`가 OpenArm 양팔 베이스 XML(어떤
태스크든)을 `MjSpec`으로 읽어 카메라 3개를 추가하고 컴파일된 `MjModel`을
돌려준다. **XML 텍스트로 왕복하지 않는다** -- 이 베이스 모델은 여러
서브파일을 `<attach>`로 합성한 것이라 `MjSpec.to_xml()` 직렬화가
깨진다(중첩 `<default>` class 누락, meshdir 경로 붕괴 -- 둘 다 실측
확인). 대신 호출자는 돌아온 `MjModel`을 기존 `Sim` 객체의 `.model`/
`.data`에 그대로 끼워넣으면 된다 -- 바디/조인트/액추에이터 id/순서가
원본과 완전히 동일하게 유지된다(카메라만 새로 추가되므로,
`tests/test_roco_interface.py::test_build_roco_model_adds_cameras_preserves_ids`가
이걸 직접 확인한다).

`<attach>`로 들여온 바디(예: `openarm_right_ee_base_link`)에도 `MjSpec`
API로는 자식(카메라)을 붙일 수 있다 -- 이전 작업에서 "안 된다"고 기록했던
건 raw XML 텍스트 편집 기준으로만 맞는 말이었다(자세한 근거는
`sim/roco_interface.py` 모듈 docstring).

렌더링이 필요한 스크립트/테스트만 `MUJOCO_GL=osmesa`를 설정한다
(`PIPELINE.md`의 "MUJOCO_GL=osmesa + PyTorch Adam 세그폴트" 함정을
피하려고 -- 렌더링 안 하는 코드는 `MUJOCO_GL`을 아예 설정하지 않는다).

## 4. 그리퍼 값 매핑

`GripperMapping(closed_value, open_value, target_range=(0.06, 0.30))`가
raw 관절 각도(라디안) <-> RoCo "열림 비율"(공식 데이터 관측 범위
0.06~0.30)을 선형으로 매핑한다. `default_gripper_mapping(model, side)`는
모델의 실제 joint range에서 0에 가까운 쪽을 open, 먼 쪽을 closed로
자동 판정한다(하드코딩 없음 -- XML이 바뀌어도 따라감).

## 5. 좌표 프레임 변환

`WorkspaceTransform(source, target=ROCO_OFFICIAL_LEFT_EE_RANGE)`가 축별
(x/y/z) 선형(affine) 변환으로 OpenArm 작업 공간을 공식 왼팔 EE 범위
(`ROCO_OFFICIAL_LEFT_EE_RANGE`: x -0.01~0.82, y -0.14~0.76, z 1.07~1.80)로
옮긴다. `source`는 반드시 호출자가 줘야 한다(태스크마다 실제 작업공간
관측 범위가 다르므로 기본값을 박아두지 않음). `.apply()`/`.inverse()`는
클리핑하지 않는다 -- source 범위 밖 입력도 그대로 외삽돼서 돌아온다.

## 6. 왼팔-액티브 관례와의 차이 (정직하게 기록)

공식 RoCo 데이터는 **왼팔=액티브, 오른팔=수동 정지**(y 4cm·z 3cm 이내,
그리퍼 고정)를 전제한다. 이 저장소의 기존 `peg_in_hole`(OpenArm)은
**정반대**(오른팔=Actuator/움직임, 왼팔=Stabilizer/고정,
`sim/peg_in_hole_openarm_env.py` 참고)다. `roco_interface.py`의 스키마
자체는 어느 팔이 움직이는지 가정하지 않으므로 문제가 없지만, **이
저장소의 기존 태스크로 만든 데이터를 그대로 변환하면
`scripts/validate_against_roco.py`의 "왼팔 액티브/오른팔 정지" 통계
검사를 통과하지 못한다** -- 버그가 아니라 두 관례가 다르다는 사실
자체다(사용자 지시 "태스크 자체는 만들지 않는다"와 일치 -- 액티브
암이 왼쪽인 새 태스크는 이번 범위 밖). 실측(`peg_in_hole` 데모 2
에피소드, 202프레임): 10개 검사 중 4개 PASS(각도 연속성, 왼팔 ry/rz
범위, 왼팔 그리퍼 범위), 6개 FAIL(에피소드 길이/수, 왼팔 rx 범위,
오른팔 정지 3종, 초반 유지->이동) -- 전부 "오른팔이 액티브인 데모라서"
예상된 결과다.

## 7. LeRobot v3.0 writer (`pipeline/to_lerobot_v3.py`)

기존 `pipeline/to_lerobot.py`(npz 기반, v2.1-approx)는 그대로 둔다 --
이 writer는 별도 경로(`--out-root`)에 LeRobot v3.0 네이티브 레이아웃
(`data/chunk-*/file-*.parquet` + `videos/*/chunk-*/file-*.mp4` + h264,
`meta/info.json`)으로 쓴다. 44/14차원 state/action 외에 추가 열:
`scenario`, `failure_type`, `phase`(normal/failure/recovery),
`loss_weight`, `source`, `episode_success`. 실제 `lerobot` 패키지의
`LeRobotDataset`을 writer/리더 양쪽에 쓴다(`--verify`가 재로딩까지
확인).

알아둘 환경 이슈(의존성 버전, 이 컨테이너 기준):
- `datasets==4.0.0`(3.0.1엔 `ParquetConfig.filters`가 없어서 리로드 실패,
  5.1.0은 `huggingface_hub`와 안 맞음), `huggingface_hub==0.26.5`(최신
  2.4.0은 `httpcore2` import 에러), `torchcodec`(설치된 `torchvision`에
  `io.VideoReader`가 없어서 video_backend가 `pyav`로 떨어져도 디코드가
  깨짐 -- `torchcodec`을 깔면 `get_safe_default_codec()`이 그걸 우선
  선택해서 해결됨).
- numpy 2.4.6에서 커스텀 스칼라(shape=(1,)) feature가 HF `datasets`의
  `Value.encode_example()`(바로 `float()`/`bool()` 호출)과 안 맞는 버그가
  있어서, `save_episode()` 직전에 `episode_buffer`의 해당 값들을
  `.item()`으로 스칼라로 바꿔주는 워크어라운드가 writer 안에 있다(환경
  특정 버전 조합 버그, 설계 결함 아님 -- 코드 주석에 자세히 기록).
- `dataset.finalize()`를 전체 에피소드 작성 후 반드시 호출해야 한다
  (안 그러면 parquet footer 메타데이터가 안 써져서 리로드가
  `FileNotFoundError`로 깨짐, `metadata_buffer_size=10` 배칭 때문).

## 8. 검증 (`scripts/validate_against_roco.py`)

```
python scripts/validate_against_roco.py --dataset-root <lerobot_v3 데이터셋 경로>
```

pyarrow로 parquet을 직접 읽는다(lerobot 패키지도, 비디오 디코드도 필요
없음). 공식 통계(200 에피소드/121,454프레임, 왼팔 회전/그리퍼 범위,
오른팔 정지 기준, 초반 유지 패턴)와 비교해 각 항목 PASS/FAIL을 출력하고
하나라도 FAIL이면 exit code 1을 반환한다. 각 검사의 허용 오차는 스크립트
상단에 상수로 명시돼 있다(예: 평균 길이는 공식 평균의 ±40%, 회전/그리퍼
범위는 공식 범위 폭의 5%, 오른팔 정지 한도는 공식 한도의 +25%).

## 9. 현황 조사 요약 (재사용 vs 부족)

| 영역 | 재사용 | 부족/새로 만든 것 |
|---|---|---|
| 태스크 레지스트리 (`sim/task_registry.py`) | `TASK_REGISTRY`/`TaskConfig`/`load_task_config()` 그대로 재사용 (수정 없음) | 이번 작업은 레지스트리에 새 태스크를 등록하지 않음 (범위 밖) |
| OpenArm 환경 규칙 (`sim/peg_in_hole_openarm_env.py`) | 바디/조인트 이름 규칙(`openarm_{side}_joint{i}`, `openarm_{side}_ee_base_link`, `openarm_{side}_finger_joint1`)을 그대로 따름 | 없음 -- VX300s 시절 `sim/stabilizer.py`는 끌어오지 않음(사용자 지시) |
| 언어 라벨링 (`pipeline/language_labeling.py`) | 건드리지 않음, 이번 작업과 무관(RoCo writer는 `task` 필드에 시나리오 문자열만 넣음) | 해당 없음 |
| 데이터 변환기 (`pipeline/to_lerobot.py`, `pipeline/episode_io.py`) | 기존 npz 기반 writer/스키마는 그대로 유지(수정 없음) | LeRobot v3.0 네이티브 writer가 없어서 `pipeline/to_lerobot_v3.py`를 새로 추가 |
| 시각 랜덤화 (`ard-gen-v2` 브랜치 `sim/visual_randomization.py`) | **가져오지 않음** -- 순수 시각(재질/조명/잡동사니/테이블 높이) 랜덤화라 물리와 무관하고, main 브랜치에 없는 XML 애셋 항목이 필요함. 이번 작업은 스키마/인터페이스 기반 작업이라 도메인 랜덤화가 필요 없다고 판단 | 필요해지면(예: 실제 RoCo 스타일 비주얼 다양성이 필요한 시점) 별도로 머지 검토 필요 |

## 10. 알려진 미검증 항목

- `scripts/validate_against_roco.py`는 `peg_in_hole` 데모로만 실행해봤다
  (그 데모가 공식 관례와 반대 팔 구성이라는 걸 알고 한 것) -- 왼팔이
  액티브인 실제 태스크가 생기기 전까지는 "통과하는 데이터"로 검증할
  방법이 없다.
- 각도 연속성 unwrap은 짐벌락 근처(`ry` -> ±pi/2)나 두 축이 동시에
  분기 전환하는 경우까지는 다루지 못한다(`roco_interface.py` 모듈
  docstring에 기록) -- 공식 데이터 `ry` 범위(-1.26~1.53)가 그 경계에
  닿지 않는다는 관측에 기반한 가정.
- `pipeline/to_lerobot_v3.py --verify`는 2-에피소드/202-프레임 소규모
  데모로만 확인했다 -- 대규모(수백 에피소드) 작성 시 `datasets`
  라이브러리의 메모리/시간 특성은 별도 확인이 필요하다.
