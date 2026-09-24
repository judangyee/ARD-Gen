# 출처

이 디렉터리는 [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco)
(v2, Apache License 2.0)의 MuJoCo 자산을 그대로 가져온 것이다. 라이선스
원문은 `LICENSE`에 있다. ARD-Gen이 직접 만든 파일이 아니다 -- 수정 없이
복사만 했고, 필요한 파일만 골라서 가져왔다(전체 저장소가 아님):

- `openarm_bimanual.xml` -- 양팔 로봇 본체(관절/그리퍼/액추에이터).
- `pedestal/openarm_pedestal.xml`, `pedestal/pedestal.xml` -- 받침대에
  양팔을 `<attach>`로 붙인 씬 + 기본 "home" 키프레임.
- `assets/visual/{arm,gripper,body}/`, `assets/collision/` -- 위 두 파일이
  참조하는 메시만(태스크 전용 `visual/cell/` 등은 제외).

상위 `assets/bimanual_openarm.xml`이 `pedestal/openarm_pedestal.xml`을
`<attach>`로 가져다 쓰고, 바닥/조명/카메라/workpiece(범용 파지 대상)만
추가한다.
