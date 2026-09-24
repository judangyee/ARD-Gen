# 출처

이 디렉터리는 [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco)
**v0.3** 디렉터리(Apache License 2.0)의 MuJoCo 자산을 그대로 가져온 것이다.
라이선스 원문은 `LICENSE`에 있다. ARD-Gen이 직접 만든 파일이 아니다 --
수정 없이 복사만 했다:

- `openarm_bimanual.xml` (원본 파일명은 `openarm_bimanual.mjcf.xml`) --
  양팔 로봇 + 받침대(pedestal)까지 전부 들어있는 단일 파일. v1/v2와 달리
  `<attach>`로 조립하는 구조가 아니라서 상위 파일에서 `<include>`로 통째로
  가져오면 끝(자세한 이유는 `assets/bimanual_openarm.xml` 상단 docstring
  참고).
- `meshes/` -- 위 파일이 참조하는 메시 전부(v0.3은 v2와 달리 visual/collision
  분리가 없고 메시 개수도 적어서 통째로 가져왔다, 14개 파일).

## 왜 v2가 아니라 v0.3인가

처음엔 v2로 만들었는데(그 커밋은 나중에 되돌렸다), 사용자가 "깃헙에서
본 손이랑 다르다"고 해서 v1/v0.3의 그리퍼(평행 조, 레일식/핀 모양)를
같이 렌더해서 보여주고 그중 v0.3을 골랐다. v0.3은 v1/v2보다 이른
프로토타입이라 관절 배치와 몸통 형상 자체가 다르고, 액추에이터도
`<position>`이 아니라 `<motor>`(순수 토크)만 있다 -- 이 차이를 상위
`assets/bimanual_openarm.xml`이 어떻게 다뤘는지는 그 파일 docstring 참고.
