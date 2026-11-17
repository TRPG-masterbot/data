"""「잠긴 종」의 게임 데이터·주사위·판정·전투를 묶은 패키지입니다.

입력: game/data의 정의 자료와 사용자 행동에 대응하는 선택지 ID입니다.
처리: Engine이 상태·판정을 담당하고 CombatMixin·SocialMixin이 전투·NPC 상태를 보탭니다.
출력: GameState의 현재 값, 화면에 표시할 Event와 ChoiceView입니다.

gameplay.py가 이 패키지의 결과를 대화·검색·세션에 연결하고 game_ui.py가 화면에 표시합니다.
규칙 수치와 행동 결과는 Python 코드·JSON 자료에서 처리하며 생성 모델은 전달 자료로 응답을 만듭니다.

This work includes material from the System Reference Document 5.2
(“SRD 5.2”) by Wizards of the Coast LLC, available at
https://www.dndbeyond.com/srd. The SRD 5.2 is licensed under the
Creative Commons Attribution 4.0 International License, available at
https://creativecommons.org/licenses/by/4.0/legalcode.

게임용 규칙 데이터와 판정 코드를 조정했습니다. 시나리오 행동의 보완 조건은
각 JSON의 source·_decisions에서 원문 근거와 구분합니다."""
