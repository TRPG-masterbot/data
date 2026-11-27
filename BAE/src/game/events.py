"""엔진과 화면 사이에서 주고받는 결과·선택지의 자료형입니다.

입력: 엔진에서 확정한 사건의 종류·설명·부가 자료, 또는 현재 가능한 행동 정보입니다.
처리: dataclass가 서로 관련된 값을 Event와 ChoiceView로 묶습니다.
출력: 사건은 Event 목록, 사용자가 고를 행동은 ChoiceView 목록으로 전달됩니다.

engine.py·combat.py·social.py가 만들고 gameplay.py와 game_ui.py가 읽습니다.
화면에 보이는 label은 설명이고 id는 실제 행동을 선택할 때 사용하는 식별자입니다."""
from dataclasses import dataclass, field


@dataclass
class Event:
    """처리된 사건 하나를 설명하는 자료입니다.

    kind는 사건 종류, text는 표시할 설명, data는 카드 ID·굴림 값·대상 같은 부가 정보입니다.
    여러 효과가 일어나면 엔진은 Event들을 실행 순서대로 목록에 담아 반환합니다."""
    kind: str                  # card / say / check / damage / item / gold / flag / move / rest / end / status / combat
    text: str
    data: dict = field(default_factory=dict)


@dataclass
class ChoiceView:
    """현재 화면에 표시할 수 있는 행동 하나의 정보입니다.

    id는 실행에 쓰는 값, label과 check_text는 읽을 설명, actors는 행동 가능한 캐릭터입니다.
    recommended는 추천 캐릭터 ID이고 key는 화면용 고정 키입니다.
    이 자료를 보여 주는 것과 실제 Engine.choose를 호출하는 것은 별도 단계입니다."""
    id: str
    label: str
    actors: list               # 이 선택지를 할 수 있는 캐릭터 id (파티 행동이면 [])
    recommended: str = ""      # 보완 A: 판정에 가장 유리한 캐릭터
    check_text: str = ""
    key: str = ""              # 화면에 보여 줄 고정 키 (턴이 바뀌어도 같은 행동은 같은 키)
