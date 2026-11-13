"""엔진이 화면·서술기(AI)에 넘기는 결과와 선택지 형식.

엔진은 화면을 직접 그리지 않는다. 결과는 Event 목록, 고를 수 있는 행동은 ChoiceView 목록으로만 내보내고,
CLI(game/cli.py)·웹 서버(server/webapp.py)·서술기(game/narrator.py)가 각자 이 형식을 읽어 보여 준다.
"""
from dataclasses import dataclass, field


@dataclass
class Event:
    """일어난 일 하나. text는 화면에 그대로 보여 줄 문장, data는 화면 연출용 값 (주사위 숫자, 카드 id 등)."""
    kind: str                  # card / say / check / damage / item / gold / flag / move / rest / end / status / combat
    text: str
    data: dict = field(default_factory=dict)


@dataclass
class ChoiceView:
    """지금 고를 수 있는 행동 하나. 화면은 label과 key를 보여 주고, 고르면 id를 Engine.choose에 넘긴다."""
    id: str
    label: str
    actors: list               # 이 선택지를 할 수 있는 캐릭터 id (파티 행동이면 [])
    recommended: str = ""      # 보완 A: 판정에 가장 유리한 캐릭터
    check_text: str = ""
    key: str = ""              # 화면에 보여 줄 고정 키 (턴이 바뀌어도 같은 행동은 같은 키)
