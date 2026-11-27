"""이미 공개한 행동의 이름과 실제 실행 ID가 바뀌지 않도록 연결합니다."""

import re


def _normalized(text):
    return re.sub(r"[\W_]+", "", text).casefold()


def exact_action(game, message):
    """현재 선택지의 정확한 이름은 직접 연결하고, 지난 행동은 대신 실행하지 않습니다.

    읽기 전용이며 공개되지 않은 행동은 비교하지 않습니다. 문장이 달라 뜻의
    해석이 필요하면 None을 반환해 기존 분류·연결 경로가 처리하게 합니다.
    """
    target = _normalized(message)
    available = {row.id: row for row in game.choices()}
    matches = [row.id for row in available.values() if _normalized(row.label) == target]
    if len(matches) == 1:
        return {"action_id": matches[0], "blocked_action": None}

    state = game.engine.s
    executed = {row.get("choice") for row in state.log if isinstance(row, dict)}
    known = [row for identifier, row in game.content.choices.items()
             if identifier in executed and identifier not in available
             and _normalized(row.get("label", "")) == target]
    if len(known) != 1:
        return None
    definition = known[0]
    destinations = {
        effect["enter"]
        for branch in ("effects", "success", "failure")
        for effect in definition.get(branch, [])
        if isinstance(effect, dict) and "enter" in effect
    }
    if state.room in destinations:
        reason = "이미 그 목적지에 도착했습니다. 현재 장소의 다음 행동을 선택해 주세요."
    elif (definition["id"], game.engine.hero.id) in state.failed:
        reason = "이미 실패한 판정입니다. 같은 조건에서 다시 굴릴 수 없습니다."
    elif definition["room"] != state.room:
        reason = "이전 장소의 행동입니다. 현재 장소에서 실행할 행동을 선택해 주세요."
    else:
        reason = "이미 처리했거나 현재 조건에서 실행할 수 없는 행동입니다."
    return {"action_id": "CLARIFY", "blocked_action": {
        "id": definition["id"], "blocked_reason": reason,
    }}


def action_destination(definition):
    """성공·실패가 같은 곳으로 이동하는 행동의 목적지만 공개합니다."""
    destinations = {
        effect["enter"]
        for branch in ("effects", "success", "failure")
        for effect in definition.get(branch, [])
        if isinstance(effect, dict) and "enter" in effect
    }
    return next(iter(destinations)) if len(destinations) == 1 else None
