"""공개된 행동의 물품 조건을 읽기 전용으로 설명합니다.

장비 외 조건이 충족되지 않은 선택지와 해석이 모호한 조건식은 표시하지 않습니다.
엔진 조건·선택지·소지품을 바꾸지 않고 기존 자료의 물품 이름과 개수만 사용합니다.
"""
import html


def _contains_item(condition):
    if not isinstance(condition, dict) or len(condition) != 1:
        return False
    if "item" in condition:
        return True
    if "not" in condition:
        return _contains_item(condition["not"])
    return any(_contains_item(row) for key in ("all", "any") for row in condition.get(key, []))


def _only_items(condition, items):
    if not isinstance(condition, dict) or len(condition) != 1:
        return False
    if "item" in condition:
        return condition["item"] in items
    if "not" in condition:
        return _only_items(condition["not"], items)
    for key in ("all", "any"):
        if key in condition:
            rows = condition[key]
            return isinstance(rows, list) and bool(rows) and all(_only_items(row, items) for row in rows)
    return False


def _equipment_condition(condition, engine):
    """(물품식, 나머지 조건 충족 여부). 섞인 any/not은 보수적으로 공개하지 않습니다."""
    if condition is None:
        return None, True
    if not isinstance(condition, dict) or len(condition) != 1:
        return None, False
    if _only_items(condition, engine.c.items):
        return condition, True
    if "all" in condition and isinstance(condition["all"], list):
        pieces = []
        for child in condition["all"]:
            equipment, ready = _equipment_condition(child, engine)
            if not ready:
                return None, False
            if equipment is not None:
                pieces.append(equipment)
        return ({"all": pieces} if pieces else None), True
    if _contains_item(condition):
        return None, False
    try:
        return None, bool(engine.cond(condition))
    except (KeyError, TypeError, ValueError):
        return None, False


def _describe(condition, content, inventory):
    if "item" in condition:
        identifier = condition["item"]
        name = content.items[identifier]["name"]
        count = inventory.get(identifier, 0)
        return f"{name} (가방 {count}개)"
    if "not" in condition:
        return "보유하지 않아야 함: (" + _describe(condition["not"], content, inventory) + ")"
    key = "all" if "all" in condition else "any"
    joiner = " 및 " if key == "all" else " 또는 "
    return "(" + joiner.join(_describe(row, content, inventory) for row in condition[key]) + ")"


def equipment_rows(game):
    """현재 행동과 장비만 부족한 공개 행동을 설명하며 원래 선택지 목록은 건드리지 않습니다."""
    engine, content = game.engine, game.content
    state = engine.s
    pending = game.pending or {}
    rows = []
    pending_id = pending.get("choice")
    current = content.choices.get(pending_id, {})
    equipment, ready = _equipment_condition(current.get("requires"), engine)
    if current and equipment is not None and ready:
        rows.append({"label": pending.get("label", current["label"]), "selected": True,
            "satisfied": bool(engine.cond(equipment)), "requirement": _describe(equipment, content, state.inventory)})
    if pending or state.pending or state.combat or state.ended:
        return rows
    available = {choice.id for choice in game.choices()}
    known = game._known_npcs()
    people = game.npc_cards()
    for choice in content.choices_in_room(state.room):
        if choice["id"] in available or choice["id"] in state.done:
            continue
        equipment, ready = _equipment_condition(choice.get("requires"), engine)
        if equipment is None or not ready or engine.cond(equipment):
            continue
        # 공개 조건과 실행자 조건은 일반 선택지 판정과 동일하게 유지합니다.
        person = engine.npc_for_choice(choice["id"])
        if person and not engine.npc_can_talk(person):
            continue
        if choice.get("actor", "any") != "party" and not engine._actor_pool(choice):
            continue
        target = people.get(choice.get("target_npc"), {})
        if target and target["id"] not in known:
            continue
        label = choice["label"]
        if game._public_text(label) != label:
            continue
        rows.append({"label": label, "selected": False, "satisfied": False,
            "requirement": _describe(equipment, content, state.inventory)})
    return rows


def equipment_requirements_panel(game):
    """접힌 설명 상자에 가방과 공개된 장비 조건만 표시합니다."""
    inventory = game.engine.s.inventory
    owned = [f"{item['name']} {inventory[identifier]}개" for identifier, item in game.content.items.items()
             if isinstance(inventory.get(identifier), (int, float)) and inventory[identifier] > 0]
    lines = ['<details class="game-equipment-conditions"><summary>장비·실행 조건</summary>',
             '<p>현재 가방: ' + html.escape(" · ".join(owned) or "없음") + '</p>']
    for row in equipment_rows(game):
        status = "선택한 행동 · 장비 충족" if row["selected"] and row["satisfied"] else "장비 조건 미충족"
        lines.append('<p><strong>' + html.escape(row["label"]) + '</strong><br>'
            + html.escape(status + ": " + row["requirement"]) + '</p>')
    lines.append('<p>가방에 물품이 있어도 선택한 행동과 필요한 다른 장비에 따라 실행 조건이 달라집니다.</p></details>')
    return "".join(lines)
