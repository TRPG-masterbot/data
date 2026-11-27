"""출력에 필요한 실행 조건과 짧은 상태 조회를 구성합니다.

입력은 현재 PlaySession과 사용자 질문입니다. 가격·수량·휴식 제한은 원본
엔진의 값으로 계산하며, 조회 과정에서는 구매나 시간 경과를 실행하지 않습니다.
"""

import re

from game.engine import LONG_REST_GAP_MIN, fmt_money


def availability(game, action_id):
    """선택지 표시·추천·실행이 함께 사용할 현재 실행 조건을 반환합니다."""
    state = game.engine.s
    definition = game.content.choices.get(action_id, {})
    if action_id.startswith("buy:"):
        item = game.content.items.get(action_id[4:], {})
        cost = item.get("price_cp")
        if state.stock.get(action_id[4:]) == 0:
            return {"available": False, "reason": "현재 판매 재고가 없습니다.", "cost_cp": cost}
        if type(cost) is int and state.gold_cp < cost:
            return {"available": False, "reason": "소지금이 부족합니다.", "cost_cp": cost}

    rest = next(
        (effect["long_rest"] for effect in definition.get("effects", []) if "long_rest" in effect),
        None,
    )
    if rest is not None:
        cost = rest["cost_cp_per_person"] * sum(
            "dead" not in person.conditions for person in state.party
        )
        last_rest = state.vars.get("last_long_rest_end")
        remaining = (
            max(0, LONG_REST_GAP_MIN - (state.minutes - last_rest))
            if type(last_rest) is int
            else 0
        )
        reason = ""
        if remaining:
            reason = f"다음 긴 휴식까지 {remaining // 60}시간 {remaining % 60}분 남았습니다."
        elif state.gold_cp < cost:
            reason = f"숙박비 {fmt_money(cost)}에 필요한 소지금이 부족합니다."
        elif game.engine.hero.hp < 1:
            reason = "의식과 HP를 회복한 뒤 긴 휴식을 시작할 수 있습니다."
        return {"available": not reason, "reason": reason, "cost_cp": cost}

    return {"available": True, "reason": ""}


def public_services(game):
    """현재 장소가 제공하는 서비스와 지금의 이용 가능 여부를 구분합니다."""
    rows = []
    for choice in game.engine.available_choices():
        definition = game.content.choices.get(choice.id, {})
        is_rest = any("long_rest" in effect for effect in definition.get("effects", []))
        if not (choice.id.startswith("buy:") or is_rest):
            continue
        if not game._story_choice_allowed(choice.id):
            continue
        rows.append(
            {
                "id": choice.id,
                "label": choice.label,
                "kind": "rest" if is_rest else "purchase",
                **availability(game, choice.id),
            }
        )
    if game.current_place()["id"] == "village_market:forge":
        rows.insert(
            0,
            {
                "id": "shop:catalogue",
                "label": "판매 목록과 구입 조건 확인",
                "kind": "catalogue",
                "available": True,
                "reason": "",
            },
        )
    return rows


def _known_item(game, message):
    """가방 또는 현재 상점의 공개 물품 이름만 질문에 연결합니다."""
    candidates = set(game.engine.s.inventory)
    if game.current_place()["id"] == "village_market:forge":
        candidates.update(
            key for key, item in game.content.items.items() if type(item.get("price_cp")) is int
        )
    found = []
    for item_id in candidates:
        item = game.content.items[item_id]
        name = item["name"]
        aliases = {name, name.split("(", 1)[0].strip()}
        if any(alias and alias.casefold() in message.casefold() for alias in aliases):
            found.append(item_id)
    return found[0] if len(found) == 1 else None


def _purchase_quantity(message):
    """질문에서 명시한 추가 구매 개수를 읽고, 모호하면 추측하지 않습니다."""
    text = message.replace("한 개", "1개").replace("하나", "1개")
    text = text.replace("두 개", "2개").replace("둘", "2개")
    patterns = (
        r"(\d+)\s*개\s*(?:를\s*)?(?:더|추가로?)\s*(?:사|구매|구입)",
        r"(?:더|추가로?)\s*(\d+)\s*개\s*(?:사|구매|구입)",
        r"(\d+)\s*개\s*(?:를\s*)?(?:사면|살|구매하면|구입하면)",
    )
    for pattern in patterns:
        matched = re.search(pattern, text)
        if matched:
            value = int(matched[1])
            return value if 0 < value <= 999 else None
    return None


def focused_information(game, scope, message=""):
    """질문한 물품의 수량·가격부터 답하고, 필요할 때만 전체 목록을 표시합니다."""
    if scope == "contracts":
        # 서비스 요금과 의뢰 보상은 서로 다른 거래입니다. NPC가 실제로 공개한
        # 의뢰 영수증만 읽으며 숨은 선택지에서 미래 보상 조건을 끌어오지 않습니다.
        offer = game._confirmed_offer()
        if not offer:
            return "현재 대화 상대가 공개한 의뢰 조건은 아직 없습니다. 먼저 부탁의 내용을 들어보세요."
        stages = {"offered": "제안됨 · 아직 수락 전", "accepted": "수락함 · 아직 완료 전",
                  "in_progress": "수행 중", "completed": "완료됨"}
        rows = ["**공개된 의뢰 · " + offer["title"] + "**",
                "진행: " + stages[offer["status"]], offer["purpose"]]
        grants = offer.get("on_acceptance", [])
        for grant in grants:
            if grant.get("kind") == "money" and type(grant.get("amount_cp")) is int:
                prefix = "수락하면서 받은 준비금" if offer["status"] != "offered" else "수락 시 준비금"
                rows.append(prefix + ": " + fmt_money(grant["amount_cp"]))
        rows.append("별도의 완료 보상 조건은 이 의뢰의 공개 기록에 없습니다.")
        return "\n\n".join(rows)

    item_id = _known_item(game, message)
    if item_id and scope in {"services", "state"}:
        item = game.content.items[item_id]
        quantity = game.engine.s.inventory.get(item_id, 0)
        rows = [f"**{item['name']}**", f"현재 소지: {quantity}개"]
        price = item.get("price_cp")
        at_shop = game.current_place()["id"] == "village_market:forge"
        if scope == "services" and at_shop and type(price) is int:
            rows.append(f"단가: {fmt_money(price)}")
            stock = game.engine.s.stock.get(item_id)
            if type(stock) is int:
                rows.append(f"현재 판매 재고: {stock}개")
            extra = _purchase_quantity(message)
            if extra is not None:
                cost = price * extra
                rows.append(f"{extra}개를 더 구입하면 총 {quantity + extra}개가 됩니다.")
                rows.append(f"구매 금액: {fmt_money(cost)}")
                if game.engine.s.gold_cp < cost:
                    rows.append("현재 소지금으로는 그 수량을 구입할 수 없습니다.")
                if type(stock) is int and extra > stock:
                    rows.append("현재 재고로는 그 수량을 구입할 수 없습니다.")
            elif stock == 0:
                rows.append("현재는 구입할 수 없습니다.")
            rows.append("이 답변은 조회이며 아직 구매하지 않았습니다.")
        return "\n\n".join(rows)

    if scope == "services":
        services = public_services(game)
        rows = ["**제공 서비스 · " + game.current_place()["name"] + "**"]
        for service in services:
            state = "이용 가능" if service["available"] else service["reason"]
            rows.append("- " + service["label"] + " — " + state)
        if not services:
            rows.append("이 장소에는 등록된 판매·휴식 서비스가 없습니다.")
        else:
            rows.append("목록에 없는 서비스의 제공 여부와 가격은 확인되지 않았습니다.")
        return "\n\n".join(rows)

    # 장비 요건·전체 상태는 기존의 정확한 상태 표시를 그대로 사용합니다.
    return None
