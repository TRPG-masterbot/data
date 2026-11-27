"""출력 어댑터의 시작 가방 구성과, 관찰한 장소의 준비물 안내입니다.

팀의 starting_equipment·캐릭터·선택지·가격 자료는 수정하지 않습니다.
새 가방은 원본 시작 장비를 적용한 직후 한 번만 구성합니다. 준비물 안내는
실행 조건과 현재 가방을 읽기만 하며 구매·물품 지급·진행 해금을 하지 않습니다.
"""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path


PROFILE_ID = "food_and_medical_bag.v1"
STARTING_BAG = {"rations": 2, "healers_kit": 1}
CASTING_ITEMS = {"spellbook", "holy_symbol"}
LIGHT_ITEMS = {"torch", "hooded_lantern", "lamp", "candle"}


def apply_preparation_loadout(engine):
    """원본 시작 장비 뒤 새게임 가방만 바꿉니다. 적용했으면 True입니다.

    무기·방어구·주문·슬롯·숙련·금액은 그대로 둡니다. 기존 주문책·성표는
    별도 장착 기록으로 옮깁니다. 미휴대 물품을 판매하거나 추가 금액을 주지
    않습니다. 불러온 게임·체크포인트·이미 진행한 상태에는 재지급하지 않습니다.
    """
    state, hero = engine.s, engine.hero
    setup = state.vars.get("setup")
    if not isinstance(setup, dict) or setup.get("starting_equipment_applied") is not True:
        raise ValueError("원본 시작 장비를 먼저 적용해야 합니다.")
    profile = setup.get("preparation_profile")
    if isinstance(profile, dict) and profile.get("id") == PROFILE_ID:
        return False
    if (state.log or state.done or state.minutes or state.room != engine.c.start_room
            or state.scene != engine.c.start_scene or state.pending or state.combat or state.ended):
        return False
    if any(identifier not in engine.c.items for identifier in STARTING_BAG):
        raise ValueError("현재 원본에 식량·응급 처치 도구가 정의되어 있지 않습니다.")
    original_setup, original_bag = deepcopy(setup), deepcopy(state.inventory)
    equipped = deepcopy(setup.get("equipped_items") or {})
    moved = {}
    for identifier, quantity in original_bag.items():
        if identifier in CASTING_ITEMS or identifier.startswith("arcane_focus_"):
            equipped[identifier] = equipped.get(identifier, 0) + quantity
            moved[identifier] = quantity
    # tools는 소유한 도구이고 tool_profs는 숙련입니다. 가방에서 뺀 도구가
    # Character.tools를 통해 여전히 사용 가능한 것으로 남지 않게 합니다.
    original_tools = list(hero.tools)
    hero.tools = [identifier for identifier in hero.tools
                  if identifier not in engine.c.items or identifier in STARTING_BAG or identifier in equipped]
    state.inventory = deepcopy(STARTING_BAG)
    kit_uses = engine.c.items["healers_kit"].get("uses")
    if type(kit_uses) is int and kit_uses > 0:
        state.vars["healers_kit_uses"] = STARTING_BAG["healers_kit"] * kit_uses
    setup["equipped_items"] = equipped
    setup["preparation_profile"] = {
        "id": PROFILE_ID, "applied_once": True, "starting_bag": deepcopy(STARTING_BAG),
        "original_startup": original_setup, "original_carried_items": original_bag,
        "casting_items_moved_to_equipment": moved, "original_owned_tools": original_tools,
        "supply_adjustment": {key: quantity - original_bag.get(key, 0)
                              for key, quantity in STARTING_BAG.items()},
        "not_carried_from_original_bag": {key: quantity for key, quantity in original_bag.items()
                                          if key not in STARTING_BAG and key not in moved},
        "source": "output_adapter_user_selected_starting_bag",
        "note": "가방은 식량 2개·응급 처치 도구 1개. 기본 장착과 시작 금액은 원본대로 유지한다."}
    return True


def _positive_item_options(condition, engine):
    """원본 조건의 양수 물품 조합만 읽습니다. 그 외 조건은 엔진으로 확인합니다."""
    if condition is None:
        return [frozenset()]
    if not isinstance(condition, dict) or len(condition) != 1:
        return []
    if "item" in condition:
        return [frozenset([condition["item"]])] if condition["item"] in engine.c.items else []
    if "all" in condition:
        children = [_positive_item_options(row, engine) for row in condition["all"]]
        if not children or any(not rows for rows in children):
            return []
        return list(dict.fromkeys(frozenset().union(*rows) for rows in product(*children)))[:32]
    if "any" in condition:
        return list(dict.fromkeys(option for row in condition["any"]
                                  for option in _positive_item_options(row, engine)))[:32]
    try:
        return [frozenset()] if engine.cond(condition) else []
    except (KeyError, TypeError, ValueError):
        return []


def _entered_rooms(choice):
    return list(dict.fromkeys(row["enter"] for key in ("effects", "success", "failure")
                             for row in choice.get(key, []) if isinstance(row.get("enter"), str)))


def _shop_options(game, missing_options):
    """판매 목록·현재 재고·가격을 확인합니다. 알지 못하는 인물은 이름을 공개하지 않습니다."""
    content, state = game.content, game.engine.s
    path = Path(__file__).with_name("transaction_bindings.json")
    bindings = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    shops = []
    for row in bindings.get("shops", []):
        menu = content.choices.get(row.get("menu_action_id"), {})
        seller = content.cards.get(row.get("seller_card_id"), {})
        if (not seller or menu.get("room") != row.get("room_id")
                or not any(effect.get("show") == row.get("menu_card_id") for effect in menu.get("effects", []))):
            continue
        place = game._person_place(seller) if hasattr(game, "_person_place") else row["room_id"]
        shops.append({"room_id": row["room_id"], "location_id": place,
                      "seller_card_id": row["seller_card_id"], "source_seller_name": row["seller_npc"],
                      "seller_name": row["seller_npc"] if row["seller_card_id"] in state.shown else None,
                      "source_menu_action_id": row["menu_action_id"], "source_menu_card_id": row["menu_card_id"]})
    options = []
    for missing in missing_options:
        products = []
        for identifier in sorted(missing):
            item = content.items[identifier]
            stock = state.stock.get(identifier, item.get("stock"))
            price = item.get("price_cp")
            if (identifier not in content.shop_order or type(price) is not int or price < 0
                    or stock is not None and (type(stock) is not int or stock <= 0)):
                break
            products.append({"item_id": identifier, "name": item["name"], "quantity": 1, "price_cp": price})
        else:
            total = sum(row["price_cp"] for row in products)
            options.append({"items": products, "total_cp": total, "affordable": state.gold_cp >= total,
                            "shops": deepcopy(shops)})
    return sorted(options, key=lambda row: (row["total_cp"], len(row["items"])))


def well_preparation_hint(game, inspected_action_id=None):
    """실제 우물 관찰 뒤 준비물의 사실·플레이어 생각을 돌려줍니다. 상태는 바꾸지 않습니다.

    밧줄 없는 판정 하강과 판정 없는 장비 하강을 구분합니다. 광원은 원본에
    활성 조건이 없으면 권고로만 쓰며, 암시야와 구현된 조명 주문을 대안으로
    남깁니다. 특정 인물 대사나 새로운 강제 구매 조건은 만들지 않습니다.
    """
    engine, content = game.engine, game.content
    state = engine.s
    identifier = inspected_action_id or "ws_open_well"
    inspection = content.choices.get(identifier, {})
    if (not inspection or inspection.get("operation") != "inspect" or inspection.get("room") != state.room
            or identifier not in state.done or state.pending or state.combat or state.ended):
        return None
    shown = [row["show"] for row in inspection.get("effects", []) if row.get("show")]
    if shown and not any(engine.hero.id in state.shown.get(card_id, []) for card_id in shown):
        return None
    # 관찰한 입구에서 이어지는 첫 하강 구간의 정의만 읽습니다. 먼 장면의
    # 물품·보상·등장인물이나 원본 비밀 카드를 끌어오지 않습니다.
    entrances = [row for row in content.choices_in_room(state.room)
                 if row.get("next_action_id") == identifier and row.get("operation") == "enter"]
    destination_ids = {room for row in entrances for room in _entered_rooms(row)}
    candidates = [row for room in destination_ids for row in content.choices_in_room(room)
                  if any(target != state.room for target in _entered_rooms(row))]
    if not candidates:
        return None
    safe, risky = [], []
    for choice in candidates:
        if choice.get("actor", "any") != "party" and not engine._actor_pool(choice):
            continue
        options = _positive_item_options(choice.get("requires"), engine)
        if not options:
            continue
        if choice.get("check"):
            risky.append({"action_id": choice["id"], "check": deepcopy(choice["check"]),
                          "currently_condition_ready": bool(engine.cond(choice.get("requires")))})
        elif any(options):
            safe.append({"action_id": choice["id"], "item_options": [sorted(row) for row in options]})
    item_options = list(dict.fromkeys(frozenset(row) for action in safe for row in action["item_options"]))
    missing_options = [frozenset(item for item in option if not state.has_item(item)) for option in item_options]
    safe_ready = bool(missing_options) and any(not option for option in missing_options)
    purchases = _shop_options(game, [option for option in missing_options if option])
    suggested = []
    for shop in purchases[0]["shops"] if purchases else []:
        suggested.append({"id": shop["location_id"], "name": "대장간",
            "npc_name": shop["source_seller_name"], "npc_card_id": shop["seller_card_id"],
            "native_room_id": shop["room_id"], "source_menu_action_id": shop["source_menu_action_id"],
            "source_menu_card_id": shop["source_menu_card_id"],
            "discovery_basis": "player_observed_preparation_need_and_source_shop",
            "introduced_by_npc": False})
    hero = engine.hero
    spells = content.rules.get("spells", {})
    light_spells = [identifier for identifier in [*hero.cantrips, *hero.spells]
                    if spells.get(identifier, {}).get("kind") in {"light", "illumination"}]
    lighting = {"darkvision_ft": hero.darkvision, "light_spells": light_spells,
                "owned_light_items": [key for key in sorted(LIGHT_ITEMS) if state.has_item(key)],
                "active_light_state": "not_defined_by_source", "mandatory_purchase": False}
    lines = ["우물 아래는 깊고 어둡다. 내려가기 전에 가방과 안전하게 내려갈 방법을 확인해야겠다."]
    if safe_ready:
        lines.append("판정 없이 하강할 장비는 갖췄다. 내려가서 장비를 실제로 고정하는 행동을 선택하자.")
    elif purchases:
        cheapest = purchases[0]
        names = " · ".join(row["name"] for row in cheapest["items"])
        lines.append("판정 없이 내려갈 준비에 부족한 물품: " + names + ".")
        known = next((shop["seller_name"] for shop in cheapest["shops"] if shop["seller_name"]), None)
        lines.append((known + "의 대장간" if known else "마을 대장간") + " 판매 목록을 확인해 보자.")
    if any(row["currently_condition_ready"] for row in risky):
        lines.append("장비 없이 시도하는 하강도 있지만 판정 실패와 추락 위험을 감수해야 한다.")
    if hero.darkvision > 0:
        lines.append("나는 암시야가 있지만, 안전하게 내려갈 장비는 별개로 챙겨야겠다.")
    elif light_spells:
        lines.append("배운 조명 주문도 시야를 확보하는 대안이 될 수 있다.")
    elif not lighting["owned_light_items"]:
        lines.append("어둠을 살필 광원도 함께 확인해 두는 편이 좋겠다.")
    else:
        lines.append("광원은 챙겼다. 가진 장비와 실제 사용할 방법을 확인하자.")
    return {"kind": "player_preparation_thought", "text": "\n\n".join(lines), "source_action_id": identifier,
            "safe_descent_methods": safe, "safe_descent_equipment_ready": safe_ready,
            "safe_descent_missing_options": [sorted(row) for row in missing_options],
            "risk_descent_alternatives": risky, "purchase_options": purchases, "lighting": lighting,
            "suggested_locations": suggested,
            "scope": "observed_entrance_and_first_descent_segment", "changes_state": False}
