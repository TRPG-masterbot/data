"""엔진의 현재 상태·실제 구매 기록·공개 이동 정보를 출력과 화면에 연결합니다.

검색 결과나 인물 설정을 만들거나 선별하지 않습니다.
"""
from copy import deepcopy
import json
from game.content import npc_name


def native_conversation_history(history):
    """게임 상태 화면으로 처리한 사용자 행만 모델의 대화 맥락에서 뺍니다.

    저장·화면·기억 제외용 원본 이력은 바꾸지 않습니다. 다른 표식의 사용자
    발언과 정상 NPC 대화 쌍은 유지하며, 모델용 복사본에는 처리 표식을 넣지 않습니다.
    """
    result = []
    for row in history or []:
        if not isinstance(row, dict):
            continue
        if row.get("role") == "user" and row.get("answered_by") == "game_state":
            continue
        copied = deepcopy(row)
        copied.pop("answered_by", None)
        result.append(copied)
    return result



def current_scene_context(payload):
    """현재 관찰·공개 배경·실제 이동 선택지를 구분해 복사합니다.

    available_actions가 목록이면 실제 가능한 move 선택지가 출구의 기준입니다.
    그 목록이 없을 때만 명시된 scene_context.exits를 사용합니다. 출구 None은
    자료 미제공, []는 제공된 목록에 공개 이동 선택지가 없음을 뜻합니다.
    시나리오 카드의 본문은 이 상태 자료에 추가하지 않습니다. 설정 근거는 팀 입력의
    검색 결과로 전달하고, 생성된 GM 문장도 현재 관찰로 다시 수집하지 않습니다.
    """
    scene = payload.get("scene_context") or {}
    scene = scene if isinstance(scene, dict) else {}
    result = {"room_id": scene.get("room_id"), "room_name": scene.get("room_name"), "exits": None}
    result["people"] = [{key: deepcopy(row[key]) for key in
        ("card_id", "name", "label", "role", "presence", "approach_action", "met") if key in row}
        for row in scene.get("people", []) if isinstance(row, dict)
        and isinstance(row.get("name"), str) and row["name"].strip()]
    observation = scene.get("observation")
    if isinstance(observation, str) and observation.strip():
        result["observation"] = observation.strip()
    raw_exits = scene.get("exits")
    scene_exits = {row["id"]: row for row in raw_exits
                   if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"].startswith("move:")} if isinstance(raw_exits, list) else {}
    actions = payload.get("available_actions")
    source = actions if isinstance(actions, list) else raw_exits if isinstance(raw_exits, list) else None
    if source is not None:
        result["exits"] = []
        seen = set()
        for row in source:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].startswith("move:"):
                continue
            identifier = row["id"]
            if identifier in seen or not identifier[5:]:
                continue
            seen.add(identifier)
            public = scene_exits.get(identifier, {})
            label = row.get("label") if isinstance(row.get("label"), str) else public.get("label")
            destination = public.get("destination") or row.get("destination")
            if not isinstance(destination, str) or not destination:
                destination = label.split(":", 1)[-1].strip() if isinstance(label, str) and label.strip() else None
            result["exits"].append({"id": identifier, "destination_id": identifier[5:],
                                    "destination": destination, "label": label})
    return result



def player_snapshot(engine):
    """실제 캐릭터·가방·장착 장비·사용 자원·행동 기록을 복사한 상태 자료를 만듭니다.

    물품 이름과 주문 이름은 Content에서 읽어 ID와 함께 넣습니다.
    현재 가진 물품과 실행했던 행동 기록을 따로 담아 모델이 두 사실을 구별하게 합니다."""
    hero = engine.hero
    item = lambda identifier, quantity: {"item_id": identifier,
        "item_name": engine.c.items.get(identifier, {}).get("name", identifier), "quantity": quantity}
    equipped = engine.s.vars.get("setup", {}).get("equipped_items", {})
    # 원본 감사 로그의 events에는 열람한 카드 전문도 섞입니다. 모델에는 실행 ID만
    # 전달하고, 수량·위치·성공 여부는 현재 상태와 구조화된 이번 결과로 확인합니다.
    actions = [{"source_log_index": index, "action_id": row.get("choice"),
                "actor_id": row.get("actor"), "event_count": len(row.get("events", []))}
               for index, row in enumerate(engine.s.log) if isinstance(row, dict)]
    spells = [row for row in actions if isinstance(row.get("action_id"), str)
              and row["action_id"].startswith("cmb:cast:") and row.get("actor_id") in (None, hero.id)]
    spell_names = {sid: engine.c.rules.get("spells", {}).get(sid, {}).get("name", sid)
                   for sid in hero.spells + hero.cantrips}
    return {"id": hero.id, "name": hero.name, "hp": hero.hp, "max_hp": hero.max_hp,
        "cls": hero.cls, "cls_ko": hero.cls_ko, "level": hero.level,
        "spellcasting": hero.spellcasting, "cantrips": deepcopy(hero.cantrips),
        "spells": deepcopy(hero.spells), "spell_names": spell_names,
        "features": deepcopy(hero.features), "resources": deepcopy(hero.resources),
        "capabilities_complete": True, "completed_actions": actions,
        "completed_spell_actions": spells, "action_log_complete": True,
        "gold_cp": engine.s.gold_cp,
        "inventory": [item(key, value) for key, value in engine.s.inventory.items()],
        "equipped_items": [item(key, value) for key, value in equipped.items()]}



def model_state(engine, snapshot):
    """플레이어 스냅샷에서 이름·위치·금액·장비를 짧은 한국어 사전으로 정리합니다."""
    from game.engine import fmt_money
    return {"플레이어 이름": snapshot["name"], "현재 장소": engine.c.room_name(engine.s.room),
        "플레이어 소지금": fmt_money(snapshot["gold_cp"]),
        "플레이어 장착 장비": [row["item_name"] for row in snapshot["equipped_items"]],
        "플레이어가 보유한 소지품": [{"이름": row["item_name"], "수량": row["quantity"]}
                              for row in snapshot["inventory"]]}



def player_party(engine):
    """거래의 from/to에 사용할 플레이어 식별 정보를 반환합니다."""
    return {"kind": "player", "id": engine.hero.id, "name": engine.hero.name}



def npc_party(name, card_id=""):
    """거래 당사자 이름과 카드 ID를 묶고, 이름이 없으면 미확인 당사자로 표시합니다."""
    return {"kind": "npc", "name": name, "card_id": card_id} if name else {"kind": "unknown"}



def new_ledger():
    """구매 내역을 쌓을 빈 거래 장부와 형식·완전성 표식을 만듭니다."""
    return {"schema": "transaction_ledger_v1", "complete": True, "entries": []}



def shop_context(engine, bindings):
    """원본 NPC 표시와 장면에 정의된 공개 상점 메뉴·완료 기록만 연결합니다."""
    matches = []
    for binding in bindings.get("shops", []):
        if binding.get("room_id") != engine.s.room or not engine.c.rooms[engine.s.room].get("shop"):
            continue
        card = engine.c.cards.get(binding.get("seller_card_id"), {})
        action = engine.c.choices.get(binding.get("menu_action_id"), {})
        menu_id = binding.get("menu_card_id")
        if (npc_name(card) != binding.get("seller_npc") or card.get("room") != engine.s.room
                or action.get("room") != engine.s.room
                or not any(effect.get("show") == menu_id for effect in action.get("effects", []))
                or menu_id not in engine.s.shown or binding["menu_action_id"] not in engine.s.done
                or not engine.eligible(binding["seller_card_id"])
                or not engine.npc_can_talk(binding["seller_card_id"])):
            continue
        matches.append(deepcopy(binding))
    if len(matches) != 1:
        return {"status": "shop_context_unknown", "seller_npc": None}
    return {**matches[0], "status": "verified_menu_owner"}



def before_action(engine, active_npc, action_id, bindings):
    """행동 실행 전의 금액·수량·재고·로그 길이를 복사해 거래 대조 기준을 만듭니다."""
    context = shop_context(engine, bindings) if action_id.startswith("buy:") else {}
    return {"action_id": action_id, "available": action_id in {row.id for row in engine.available_choices()},
        "room_id": engine.s.room, "active_npc": active_npc, "shop_context": context,
        "gold_cp": engine.s.gold_cp, "inventory": deepcopy(engine.s.inventory),
        "stock": deepcopy(engine.s.stock), "log_length": len(engine.s.log)}



def record_action(ledger, before, engine, events):
    """실제 구매 전후 차이와 엔진 로그·Event를 대조해 장부 항목을 만듭니다.

    상품 수량 증가, 지출, 재고 감소와 buy 행동 기록이 맞는지 확인합니다.
    같은 로그 위치는 중복 기록하지 않으며, 판매자가 불명확한 경우도 사유를 구분해 남깁니다.
    검증한 기록은 ledger에 추가하고 반환합니다."""
    action_id = before["action_id"]
    if not action_id.startswith("buy:"):
        return None
    index = before["log_length"]
    if len(engine.s.log) != index + 1 or engine.s.log[index].get("choice") != action_id:
        return None
    old = next((row for row in ledger["entries"] if row["source_log_index"] == index), None)
    if old is not None:
        if old["action_id"] != action_id:
            raise ValueError("거래 기록의 log 위치가 충돌합니다.")
        return deepcopy(old)
    item_id = action_id.split(":", 1)[1]
    item = engine.c.items.get(item_id, {})
    price = item.get("price_cp")
    stock_before, stock_after = before["stock"].get(item_id), engine.s.stock.get(item_id)
    quantity_before = before["inventory"].get(item_id, 0)
    quantity_after = engine.s.inventory.get(item_id, 0)
    spent = before["gold_cp"] - engine.s.gold_cp
    # 구매 로그만 보지 않고 물품 +1, 가격만큼 지출, 재고 감소와 item Event를 함께 대조합니다.
    executed = (before["available"] and engine.s.room == before["room_id"]
        and len(engine.s.log) == index + 1 and engine.s.log[index].get("choice") == action_id
        and item_id in engine.c.shop_order and type(price) is int and price >= 0
        and quantity_after - quantity_before == 1 and spent == price
        and (stock_after is None if stock_before is None else stock_after == stock_before - 1)
        and any(event.kind == "item" and event.data.get("item") == item_id for event in events))
    context = before.get("shop_context", {})
    seller = context.get("seller_npc") if context.get("status") == "verified_menu_owner" else None
    seller_party = npc_party(seller, context.get("seller_card_id", ""))
    buyer_party = player_party(engine)
    confirmed = bool(executed)
    status = "confirmed_purchase" if confirmed and seller else "counterparty_unknown" if confirmed else "state_delta_mismatch"
    record = {"kind": "purchase", "entry_id": f"tx:{index}", "source_log_index": index,
        "action_id": action_id, "room_id": before["room_id"], "item_id": item_id,
        "item_name": item.get("name", item_id), "quantity": 1 if confirmed else None,
        "unit_price_cp": price, "spent_cp": spent if confirmed else None,
        "confirmed": confirmed, "status": status, "from": seller_party, "to": buyer_party,
        "knowledge_npcs": [seller] if confirmed and seller else [],
        "before": {"player_quantity": quantity_before, "player_gold_cp": before["gold_cp"], "stock_quantity": stock_before},
        "after": {"player_quantity": quantity_after, "player_gold_cp": engine.s.gold_cp, "stock_quantity": stock_after},
        "transfers": [{"item_id": item_id, "item_name": item.get("name", item_id), "quantity": 1 if confirmed else None,
            "player_quantity": quantity_after, "transfer_completed": confirmed,
            "npc_current_quantity": stock_after if seller and type(stock_after) is int else None,
            "npc_custody_status": "shop_stock" if seller and type(stock_after) is int else "stock_quantity_unknown",
            "from": seller_party, "to": buyer_party}],
        "payments": [{"gold_cp": -spent if confirmed else None, "direction": "player_pays",
            "payment_completed": confirmed, "from": buyer_party, "to": seller_party}]}
    ledger["entries"].append(deepcopy(record))
    return record



def restore_ledger(value, engine_log):
    """저장된 거래 장부의 형식과 각 항목이 가리키는 엔진 로그를 검사합니다.

    기록 없는 예전 저장은 불완전한 장부로 시작합니다. 장부 형식·중복 위치·구매 로그가
    맞지 않으면 복원을 거절하며, 확인되지 않은 구매 이력을 새로 채우지 않습니다."""
    if value is None:
        return {**new_ledger(), "complete": False}
    if (not isinstance(value, dict) or value.get("schema") != "transaction_ledger_v1"
            or type(value.get("complete")) is not bool or not isinstance(value.get("entries"), list)):
        raise ValueError("거래 기록 형식이 올바르지 않습니다.")
    result, indices = deepcopy(value), set()
    for entry in result["entries"]:
        index = entry.get("source_log_index") if isinstance(entry, dict) else None
        if (type(index) is not int or not 0 <= index < len(engine_log) or index in indices
                or entry.get("entry_id") != f"tx:{index}" or entry.get("kind") != "purchase"
                or engine_log[index].get("choice") != entry.get("action_id")):
            raise ValueError("거래 기록과 엔진 log가 일치하지 않습니다.")
        indices.add(index)
        if type(entry.get("confirmed")) is not bool or not isinstance(entry.get("knowledge_npcs"), list):
            raise ValueError("거래 확인 자료형이 올바르지 않습니다.")
        if entry["confirmed"] and (type(entry.get("quantity")) is not int or entry["quantity"] < 1
                or type(entry.get("spent_cp")) is not int or entry["spent_cp"] < 0):
            raise ValueError("거래 수량·금액이 올바르지 않습니다.")
    buys = {i for i, row in enumerate(engine_log) if str(row.get("choice", "")).startswith("buy:")}
    result["complete"] = result["complete"] and buys <= indices
    return result



def shop_state(engine, npc_name, ledger, bindings):
    """현재 NPC가 실제 판매자인지 확인하고 그 인물이 아는 구매 기록을 추립니다.

    현재 재고·플레이어 수량과 과거 구매 수량·지출액을 따로 묶습니다.
    장부에 빠진 구매가 있으면 과거 누적값을 미확인으로 표시합니다."""
    context = shop_context(engine, bindings)
    log_buys = {i for i, row in enumerate(engine.s.log) if str(row.get("choice", "")).startswith("buy:")}
    complete_ledger = ledger["complete"] and log_buys <= {row["source_log_index"] for row in ledger["entries"]}
    base = {"seller_npc": context.get("seller_npc"), "seller_card_id": context.get("seller_card_id"),
        "room_id": engine.s.room, "menu_card_id": context.get("menu_card_id"),
        "status": context["status"], "ledger_complete": complete_ledger, "items": [], "transactions": [],
        "buyback_policy": deepcopy(context.get("buyback_policy")),
        "buyback_price_cp": context.get("buyback_price_cp"),
        "refund_policy": deepcopy(context.get("refund_policy"))}
    if context.get("seller_npc") != npc_name:
        return {**base, "status": "not_counterparty" if context.get("seller_npc") else context["status"],
                "seller_npc": None, "seller_card_id": None, "menu_card_id": None}
    scoped = [row for row in ledger["entries"] if npc_name in row.get("knowledge_npcs", [])]
    base["transactions"] = deepcopy(scoped)
    for item_id in engine.c.shop_order:
        item = engine.c.items[item_id]
        stock = engine.s.stock.get(item_id)
        records = [row for row in scoped if row["item_id"] == item_id and row["confirmed"]]
        unknown = any(row["room_id"] == engine.s.room and row["item_id"] == item_id
                      and (not row["confirmed"] or not row["knowledge_npcs"]) for row in ledger["entries"])
        complete = complete_ledger and not unknown
        base["items"].append({"item_id": item_id, "item_name": item["name"], "unit_price_cp": item["price_cp"],
            "stock_quantity": stock, "stock_status": "finite" if type(stock) is int else "unbounded_or_unspecified",
            "player_quantity": engine.s.inventory.get(item_id, 0),
            "purchased_quantity": sum(row["quantity"] for row in records) if complete else None,
            "spent_cp": sum(row["spent_cp"] for row in records) if complete else None,
            "purchase_status": "recorded_purchase" if complete and records else "no_recorded_purchase" if complete else "history_unknown"})
    return base



def render_confirmed_state(payload, engine_result=None, npc_name=None, resolve_speaker=None):
    """명시적인 상태 조회에 엔진의 현재 값만 표시합니다. NPC 설정은 추론하지 않습니다."""
    if not payload.get("needs_state"):
        return ""
    from game.engine import fmt_money
    state = (engine_result or {}).get("state_after") or payload.get("game_state") or {}
    snapshot = payload.get("player_snapshot") or {}
    hero = state.get("hero") or snapshot
    topics = set(payload.get("state_topics") or ["location", "money", "hp", "items", "resources"])
    rows = []
    if "location" in topics:
        scene = current_scene_context(payload)
        value = scene.get("room_name") or state.get("current_room")
        if value:
            rows.append(("현재 장소", value))
    if "money" in topics and type(state.get("gold_cp", snapshot.get("gold_cp"))) is int:
        rows.append(("소지금", fmt_money(state.get("gold_cp", snapshot.get("gold_cp")))))
    if "hp" in topics and type(hero.get("hp")) is int:
        maximum = hero.get("max_hp", hero.get("hp_max"))
        rows.append(("HP", f"{hero['hp']}/{maximum}" if type(maximum) is int else str(hero["hp"])))
    if "items" in topics:
        inventory = snapshot.get("inventory")
        if isinstance(inventory, list):
            rows.append(("휴대품", ", ".join(str(item.get("item_name", item.get("item_id", "물품")))
                + " " + str(item["quantity"]) + "개" for item in inventory
                if type(item.get("quantity")) is int and item["quantity"] > 0) or "없음"))
        elif isinstance(state.get("inventory"), dict):
            rows.append(("휴대품", ", ".join(str(key) + " " + str(value) + "개"
                for key, value in state["inventory"].items() if type(value) is int and value > 0) or "없음"))
    if "resources" in topics or "rest" in topics:
        for key, label in (("resources", "현재 자원"), ("hit_dice", "회복 주사위")):
            value = hero.get(key)
            if value is not None:
                rows.append((label, json.dumps(value, ensure_ascii=False)))
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return ("**게임 상태 확인**\n\n항목 | 현재 값\n---|---\n" +
            "\n".join(cell(label) + " | " + cell(value) for label, value in rows)) if rows else ""
