"""게임 엔진의 현재 값과 실제 거래 기록을 출력용 자료로 바꿉니다.

입력: 엔진, 전달용 payload, 거래 연결 정보, 행동 전 상태와 실행 후 Event 목록입니다.
처리: 플레이어 상태·공개 출구를 복사하고 구매의 전후 수량·금액·재고를 대조합니다.
생성된 문장의 금전·물품·체력 표현은 정규식으로 찾아 상태 표시 범위를 정합니다.
출력: 모델용 상태 사전, 확인된 거래 장부, 현재 NPC에게 해당하는 상점 자료입니다.

gameplay.py와 output_code.py가 읽는 보조 자료이며 거래의 기본 구조는
dialogue_projection.transaction_state를 사용합니다. 표현 검사는 의미 전체를 검증하는
모델이 아니라, 확인된 엔진 값을 어떤 영역에 표시할지 정하는 보조 절차입니다."""
from copy import deepcopy
import re

from dialogue_projection import transaction_state


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


def mentioned_context_items(message, payload, npc_name):
    """현재 인물에게 허용된 물품 이름을 찾는 표시 보조 함수입니다.

    공백·대소문자·영문 괄호를 무시하고 한글 이름의 마지막 두 글자 이상
    명사도 비교합니다. 동명이거나 같은 꼬리 이름은 함께 잡힐 수 있으며,
    동의어나 대명사의 뜻을 추측하지 않습니다.
    """
    normalize = lambda value: re.sub(r"\s+", "", value).casefold()
    text = normalize(message)
    snapshot = payload.get("player_snapshot") or {}
    candidates = [row for key in ("inventory", "equipped_items") for row in snapshot.get(key, [])
                  if type(row.get("quantity")) is int and row["quantity"] > 0]
    dialogue = payload.get("dialogue_context") or {}
    ledger = dialogue.get("transactions") or {}
    if ledger.get("npc_name") == npc_name:
        for transaction in ledger.get("transactions", []):
            transfers = transaction.get("transfers", [])
            past = any(row.get("transfer_completed") is True for row in transfers)
            past |= any(row.get("payment_completed") is True for row in transaction.get("payments", []))
            if transaction.get("action_currently_available") or transaction.get("action_completed") or past:
                candidates.extend(transfers)
    shop = dialogue.get("shop") or {}
    if shop.get("seller_npc") == npc_name and shop.get("status") == "verified_menu_owner":
        candidates.extend(shop.get("items", []))
    matched = set()
    for item in candidates:
        label = item.get("item_name")
        if not isinstance(label, str) or not item.get("item_id"):
            continue
        label = re.sub(r"\([^)]*[A-Za-z][^)]*\)", "", label).strip()
        aliases = {normalize(label)}
        tail = re.split(r"\s+|의", label)[-1]
        if re.fullmatch(r"[가-힣]{2,}", tail):
            aliases.add(normalize(tail))
        if any(len(alias) >= 2 and alias in text for alias in aliases):
            matched.add(item["item_id"])
    return sorted(matched)


def current_scene_context(payload):
    """현재 관찰·공개 배경·실제 이동 선택지를 구분해 복사합니다.

    available_actions가 목록이면 실제 가능한 move 선택지가 출구의 기준입니다.
    그 목록이 없을 때만 명시된 scene_context.exits를 사용합니다. 출구 None은
    자료 미제공, []는 제공된 목록에 공개 이동 선택지가 없음을 뜻합니다.
    conditional_background는 이미 공개된 장소 설명이며 현재 시각·지각의 확정값이
    아닙니다. 출처와 조건을 유지하고, 생성된 GM 문장을 여기서 수집하지 않습니다.
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
        result["observation"] = observation.strip()[:240]
    background = scene.get("conditional_background")
    result["conditional_background"] = [
        {"source_card_id": row["source_card_id"], "text": row["text"]}
        for row in (background if isinstance(background, list) else [])
        if isinstance(row, dict) and isinstance(row.get("source_card_id"), str)
        and isinstance(row.get("text"), str) and row["text"].strip()]
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


def draft_state_topics(text, payload, npc_name):
    """생성한 대사에 상태 영역의 표현이 있는지 순수 Python으로 확인합니다.

    생성된 문장에서 금전 업무의 어휘·단위를 찾고, 현재 공개된 물품 이름과
    수치 HP도 함께 확인합니다. 해당 분야의 내용은 엔진 상태 표로 표시합니다.
    대명사나 어휘 목록 밖 동의어는 이 검사에서 놓칠 수 있습니다.
    길 안내는 현재 공개 장소 자료가 있을 때 일반적인 이동 조건·명령 표현으로
    찾습니다. 과거 이야기의 장소 이름만으로는 분기하지 않습니다. 인용·은유,
    새 표현의 뜻까지 판정하는 검사가 아니므로 의미적 정확성을 보장하지 않습니다.
    """
    if not isinstance(text, str):
        return []
    topics = []
    finance = (r"돈(?=[^가-힣]|[은을이가도에만과으]|$)|선금|보수(?=[^가-힣]|[는은가를로]|$)|보상|금액|지급|지불|수령|"
               r"구매|구입|환불|급여|가격|대금|잔액|매입|되팔|매각|재판매|골드|실버|코퍼|금화|은화|"
               r"(?<![a-z])(?:gp|sp|cp)(?![a-z])")
    requested = payload.get("state_topics")
    money_scope = (isinstance(requested, list) and "money" in requested
        and all(isinstance(topic, str) and topic in {"money", "items", "hp", "rest", "location", "resources"}
                for topic in requested))
    dialogue = payload.get("dialogue_context") or {}
    snapshot = payload.get("player_snapshot") or {}
    contract = dialogue.get("contract") or {}
    advance = contract.get("advance") if isinstance(contract, dict) else None
    advance = advance if isinstance(advance, dict) else {}
    shop = dialogue.get("shop") or {}
    ledger = dialogue.get("transactions") or {}
    payment_context = ledger.get("npc_name") in (None, npc_name) and any(
        (row.get("action_currently_available") or row.get("action_completed")
         or any(payment.get("payment_completed") is True for payment in row.get("payments", [])))
        and any(type(payment.get("gold_cp")) is int for payment in row.get("payments", []))
        for row in ledger.get("transactions", []))
    financial_context = (type(snapshot.get("gold_cp")) is int or type(advance.get("gold_cp")) is int
        or (shop.get("status") == "verified_menu_owner" and shop.get("seller_npc") == npc_name)
        or payment_context)
    # 금전 범위가 명시된 문맥에서 독립된 전달·수령 술어만 보탭니다.
    # '알려드릴게요', '기다려주세요'처럼 앞말에 붙은 보조 표현은 제외합니다.
    transfer = (r"(?<![가-힣A-Za-z0-9_])(?:안\s*)?(?:"
        r"줬|주(?:었|겠|지\s*(?:않|못)|셨)|줄\s*(?:게|께|거|겁|테|예정)|"
        r"드렸|드린\s*(?:적|것|거|게)|드리(?:겠|지\s*(?:않|못))|드릴\s*(?:게|거|겁|테|예정)|"
        r"받(?:았|겠|은|을|으셨|으신|지\s*(?:않|못)))")
    # 앞의 일반 조건절만 검사 사본에서 제외합니다. 실제 금액이나 거래 술어는 유지합니다.
    # 원래 대사와 전달·수령 검사에 사용할 text는 바꾸지 않습니다.
    claim_text = text
    condition = re.match(r"^\s*[^.!?\n]*?(?:더라도|다면|해도|여도|셔도|어도|아도|면)(?=\s)", text)
    if condition:
        clause, remainder = condition.group(), text[condition.end():]
        amount_or_currency = r"\d|골드|실버|코퍼|금화|은화|원|(?<![a-z])(?:gp|sp|cp)(?![a-z])"
        money_action = r"지급|지불|수령|구매|구입|환불|매입|매각|재판매|되팔"
        # 조건절 예외가 독립된 현재형 전달·거래 주장까지 통과시키지 않게 합니다.
        independent_transaction = (r"(?<![가-힣A-Za-z0-9_])(?:안\s*)?(?:"
            r"드리|드립|드려|드린|주(?:겠|었|는|고|며|오|지|십|셨|신|세|시)|줘|줍|준|줄|"
            r"받|건네|건넵|건넨|전달|판매|팔|팝|판|삽|사(?:는|고|며|겠)|"
            r"구매|구입|수령|지급|지불|돌려(?:주|드))")
        if (remainder.strip() and re.search(finance, clause, re.IGNORECASE)
                and not re.search(amount_or_currency, clause, re.IGNORECASE)
                and not re.search(money_action, clause)
                and not re.search(finance, remainder, re.IGNORECASE)
                and not re.search(independent_transaction, remainder)
                and not re.search(transfer, text)):
            claim_text = remainder
    if re.search(finance, claim_text, flags=re.IGNORECASE):
        topics.append("money")
    if money_scope and financial_context and re.search(transfer, text) and "money" not in topics:
        topics.append("money")
    if mentioned_context_items(text, payload, npc_name):
        topics.append("items")
    if re.search(r"(?:hp|체력|생명력)[^\d\n]{0,12}\d+|\d+(?:\s*/\s*\d+)?\s*(?:hp|체력|생명력)",
                 text, flags=re.IGNORECASE):
        topics.append("hp")
    resource_words = r"슬롯|직업\s*(?:기능|능력)|클래스\s*자원|남은\s*자원|사용\s*횟수|\bslots?\b"
    resource_claim = r"현재|남[은았아]|소모|소진|사용[했한]|시전[했한]|썼|완료|없|있|회복|\d|한\s*번|두\s*번"
    snapshot = payload.get("player_snapshot") or {}
    resource_names = list((snapshot.get("resources") or {}).keys()) + snapshot.get("features", [])
    resource_names += snapshot.get("spells", []) + snapshot.get("cantrips", [])
    resource_names += list((snapshot.get("spell_names") or {}).values())
    normalize = lambda value: re.sub(r"[\s_]", "", str(value)).casefold()
    named_resource = any(len(normalize(name)) >= 3 and normalize(name) in normalize(text) for name in resource_names)
    spell_completion = re.search(r"주문|시전|\b(?:spell|cantrip)\b", text, re.IGNORECASE) and re.search(
        r"소모|소진|사용[했한]|시전[했한]|썼|완료", text)
    if spell_completion or ((re.search(resource_words, text, re.IGNORECASE) or named_resource)
                            and re.search(resource_claim, text)):
        topics.append("resources")
    scene = current_scene_context(payload)
    has_scene = bool(scene.get("room_id") or scene.get("room_name") or scene.get("exits") is not None)
    if has_scene:
        navigation = (r"(?:나가|나오|들어가|들어오|돌아가|돌아오|올라가|올라오|내려가|내려오|건너가)려면|"
                      r"(?<![가-힣])(?:가|오)려면|(?:으로|로)\s*(?:가|오)려면|"
                      r"(?:이동하|올라가|올라오|내려가|내려오|들어가|들어오|나가|나오|돌아가|돌아오|건너가|따라가|따라오|지나가|통과하|향하|가|오)(?:세요|십시오)|"
                      r"(?:따라|통해|거쳐|타고)[^.!?\n]{0,16}(?:가면|오면|나오면|됩니다)|"
                      r"(?:으로|로)\s*(?:이동하면|가면|오면|향하면|나가면|들어가면|올라가면|올라오면|내려가면|내려오면)")
        spatial_cue = (r"통로|출구|입구|계단|사다리|길목|방향|왼쪽|오른쪽|위쪽|아래쪽|밖으로|안으로|위로|아래로|"
                       r"(?:^|[^가-힣])(?:그|이|저)?\s*(?:길|문)(?=[은을이가으로에]|\s|$)")
        labels = [scene.get("room_name")] + [row.get("destination") for row in scene.get("exits") or []]
        # 길 안내는 같은 문장 안의 공개 장소·공간 단서와 함께 확인합니다.
        # 인사말의 '가세요'나 보고된 인용 대사를 독립된 이동 지시로 바꾸지 않습니다.
        unquoted = re.sub(r'''"[^"\n]*"|'[^'\n]*'|“[^”\n]*”|「[^」\n]*」|『[^』\n]*』''', "", text)
        for sentence in re.split(r"[.!?\n]+", unquoted):
            if re.search(r"(?:라고|다고|라는|다는)[^.!?\n]*(?:들었|전해|말했|말했다|했대|한대)", sentence):
                continue
            named = any(isinstance(label, str) and len(label) >= 2 and label in sentence for label in labels)
            route = re.search(navigation, sentence) and (named or re.search(spatial_cue, sentence))
            present = re.search(r"여기|이곳|현재|지금|우리는|우리가|저희는|저희가", sentence)
            concrete_position = present and (
                (named and re.search(r"입니다|이에요|예요|있어요|있습니다|있다", sentence))
                or re.search(r"(?:안|밖|내부|외부|입구|출구|앞|뒤|옆|근처)(?:입니다|이에요|예요)", sentence)
                or re.search(r"(?:안|밖|내부|외부|입구|출구|앞|뒤|옆|근처)(?:에|에서)\s*(?:있|대화|이야기|만나|서\s*있)", sentence))
            if route or concrete_position:
                topics.append("location")
                break
    return topics


def player_snapshot(engine):
    """실제 캐릭터·가방·장착 장비·사용 자원·행동 기록을 복사한 상태 자료를 만듭니다.

    물품 이름과 주문 이름은 Content에서 읽어 ID와 함께 넣습니다.
    현재 가진 물품과 실행했던 행동 기록을 따로 담아 모델이 두 사실을 구별하게 합니다."""
    hero = engine.hero
    item = lambda identifier, quantity: {"item_id": identifier,
        "item_name": engine.c.items.get(identifier, {}).get("name", identifier), "quantity": quantity}
    equipped = engine.s.vars.get("setup", {}).get("equipped_items", {})
    actions = [{"source_log_index": index, "action_id": row.get("choice"),
                "actor_id": row.get("actor"), "events": deepcopy(row.get("events", []))}
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
    """실제 장소·공개 메뉴·인물 행동의 연결로 현재 판매자를 확인합니다.

    bindings의 상점 선언과 엔진의 공개·완료 기록이 함께 맞아야 판매자를 정합니다.
    같은 방에 있다는 사실만으로 그곳의 모든 NPC를 거래 상대라고 추정하지 않습니다."""
    matches = []
    for binding in bindings.get("shops", []):
        if binding.get("room_id") != engine.s.room or not engine.c.rooms[engine.s.room].get("shop"):
            continue
        card = engine.c.cards.get(binding.get("seller_card_id"), {})
        profile = card.get("dialogue", {})
        action = engine.c.choices.get(binding.get("menu_action_id"), {})
        menu_id = binding.get("menu_card_id")
        if (profile.get("name") != binding.get("seller_npc") or card.get("room") != engine.s.room
                or action.get("room") != engine.s.room or binding.get("menu_action_id") not in profile.get("action_ids", [])
                or not any(effect.get("show") == menu_id for effect in action.get("effects", []))
                or menu_id not in engine.s.shown
                or binding["menu_action_id"] not in engine.s.done
                or not engine.eligible(binding["seller_card_id"])
                or not engine.npc_can_talk(binding["seller_card_id"])):
            continue
        # 같은 메뉴를 다른 인물의 소유 행동으로도 선언하면 판매자를 확정하지 않습니다.
        owners = [c for c in engine.c.cards.values() if binding["menu_action_id"] in c.get("dialogue", {}).get("action_ids", [])]
        if len(owners) != 1:
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


def directed_transactions(engine, profile, npc_name, bindings):
    """정적 물품·금액 효과에 확인 가능한 주는 사람과 받는 사람을 연결합니다.

    기본 거래 구조는 dialogue_projection에서 읽습니다. 물품 보관 선언과 선금 연결이
    맞는 경우에만 NPC를 출처로 정하고, 근거가 없는 당사자는 unknown으로 남깁니다."""
    result = transaction_state(engine, profile, npc_name)
    player = player_party(engine)
    card_id = next((key for key, card in engine.c.cards.items() if card.get("dialogue", {}).get("name") == npc_name), "")
    npc = npc_party(npc_name, card_id)
    for row in result["transactions"]:
        for transfer in row["transfers"]:
            declared = any(c.get("item_id") == transfer["item_id"] and c.get("source_npc") == npc_name
                and c.get("outgoing_action") == row["action_id"] for c in profile.get("custody", []))
            transfer.update({"from": npc if declared else {"kind": "unknown"}, "to": player})
        for payment in row["payments"]:
            declared = any(b.get("npc_name") == npc_name and b.get("action_id") == row["action_id"]
                and b.get("gold_cp") == payment["gold_cp"]
                and (profile.get("quest_offer") or {}).get("advance", {}).get("gold_cp") == payment["gold_cp"]
                for b in bindings.get("payments", []))
            source, target = (npc if declared else {"kind": "unknown"}, player) if payment["gold_cp"] >= 0 else (player, {"kind": "unknown"})
            payment.update({"from": source, "to": target})
    return result
