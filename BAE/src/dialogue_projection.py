"""현재 NPC에게 전달할 공개 사실과 거래 상태를 정리합니다.

입력: 게임 엔진, 인물 프로필, context_cards.json의 사실 사전, 검색된 카드 ID입니다.
처리: 카드 공개 조건과 인물별 지식 범위를 확인해 사용할 사실을 복사합니다.
물품 지급·금액 효과는 실행 기록과 대조하여 완료·미완료·확인 불가를 구분합니다.
출력: 사실 목록, 거래 사전, 또는 모델에 전달할 한국어 상태 설명입니다.

gameplay.py가 대화 자료를 준비할 때 사용하며 state_projection.py도 거래 구조를 읽습니다.
이 모듈은 엔진 상태를 읽습니다. 현재 보유량, 과거 전달, NPC의 보관량은 서로 다른 값입니다."""
from copy import deepcopy
from collections import OrderedDict
import math
from threading import RLock
import weakref


_FACT_VECTOR_CACHE = OrderedDict()
_FACT_VECTOR_CACHE_LOCK = RLock()
_FACT_VECTOR_CACHE_LIMIT = 256


def fact_knowledge_basis(fact, knower=None):
    """자료를 아는 방식과 실제 감각 경험의 범위를 구분합니다.

    기존 direct 표시는 개인 지식이라는 뜻일 수 있어 시각 목격으로 해석하지 않습니다.
    구체 감각·경험 주체는 원본 근거에 연결된 knowledge_basis 선언만 사용합니다.
    """
    declared = fact.get("knowledge_basis")
    if isinstance(declared, dict) and declared:
        return deepcopy(declared)
    kind = fact.get("direct_or_heard")
    category = fact.get("category")
    if kind == "unknown" or category == "knowledge_limit":
        method = "인지 한계 · 확인되지 않은 사항"
    elif kind == "heard" or category == "hearsay":
        method = "전해 들은 말 · 소문"
    elif kind in {"public_report", "public_identity"} or category in {"public_reference", "public_identity"}:
        method = "공개된 보고 · 정체 정보 (관찰자와 전달 경로 미지정)"
    elif category == "relationship":
        method = "관계 · 정체 · 인적 사항에 대한 지식"
    elif category == "memory":
        method = "개인 기억 (감각 경로 미지정)"
    elif category == "event":
        method = "사건 · 상태에 대한 지식 (목격자 미지정)"
    elif kind == "offer" or category in {"character_goal", "quest_contract", "trade"}:
        method = "바람 · 의향 · 제안 또는 운영 조건"
    else:
        method = "자료에 명시된 지식 (감각 경험 미지정)"
    result = {"확인 방식": method,
              "경험 범위": "기재된 내용에 한정하며 다른 감각 경험·현재 관찰을 입증하지 않음"}
    if knower:
        result["지식 보유자"] = knower
    return result


def _actor(data, npc_name):
    """인물 이름 또는 원본 카드 ID로 facts 자료의 인물 항목을 찾습니다."""
    for name, actor in data["actors"].items():
        if npc_name in {name, actor["source_card_id"]}:
            return name, actor
    raise ValueError("실험 자료에 없는 인물입니다.")


def _eligible_card(engine, identifier):
    return identifier in engine.c.cards and bool(engine.eligible(identifier))


def eligible_facts(engine, npc_name, data, retrieved_ids=()):
    """현재 인물이 알 수 있고 엔진의 공개 조건을 통과한 사실을 복사합니다.

    engine은 카드 공개와 대화 가능 여부를, data는 사실별 아는 인물과 출처를 제공합니다.
    자기 프로필의 사실은 사용할 수 있고, 다른 카드의 사실은 공개 가능한 검색 결과와
    연결된 경우에 후보로 넣습니다. 후보 목록을 반환하며 최종 관련성 선택은 따로 합니다."""
    name, actor = _actor(data, npc_name)
    profile_id = actor["source_card_id"]
    if not _eligible_card(engine, profile_id):
        return []
    can_talk = getattr(engine, "npc_can_talk", None)
    if callable(can_talk) and not can_talk(profile_id):
        return []
    if isinstance(retrieved_ids, str):
        retrieved_ids = [retrieved_ids]
    hits = {identifier for identifier in retrieved_ids
            if isinstance(identifier, str) and _eligible_card(engine, identifier)}
    result = []
    for fact in data["facts"]:
        if name not in fact["knowers"]:
            continue
        source_id = fact["source_card_id"]
        if not _eligible_card(engine, source_id):
            continue
        own = source_id == profile_id
        focused = set(fact.get("retrieval_card_ids", [])) & hits
        if own or source_id in hits or focused:
            result.append(deepcopy(fact))
    return result


def select_fact_ids(candidates, ids, max_facts=3):
    """사실 후보에서 선택된 ID만 중복 없이 순서대로 복사합니다.

    max_facts로 개수를 제한하고, 후보에 없는 ID나 잘못된 자료형은 선택하지 않습니다.
    내용을 새로 만들거나 빈 선택을 임의의 사실로 채우지 않습니다."""
    if type(max_facts) is not int or not 0 <= max_facts <= 6:
        raise ValueError("max_facts는 0~6 정수여야 합니다.")
    if not isinstance(ids, (list, tuple)):
        return []
    by_id = {fact["id"]: fact for fact in candidates}
    chosen, seen = [], set()
    for identifier in ids:
        if not isinstance(identifier, str) or identifier not in by_id or identifier in seen:
            continue
        if len(chosen) >= max_facts:
            break
        chosen.append(deepcopy(by_id[identifier]))
        seen.add(identifier)
    return chosen


def select_relevant_facts(question, candidates, embedder, max_facts=3):
    """공개된 사실 후보를 기존 E5로 비교하고 상위 ID의 원본 사본을 반환합니다.

    공개 조건·지식 범위와 항상 유지할 관계는 호출부가 관리합니다.
    인지 한계도 같은 질문과 비교해 선택하며, 모든 한계를 자동으로 덧붙이지 않습니다.
    이 함수는 후보의 category를 임의로 제외하지 않으며, 새 사실을 만들지 않습니다.
    passage 벡터만 모델별 최대 256개 보관합니다. 질문은 매번 query로 인코딩합니다.
    반환값은 (선택한 사실 목록, 관련성 점수·캐시·실패 진단)입니다.
    인코딩 오류나 차원·유한값 검증 실패 때는 빈 선택과 오류 진단을 반환합니다.
    관련성 순위는 사실의 진위나 생성 문장의 근거 일치를 보증하지 않습니다.
    작은 실제 E5 비교에서 단일 점수 기준이 관련 자료를 누락해 절대 임계값은 적용하지 않습니다.
    """
    if type(max_facts) is not int or not 0 <= max_facts <= 6:
        raise ValueError("max_facts는 0~6 정수여야 합니다.")
    diagnostic = {"method": "e5_query_passage_dot", "candidate_count": 0,
                  "selected_ids": [], "scores": [], "cache_hits": 0,
                  "encoded_passages": 0, "ignored_candidates": 0,
                  "max_facts": max_facts, "status": "empty",
                  "selection_policy": "ranking_only_without_validated_absolute_cutoff"}
    if not isinstance(question, str) or not question.strip():
        diagnostic.update(status="invalid_input", error="question은 비어 있지 않은 문자열이어야 합니다.")
        return [], diagnostic
    if not isinstance(candidates, (list, tuple)):
        diagnostic.update(status="invalid_input", error="candidates는 사실 목록이어야 합니다.")
        return [], diagnostic
    rows, seen, duplicates = [], set(), set()
    for row in candidates:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip()
                or not isinstance(row.get("text"), str) or not row["text"].strip()):
            diagnostic["ignored_candidates"] += 1
            continue
        if row["id"] in seen:
            duplicates.add(row["id"])
        seen.add(row["id"])
        rows.append(row)
    # 같은 ID가 여러 내용을 가리킬 수 있으면 어느 행도 임의로 채택하지 않습니다.
    diagnostic["ignored_candidates"] += sum(row["id"] in duplicates for row in rows)
    rows = [row for row in rows if row["id"] not in duplicates]
    diagnostic["candidate_count"] = len(rows)
    if not rows or max_facts == 0:
        return [], diagnostic

    def vectors(value, count):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != count:
            raise ValueError("임베딩 행 수가 요청과 다릅니다.")
        result, width = [], None
        for vector in value:
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            if not isinstance(vector, (list, tuple)) or not 1 <= len(vector) <= 8192:
                raise ValueError("임베딩 벡터 형식이 잘못되었습니다.")
            converted = tuple(float(number) for number in vector)
            if not all(math.isfinite(number) for number in converted):
                raise ValueError("임베딩에 유한하지 않은 값이 있습니다.")
            norm = math.sqrt(sum(number * number for number in converted))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("임베딩 벡터의 길이가 잘못되었습니다.")
            converted = tuple(number / norm for number in converted)
            if width is not None and width != len(converted):
                raise ValueError("임베딩 차원이 서로 다릅니다.")
            width = len(converted)
            result.append(converted)
        return result

    try:
        if not callable(getattr(embedder, "encode", None)):
            raise TypeError("사용 가능한 임베딩 모델이 없습니다.")
        try:
            model_key = weakref.ref(embedder)
            hash(model_key)
        except TypeError:
            # 약한 참조를 지원하지 않는 encoder도 캐시 없이 사용할 수 있습니다.
            model_key = None
        passages = ["passage: " + row["text"].strip() for row in rows]
        found, missing = {}, []
        with _FACT_VECTOR_CACHE_LOCK:
            for text in dict.fromkeys(passages):
                key = (model_key, text)
                if model_key is not None and key in _FACT_VECTOR_CACHE:
                    found[text] = _FACT_VECTOR_CACHE[key]
                    _FACT_VECTOR_CACHE.move_to_end(key)
                    diagnostic["cache_hits"] += 1
                else:
                    missing.append(text)
        options = {"normalize_embeddings": True, "show_progress_bar": False,
                   "convert_to_numpy": True}
        query = vectors(embedder.encode(["query: " + question.strip()], **options), 1)[0]
        fresh = vectors(embedder.encode(missing, **options), len(missing)) if missing else []
        diagnostic["encoded_passages"] = len(missing)
        for text, vector in zip(missing, fresh):
            found[text] = vector
        if any(len(vector) != len(query) for vector in found.values()):
            raise ValueError("질문과 사실의 임베딩 차원이 다릅니다.")
        scored = [(sum(left * right for left, right in zip(query, found[text])), index, row["id"])
                  for index, (row, text) in enumerate(zip(rows, passages))]
        if not all(math.isfinite(score) for score, _, _ in scored):
            raise ValueError("관련성 점수가 유한하지 않습니다.")
        with _FACT_VECTOR_CACHE_LOCK:
            if model_key is not None:
                for text, vector in zip(missing, fresh):
                    _FACT_VECTOR_CACHE[(model_key, text)] = vector
                    _FACT_VECTOR_CACHE.move_to_end((model_key, text))
                while len(_FACT_VECTOR_CACHE) > _FACT_VECTOR_CACHE_LIMIT:
                    _FACT_VECTOR_CACHE.popitem(last=False)
        chosen = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_facts]
        ids = [identifier for _, _, identifier in chosen]
        diagnostic.update(status="selected", selected_ids=ids,
                          scores=[{"id": identifier, "score": round(score, 6)} for score, _, identifier in chosen],
                          candidate_scores=[{"id": identifier, "score": round(score, 6)}
                                            for score, _, identifier in sorted(scored, reverse=True)[:6]])
        return select_fact_ids(rows, ids, max_facts), diagnostic
    except Exception as error:
        diagnostic.update(status="encoder_error", error=(type(error).__name__ + ": " + str(error))[:400])
        return [], diagnostic


def _effect_rows(content, effects, phase, path, conditional=False, active_sets=()):
    """물품·금액 효과를 찾아 실행 단계와 JSON 내부 경로를 함께 돌려주는 생성기입니다.

    공유 효과 묶음과 조건 분기도 따라갑니다. 조건 분기는 과거에 어느 쪽이 실행됐는지
    별도 판단이 필요하므로 conditional 표시를 남깁니다."""
    for index, effect in enumerate(effects):
        if not isinstance(effect, dict):
            continue
        here = f"{path}[{index}]"
        if "give" in effect or "gold_cp" in effect:
            yield phase, effect, here, conditional
        elif "use" in effect:
            name = effect["use"]
            if name not in active_sets and name in content.effect_sets:
                yield from _effect_rows(content, content.effect_sets[name], phase,
                                        here + ".use:" + name, conditional, (*active_sets, name))
        elif "if" in effect:
            # 지금의 조건이 과거 실행 순간에도 같았다고 가정하지 않습니다.
            for branch in ("then", "else"):
                yield from _effect_rows(content, effect.get(branch, []), phase,
                                        here + "." + branch, True, active_sets)
        elif "combat" in effect:
            for key, branch in effect.items():
                if key.startswith("on_") and isinstance(branch, list):
                    yield from _effect_rows(content, branch, phase, here + "." + key,
                                            True, active_sets)


def _phase_completed(engine, choice, phase):
    """행동 로그·성공 기록·실패 기록으로 해당 실행 단계의 완료 여부를 구합니다.

    반환값 True는 완료 증거가 있음, False는 미실행·미완료, None은 이 기록만으로
    분기를 확정할 수 없음을 뜻합니다. 현재 소지품만으로 과거 실행을 역산하지 않습니다."""
    identifier = choice["id"]
    done = identifier in engine.s.done
    attempted = done or any(row.get("choice") == identifier for row in engine.s.log)
    failed = any(failure[0] == identifier for failure in engine.s.failed)
    unresolved = bool(engine.s.pending and engine.s.pending.get("choice") == identifier)
    if phase == "before":
        return attempted
    if not attempted and not failed:
        return False
    if unresolved:
        return False
    if phase == "effects":
        return attempted if "check" not in choice else None
    if phase == "success":
        if done:
            return True
        return False if failed else None
    if phase == "failure":
        if failed:
            return True
        return False if done else None
    return None


def _npc_custody(engine, profile, npc_name, action_id, item_id, phase, path, conditional, completed):
    """명시된 초기 보관량과 유일한 지급 경로로 NPC의 남은 보관량을 구합니다.

    프로필 선언, 반복 횟수, 효과 경로, 완료 기록이 맞는 경우에만 수량을 계산합니다.
    다른 지급 경로나 충돌이 있으면 사유와 함께 미확인 값을 반환합니다.
    물건을 보관한다는 사실과 이야기상 소유권을 같은 뜻으로 취급하지 않습니다."""
    result = {"npc_current_quantity": None, "npc_custody_status": "not_declared"}

    def unknown(status):
        return {**result, "npc_custody_status": status}

    declarations = profile.get("custody", [])
    if not isinstance(declarations, list):
        return unknown("invalid_custody_declaration")
    declarations = [row for row in declarations if isinstance(row, dict) and row.get("item_id") == item_id]
    if not declarations:
        return result
    if len(declarations) != 1:
        return unknown("duplicate_custody_declaration")
    declaration = declarations[0]
    if declaration.get("source_npc") != npc_name:
        return unknown("source_npc_not_declared")
    initial = declaration.get("initial_quantity")
    if type(initial) is not int or initial < 0:
        return unknown("initial_quantity_unknown")
    if declaration.get("closed_unique_path") is not True:
        return unknown("closed_path_not_declared")
    if declaration.get("outgoing_action") != action_id or action_id not in profile.get("action_ids", []):
        return unknown("outgoing_action_not_declared")
    # 같은 고유 물품을 두 인물이 처음부터 보관한다는 중복 자료는 확정하지 않습니다.
    all_declarations = []
    for card in engine.c.cards.values():
        rows = card.get("dialogue", {}).get("custody", [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            return unknown("invalid_custody_declaration")
        all_declarations.extend(row for row in rows if row.get("item_id") == item_id)
    if len(all_declarations) > 1:
        return unknown("duplicate_custody_declaration")
    choice = engine.c.choices[action_id]
    if (choice.get("once") is not True or "check" in choice or phase != "effects" or conditional):
        return unknown("conditional_or_repeatable_path")
    # 선언 이후 다른 지급 경로가 추가되거나 효과가 중복되면 기존 추정을 중단합니다.
    paths = []
    for identifier, candidate in engine.c.choices.items():
        for candidate_phase in ("before", "effects", "success", "failure"):
            for _, effect, effect_path, nested in _effect_rows(
                    engine.c, candidate.get(candidate_phase, []), candidate_phase, candidate_phase):
                if effect.get("give") == item_id:
                    paths.append((identifier, candidate_phase, effect_path, nested, effect.get("n", 1)))
    if len(paths) != 1 or paths[0][:4] != (action_id, phase, path, False):
        return unknown("non_unique_outgoing_path")
    quantity = paths[0][4]
    if type(quantity) is not int or not 0 < quantity <= initial:
        return unknown("invalid_outgoing_quantity")
    log_count = sum(row.get("choice") == action_id for row in engine.s.log)
    done = action_id in engine.s.done
    if log_count > 1:
        return unknown("duplicate_outgoing_record")
    if completed is None or done != completed:
        return unknown("completion_record_conflict")
    player_quantity = engine.s.inventory.get(item_id, 0)
    if type(player_quantity) is not int or player_quantity < 0:
        return unknown("invalid_player_quantity")
    if (not done and player_quantity > 0) or (done and player_quantity > quantity):
        return unknown("unique_item_state_conflict")
    # 지급 뒤 플레이어에게 물품이 없더라도 반환으로 추정하지 않습니다.
    current = initial - quantity if done else initial
    status = ("outgoing_completed_player_item_absent" if done and player_quantity == 0
              else "outgoing_completed" if done else "declared_initial_custody")
    return {"npc_current_quantity": current, "npc_custody_status": status}


def transaction_state(engine, profile, npc_name):
    """인물에게 연결된 선택지의 물품 지급·금액 효과를 현재 상태와 묶습니다.

    입력은 engine, 인물 프로필, 현재 이름입니다. 각 행동의 가능 여부·완료 여부와 함께
    전달 수량·플레이어 현재 수량·NPC 보관량·효과 출처를 사전으로 반환합니다.
    과거 조건 분기를 확정할 수 없으면 *_completed를 None으로 둡니다.
    상점 구매·숙박은 별도 처리 경로의 자료이며 이 함수의 give/gold_cp 범위와 구분됩니다."""
    if profile.get("name", npc_name) != npc_name:
        raise ValueError("현재 인물과 프로필의 이름이 다릅니다.")
    available = {choice.id for choice in engine.available_choices()}
    transactions, missing = [], []
    for identifier in dict.fromkeys(profile.get("action_ids", [])):
        choice = engine.c.choices.get(identifier)
        if choice is None:
            missing.append(identifier)
            continue
        transfers, payments = [], []
        for phase in ("before", "effects", "success", "failure"):
            for _, effect, path, conditional in _effect_rows(
                    engine.c, choice.get(phase, []), phase, phase):
                completed = _phase_completed(engine, choice, phase)
                # 조건 안의 효과는 행동 실행 기록만으로 어느 분기가 적용됐는지 확정할 수 없습니다.
                if conditional and completed is not False:
                    completed = None
                if "give" in effect:
                    item_id = effect["give"]
                    transfers.append({"item_id": item_id,
                        "item_name": engine.c.items.get(item_id, {}).get("name", item_id),
                        "quantity": effect.get("n", 1),
                        "player_quantity": engine.s.inventory.get(item_id, 0),
                        "transfer_completed": completed,
                        "completion_evidence": "engine_action_phase" if completed is not None else "not_determinable",
                        "effect_path": path,
                        **_npc_custody(engine, profile, npc_name, identifier, item_id,
                                       phase, path, conditional, completed)})
                if "gold_cp" in effect:
                    amount = effect["gold_cp"]
                    payments.append({"gold_cp": amount,
                        "direction": "player_receives" if amount > 0 else "player_pays" if amount < 0 else "no_change",
                        "payment_completed": completed,
                        "completion_evidence": "engine_action_phase" if completed is not None else "not_determinable",
                        "effect_path": path})
        if transfers or payments:
            transactions.append({"action_id": identifier, "label": choice["label"],
                "action_currently_available": identifier in available,
                "action_completed": identifier in engine.s.done,
                "transfers": transfers, "payments": payments})
    return {"npc_name": npc_name, "scope": "give_and_gold_cp_effects_only",
            "player_gold_cp": engine.s.gold_cp, "transactions": transactions,
            "missing_action_ids": missing}


def render_transaction_state(state, npc_name=None):
    """거래 사전을 모델이 읽을 한국어 상태 설명으로 바꿉니다.

    현재 보유량, 전달 완료, 실행 가능 여부를 각각 표현하며, None은 미확인으로 표시합니다.
    질문에 답을 생성하는 함수가 아니라 transaction_state의 자료를 문장으로 정리합니다."""
    if isinstance(state, list):
        rows, header = state, {}
    elif isinstance(state, dict):
        rows = state.get("transactions", [state])
        header = state if "transactions" in state else {}
    else:
        raise TypeError("거래 상태는 객체 또는 목록이어야 합니다.")
    if not isinstance(rows, list):
        raise TypeError("transactions는 목록이어야 합니다.")

    def status(value, true_text, false_text):
        return true_text if value is True else false_text if value is False else "확정 여부 미확인"

    lines = []
    if header.get("scope"):
        scope = {"give_and_gold_cp_effects_only": "물품 지급과 gold_cp 효과만 포함"}.get(header["scope"], header["scope"])
        lines.append("상태 자료의 범위: " + str(scope) + ".")
    if "player_gold_cp" in header:
        lines.append("플레이어의 현재 소지금: " + str(header["player_gold_cp"]) + "cp.")
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("각 거래는 객체여야 합니다.")
        speaker = row.get("source_npc", row.get("npc_name", header.get("npc_name", npc_name or "NPC")))
        action_id = str(row.get("action_id", "ID 미제공"))
        prefix = str(speaker) + " ↔ 플레이어 / 행동 " + action_id
        action_states = []
        if row.get("label"):
            action_states.append("행동명: " + str(row["label"]))
        if "action_completed" in row:
            action_states.append(status(row["action_completed"], "행동 완료", "행동 미완료"))
        if "action_currently_available" in row:
            action_states.append(status(row["action_currently_available"], "현재 실행 가능", "현재 실행 불가"))
        if action_states:
            lines.append(prefix + ": " + "; ".join(action_states) + ".")
        transfers = row.get("transfers", [])
        if "item" in row or "item_id" in row:
            transfers = [row, *transfers]
        for transfer in transfers:
            item = transfer.get("item_name", transfer.get("item", transfer.get("item_id", "물품명 미제공")))
            details = ["물품: " + str(item)]
            if "item_id" in transfer and transfer.get("item_name"):
                details.append("물품 ID: " + str(transfer["item_id"]))
            if "quantity" in transfer:
                details.append("행동의 물품 수량: " + str(transfer["quantity"]) + "개")
            if "transfer_completed" in transfer:
                details.append(status(transfer["transfer_completed"], "전달 완료", "전달 미완료"))
            if "player_current_quantity" in transfer or "player_quantity" in transfer:
                quantity = transfer.get("player_current_quantity", transfer.get("player_quantity"))
                details.append("플레이어 현재 수량: " + (str(quantity) + "개" if quantity is not None else "미확인"))
            for key, label in (("completion_evidence", "확정 근거"), ("effect_path", "효과 경로")):
                if key in transfer:
                    details.append(label + ": " + str(transfer[key]))
            transfer_prefix = str(speaker) + " → 플레이어 / 행동 " + action_id
            lines.append(transfer_prefix + ": " + "; ".join(details) + ".")
        payments = row.get("payments", [])
        if "gold_cp" in row:
            payments = [row, *payments]
        for payment in payments:
            details = ["금액: " + str(payment.get("gold_cp", "미제공")) + "cp"]
            if payment.get("direction"):
                direction = {"player_receives": "플레이어가 받음", "player_pays": "플레이어가 지불함", "no_change": "금액 변화 없음"}.get(payment["direction"], payment["direction"])
                details.append("금액 방향: " + str(direction))
            if "payment_completed" in payment:
                details.append(status(payment["payment_completed"], "지급 완료", "지급 미완료"))
            for key, label in (("completion_evidence", "확정 근거"), ("effect_path", "효과 경로")):
                if key in payment:
                    details.append(label + ": " + str(payment[key]))
            amount = payment.get("gold_cp")
            payer_is_player = (payment.get("direction") == "player_pays"
                               or (isinstance(amount, (int, float)) and amount < 0))
            payment_prefix = (("플레이어 → " + str(speaker)) if payer_is_player
                              else (str(speaker) + " → 플레이어")) + " / 행동 " + action_id
            lines.append(payment_prefix + ": " + "; ".join(details) + ".")
    if header.get("missing_action_ids"):
        lines.append("자료에서 찾지 못한 행동 ID: " + ", ".join(map(str, header["missing_action_ids"])) + ".")
    return "\n".join(lines)
