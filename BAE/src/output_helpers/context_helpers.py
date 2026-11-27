# 출력 실행 보조 함수: 노트북과 같은 전역 공간에서 원본 본문을 실행합니다.

OPENING_LINES = {'scn-s1-npc-halvik': {'name': '할빅',
                       'text': '어서 오시오. 나는 이 마을의 촌장 할빅이오. 괜찮다면 마을 사정을 잠시 들어 주겠소?',
                       'source_text_sha256': 'cfdcd8cd66c9b17241528171d3d2c0c2ec7ee0607b7107d84b2ee8e9a01b1e55',
                       'speech_register': 'hao'},
 'scn-s1-npc-mera': {'name': '메라',
                     'text': '나는 이 우물을 지키던 메라요. 눈이 어두우니 조금 가까이 와서 말해 주구려. 무슨 일로 찾아왔소?',
                     'source_text_sha256': '7d8be9e059626c2154c0afa6ec69c0df3b55e9fadf8fa6d65d3df06bead59025',
                     'speech_register': 'elder'},
 'scn-q1-npc-sera': {'name': '세라',
    'text': '안녕하세요. 저는 세라예요. 남편 토브와 딸 리나가 사라졌어요. 두 사람을 찾는 일을 도와주실 수 있을까요?',
                     'source_text_sha256': '07a11e622d7a7142a0f6eb28f4af172b16c348bfe50b192195248044b6962473',
                     'speech_register': 'polite'},
 'scn-m-npc-smith': {'name': '그란',
                     'text': '나는 대장장이 그란이오. 연장을 손보러 왔소, 아니면 필요한 물건이 있소?',
                     'source_text_sha256': '95346dab2bfdf63f74bdec9a4ea8a2d2c0313dbd94e8f7cec3d24b7c59ad896d',
                     'speech_register': 'hao'},
 'scn-m-npc-innkeeper': {'name': '오단',
                         'text': '어서 오세요. 저는 이 여관의 주인 오단입니다. 묵을 곳이나 잠시 쉬어 갈 곳을 찾으시나요?',
                         'source_text_sha256': '17c3c57fce93d5da390033cf474cfa47753922f77d7e819db5d3951a1416bc4b',
                         'speech_register': 'polite'}}


def _json_text(value):
    """한글을 유지해 직렬화하고, 엔진의 set은 순서가 일정한 목록으로 바꿉니다."""
    def fallback(item):
        if isinstance(item, set):
            return sorted(item, key=str)
        raise TypeError("JSON으로 전달할 수 없는 값입니다: " + type(item).__name__)
    return json.dumps(value, ensure_ascii=False, default=fallback)


def output_speaker(payload, npc_name=None):
    """호출자가 지정한 현재 NPC를 우선합니다. 이름이 없으면 GM 장면입니다."""
    if npc_name is not None:
        return canonical_npc_name(npc_name)
    return canonical_npc_name(payload.get("npc_name") or (payload.get("actor_context") or {}).get("name") or "")


def _recent_messages(history, *, speaker_id=None):
    """모델에 필요한 최근 발언만 복사하고 원본 대화 기록은 보존합니다.

    NPC가 지정되면 그 NPC의 직접 대사만 가져옵니다. 생성된 배경·몸짓이나 다른
    인물의 발언을 현재 NPC의 기억으로 다시 넣지 않습니다. 지정이 없는 GM·추천
    호출은 기존의 일반 대화 구성을 사용합니다.
    """
    messages = []
    for row in normalize_history(history)[-HISTORY_MESSAGES:]:
        if row.get("knowledge_only") is True or row.get("kind") == "card":
            continue  # 참고 카드나 예시를 실제 발언 이력으로 승격하지 않습니다.
        content = row["content"]
        if row["role"] == "user" and speaker_id and row.get("input_kind") == "action":
            # 실제 입력 메타데이터가 행동이라고 할 때만 구분합니다. 문장에 이름이
            # 있거나 '간다'로 끝난다는 이유로 과거 입력의 종류를 추측하지 않습니다.
            content = _json_text({"input_kind": "action", "text": content})
        elif row["role"] == "assistant":
            # 동일 대사를 aggregate text와 blocks에 두 번 넣지 않습니다.
            record = {"trust": "previous_utterance"}
            if isinstance(row.get("blocks"), list) and row["blocks"]:
                blocks = [block for block in row["blocks"] if isinstance(block, dict)]
                if speaker_id:
                    blocks = [block for block in blocks if block.get("kind") == "speech"
                              and block.get("speaker_id") == speaker_id]
                    if not blocks:
                        continue
                record["blocks"] = [{key: deepcopy(block[key]) for key in
                    ("kind", "speaker_id", "speaker", "text") if key in block}
                    for block in blocks]
            else:
                if speaker_id and (isinstance(row.get("blocks"), list)
                        or row.get("speaker_id") != speaker_id or row.get("kind") == "narration"):
                    continue  # 직접 대사나 화자를 확인할 수 없는 이력은 NPC에게 넘기지 않습니다.
                record["text"] = content
                for key in ("speaker_id", "speaker", "kind"):
                    if key in row:
                        record[key] = deepcopy(row[key])
            content = _json_text(record)
        messages.append({"role": row["role"], "content": content})
    return messages


def _speaker_context(payload):
    """실제 장면의 허용 화자와 대화 단계만 연결하며 인적 사항은 만들지 않습니다."""
    from response_guard import allowed_speaker_map
    allowed = allowed_speaker_map(payload)
    return {"allowed_speakers": [{"speaker_id": identifier, "name": name}
                                 for identifier, name in allowed.items()],
            "dialogue_state": deepcopy(payload.get("dialogue_state") or {})}


def _with_speaker_whitelist(payload, npc_name=None):
    """엔진 목록이 없는 직접 JSON 호출에는 명시된 현재 화자의 ID만 연결합니다.

    인적 사항을 만들지 않고 호출자의 ID 또는 검색 원문의 공개 이름 머리말만 읽습니다.
    엔진이 allowed_speakers를 명시했으면 빈 목록도 그대로 존중합니다.
    """
    result = deepcopy(payload)
    # 명시된 현재 대화 상대를 검증 계층에도 전달합니다. 새 인물 정보는 만들지 않습니다.
    if npc_name is not None:
        result["npc_name"] = canonical_npc_name(npc_name)
    if "allowed_speakers" in payload:
        return result
    speaker = output_speaker(payload, npc_name)
    actor = payload.get("actor_context") or {}
    identifier = payload.get("speaker_id") or actor.get("speaker_id") or actor.get("card_id")
    allowed = []
    if speaker and isinstance(identifier, str) and identifier:
        allowed = [{"speaker_id": identifier, "name": speaker}]
    elif speaker:
        for row in (payload.get("retrieval") or {}).get("scenario_results", []):
            identifier = row.get("card_id") or (row.get("metadata") or {}).get("card_id")
            heading, separator, _ = row.get("text", "").partition(" —")
            if separator and heading.strip() == speaker and isinstance(identifier, str) and identifier:
                allowed = [{"speaker_id": identifier, "name": speaker}]
                break
    result["allowed_speakers"] = allowed
    return result


def _utterance_task(payload):
    """엔진의 역할·단계를 작은 턴 계약으로 묶습니다. 인물별 대사를 만들지 않습니다.

    첫 만남의 마지막 입력은 대개 '누구와 이야기한다'는 행동 표시문입니다.
    이를 이미 한 대사로 해석하지 않도록, 이번에 누가 어떤 목적으로 말하는지
    자료와 별도로 명시합니다. 근거 참조는 이미 받은 I3의 문장에만 연결합니다.
    """
    from response_guard import response_requirements
    requirements = response_requirements(payload)
    dialogue = payload.get("dialogue_state") or {}
    phase = requirements["phase"]
    input_kind = dialogue.get("last_input_kind")
    if input_kind not in {"speech", "action"}:
        input_kind = "action" if phase in {"first_contact", "action_result"} else "speech"
    # 응답 방식이 단계보다 먼저입니다. 예를 들어 첫 인사 전 규칙 질문을
    # 받았어도 GM의 규칙 설명을 NPC 자기소개로 바꾸지 않습니다.
    if requirements["mode"] == "rule_explanation":
        task = "팀 룰 근거로 현재 규칙 질문을 짧게 설명한다. 실제 게임의 판정 결과는 엔진 확정 자료와 구분한다."
    elif requirements["mode"] == "scene_dialogue":
        task = "이번 입력에 관련된 공개 장면과 엔진 확정 결과를 필요한 만큼 설명한다. 대사가 필요하면 현재 허용된 인물만 말한다."
    elif phase == "first_contact":
        task = ("현재 NPC가 처음 인사하는 차례다. 원문의 역할에 맞춰 자신을 소개하고 상대의 용건을 받는다. "
                "첫 만남의 진행 안내가 제공됐으면 지금 이용할 수 있는 도움과 지정된 다음 준비 대상·장소를 짧게 연결한다. "
                "이미 확정된 행동이 있다면 그 결과를 반영한다.")
        if input_kind == "speech":
            task += " 실제 질문을 받았다면 짧게 소개한 뒤 그 질문에 답한다."
    elif phase == "action_result":
        task = ("이번에 누가 무엇을 실행했고 어떤 결과가 확정됐는지에 반응한다. "
                "현재 NPC는 상대방의 행동에 자기 입장에서 답한다. "
                "수락·제안과 실제 전달 완료를 구분한다. 전달이 확정됐으면 보낸 쪽·받은 쪽·물건·수량을 유지하며, "
                "전달 자료에 있는 용도를 자연스럽게 설명한다.")
    else:
        task = "현재 질문이나 제안에 먼저 1~3문장으로 직접 답한다."
    confirmed_event = payload.get("engine_result")
    confirmed_event = confirmed_event if isinstance(confirmed_event, dict) else {}
    handovers = payload.get("item_handovers") or confirmed_event.get("item_handovers") or []
    handovers = handovers if isinstance(handovers, list) else []
    if phase == "action_result" and any(isinstance(row, dict) and row.get("confirmed") is True for row in handovers):
        # 인물이나 아이템 이름을 조건으로 검사하지 않고 실제 전달 영수증에 반응합니다.
        task += " 이번에는 전달이 완료됐으므로 건넨 물건과 자료에 명시된 용도를 먼저 설명한다."
    if requirements["mode"] == "npc_dialogue":
        task += " 현재 NPC가 1~3문장으로 직접 말한다."
        if payload.get("guidance_rendered_separately") is True:
            task += " 질문과 관계있는 이미 전달한 공개 안내와 단서를 근거로 답한다."
        else:
            task += " 진행 안내는 제공된 allowed_guidance의 다음 대상만 짧게 연결한다."
    target = requirements["target_speaker_id"]
    speaker_sources = []
    for index, row in enumerate((payload.get("retrieval") or {}).get("scenario_results", []), 1):
        identifier = row.get("card_id") or (row.get("metadata") or {}).get("card_id")
        if target and identifier == target:
            speaker_sources.append("S" + str(index))
    context_ids = [prefix + str(index) for prefix, key in (("R", "rule_results"), ("S", "scenario_results"))
        for index, _ in enumerate((payload.get("retrieval") or {}).get(key, []), 1)]
    if requirements["mode"] == "npc_dialogue" and phase == "first_contact":
        context_ids = speaker_sources  # 첫 인사는 현재 인물의 소개·용건에 집중합니다.
        if not speaker_sources:
            # 검색되지 않은 인물 설정을 채우기 위해 다른 카드나 기억을 뒤지지 않습니다.
            task = "현재 NPC가 처음 말을 건넨다. 제공된 이름으로 짧게 인사하고, 자료에 없는 개인 사정이나 부탁은 만들지 않는다."
    return {"phase": phase, "mode": requirements["mode"], "input_kind": input_kind,
        "input_meaning": "플레이어의 행동 표시문" if input_kind == "action" else "플레이어가 한 말",
        "speaker": {"speaker_id": target, "name": requirements["target_name"] or "게임 마스터"},
        "listener": _npc_view(_engine_data(dialogue.get("listener"))) if requirements["mode"] == "npc_dialogue" else _engine_data(dialogue.get("listener")),
        "performer": _npc_view(_engine_data(payload.get("action_subject") or dialogue.get("performer"))) if requirements["mode"] == "npc_dialogue" else _engine_data(payload.get("action_subject") or dialogue.get("performer")),
        "speaker_source_ids": speaker_sources, "context_source_ids": context_ids, "task": task,
        "output": {key: requirements[key] for key in ("speech_required", "narration_allowed", "max_blocks")}}


def _npc_view(value):
    """엔진의 플레이어 이름을 NPC가 이미 들은 정보로 제공하지 않습니다.

    typed player 역할과 엔진 hero·player 구조의 식별 이름만 모델용 사본에서 뺍니다. 실제 사용자 발언,
    팀 근거·저장 이력·게임 상태·화면의 캐릭터 이름은 바꾸지 않습니다.
    """
    if isinstance(value, dict):
        result = {key: _npc_view(child) for key, child in value.items()}
        for key in ("hero", "player"):
            if isinstance(result.get(key), dict):
                for name_key in ("id", "name", "speaker_id", "speaker"):
                    result[key].pop(name_key, None)
        if value.get("kind") == "player":
            for key in ("id", "name", "speaker_id", "speaker"):
                result.pop(key, None)
        return result
    if isinstance(value, list):
        return [_npc_view(child) for child in value]
    return deepcopy(value)


def _delivered_guidance_context(payload):
    """현재 NPC가 실제로 전달한 공개 안내만 작은 사실 목록으로 복사합니다.

    연결부가 메시지·영수증을 대조한 delivered_guidance를 받습니다. 아직 표시하지
    않은 allowed_guidance 후보나 다른 NPC의 대화를 현재 화자의 기억으로 보충하지 않습니다.
    """
    rows = payload.get("delivered_guidance", [])
    if not isinstance(rows, list):
        raise ValueError("이미 전달한 방문 안내는 목록이어야 합니다.")
    if not rows:
        return []
    from response_guard import response_requirements

    speaker_id = response_requirements(payload)["target_speaker_id"]
    if not speaker_id:
        return []
    fields = (
        "guidance_id", "source_npc_id", "source_npc_name", "target_npc_id",
        "target_npc_name", "room_id", "room_name", "purpose", "message_index",
    )
    result = []
    for row in rows:
        if not isinstance(row, dict) or row.get("source_npc_id") != speaker_id:
            continue
        if (not all(isinstance(row.get(key), str) and row[key].strip() for key in
                    ("guidance_id", "target_npc_id", "target_npc_name", "room_id", "room_name"))
                or type(row.get("message_index")) is not int or row["message_index"] < 0):
            raise ValueError("이미 표시한 방문 안내에는 실제 영수증·인물·장소가 필요합니다.")
        result.append({**{key: deepcopy(row[key]) for key in fields if key in row},
                       "status": "already_delivered_public_guidance"})
    return result


def _story_context(payload):
    """현재 진행 단계와 안내만 복사합니다. 원문 카드나 새 인물 설정을 만들지 않습니다."""
    value = payload.get("story_progression")
    if payload.get("guidance_rendered_separately") is True:
        # 안내의 실제 발언·해금은 연결부가 한 번 처리합니다. 모델에게 목적지를
        # 다시 주지 않아 같은 이동 지시를 다른 말로 반복하거나 바꾸지 않습니다.
        result = {"guidance_rendered_separately": True}
        if isinstance(value, dict):
            for key in ("stage", "reference_required"):
                if key in value:
                    result[key] = deepcopy(value[key])
        return result
    return _engine_data(value) if isinstance(value, dict) else {}


def _resolved_public_knowledge(payload, engine_result=None):
    """이번에 공개된 단서·사실·소문만 복사합니다. 인물 카드 전체는 제외합니다.

    입력은 연결부의 revealed_knowledge 목록이며, 반환값은 대사에 필요한 짧은
    공개 근거입니다. 실제 단서 원문의 표시·저장은 연결부가 따로 담당합니다.
    """
    rows = payload.get("revealed_knowledge")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise ValueError("이번에 공개한 지식은 목록이어야 합니다.")
    revealed = None
    if isinstance(engine_result, dict):
        revealed = set(engine_result.get("revealed_card_ids") or [])
    result = []
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") not in {"clue", "fact", "rumor"}:
            continue
        identifier, text = row.get("card_id"), row.get("text")
        if not isinstance(identifier, str) or not isinstance(text, str) or not text.strip():
            raise ValueError("공개 지식에는 카드 ID와 실제 공개 문장이 필요합니다.")
        if revealed is not None and identifier not in revealed:
            raise ValueError("이번 실행에서 공개하지 않은 지식은 전달할 수 없습니다.")
        result.append({"card_id": identifier, "title": row.get("title") or "단서",
                       "kind": row["kind"], "text": text,
                       "epistemic_status": "rumor" if row["kind"] == "rumor" else "public_evidence"})
    return deepcopy(result)


def _confirmed_outcome_facts(payload, engine_result):
    """확정 사건을 작은 결과 자료로 읽습니다. 원본 엔진·검색 자료는 바꾸지 않습니다.

    새 연결부는 outcome_facts를 전달합니다. 이전 호출도 실제 공개 이벤트와
    현재 관찰만으로 표시할 수 있게 하며, 카드의 조건문·과거 대사는 보충하지 않습니다.
    """
    confirmed = validate_engine_result(engine_result)
    if confirmed.get("confirmed") is not True:
        raise ValueError("확정된 실행만 결과 화면에 표시할 수 있습니다.")
    supplied = payload.get("outcome_facts")
    if supplied is not None:
        if not isinstance(supplied, dict):
            raise ValueError("실행 결과 자료는 객체여야 합니다.")
        facts = deepcopy(supplied)
    else:
        current = _current_action_context(payload, confirmed)
        changes = current["changes"]
        facts = {
            "action_id": changes.get("executed_action_id"),
            "action_label": payload.get("player_input", ""),
            "room_before": changes.get("room_before"),
            "room_after": changes.get("room_after"),
            "elapsed_minutes": 0,
            "ordered_events": current["confirmed_events"],
            "changes": changes,
            "observation": current["current_observation"].get("observation", ""),
        }
    elapsed = facts.get("elapsed_minutes", 0)
    if type(elapsed) is not int or elapsed < 0:
        raise ValueError("경과 시간은 0 이상의 정수 분이어야 합니다.")
    events = facts.get("ordered_events", [])
    if not isinstance(events, list) or any(not isinstance(row, dict) for row in events):
        raise ValueError("확정 사건은 발생 순서대로 담긴 객체 목록이어야 합니다.")
    observation = facts.get("observation", "")
    if not isinstance(observation, str):
        raise ValueError("현재 관찰은 실제 공개 문장이어야 합니다.")
    facts["elapsed_minutes"] = elapsed
    facts["ordered_events"] = _engine_data(events)
    facts["observation"] = observation
    facts["changes"] = _engine_data(facts.get("changes") or {})
    return facts


def _speech_style(identifier):
    """모든 NPC 생성 경로에 같은 한국어 말투 설명을 제공합니다.

    영문 분류값은 원본 인사 설정에 보존하고 모델에는 의미가 명확한 설명을
    전달합니다. 인물별 정답 대사를 만들거나 카드의 성격 문장을 바꾸지 않습니다.
    """
    register = OPENING_LINES.get(identifier, {}).get("speech_register")
    descriptions = {
        "polite": "상대를 존중하는 공손한 존댓말. 한 응답 안에서 같은 높임말 수준을 유지한다.",
        "hao": "상대를 존중하는 하오체. 말끝을 '~하오', '~겠소', '~이오' 계열로 일관되게 맞춘다.",
        "elder": "차분하고 공손한 옛말투. 상대를 낮추지 않고 높임말 수준과 말끝을 일정하게 유지한다."}
    return {"description": descriptions.get(register, "최근 직접 대사의 높임말 수준과 종결형을 유지한다.")}


def _current_interaction_contract(payload):
    """화면에 실제 공개된 현재 요청만 작은 읽기 자료로 제공합니다.

    제안·수락·지급은 서로 다른 상태입니다. 모델의 대사나 선언으로 상태를
    바꾸지 않으며, 공개하지 않은 선택지의 본문이나 보상을 여기서 추출하지 않습니다.
    """
    offer = payload.get("confirmed_offer")
    public = None
    if isinstance(offer, dict) and offer.get("status") in {
            "offered", "accepted", "in_progress", "completed"}:
        issuer = offer.get("issuer") or {}
        speaker = output_speaker(payload)
        if not speaker or issuer.get("name") == speaker:
            public = _npc_view(_engine_data({key: deepcopy(offer[key]) for key in (
                "event_id", "request_id", "source_choice_id", "issuer", "status", "purpose",
                "acceptance_label", "on_acceptance") if key in offer}))
            # 수락 문구의 '찾는다'를 의뢰인이 할 일로 옮기지 않도록 역할을 명시합니다.
            # 이는 공개 의뢰의 수락 주체이며, 수락·지급 완료를 새로 선언하지 않습니다.
            public["acceptor"] = {"kind": "player"}
            public["performer"] = {"kind": "player"}
    return {"state_authority": "engine", "dialogue_commits_effects": False,
            "public_request": public,
            "scope": "공개한 조건과 실제 실행은 구분한다. 확인된 보상·물품·행동 범위에서 제안하고 완료는 엔진 결과를 따른다."}


def _revealed_public_references(payload, engine_result):
    """이번에 실제 공개한 카드 ID와 일치하는 원본 I3 문장만 연결합니다.

    카드의 본문·순위·출처는 수정하지 않습니다. 공개된 지식과 실행된 효과를
    분리하며, 검색되지 않은 본문을 별도 카드 캐시에서 보충하지 않습니다.
    """
    event = validate_engine_result(engine_result)
    identifiers = event.get("revealed_card_ids") or []
    if not isinstance(identifiers, list) or any(not isinstance(value, str) for value in identifiers):
        raise ValueError("공개한 카드 ID는 문자열 목록이어야 합니다.")
    requested = list(dict.fromkeys(identifiers))
    _, sources = build_context(payload)
    provided = []
    matched = []
    for source in sources:
        metadata = source.get("metadata") or {}
        identifier = metadata.get("card_id")
        if not identifier:
            rows = payload["retrieval"].get("scenario_results", [])
            if source["label"].startswith("S"):
                identifier = rows[int(source["label"][1:]) - 1].get("card_id")
        if source["label"].startswith("S") and identifier in requested:
            provided.append(deepcopy(source))
            matched.append(identifier)
    return provided, {"revealed_card_ids": requested, "provided_card_ids": list(dict.fromkeys(matched)),
                      "missing_card_ids": [identifier for identifier in requested if identifier not in matched]}


def _with_action_speaker_scope(payload, engine_result):
    """행동에 실제 응답할 인물만 발화 권한으로 사용합니다.

    현재 방에 보인다는 이유만으로 새 NPC 대사를 만들지 않습니다. 방의 사람
    목록은 관찰 정보로 유지하고, 엔진이 정한 응답 화자는 기존 허용 목록과
    교집합으로 제한합니다. 원본 payload와 엔진 기록은 바꾸지 않습니다.
    """
    result = deepcopy(payload)
    task = _utterance_task(result)
    if (task["phase"] != "action_result" or task["mode"] != "scene_dialogue"
            or not isinstance(engine_result, dict) or engine_result.get("confirmed") is not True):
        return result
    from response_guard import allowed_speaker_map
    allowed = allowed_speaker_map(result)
    responding = engine_result.get("reply_speakers") or []
    if not isinstance(responding, list):
        raise ValueError("이번 행동의 응답 인물은 목록이어야 합니다.")
    selected = []
    for row in responding:
        if not isinstance(row, dict) or allowed.get(row.get("speaker_id")) != row.get("name"):
            raise ValueError("이번 행동의 응답 인물이 현재 허용 인물과 다릅니다.")
        if row not in selected:
            selected.append({"speaker_id": row["speaker_id"], "name": row["name"]})
    result["allowed_speakers"] = selected
    return result


def _npc_turn_roles(payload, contract):
    """누가 말하고 누가 의뢰를 맡는지, 공개된 역할과 목적만 묶습니다.

    가족 관계 등 인물 설정을 대화에서 다시 추출하지 않습니다. 발언 주인과
    의뢰 수행자는 매번 같은 엔진 역할을 사용하고 설정은 팀 근거로 확인합니다.
    """
    roles = {"speaker": deepcopy(contract["speaker"]), "listener": {"kind": "player"},
             "user_utterance_owner": "listener", "assistant_utterance_owner": "speaker"}
    public = _current_interaction_contract(payload)["public_request"]
    if public is not None:
        roles["request"] = {key: deepcopy(public[key]) for key in
            ("request_id", "issuer", "acceptor", "performer", "status", "purpose") if key in public}
    return _npc_view(roles)


def _completed_transfer_roles(payload, contract):
    """확정 영수증의 전달 방향을 현재 화자·상대 기준으로 읽습니다. 문장을 추측하지 않습니다."""
    event = payload.get("engine_result") or {}
    handovers = payload.get("item_handovers") or event.get("item_handovers") or []
    transfers = []
    speaker = contract["speaker"]

    def participant(party):
        if not isinstance(party, dict):
            return "other"
        if party.get("kind") == "player":
            return "listener"
        if (party.get("speaker_id") == speaker["speaker_id"]
                or party.get("card_id") == speaker["speaker_id"]
                or party.get("name") == speaker["name"]):
            return "speaker"
        return "other"

    for row in handovers if isinstance(handovers, list) else []:
        if not isinstance(row, dict) or row.get("confirmed") is not True:
            continue
        giver = participant(row.get("from") or {
            "card_id": row.get("source_card_id"), "name": row.get("source_npc"),
        })
        recipient = participant(row.get("to"))
        if "speaker" not in {giver, recipient}:
            continue
        transfers.append({
            "item_id": row.get("item_id"), "item_name": row.get("item_name"),
            "quantity": row.get("quantity"), "giver": giver, "recipient": recipient,
            "speaker_role": "giver" if giver == "speaker" else "recipient",
            "listener_role": "giver" if giver == "listener" else
                             "recipient" if recipient == "listener" else "other",
            "purpose": row.get("explanation", ""), "completed": True,
        })
    return transfers


def _npc_confirmed_fact_message(payload, contract):
    """현재 화자의 실제 전달·공개 안내를 마지막에 놓을 짧은 한국어 사실로 바꿉니다.

    모델의 답변 예시를 만들지 않습니다. 영수증의 주체·수량과 실제 표시한 안내만
    현재 관점으로 읽으며, 아직 공개하지 않은 후보와 검색 소문은 여기에 넣지 않습니다.
    """
    if contract["mode"] != "npc_dialogue":
        return None
    facts = []
    # 대화에서 감사·약속이 나와도 의뢰의 단계는 실제 실행 기록으로만 정합니다.
    # 카드별 예외 문장을 두지 않고 모든 의뢰에 같은 주체·시점 정보를 제공합니다.
    request = _current_interaction_contract(payload)["public_request"]
    if request is not None:
        status = request["status"]
        stages = {"offered": "플레이어에게 제안했으며 아직 수락되지 않았다",
                  "accepted": "플레이어가 수락했으며 아직 완료되지 않았다",
                  "in_progress": "플레이어가 수행 중이며 아직 완료되지 않았다",
                  "completed": "플레이어가 수행을 완료했다"}
        issuer = (request.get("issuer") or {}).get("name") or contract["speaker"]["name"]
        facts.append(issuer + "은 의뢰인이고, 의뢰를 수행할 사람은 플레이어다.")
        facts.append("의뢰의 현재 단계: " + stages[status] + ".")
        if request.get("purpose"):
            facts.append("공개된 의뢰 목적: " + request["purpose"])
        # 보상 금액만 주면 지급 시점이 뒤집힐 수 있어 실제 공개된 지급 조건도
        # 같은 자료에 둡니다. 아직 제안되지 않은 완료 보상은 여기서 보충하지 않습니다.
        for grant in request.get("on_acceptance", []):
            if grant.get("kind") == "money" and type(grant.get("amount_cp")) is int:
                from game.engine import fmt_money
                timing = "수락할 때 지급할" if status == "offered" else "수락할 때 이미 지급한"
                facts.append("공개된 지급 조건: " + timing + " 준비금 "
                             + fmt_money(grant["amount_cp"]) + ".")
    state = _current_public_state(payload)
    inventory = state.get("player", {}).get("inventory", state.get("inventory"))
    for transfer in _completed_transfer_roles(payload, contract):
        name, quantity = transfer.get("item_name"), transfer.get("quantity")
        if not isinstance(name, str) or not name.strip():
            continue
        amount = f" {quantity}개" if type(quantity) is int and quantity > 0 else ""
        if transfer["giver"] == "speaker" and transfer["recipient"] == "listener":
            facts.append(f"현재 화자가 상대에게 {name}{amount}를 이미 건넸다.")
            held = (inventory.get(transfer["item_id"], 0) if isinstance(inventory, dict) else
                    next((row.get("quantity", 0) for row in inventory or []
                          if isinstance(row, dict) and row.get("item_id") == transfer["item_id"]), 0))
            if isinstance(held, (int, float)) and held > 0:
                facts.append(f"지금 {name}을 가진 사람은 상대다.")
        elif transfer["giver"] == "listener" and transfer["recipient"] == "speaker":
            facts.append(f"현재 화자가 상대에게서 {name}{amount}를 이미 받았다.")
        else:
            continue
        if transfer.get("purpose"):
            facts.append("이 전달의 확인된 용도: " + transfer["purpose"])
    for guide in _delivered_guidance_context(payload):
        facts.append("현재 화자의 안내로 상대에게 " + guide["target_npc_name"]
                     + " · " + guide["room_name"] + " 방문 정보가 이미 전달됐다.")
        if guide.get("purpose"):
            facts.append("이 안내의 확인된 목적: " + guide["purpose"])
    if not facts:
        return None
    return {"role": "system", "content":
        "이번 대화의 확정 사실을 현재 상태의 기준으로 삼아, 검색된 소문·예시보다 우선해 현재 질문에 답한다.\n"
        + _json_text({"이번 대화의 확정 사실": facts})}


def _npc_current_scene(payload):
    """현재 장소의 공개 관찰·실제 인물만 복사하고 다른 장소는 보충하지 않습니다."""
    scene = payload.get("scene_context") or {}
    if not isinstance(scene, dict):
        raise ValueError("현재 장면 자료는 dict여야 합니다.")
    result = {key: deepcopy(scene[key]) for key in ("room_id", "room_name", "observation") if key in scene}
    result["people"] = [{key: deepcopy(person[key]) for key in ("card_id", "name", "label", "presence")
                         if key in person} for person in scene.get("people", []) if isinstance(person, dict)]
    return _npc_view(_engine_data(result))


def _npc_dialogue_affordances(payload):
    """연결부가 확인한 현재 서비스와 공개 장비 조건을 대사에도 전달합니다.

    목록이 없는 호출은 미제공으로 남깁니다. 빈 목록과 미제공은 다릅니다.
    전체 시나리오의 숨은 행동·보상·조건은 여기서 찾아 채우지 않습니다.
    """
    value = payload.get("dialogue_affordances")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("현재 이용 조건은 dict여야 합니다.")
    result = {"services_complete": value.get("services_complete") is True}
    fields = {"services": ("id", "label", "kind", "price_cp", "item_id", "item_name", "quantity",
                           "available", "reason"),
              "equipment_checks": ("id", "label", "requirements", "missing", "available")}
    for key, allowed in fields.items():
        rows = value.get(key)
        if rows is None:
            continue
        if not isinstance(rows, list):
            raise ValueError("현재 이용 조건의 " + key + "는 목록이어야 합니다.")
        result[key] = [{field: _engine_data(row[field]) for field in allowed if field in row}
                       for row in rows if isinstance(row, dict)]
    return _npc_view(result)


def _current_public_state(payload):
    """현재 장소·소지품·수치만 읽습니다. 과거 행동 목록과 내부 조건은 제외합니다."""
    state = payload.get("engine_snapshot") or payload.get("model_state") or payload.get("game_state") or {}
    result = {key: deepcopy(state[key]) for key in
        ("scene", "current_room", "inventory", "gold_cp", "money", "현재 장소", "플레이어 소지금") if key in state}
    hero = state.get("hero") if isinstance(state.get("hero"), dict) else {}
    result["player"] = {key: deepcopy(hero[key]) for key in
        ("name", "hp", "max_hp", "hp_max", "resources", "conditions") if key in hero}
    snapshot = payload.get("player_snapshot")
    if isinstance(snapshot, dict):
        # 이름·ID·전체 행동 이력을 빼고 지금 가방의 표시 이름·수량만 유지합니다.
        for key in ("hp", "max_hp", "gold_cp", "resources", "conditions"):
            if key in snapshot:
                result["player"][key] = deepcopy(snapshot[key])
        if isinstance(snapshot.get("inventory"), list):
            result.pop("inventory", None)
            result["player"]["inventory"] = [{key: deepcopy(row[key]) for key in
                ("item_id", "item_name", "quantity") if key in row}
                for row in snapshot["inventory"] if isinstance(row, dict)]
    elif "플레이어가 보유한 소지품" in state:
        result["player"]["inventory"] = deepcopy(state["플레이어가 보유한 소지품"])
    return _engine_data(result)


def _current_action_context(payload, engine_result):
    """이번에 확정된 사건과 현재 관찰만 모델용 자료로 구성합니다.

    원본 엔진 기록은 그대로 보존합니다. 이전 생성 서술·과거 실행 목록·검색된
    배경 카드가 이번 행동의 결과를 대신하지 않도록 별도 구성 경로를 사용합니다.
    """
    confirmed = validate_engine_result(engine_result)
    if confirmed.get("confirmed") is not True:
        raise ValueError("행동 서술에는 확정된 실행 결과가 필요합니다.")
    events = []
    records = confirmed.get("public_event_records")
    has_records = isinstance(records, list)
    for row in records if has_records else []:
        if (not isinstance(row, dict) or row.get("kind") == "card"
                or row.get("kind") in {"say", "interaction", "scene_cue"}
                or row.get("knowledge_only") is True
                or (row.get("data") or {}).get("knowledge_only") is True):
            continue
        events.append({"kind": row.get("kind"), "text": row.get("text", ""),
            "data": _engine_data(row.get("data") or {})})
    # say에는 생략된 주어와 옛 진행 안내가 섞일 수 있습니다. 확정된 숫자 변화와
    # 전달 영수증을 사용하고, 뺀 say 문장을 public_events에서 되살리지 않습니다.
    if not events and not has_records:
        events = [{"kind": "public", "text": text} for text in confirmed.get("public_events", [])
                  if isinstance(text, str) and text.strip()]
    scene = payload.get("scene_context") or {}
    observed = {key: deepcopy(scene[key]) for key in
        ("room_id", "room_name", "observation", "people", "aftermath", "exits") if key in scene}
    return {"event_id": confirmed.get("event_id"),
        "performed_action_ids": deepcopy(confirmed.get("executed_action_ids") or []),
        "performer": _engine_data(payload.get("action_subject") or {}),
        "confirmed_events": events,
        "changes": _engine_data(confirmed.get("engine_delta") or payload.get("engine_delta") or {}),
        "item_handovers": _engine_data(confirmed.get("item_handovers") or []),
        "revealed_card_ids": deepcopy(confirmed.get("revealed_card_ids") or []),
        "current_state": _current_public_state(payload),
        "current_observation": _engine_data(observed)}


def _build_current_action_messages(payload, speaker, contract, engine_result):
    """실행 결과의 새 설명을 만들며 과거 장면이나 무관한 검색 카드를 반복하지 않습니다."""
    sources, coverage = _revealed_public_references(payload, engine_result)
    data = {"이번 실행과 현재 관찰": _current_action_context(payload, engine_result),
        "읽기 전용 상호작용 계약": _current_interaction_contract(payload),
        "이번 화자": deepcopy(contract["speaker"]),
        "허용된 화자": _speaker_context(payload)["allowed_speakers"]}
    if contract["mode"] == "npc_dialogue":
        # 단서의 실제 문장만 전달합니다. 인물 카드 전체를 현재 대사로 옮기지 않습니다.
        knowledge = _resolved_public_knowledge(payload, engine_result)
        data["이번에 공개한 지식"] = knowledge
        # 전달 방향은 마지막 확정 사실 문맥에서 한 번만 설명합니다.
        data["이번 실행과 현재 관찰"].pop("item_handovers", None)
        coverage["provided_resolved_card_ids"] = [row["card_id"] for row in knowledge]
        data["공개 지식의 전달 범위"] = coverage
        # provided_sources는 실제 요청에 들어간 자료만 뜻합니다. 원본 검색 목록은
        # payload와 기존 coverage에 남기고, 빼 놓은 카드 전체를 제공했다고 기록하지 않습니다.
        sources = [{"label": row["card_id"], "title": row["title"], "text": row["text"],
                    "metadata": {"card_id": row["card_id"], "kind": row["kind"],
                                 "epistemic_status": row["epistemic_status"]},
                    "authority": "reported_rumor" if row["kind"] == "rumor"
                                 else "confirmed_public_knowledge"} for row in knowledge]
        data["말투"] = _speech_style(contract["speaker"]["speaker_id"])
        data["이번 대화의 역할"] = _npc_turn_roles(payload, contract)
        affordances = _npc_dialogue_affordances(payload)
        if affordances is not None:
            data["현재 장소의 이용 조건"] = affordances
        data["진행 안내"] = _story_context(payload)
        data = _npc_view(data)
    else:
        # GM의 현재 사건은 엔진이 실행한 변화와 실제 관찰로만 설명합니다.
        # 공개 카드의 소문·조건·관찰 방법은 알아낸 정보로 보존되지만, 이동하거나
        # 뚜껑을 열었다는 이유로 실제 발생한 사건이나 추가 조사로 승격하지 않습니다.
        # 원본 검색 결과와 coverage는 결과 진단에서 별도로 그대로 확인합니다.
        sources = []
    instruction = ("이번 실행과 현재 관찰에 있는 행동·변화·눈앞의 요소만 짧은 장면으로 표현한다. "
        "새 조사나 이동, 시각·날씨·소문을 덧붙이지 않고 현재 사건 자체를 서술한다.")
    if contract["mode"] == "npc_dialogue":
        instruction = ("현재 NPC가 확정된 이번 결과에 자기 입장에서 1~3문장으로 직접 답한다. "
            "의뢰 수행자는 역할 자료를 따르고, 전달이 완료됐다면 실제 보낸 쪽·받은 쪽·물건과 용도를 유지한다. "
            "진행 안내는 제공된 allowed_guidance와 연결한다.")
        if payload.get("guidance_rendered_separately") is True:
            instruction = ("현재 NPC가 확정된 결과에 자기 입장에서 1~3문장으로 반응한다. "
                "의뢰와 전달의 주체·물건·용도를 유지한다. "
                "질문과 관계있는 이미 전달한 공개 안내와 단서를 근거로 답한다.")
    facts = _npc_confirmed_fact_message(payload, contract)
    return [{"role": "system", "content": NPC_CONVERSATION_SYSTEM if contract["mode"] == "npc_dialogue" else OUTPUT_SYSTEM},
        {"role": "system", "content": "이번에 확정된 실행 자료입니다.\n" + _json_text(data)},
        {"role": "system", "content": instruction + "\n" + _json_text(contract)},
        *([facts] if facts else []),
        {"role": "user", "content": payload["player_input"]}], sources


def _plain_npc_conversation(history, speaker_id):
    """실제로 주고받은 현재 NPC의 대사만 평문 이력으로 복사합니다.

    화면 표시용 집계 문자열과 blocks를 중복해서 넣지 않습니다. 다른 인물의 대사,
    GM 설명·카드·예시는 현재 NPC가 한 말로 바꾸지 않습니다. 행동 입력은 원래
    입력 종류가 기록돼 있을 때만 별도로 표시하며 문장 표현으로 추측하지 않습니다.
    """
    recent = []
    for row in normalize_history(history)[-HISTORY_MESSAGES:]:
        if row.get("knowledge_only") is True or row.get("kind") == "card":
            continue
        if row["role"] == "user":
            content = (_json_text({"input_kind": "action", "text": row["content"]})
                if row.get("input_kind") == "action" else row["content"])
        elif isinstance(row.get("blocks"), list):
            parts = [block["text"] for block in row["blocks"] if isinstance(block, dict)
                and block.get("kind") == "speech" and block.get("speaker_id") == speaker_id
                and isinstance(block.get("text"), str) and block["text"].strip()]
            if not parts:
                continue
            content = "\n".join(parts)
        elif row.get("speaker_id") == speaker_id and row.get("kind") != "narration":
            content = row["content"]
        else:
            continue
        recent.append({"role": row["role"], "content": content})
    return recent


def _reference_requested(payload):
    """연결부에서 정한 사실 확인 여부를 본문 모델의 선택보다 먼저 적용합니다."""
    values = [payload.get("reference_required"), payload.get("needs_information"),
              _story_context(payload).get("reference_required")]
    if any(value is True for value in values):
        return True
    # 직접 JSON을 받는 호출도 기존 근거를 보유합니다. 분류가 없으면 원문 확인을
    # 기본으로 하고, 연결부가 명시한 잡담(False)만 자료 없는 대화로 시작합니다.
    return not any(value is False for value in values)


def _engine_data(value):
    """엔진 전달값에서 참고 카드 본문을 제외하고 실행 데이터만 복사합니다.

    이 경계는 팀 retrieval에는 적용하지 않습니다. 문장 의미를 분류하거나
    지식을 재검색하지 않고 카드로 표시된 구조·본문 필드만 제외합니다.
    """
    missing = object()
    blocked_fields = {"public_narratives", "conditional_background", "player_text", "gm_only_text", "vars",
                      "flags", "visited", "done", "once_done",
                      "card_text", "profile_text", "background_text"}
    def copy(item):
        if isinstance(item, dict):
            if item.get("knowledge_only") is True or item.get("kind") == "card":
                return missing
            result = {}
            for key, child in item.items():
                if key in blocked_fields:
                    continue
                copied = copy(child)
                if copied is not missing:
                    result[key] = copied
            return result
        if isinstance(item, list):
            result = []
            for child in item:
                copied = copy(child)
                if copied is not missing:
                    result.append(copied)
            return result
        return deepcopy(item)
    result = copy(value)
    return None if result is missing else result
