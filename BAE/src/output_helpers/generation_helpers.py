# 출력 실행 보조 함수: 노트북과 같은 전역 공간에서 원본 본문을 실행합니다.

"""현재 확인된 사실을 근거로 대사와 장면을 생성합니다.

첫 인사는 원본과 연결된 준비 문구를 유지하며, 수락·전달은 실제 역할과 영수증으로
공통 확인 문구를 구성합니다. 일반 대화와 새 지식 설명은 모델이 생성합니다. 실행 사실은 엔진에서 읽고
이전 생성 문장을 현재 사건의 근거로 재사용하지 않습니다. 형식 검사와 실제
생성 비용은 기록하며, 그 검사를 사실 정확도 점수로 표시하지 않습니다.
"""


def _prompt_characters(messages):
    """요청 크기 제한용 문자 수입니다. 실제 입력 토큰 수는 Ollama 응답에서 받습니다."""
    return len(_json_text(messages))


def generate_answer(messages, max_new_tokens=MAX_NEW_TOKENS, sample=True):
    """짧은 분류·매핑 등 문자열이 필요한 기존 호출의 연결 함수입니다."""
    return run_generation(messages, max_new_tokens=max_new_tokens, sample=sample)["text"]


OUTPUT_POLICY = {"first_contact": "source_bound_prepared_or_guided_generation", "confirmed_acceptance_and_handover": "deterministic_role_receipt_acknowledgement",
                 "critical_transaction_model_requests": 0, "ordinary_dialogue": "model_generated",
                 "free_dialogue": "explicit_information_route_then_optional_reference", "automatic_semantic_judge": False,
                 "ordinary_npc_max_model_requests": 2}


class OutputFormatError(ValueError):
    def __init__(self, message, raw_response=""):
        super().__init__(message)
        self.raw_response = raw_response


def _blocks_text(blocks):
    """기존 문자열 소비자를 위한 표시 집계이며 구조화 원본은 blocks에 남습니다."""
    return "\n\n".join(("**" + row["speaker"] + "**\n" if row["kind"] == "speech" else "")
                        + row["text"] for row in blocks)


class _ResponseStream:
    """닫힌 블록의 구조·화자를 확인한 뒤 누적 목록을 화면으로 보냅니다.

    미완성 JSON은 숨기며 chunk에는 지금까지 확인한 blocks 전체가 들어갑니다.
    블록 내부의 글자 스트림이나 문장별 사실 검증을 뜻하지 않습니다.
    """
    def __init__(self, payload, engine_result=None):
        from response_guard import allowed_speaker_map, response_requirements
        self.payload, self.engine_result = payload, engine_result
        self.allowed = allowed_speaker_map(payload)
        self.max_blocks = response_requirements(payload)["max_blocks"]
        self.turn_id = secrets.token_urlsafe(12)
        self.enabled = _output_stream_sink.get() is not None
        self.restart()

    def packet(self, event, blocks=None, block_id=None):
        rows = self.blocks if blocks is None else blocks
        return {"event": event, "turn_id": self.turn_id, "blocks": deepcopy(rows),
                "block_id": block_id, "text": _blocks_text(rows),
                "speaker": "", "kind": "structured_blocks"}

    def restart(self):
        self.blocks, self.hold = [], False
        _emit_output_stream(self.packet("start"))

    def update(self, text):
        if not self.enabled or self.hold:
            return
        # JSON 본문이나 미완성 문자열은 보내지 않고 닫힌 블록만 읽습니다.
        prefix = re.match(r'\s*\{\s*"blocks"\s*:\s*\[', text)
        if prefix is None:
            return
        from response_guard import validate_response_block, unique_json_keys
        decoder, position, index = json.JSONDecoder(object_pairs_hook=unique_json_keys), prefix.end(), 0
        while index < self.max_blocks:
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text) or text[position] == "]":
                return
            try:
                row, position = decoder.raw_decode(text, position)
            except ValueError:
                return  # 다음 조각에서 블록이 완성될 수 있습니다.
            if index >= len(self.blocks):
                try:
                    block = validate_response_block(row, self.allowed, self.payload, self.engine_result)
                except ValueError:
                    self.hold = True
                    return
                self.blocks.append(block)
                _emit_output_stream(self.packet("chunk", block_id=index))
            index += 1
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text) or text[position] != ",":
                return
            position += 1

    def finish(self, blocks):
        self.blocks = deepcopy(blocks)
        _emit_output_stream(self.packet("end", block_id=len(blocks) - 1))

    def fail(self):
        self.blocks = []
        _emit_output_stream(self.packet("error"))


def _attach_output_error(error, requested, messages, payload, stages, trace):
    """오류·취소에서도 완료된 요청과 실제 미완료 단계를 구분해 보존합니다."""
    error.messages = deepcopy(requested)
    error.context_messages = deepcopy(messages)
    error.draft_messages = deepcopy(trace.get("draft_messages", []))
    error.review_messages = deepcopy(trace.get("review_messages", []))
    error.meaning_review = deepcopy(getattr(error, "meaning_review", trace.get("meaning_review")))
    error.reply_to = payload["player_input"]
    error.retrieval_query = (payload.get("input_trace") or {}).get("retrieval_query", payload["player_input"])
    error.stage_metrics = deepcopy(stages)
    error.active_stage = trace.get("active_stage")
    error.backend = diagnostics_metadata()
    if stages:
        error.backend["options"] = deepcopy(stages[0].get("request_options"))
        error.raw_response = stages[-1].get("text", "")
        error.draft_response = stages[0].get("text", "")
    error.versions = _version_metadata()


def _generate_public(messages, payload, speaker, *, engine_result=None, cancel_event=None, kind="dialogue", provided_sources=None):
    """본문 생성과 최대 1회의 형식 교정을 실행하고 실제 요청·비용을 보존합니다.

    일반 오류와 사용자 취소는 모두 원래 예외로 호출자에게 전달합니다.
    취소된 화면에 오류 알림을 보내다 실패해도 원래 예외와 진단을 덮지 않습니다.
    """
    from response_guard import parse_response_blocks, response_blocks_schema
    response_format = response_blocks_schema(payload)
    def inspect(generated):
        limit = generated.get("generation_limit", MAX_NEW_TOKENS)
        if (generated.get("done_reason") == "length"
                or generated.get("output_tokens", 0) >= limit):
            raise ValueError("응답이 생성 한도에 도달했습니다.")
        return parse_response_blocks(generated["text"], payload, engine_result)
    stream = None
    stages = []
    requested = messages
    try:
        # 첫 start 알림에서도 화면이 취소될 수 있으므로 생성과 같은 예외 경계에 둡니다.
        stream = _ResponseStream(payload, engine_result)
        generated = run_generation(requested,
            on_text=None, cancel_event=cancel_event,
            response_format=response_format)
        generated["stage"] = "generation"
        generated["messages"] = deepcopy(requested)
        generated["provided_sources"] = deepcopy(provided_sources or [])
        first = generated
        stages.append(generated)
        try:
            blocks, report = inspect(generated)
        except (ValueError, TypeError) as error:
            stream.restart()
            # 실패한 초안은 진단에만 남기고, 실제 질문을 마지막 발언으로 유지합니다.
            repair = deepcopy(messages)
            repair.insert(len(repair) - 1, {"role": "system",
                "content": "같은 자료와 허용 화자 목록을 유지하고 언어·블록 형식만 한 번 고쳐 주세요. "
                    + str(error) + " 더 짧고 완결된 blocks JSON만 반환하세요. text는 한국어 본문입니다."})
            requested = repair
            generated = run_generation(requested, sample=False,
                on_text=None, cancel_event=cancel_event,
                response_format=response_format)
            generated["stage"] = "format_repair"
            generated["messages"] = deepcopy(requested)
            generated["provided_sources"] = deepcopy(provided_sources or [])
            stages.append(generated)
            try:
                blocks, report = inspect(generated)
            except (ValueError, TypeError) as error:
                failure = OutputFormatError("응답의 블록·화자·언어 형식을 확인할 수 없습니다.", generated["text"])
                failure.request_options = deepcopy(generated.get("request_options"))
                raise failure from error
        stream.finish(blocks)
        return blocks, generated, first, stages, report, requested
    except BaseException as error:
        error.messages = deepcopy(requested)
        error.context_messages = deepcopy(messages)
        error.reply_to = payload["player_input"]
        error.retrieval_query = (payload.get("input_trace") or {}).get("retrieval_query", payload["player_input"])
        error.stage_metrics = deepcopy(stages)
        error.backend = diagnostics_metadata()
        if stages:
            error.backend["options"] = deepcopy(stages[0].get("request_options"))
        error.versions = _version_metadata()
        if stages:
            error.raw_response = stages[-1].get("text", "")
            error.draft_response = stages[0].get("text", "")
        if stream is not None:
            try:
                stream.fail()
            except BaseException:
                # 닫힌 화면의 후속 알림 실패보다 원래 생성 오류·취소 기록을 우선합니다.
                pass
        raise


def _result(blocks, generated, first, stages, report, messages, payload, speaker, sources, requested):
    """화자별 본문과 기존 API용 문자열, 실제 생성·요청 진단을 함께 반환합니다."""
    seconds = round(sum(row["seconds"] for row in stages), 3)
    trace = generated.get("_output_review") or {}
    meaning_seconds = sum(row["seconds"] for row in stages if row.get("stage") == "semantic_review")
    repair_seconds = sum(row["seconds"] for row in stages if row.get("stage") in {"source_rebuild", "format_repair"})
    text = _blocks_text(blocks)
    result = {"text": text, "blocks": deepcopy(blocks),
        "speech": "\n".join(row["text"] for row in blocks if row["kind"] == "speech"),
        "narration": "\n".join(row["text"] for row in blocks if row["kind"] == "narration"),
        "payload": payload, "player_input": payload.get("player_input", ""),
        "suggestions": [], "response_format": "blocks", "unstructured_text": text,
        "model_generated": bool(stages), "engine_state_topics": [], "state_claim_topics": [],
        "response_origin": trace.get("response_origin", "model_grounded_generation"),
        "meaning_review": deepcopy(trace.get("meaning_review")),
        "raw_text": generated["text"], "draft_text": first["text"], "response_guard": report,
        "source_ids": [], "selected_source_ids": [], "sources": [], "provided_sources": sources,
        "warnings": [], "status": "format_valid", "seconds": seconds, "generation_seconds": seconds,
        "input_tokens": sum(row["input_tokens"] for row in stages),
        "output_tokens": sum(row["output_tokens"] for row in stages), "stage_metrics": stages,
        "messages": deepcopy(requested), "context_messages": deepcopy(messages),
        "request_trace": deepcopy(trace.get("request_trace", [])),
        "draft_messages": deepcopy(trace.get("draft_messages", [])),
        "reply_to": payload["player_input"],
        "retrieval_query": (payload.get("input_trace") or {}).get("retrieval_query", payload["player_input"]),
        "history_messages": max(0, len(messages) - 4),
        # 기존 review_*에는 실제 후속 요청 비용을 보존하고, 두 종류의 비용도 분리합니다.
        "review_enabled": any(row.get("stage") == "semantic_review" for row in stages),
        "review_seconds": meaning_seconds + repair_seconds,
        "meaning_review_seconds": meaning_seconds, "format_repair_seconds": repair_seconds,
        "review_messages": deepcopy(trace.get("review_messages", [])), "state_reasoning": False,
        "actual_model_calls": len(stages),
        "generation_token_limit": generated.get("generation_limit", MAX_NEW_TOKENS),
        "backend": {**diagnostics_metadata(), "options": deepcopy(first.get("request_options"))},
        "request_options": deepcopy(generated.get("request_options")),
        "versions": _version_metadata()}
    if trace.get("prepared_context") is not None:
        result["prepared_context"] = deepcopy(trace["prepared_context"])
    # 권한 경계와 근거 전달 여부를 기록합니다. 자유 본문의 의미 검증 결과가 아닙니다.
    current = payload.get("engine_result")
    coverage = None
    if (isinstance(current, dict) and current.get("confirmed") is True
            and (payload.get("dialogue_state") or {}).get("phase") == "action_result"):
        _, coverage = _revealed_public_references(payload, current)
    result["authority_boundary"] = {"state_authority": "engine", "dialogue_committed_effects": False,
        "public_request": _current_interaction_contract(payload)["public_request"],
        "knowledge_reference_coverage": coverage,
        "scope": "구조화 실행 권한과 자료 전달 범위이며, 대사 내용의 사실 정확도 판정은 아님"}
    return result


def _npc_conversation_schema(payload):
    """잡담 답변과 원문 확인 요청을 같은 첫 요청 안에서 구분합니다.

    이 선택은 사실 검증 결과가 아닙니다. 본문이 있으면 기존 화자·언어·블록 검사를
    그대로 적용하고, 원문 확인을 고르면 아직 사용자에게 보여 줄 본문은 없습니다.
    """
    from response_guard import response_blocks_schema
    body = deepcopy(response_blocks_schema(payload)["properties"]["blocks"])
    body["minItems"] = 0  # consult_reference의 빈 목록만 검사 함수에서 허용합니다.
    return {"type": "object", "properties": {
        "mode": {"type": "string", "enum": ["reply", "consult_reference"]}, "blocks": body},
        "required": ["mode", "blocks"], "additionalProperties": False}


def _inspect_npc_conversation(generated, payload, engine_result=None, *, envelope=True):
    """한 요청의 완결 여부와 형식만 확인합니다. 답의 의미 정확도 판정이 아닙니다."""
    from response_guard import parse_response_blocks, unique_json_keys
    limit = generated.get("generation_limit", MAX_NEW_TOKENS)
    if generated.get("done_reason") == "length" or generated.get("output_tokens", 0) >= limit:
        raise ValueError("응답이 생성 한도에 도달했습니다.")
    raw = generated["text"]
    if not envelope:
        blocks, report = parse_response_blocks(raw, payload, engine_result)
        return "reply", blocks, report
    value = json.loads(raw, object_pairs_hook=unique_json_keys)
    if not isinstance(value, dict) or set(value) != {"mode", "blocks"}:
        raise ValueError("대화 응답은 mode와 blocks로 구성해야 합니다.")
    mode = value["mode"]
    if mode == "consult_reference":
        if value["blocks"] != []:
            raise ValueError("원문 확인 요청에는 공개할 대사를 넣지 않습니다.")
        return mode, None, None
    if mode != "reply":
        raise ValueError("대화 응답 mode를 확인할 수 없습니다.")
    blocks, report = parse_response_blocks(_json_text({"blocks": value["blocks"]}), payload, engine_result)
    return mode, blocks, report


def _generate_npc_conversation(payload, history, speaker, *, engine_result=None, cancel_event=None):
    """현재 대화에 먼저 답하고, 필요할 때만 전달받은 원문을 확인합니다.

    새로운 검색·분류 모델이나 의미 검토 모델은 호출하지 않습니다. 사실 질문은
    첫 요청부터 팀 RAG 원문을 확인하고 잡담은 실제 대화로 먼저 답합니다. 형식 복구도 같은
    두 요청 예산에 포함하므로 세 번째 자동 요청은 없습니다. 중간 선택·실패 본문은
    화면에 보내지 않으며, 실제 요청별 입력·출처·비용은 진단 사본으로 보존합니다.
    """
    from response_guard import response_blocks_schema
    # 연결부에서 사실 질문이나 진행 설명으로 정한 입력은 원문 확인을 생략할 수 없습니다.
    # 분류 결과가 없는 외부 직접 호출도 원문을 먼저 제시하는 쪽으로 처리합니다.
    required = _reference_requested(payload)
    if required:
        messages, sources = build_npc_conversation_messages(payload, history, speaker,
            include_reference=True, engine_result=engine_result)
        blocks, generated, first, stages, report, requested = _generate_public(
            messages, payload, speaker, engine_result=engine_result, cancel_event=cancel_event,
            provided_sources=sources)
        stages[0]["stage"] = "reference_answer"
        generated["_output_review"] = {"response_origin": "model_required_reference",
            "draft_messages": deepcopy(messages), "review_messages": [],
            "request_trace": [{"stage": row["stage"], "messages": deepcopy(row.get("messages", [])),
                "provided_sources": deepcopy(row.get("provided_sources", [])), "completed": True}
                for row in stages]}
        result = _result(blocks, generated, first, stages, report, messages,
            payload, speaker, sources, requested)
        result["history_messages"] = max(0, len(messages) - 3)
        return result
    messages, sources = build_npc_conversation_messages(payload, history, speaker,
        include_reference=False, engine_result=engine_result)
    first_messages = deepcopy(messages)
    requested, actual_sources, stages, stream = messages, sources, [], None
    trace = {"response_origin": "model_conversation", "draft_messages": deepcopy(messages),
        "review_messages": [], "request_trace": [], "active_stage": None,
        "meaning_review": {"verdict": "not_performed", "issues": [], "model_assisted": False,
            "scope": "conversation_reference_selection_and_format_only", "draft_format_error": None}}

    def request(stage, sent, provided, schema, *, repair=False):
        nonlocal requested, actual_sources
        if len(trace["request_trace"]) >= 2:
            raise OutputFormatError("자동 대화 요청 한도에 도달했습니다.")
        requested, actual_sources = sent, provided
        trace["active_stage"] = stage
        row = {"stage": stage, "messages": deepcopy(sent), "provided_sources": deepcopy(provided),
               "completed": False}
        trace["request_trace"].append(row)
        options = {"on_text": None, "cancel_event": cancel_event, "response_format": schema}
        if repair:
            options["sample"] = False
        generated = run_generation(sent, **options)
        generated["stage"] = stage
        generated["messages"] = deepcopy(sent)
        generated["provided_sources"] = deepcopy(provided)
        stages.append(generated)
        row["completed"] = True
        trace["active_stage"] = None
        return generated

    try:
        stream = _ResponseStream(payload, engine_result)
        first = generated = request("conversation", messages, sources, _npc_conversation_schema(payload))
        try:
            mode, blocks, report = _inspect_npc_conversation(generated, payload, engine_result)
        except (ValueError, TypeError) as error:
            # 잘못된 초안이나 그 오류 문장을 사실로 다시 주입하지 않습니다. 기존 원문과
            # 최신 실제 입력으로 한 번만 다시 작성하고 남은 확인 요청은 허용하지 않습니다.
            trace["meaning_review"]["draft_format_error"] = str(error)
            messages, sources = build_npc_conversation_messages(payload, history, speaker,
                include_reference=True, engine_result=engine_result)
            repair = deepcopy(messages)
            repair.insert(len(repair) - 1, {"role": "system", "content":
                "현재 질문에 한국어 대사로 짧게 답하고, 허용된 화자의 완결된 blocks JSON만 반환하세요."})
            trace["response_origin"] = "model_conversation_format_repair"
            generated = request("format_repair", repair, sources, response_blocks_schema(payload), repair=True)
            try:
                _, blocks, report = _inspect_npc_conversation(generated, payload, engine_result, envelope=False)
            except (ValueError, TypeError) as error:
                raise OutputFormatError("응답의 블록·화자·언어 형식을 확인할 수 없습니다.", generated["text"]) from error
        else:
            if mode == "consult_reference":
                messages, sources = build_npc_conversation_messages(payload, history, speaker,
                    include_reference=True, engine_result=engine_result)
                trace["response_origin"] = "model_reference_answer"
                generated = request("reference_answer", messages, sources, response_blocks_schema(payload))
                try:
                    _, blocks, report = _inspect_npc_conversation(generated, payload, engine_result, envelope=False)
                except (ValueError, TypeError) as error:
                    raise OutputFormatError("원문 확인 뒤 응답의 블록·화자·언어 형식을 확인할 수 없습니다.", generated["text"]) from error
        generated["_output_review"] = trace
        stream.finish(blocks)
        result = _result(blocks, generated, first, stages, report, first_messages,
            payload, speaker, actual_sources, requested)
        # 이 경로는 공통·참고 system 2개와 현재 user 1개를 사용합니다.
        result["history_messages"] = max(0, len(first_messages) - 3)
        return result
    except BaseException as error:
        _attach_output_error(error, requested, first_messages, payload, stages, trace)
        error.request_trace = deepcopy(trace["request_trace"])
        error.provided_sources = deepcopy(actual_sources)
        error.response_origin = trace["response_origin"]
        if stream is not None:
            try:
                stream.fail()
            except BaseException:
                pass  # 후속 화면 알림 실패보다 원래 오류·취소를 보존합니다.
        raise


def _prepared_opening(payload, speaker, sources, *, engine_result=None, cancel_event=None, force_prepared=False):
    """첫 인사의 원본 연결을 확인하고 준비 문구 또는 안내 생성으로 분기합니다.

    카드 ID·공개 이름·원문 해시가 모두 일치해야 합니다. 현재 역할과 다음 장소의
    안내가 필요한 첫 만남은 검증 후 생성 경로에 맡깁니다. 준비 문구를 그대로
    표시한 경우에는 모델 요청·토큰·생성 시간이 0입니다.
    """
    from response_guard import parse_response_blocks
    task = _utterance_task(payload)
    if not (task["mode"] == "npc_dialogue" and task["phase"] == "first_contact"
            and task["input_kind"] == "action"):
        return None
    stream = None
    trace = {"response_origin": "prepared_opening", "draft_messages": [], "review_messages": [],
        "active_stage": "prepared_opening", "meaning_review": {"verdict": "not_performed", "issues": [],
            "scope": "prepared_opening_source_identity_and_format_only", "model_assisted": False,
            "draft_format_error": None}}
    try:
        _raise_if_cancelled(cancel_event)
        identifier, name = task["speaker"]["speaker_id"], task["speaker"]["name"]
        registry = globals().get("OPENING_LINES")
        prepared = registry.get(identifier) if isinstance(registry, dict) else None
        rows = [row for row in payload["retrieval"].get("scenario_results", [])
            if (row.get("card_id") or (row.get("metadata") or {}).get("card_id")) == identifier]
        texts = {row.get("text") for row in rows if isinstance(row.get("text"), str)}
        if not identifier or len(texts) != 1 or not isinstance(prepared, dict):
            raise OutputFormatError("현재 NPC의 원본 카드와 연결된 첫 인사 문구가 필요합니다.")
        source_hash = hashlib.sha256(next(iter(texts)).encode("utf-8")).hexdigest()
        if (prepared.get("name") != name or prepared.get("source_text_sha256") != source_hash
                or not isinstance(prepared.get("text"), str) or not prepared["text"].strip()):
            raise OutputFormatError("현재 NPC의 카드 원문이 준비된 첫 인사와 다릅니다. 첫 인사를 확인해 주세요.")
        # 안내가 필요한 첫 만남도 같은 원문 ID·해시를 확인한 후 생성 경로를 사용합니다.
        if _story_context(payload).get("first_contact_guidance") is True and not force_prepared:
            return None
        stream = _ResponseStream(payload, engine_result)
        raw = _json_text({"blocks": [{"kind": "speech", "speaker_id": identifier, "text": prepared["text"]}]})
        try:
            blocks, report = parse_response_blocks(raw, payload, engine_result)
        except (ValueError, TypeError) as error:
            raise OutputFormatError("준비된 첫 인사의 화자·본문 형식을 확인할 수 없습니다.") from error
        _raise_if_cancelled(cancel_event)
        stream.finish(blocks)
        trace["active_stage"] = None
        trace["prepared_context"] = {"speaker_id": identifier, "source_text_sha256": source_hash}
        empty_generation = {"text": "", "seconds": 0, "input_tokens": 0, "output_tokens": 0,
            "generation_limit": 0, "request_options": None, "_output_review": trace}
        return _result(blocks, empty_generation, empty_generation, [], report, [], payload,
            speaker, sources, [])
    except BaseException as error:
        _attach_output_error(error, [], [], payload, [], trace)
        if stream is not None:
            try:
                stream.fail()
            except BaseException:
                pass
        raise


def _prepared_confirmed_interaction(payload, speaker, sources, *, engine_result=None):
    """확정된 수락·전달은 역할과 영수증으로 구성하며 모델을 호출하지 않습니다.

    대사의 자연스러움은 일반 생성 경로가 담당합니다. 의뢰를 맡은 사람과 물품의
    주고받는 방향은 여기서 공통 문구로 보존합니다. 카드 문장을 복사하거나
    영수증의 제삼자 설명을 NPC 대사로 재작성하지 않으며 게임 효과도 실행하지 않습니다.
    """
    from response_guard import parse_response_blocks
    task = _utterance_task(payload)
    event = engine_result if isinstance(engine_result, dict) else {}
    executed = event.get("executed_action_ids") or []
    if (task["mode"] != "npc_dialogue" or task["phase"] != "action_result"
            or task["input_kind"] != "action" or event.get("confirmed") is not True or not executed):
        return None
    name = task["speaker"]["name"]
    identifier = task["speaker"]["speaker_id"]
    requests = [row for row in (payload.get("interaction_context") or {}).get("requests", [])
        if row.get("request_id") in executed and row.get("status") == "accepted"
        and (row.get("request_issuer") or {}).get("kind") == "npc"
        and (row.get("request_issuer") or {}).get("name") == name
        and (row.get("request_acceptor") or {}).get("kind") == "player"]
    public = _current_interaction_contract(payload)["public_request"]
    accepted_ids = list(dict.fromkeys(row["request_id"] for row in requests))
    if (isinstance(public, dict) and public.get("status") == "accepted"
            and public.get("source_choice_id") in executed
            and (public.get("issuer") or {}).get("name") == name):
        if public["source_choice_id"] not in accepted_ids:
            accepted_ids.append(public["source_choice_id"])
    accepted = payload.get("action_kind") == "accept" and bool(accepted_ids)

    transfers, seen = [], set()
    for row in event.get("item_handovers", []):
        if not isinstance(row, dict) or row.get("confirmed") is not True or row.get("action_id") not in executed:
            continue
        origin, destination = row.get("from") or {}, row.get("to") or {}
        outgoing = origin.get("kind") == "npc" and origin.get("name") == name and destination.get("kind") == "player"
        incoming = destination.get("kind") == "npc" and destination.get("name") == name and origin.get("kind") == "player"
        if not outgoing and not incoming:
            continue  # 다른 당사자의 영수증을 현재 NPC의 전달로 바꾸지 않습니다.
        quantity = row.get("quantity")
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("확정된 전달 영수증에는 양의 정수 수량이 필요합니다.")
        item_name = row.get("item_name")
        item_name = item_name.strip() if isinstance(item_name, str) and item_name.strip() else "물품"
        direction = "npc_to_player" if outgoing else "player_to_npc"
        key = (row["action_id"], row.get("item_id"), direction, quantity)
        if key in seen:
            continue
        seen.add(key)
        transfers.append({"action_id": row["action_id"], "item_id": row.get("item_id"),
                          "item_name": item_name, "quantity": quantity, "direction": direction})
    if not accepted and not transfers:
        return None

    register = OPENING_LINES.get(identifier, {}).get("speech_register", "polite")
    styles = {
        "polite": {"accept": "부탁을 맡아 주셔서 고마워요.", "give": "{item} {quantity}개를 건네드렸어요.",
                   "receive": "{item} {quantity}개를 잘 받았어요.", "money": "약속한 {amount}를 건네드렸어요.",
                   "preparation": "준비에 관해서는 {person}에게 물어보셔도 좋아요.",
                   "investigation": "괜찮으시면 {person}에게도 이야기를 물어보세요."},
        "hao": {"accept": "부탁을 맡아 주어 고맙소.", "give": "{item} {quantity}개를 건네드렸소.",
                "receive": "{item} {quantity}개를 잘 받았소.", "money": "약속한 {amount}를 건네드렸소.",
                "preparation": "준비에 관해서는 {person}에게 물어보아도 좋겠소.",
                "investigation": "괜찮다면 {person}에게도 이야기를 물어보시오."},
        "elder": {"accept": "부탁을 맡아 주어 고맙구려.", "give": "{item} {quantity}개를 건네드렸구려.",
                  "receive": "{item} {quantity}개를 잘 받았구려.", "money": "약속한 {amount}를 건네드렸구려.",
                  "preparation": "준비에 관해서는 {person}에게 물어보아도 좋겠구려.",
                  "investigation": "괜찮다면 {person}에게도 이야기를 물어보구려."}}
    style = styles.get(register, styles["polite"])
    lines = [style["accept"]] if accepted else []
    for row in transfers:
        key = "give" if row["direction"] == "npc_to_player" else "receive"
        lines.append(style[key].format(item=row["item_name"], quantity=row["quantity"]))

    # 공개한 조건과 실제 증가액이 같은 경우에만 금액을 확인합니다. 새 보상은 만들지 않습니다.
    money_cp = 0
    if accepted and isinstance(public, dict) and public.get("source_choice_id") in accepted_ids:
        grants = public.get("on_acceptance") or []
        expected = sum(row["amount_cp"] for row in grants if isinstance(row, dict)
                       and row.get("kind") == "money" and type(row.get("amount_cp")) is int and row["amount_cp"] > 0)
        change = (event.get("engine_delta") or {}).get("gold_cp_change")
        if expected > 0 and type(change) is int and change == expected:
            money_cp = change
            parts, remaining = [], money_cp
            for unit, size in (("gp", 100), ("sp", 10), ("cp", 1)):
                quantity, remaining = divmod(remaining, size)
                if quantity:
                    parts.append(str(quantity) + " " + unit)
            lines.append(style["money"].format(amount=" ".join(parts)))

    # 목적 문장을 기계적으로 NPC 대사로 바꾸지 않고, 허용된 다음 인물만 조건부로 권합니다.
    guidance = None
    for row in _story_context(payload).get("allowed_guidance", []):
        if not isinstance(row, dict) or row.get("purpose") not in {"preparation", "investigation"}:
            continue
        person = row.get("npc_name")
        if isinstance(person, str) and person.strip() and person != name:
            guidance = {"npc_name": person, "purpose": row["purpose"]}
            lines.append(style[row["purpose"]].format(person=person))
            break

    _raise_if_cancelled(None)
    raw = _json_text({"blocks": [{"kind": "speech", "speaker_id": identifier, "text": " ".join(lines)}]})
    blocks, report = parse_response_blocks(raw, payload, event)
    trace = {"response_origin": "confirmed_role_receipt", "draft_messages": [], "review_messages": [],
        "meaning_review": {"verdict": "not_performed", "issues": [], "model_assisted": False,
            "scope": "typed_roles_receipts_and_format_only"},
        "prepared_context": {"event_id": event.get("event_id"), "executed_action_ids": deepcopy(executed),
            "accepted_request_ids": accepted_ids if accepted else [], "accepted_by": "player" if accepted else None,
            "rendered_handovers": transfers, "rendered_money_cp": money_cp, "guidance": guidance,
            "speech_register": register, "receipt_explanation_rendered": False}, "request_trace": []}
    empty = {"text": "", "seconds": 0, "input_tokens": 0, "output_tokens": 0,
             "generation_limit": 0, "request_options": None, "_output_review": trace}
    stream = _ResponseStream(payload, event)
    try:
        _raise_if_cancelled(None)
        stream.finish(blocks)
    except BaseException:
        stream.fail()
        raise
    result = _result(blocks, empty, empty, [], report, [], payload, name, [], [])
    result["history_messages"] = 0
    return result


def _generate_main_result(messages, payload, speaker, sources, engine_result=None):
    """공통 생성·형식 검사와 원문이 확인된 첫 인사의 오류 복구를 연결합니다."""
    try:
        blocks, generated, first, stages, report, requested = _generate_public(messages, payload,
            speaker, engine_result=engine_result, provided_sources=sources)
    except OutputFormatError as error:
        # 안내 대사의 생성 형식이 끝내 잘못돼도 만남 자체를 막지 않습니다.
        # 복구 문구는 기존 첫 인사와 같은 원문 ID·해시 검사를 통과해야 합니다.
        fallback = _prepared_opening(payload, speaker, sources,
            engine_result=engine_result, force_prepared=True)
        if fallback is None:
            raise
        stages = deepcopy(getattr(error, "stage_metrics", []))
        seconds = round(sum(row.get("seconds", 0) for row in stages), 3)
        fallback.update(response_origin="prepared_opening_after_format_error", stage_metrics=stages,
            actual_model_calls=len(stages), seconds=seconds, generation_seconds=seconds,
            input_tokens=sum(row.get("input_tokens", 0) for row in stages),
            output_tokens=sum(row.get("output_tokens", 0) for row in stages),
            messages=deepcopy(getattr(error, "messages", messages)), context_messages=deepcopy(messages),
            raw_text=getattr(error, "raw_response", ""), draft_text=getattr(error, "draft_response", ""),
            warnings=["첫 안내 생성의 형식 오류로 원문이 확인된 소개 문구를 사용했습니다."])
        return fallback
    task = _utterance_task(payload)
    if task["phase"] == "first_contact" and _story_context(payload).get("first_contact_guidance") is True:
        stages[0]["stage"] = "guided_opening"
        origin = "model_guided_opening"
    elif task["phase"] == "action_result" and isinstance(engine_result, dict) and engine_result.get("confirmed"):
        origin = "model_current_confirmed_action"
    else:
        origin = "model_grounded_generation"
    generated["_output_review"] = {"response_origin": origin, "draft_messages": deepcopy(messages),
        "review_messages": [], "request_trace": [{"stage": row["stage"],
            "messages": deepcopy(row["messages"]), "provided_sources": deepcopy(row["provided_sources"]),
            "completed": True} for row in stages]}
    return _result(blocks, generated, first, stages, report, messages, payload,
        speaker, sources, requested)


def generation_for_display(result):
    """공개 요청과 생성 기록만 JSON으로 정리합니다. 사실 정확도 점수는 만들지 않습니다."""
    keys = ("messages", "context_messages", "draft_messages", "review_messages", "blocks", "raw_text", "draft_text",
            "response_origin", "meaning_review", "response_guard", "stage_metrics", "request_trace", "generation_seconds",
            "meaning_review_seconds", "format_repair_seconds", "input_tokens", "output_tokens")
    return _json_text({key: result.get(key) for key in keys})
