"""공개 대화와 측정된 턴 진단을 사람용 보고서·단일 JSON으로 복사합니다.

저장된 입력을 변경하지 않습니다. 환경·인증 정보, 내부 상태 사전, GM 자료나
임의의 중첩 객체는 내보내지 않고 정해진 공개 필드만 사용합니다.
"""
import json
import math
import re


SCHEMA = "output_diagnostics.v1"
_TIMINGS = ("search_seconds", "handoff_seconds", "input_seconds", "input_pipeline_seconds", "classification_seconds",
            "scope_seconds", "mapper_seconds", "modelgen_seconds", "generation_seconds",
            "repair_seconds", "meaning_review_seconds", "followup_seconds", "followup_wall_seconds", "total_seconds")
_OPTIONS = ("temperature", "top_p", "top_k", "seed", "num_ctx", "num_predict", "max_new_tokens",
            "repeat_penalty", "repetition_penalty", "presence_penalty", "frequency_penalty",
            "sample", "do_sample", "think", "thinking", "structured_response")
_PATH = re.compile(r"(?i)(?:[a-z]:[\\/][^\s\"'<>]+|/(?:content|home|users|tmp|var)/[^\s\"'<>]+)")


def _dict(value):
    return value if isinstance(value, dict) else {}


def _string(value):
    return value if isinstance(value, str) else None


def _number(value, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return value if not integer or isinstance(value, int) else None


def _scalars(value, keys):
    result = {}
    for key in keys:
        item = _dict(value).get(key)
        if item is None or isinstance(item, (str, bool)):
            result[key] = item
        elif isinstance(item, (int, float)) and math.isfinite(item):
            result[key] = item
        else:
            result[key] = None
    return result


def _numeric_map(value):
    """수량·사용 자원 등 공개 숫자 사전만 복사합니다."""
    if not isinstance(value, dict):
        return None
    result = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            result[key] = item
        elif isinstance(item, bool):
            result[key] = item
        elif isinstance(item, dict):
            nested = _numeric_map(item)
            if nested:
                result[key] = nested
    return result


def _strings(value):
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else None


def _rows(value):
    return value if isinstance(value, list) else []


def _blocks(value):
    if not isinstance(value, list):
        return None
    return [_scalars(row, ("kind", "speaker_id", "speaker", "text")) for row in value
            if isinstance(row, dict) and isinstance(row.get("text"), str)]


def _inventory(value):
    if isinstance(value, dict):
        return _numeric_map(value)
    if isinstance(value, list):
        return [_scalars(row, ("item_id", "item_name", "name", "quantity")) for row in value if isinstance(row, dict)]
    return None


def _state(value):
    if not isinstance(value, dict):
        return None
    result = _scalars(value, ("scene", "cls", "current_room", "room", "room_name", "minutes", "gold_cp", "money"))
    result["inventory"] = _inventory(value.get("inventory"))
    hero = value.get("hero")
    if isinstance(hero, dict):
        result["hero"] = _scalars(hero, ("id", "name", "hp", "max_hp", "ac", "level", "proficiency_bonus"))
        result["hero"].update(conditions=_strings(hero.get("conditions")),
                              abilities=_numeric_map(hero.get("abilities")),
                              resources=_numeric_map(hero.get("resources")),
                              hit_dice=_numeric_map(hero.get("hit_dice")))
    else:
        result["hero"] = None
    result["combat"] = _scalars(value["combat"], ("round",)) if isinstance(value.get("combat"), dict) else None
    if result["combat"] is not None:
        result["combat"]["turn"] = _numeric_map(value["combat"].get("turn"))
    return result


def _error(value):
    def message(text):
        return _PATH.sub("[local_path]", text) if isinstance(text, str) else None
    if isinstance(value, str):
        return {"type": None, "code": None, "stage": None, "message": message(value)}
    if isinstance(value, dict):
        result = _scalars(value, ("type", "code", "stage", "message"))
        result["message"] = message(result["message"])
        return result
    return None


def _guard(value):
    if not isinstance(value, dict):
        return None
    result = _scalars(value, ("status", "scope", "valid", "repaired", "reason"))
    for key in ("issues", "hard_issues"):
        rows = value.get(key)
        result[key] = [_scalars(row, ("code", "message", "stage")) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else None
    for key in ("errors", "warnings"):
        result[key] = [_scalars(row, ("code", "message", "stage")) if isinstance(row, dict) else row
                       for row in value[key] if isinstance(row, (dict, str))] if isinstance(value.get(key), list) else None
    return result


def _sources(record):
    rows = record.get("sources_public")
    if rows is None:
        context = _dict(record.get("provided_context"))
        retrieval = _dict(context.get("retrieval"))
        original = _dict(context.get("retrieved_context"))
        if retrieval:
            rows = [*_rows(retrieval.get("rule_results")), *_rows(retrieval.get("scenario_results"))]
        elif original:
            rows = [*_rows(original.get("rules")), *_rows(original.get("scenarios"))]
    if not isinstance(rows, list):
        return None
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        metadata = _dict(row.get("metadata"))
        keys = ("source", "card_id", "title", "page", "score", "text")
        result.append(_scalars({key: row.get(key, metadata.get(key)) for key in keys}, keys))
    return result


def _outcome(value):
    return _scalars(value, ("status", "success", "result", "total", "roll", "dc", "modifier", "hit",
                            "critical", "damage", "damage_type", "healed", "defeated", "action_id")) if isinstance(value, dict) else _string(value)


def _delta(value):
    if not isinstance(value, dict):
        return None
    result = _scalars(value, ("action_id", "executed_action_id", "actor_id", "room_before", "room_after",
        "gold_cp_before", "gold_cp_after", "gold_cp_delta", "gold_cp_change", "hp_before", "hp_after", "hp_delta", "hp_change", "success"))
    result["executed_action_ids"] = _strings(value.get("executed_action_ids"))
    for key in ("inventory_before", "inventory_after", "inventory_delta"):
        result[key] = _inventory(value.get(key))
    result["outcome"] = _outcome(value.get("outcome"))
    result["inventory_changes"] = [_scalars(row, ("item_id", "item_name", "quantity_change"))
                                   for row in _rows(value.get("inventory_changes")) if isinstance(row, dict)] if isinstance(value.get("inventory_changes"), list) else None
    return result


def _stages(value):
    if not isinstance(value, list):
        return None
    rows = []
    for row in value:
        if not isinstance(row, dict):
            continue
        copied = _scalars(row, ("stage", "name", "seconds", "load_seconds", "prompt_seconds", "decode_seconds",
                               "input_tokens", "output_tokens", "generation_limit", "done_reason", "model", "model_digest"))
        copied["request_options"] = _scalars(row["request_options"], _OPTIONS) if isinstance(row.get("request_options"), dict) else None
        # 단계별 실제 요청·답변·참고자료를 보존합니다. 환경이나 내부 상태는 내보내지 않습니다.
        copied["text"] = _string(row.get("text"))
        copied["messages"] = _request_messages(row.get("messages"))
        copied["provided_sources"] = _sources({"sources_public": row.get("provided_sources")})
        rows.append(copied)
    return rows


def _rolls(value):
    if not isinstance(value, list):
        return None
    result = []
    for row in value:
        if not isinstance(row, dict):
            continue
        copied = _scalars(row, ("kind", "actor", "char", "side", "roll", "mod", "modifier", "bonus",
            "dc", "ac", "total", "success", "hit", "crit", "critical", "ability", "skill", "tool",
            "method_index", "mode", "developer_override", "forced_outcome", "developer_forced_outcome", "natural_success"))
        copied["rolls"] = [_number(number, integer=True) for number in row["rolls"]] if isinstance(row.get("rolls"), list) else None
        result.append(copied)
    return result


def _request_messages(value):
    """출력 모듈이 실제 전송한 공개 메시지의 역할과 본문만 보존합니다."""
    if not isinstance(value, list):
        return None
    return [{"role": row["role"], "content": row["content"]} for row in value
            if isinstance(row, dict) and row.get("role") in {"system", "developer", "user", "assistant"}
            and isinstance(row.get("content"), str)]


def _prompt_trace(value):
    if not isinstance(value, dict):
        return None
    result = _scalars(value, ("reply_to", "retrieval_query", "phase", "model"))
    result["messages"] = _request_messages(value.get("messages"))
    result["context_messages"] = _request_messages(value.get("context_messages"))
    result["draft_messages"] = _request_messages(value.get("draft_messages"))
    result["review_messages"] = _request_messages(value.get("review_messages"))
    actor = value.get("active_npc")
    result["active_npc"] = (_scalars(actor, ("name", "card_id", "speaker_id"))
                            if isinstance(actor, dict) else _string(actor))
    result["options"] = _scalars(value["options"], _OPTIONS) if isinstance(value.get("options"), dict) else None
    result["history_roles"] = _strings(value.get("history_roles"))
    return result


def _turn(record):
    record = _dict(record)
    input_data, output, engine = (_dict(record.get(key)) for key in ("input", "output", "engine"))
    sources = _sources(record)
    timing = _dict(record.get("timing"))
    tokens = _dict(record.get("tokens"))
    return {"turn_id": record.get("turn_id") if isinstance(record.get("turn_id"), (str, int)) else None,
        "kind": _string(record.get("kind")), "status": _string(record.get("status")),
        "input": _scalars(input_data, ("text", "request", "kind", "npc_name", "original_player_input",
                                      "retrieval_query", "upstream_player_input", "retry_of_turn_id")),
        "prompt_trace": _prompt_trace(record.get("prompt_trace")),
        "timing": {key: _number(timing.get(key)) for key in _TIMINGS},
        "tokens": {key: _number(tokens.get(key), integer=True) for key in ("input", "output")},
        "sources_public": sources, "public_source_count": len(sources) if sources is not None else None,
        "sources_count": {key: _number(_dict(record.get("sources_count")).get(key), integer=True)
                          for key in ("rules", "scenarios")},
        "versions": _scalars(record.get("versions"), ("output_code_sha256", "system_prompt_sha256", "output_version",
                                                     "opening_lines_sha256", "output_policy_sha256", "conversation_prompt_sha256")),
        "output": {"raw": _string(output.get("raw")), "draft": _string(output.get("draft")),
            "shown": _string(output.get("shown")), "blocks": _blocks(output.get("blocks")),
            "guard": _guard(output.get("guard")), "response_origin": _string(output.get("response_origin")),
            "meaning_review": _meaning_review(output.get("meaning_review")), "error": _error(output.get("error"))},
        "stage_metrics": _stages(record.get("stage_metrics", output.get("stage_metrics"))),
        "recommendations": _recommendations(record.get("recommendations")),
        "engine": {"event_id": _string(engine.get("event_id")),
            "executed_action_ids": _strings(engine.get("executed_action_ids")),
            "rolls": _rolls(engine.get("rolls")),
            "before": _state(engine.get("before", engine.get("state_before"))),
            "after": _state(engine.get("after", engine.get("state_after"))),
            "delta": _delta(engine.get("delta")), "outcome": _outcome(engine.get("outcome"))}}


def _meaning_review(value):
    """모델의 근거 대조 기록이며 정답·정확도 판정으로 바꾸지 않습니다."""
    if not isinstance(value, dict):
        return None
    result = _scalars(value, ("verdict", "scope", "draft_format_error", "model_assisted"))
    result["issues"] = [_scalars(row, ("kind", "source_id", "note"))
                        for row in _rows(value.get("issues")) if isinstance(row, dict)]
    return result


def _recommendations(value):
    if not isinstance(value, dict):
        return None
    result = _scalars(value, ("called", "reason", "text", "input_tokens", "output_tokens"))
    result["suggestions"] = _strings(value.get("suggestions"))
    result["recommendation_options"] = [_scalars(row, ("kind", "text", "action_id"))
        for row in _rows(value.get("recommendation_options")) if isinstance(row, dict)] if isinstance(value.get("recommendation_options"), list) else None
    result["error"] = _error(value.get("error"))
    return result


def _meta(value):
    value = _dict(value)
    result = _scalars(value, ("schema", "model", "model_digest", "quantization", "version_hash", "backend_version", "output_version"))
    if result["model"] is None:
        result["model"] = _string(value.get("model_name"))
    if result["model_digest"] is None:
        result["model_digest"] = _string(value.get("digest"))
    for key in ("model", "backend_version", "quantization"):
        if isinstance(result[key], str):
            result[key] = _PATH.sub("[local_path]", result[key])
    if not isinstance(result["version_hash"], str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", result["version_hash"]):
        result["version_hash"] = None
    code = value.get("code_sha256")
    if isinstance(code, str):
        result["code_sha256"] = code if re.fullmatch(r"[0-9a-fA-F]{40,64}", code) else None
    elif isinstance(code, dict):
        result["code_sha256"] = {name: digest for name, digest in code.items()
            if name in {"output_code.py", "ollama_backend.py", "response_guard.py", "gameplay.py", "game_ui.py", "portrait_view.py"}
            and isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{40,64}", digest)}
    else:
        result["code_sha256"] = None
    result["options"] = _scalars(value["options"], _OPTIONS) if isinstance(value.get("options"), dict) else None
    return result


def _conversation(messages):
    result = []
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "system"}:
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(row["text"] for row in content if isinstance(row, dict)
                                and row.get("type") == "text" and isinstance(row.get("text"), str))
        if not isinstance(content, str):
            continue
        row = _scalars(message, ("role", "speaker", "speaker_id", "kind"))
        row.update(content=content, blocks=_blocks(message.get("blocks")))
        result.append(row)
    return result


def _text_block(value):
    if value is None:
        return "    미제공"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    return "\n".join("    " + line for line in text.splitlines()) if text else "    (빈 문자열)"


def _value(value, suffix=""):
    return "미측정" if value is None else str(value) + suffix


def _inline(value):
    return " ".join(str(value).splitlines())


def build_copy_report(messages, turn_records, session_meta, current_state):
    """공개 대화 전체와 턴 진단을 담은 단일 복사 문자열을 반환합니다."""
    records = turn_records if isinstance(turn_records, list) else []
    document = {"schema": SCHEMA, "session": _meta(session_meta),
        "conversation": _conversation(messages), "turns": [_turn(row) for row in records if isinstance(row, dict)],
        "current_state": _state(current_state)}
    lines = ["TRPG 전체 대화·테스트 기록", "형식: " + SCHEMA,
             "미측정 값은 null입니다. 형식 검사 결과는 사실 정확도 평가가 아닙니다.",
             "", "[실행 정보]", _text_block(document["session"]), "", "[전체 공개 대화]"]
    for index, message in enumerate(document["conversation"], 1):
        name = "플레이어" if message["role"] == "user" else message.get("speaker") or "게임 마스터"
        lines.extend([f"{index}. {_inline(name)}", _text_block(message["content"])])
    lines.extend(["", "[턴별 진단]"])
    for index, turn in enumerate(document["turns"], 1):
        timing, tokens = turn["timing"], turn["tokens"]
        lines.extend([f"{index}. 턴 {_inline(turn['turn_id'])} · {_inline(turn['kind'])} · {_inline(turn['status'])}",
            "입력", _text_block(turn["input"]),
            "시간: 검색 " + _value(timing["search_seconds"], "초")
            + " / 전달 JSON " + _value(timing["handoff_seconds"], "초")
            + " / 생성 " + _value(timing["generation_seconds"], "초")
            + " / 교정 " + _value(timing["repair_seconds"], "초")
            + " / 전체 " + _value(timing["total_seconds"], "초"),
            "토큰: 입력 " + _value(tokens["input"]) + " / 출력 " + _value(tokens["output"]),
            "전체 측정 항목", _text_block(timing),
            "출력 버전", _text_block(turn["versions"]),
            "실제 공개 모델 입력 · 질문/검색 질의/화자/단계/옵션", _text_block(turn["prompt_trace"]),
            "표시된 응답", _text_block(turn["output"]["shown"]),
            "화자별 블록", _text_block(turn["output"]["blocks"]),
            "모델 초안", _text_block(turn["output"]["draft"]),
            "모델 최종 원문", _text_block(turn["output"]["raw"]),
            "공개 검색 자료: " + _value(turn["public_source_count"], "개"), _text_block(turn["sources_public"]),
            "규칙·시나리오 검색 개수", _text_block(turn["sources_count"]),
            "언어·형식 검사", _text_block(turn["output"]["guard"]),
            "오류", _text_block(turn["output"]["error"]),
            "생성 단계 측정", _text_block(turn["stage_metrics"]),
            "추천 생성", _text_block(turn["recommendations"]),
            "실제 엔진 결과", _text_block(turn["engine"]), ""])
    lines.extend(["[현재 공개 상태]", _text_block(document["current_state"]), "", "[기계 판독용 JSON]",
                  "```json", json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), "```"])
    return "\n".join(lines)
