"""공개 대화와 측정된 턴 진단을 사람용 보고서·단일 JSON으로 복사합니다.

저장된 입력을 변경하지 않습니다. 환경·인증 정보, 내부 상태 사전, GM 자료나
임의의 중첩 객체는 내보내지 않고 정해진 공개 필드만 사용합니다.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import re

from diagnostics_live import public_diagnostics


SCHEMA = "output_diagnostics.v1"
COPY_SCHEMA = "output_diagnostics.v2"
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
        # 본문 거절의 실제 이유를 일반 오류 이름과 함께 남깁니다.
        for key in ("validation", "guard_report", "validation_attempts", "cause", "attempts"):
            if isinstance(value.get(key), (dict, list, str)):
                result[key] = public_diagnostics(value[key])
        return result
    return None


def _guard(value):
    if not isinstance(value, dict):
        return None
    result = _scalars(value, ("status", "scope", "valid", "repaired", "reason"))
    for key in ("issues", "hard_issues"):
        rows = value.get(key)
        result[key] = [public_diagnostics(row) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else None
    for key in ("errors", "warnings"):
        result[key] = [public_diagnostics(row) if isinstance(row, dict) else row
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
        if isinstance(row.get("validation"), dict):
            copied["validation"] = public_diagnostics(row["validation"])
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
        "request_id": _string(record.get("request_id")),
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
    # GPU와 캐시 준비 시간은 모델 이름·해시만으로는 알 수 없는 지연 단서입니다.
    for key in ("runtime_info", "model_details", "runtime_package"):
        result[key] = public_diagnostics(value[key]) if isinstance(value.get(key), dict) else None
    for key in ("gpu", "num_ctx", "backend", "device", "load_seconds", "prompt_seconds", "decode_seconds"):
        if key in value:
            result[key] = public_diagnostics(value[key])
    return public_diagnostics(result)


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


def build_plain_conversation(messages):
    """진단 정보 없이 공개 대화 전체만 복사합니다."""
    lines = []
    for message in _conversation(messages):
        name = "플레이어" if message["role"] == "user" else message.get("speaker") or "게임 마스터"
        lines.append(_inline(name) + "\n" + message["content"])
    return "\n\n".join(lines)


def build_diagnostics_document(messages, turn_records, session_meta, current_state, live_diagnostics=None):
    """기존 v1 공개 필드만 구성합니다. 복사 형식이 바뀌어도 원문 계약은 유지합니다."""
    records = turn_records if isinstance(turn_records, list) else []
    document = {"schema": SCHEMA, "session": _meta(session_meta),
        "conversation": _conversation(messages), "turns": [_turn(row) for row in records if isinstance(row, dict)],
        "current_state": _state(current_state)}
    live = _dict(live_diagnostics)
    if live_diagnostics is not None:
        document["live_requests"] = public_diagnostics(_rows(live.get("live_requests")))
        document["diagnostic_snapshot"] = public_diagnostics({key: live.get(key) for key in
            ("cache_available", "captured_at", "cache_age_seconds", "snapshot_at", "dropped_count", "history_count")})
    return document


def _canonical(value):
    """무손실 복원 확인에만 사용하는 일정한 JSON 표현입니다."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True, allow_nan=False)


def compact_diagnostics_document(document):
    """반복되는 긴 공개 문자열을 한 번 저장합니다. 내용이나 필드는 생략하지 않습니다."""
    counts, singleton_keys = Counter(), set()

    def count(value):
        if isinstance(value, str):
            if len(value) >= 96:
                counts[value] += 1
        elif isinstance(value, list):
            for item in value:
                count(item)
        elif isinstance(value, dict):
            if len(value) == 1:
                singleton_keys.update(value)
            for item in value.values():
                count(item)

    count(document)
    # 실제 입력이 우연히 참조와 같은 모양이어도 원문 객체로 정확히 복원합니다.
    reference_key = "$text"
    while reference_key in singleton_keys:
        reference_key += "_"
    shared = {}
    identifiers = {text: hashlib.sha256(text.encode("utf-8")).hexdigest()
                   for text, times in counts.items() if times > 1}

    def encode(value):
        if isinstance(value, str) and value in identifiers:
            identifier = identifiers[value]
            shared[identifier] = value
            return {reference_key: identifier}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        return value

    encoded = encode(document)
    return {"schema": COPY_SCHEMA, "expanded_schema": SCHEMA,
            "encoding": {"kind": "shared_text.v1", "reference_key": reference_key,
                "restore": "document 안에서 reference_key 하나만 가진 객체를 shared_text의 해당 값으로 재귀 치환합니다. 문자열의 내용은 다시 해석하지 않습니다.",
                "python": "from diagnostic_export import expand_diagnostics_document; expanded = expand_diagnostics_document(data)"},
            "document_sha256": hashlib.sha256(_canonical(document).encode("utf-8")).hexdigest(),
            "shared_text": shared, "document": encoded}


def expand_diagnostics_document(value):
    """예전 v1 문서와 새 v2 복사를 모두 읽고 기존 공개 필드를 복원합니다."""
    if not isinstance(value, dict):
        raise ValueError("진단 JSON은 객체여야 합니다.")
    if value.get("schema") == SCHEMA:
        return deepcopy(value)
    encoding = _dict(value.get("encoding"))
    if (value.get("schema") != COPY_SCHEMA or value.get("expanded_schema") != SCHEMA
            or encoding.get("kind") != "shared_text.v1"):
        raise ValueError("지원하지 않는 진단 복사 형식입니다.")
    reference_key = encoding.get("reference_key")
    shared = value.get("shared_text")
    if not isinstance(reference_key, str) or not isinstance(shared, dict):
        raise ValueError("진단 문자열 참조 정보가 없습니다.")
    for identifier, text in shared.items():
        if (not isinstance(identifier, str) or not isinstance(text, str)
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != identifier):
            raise ValueError("진단 문자열의 내용 확인값이 일치하지 않습니다.")

    def decode(item):
        if isinstance(item, dict):
            if set(item) == {reference_key}:
                identifier = item[reference_key]
                if not isinstance(identifier, str) or identifier not in shared:
                    raise ValueError("진단 문자열 참조가 끊어졌습니다.")
                return shared[identifier]
            return {key: decode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [decode(child) for child in item]
        return item

    restored = decode(value.get("document"))
    digest = hashlib.sha256(_canonical(restored).encode("utf-8")).hexdigest()
    if (not isinstance(restored, dict) or restored.get("schema") != SCHEMA
            or digest != value.get("document_sha256")):
        raise ValueError("진단 문서를 완전히 복원하지 못했습니다.")
    return restored


def build_copy_report(messages, turn_records, session_meta, current_state, live_diagnostics=None):
    """사람용 대화·측정 요약과 모든 근거를 한 번에 복사합니다."""
    document = build_diagnostics_document(messages, turn_records, session_meta, current_state, live_diagnostics)
    session = document["session"]
    lines = ["TRPG 전체 대화·테스트 기록", "형식: " + COPY_SCHEMA,
        "미측정 값은 null입니다. 형식 검사 결과는 사실 정확도 평가가 아닙니다.",
        "대화와 요약은 아래에, 모든 실제 요청·응답·근거·진행 기록은 마지막 JSON에 보존합니다.",
        "같은 긴 원문은 shared_text에 한 번 저장하고 참조합니다. 누락이나 잘라내기가 아닙니다.",
        "", "[실행 정보]",
        "출력 " + _value(session.get("output_version")) + " / 모델 " + _value(session.get("model"))
        + " / 양자화 " + _value(session.get("quantization")),
        "GPU " + _value(session.get("gpu")) + " / 모델 해시 " + _value(session.get("model_digest")),
        "실제 옵션", _text_block(session.get("options"))]
    if live_diagnostics is not None:
        requests = document["live_requests"]
        active = [row for row in requests if isinstance(row, dict) and row.get("pending")]
        lines.extend(["", "[현재 요청·진행 단계]", "진행 중 요청: " + str(len(active)) + "건",
            "대화·게임 상태는 마지막 확정 사본입니다. 진행 중 작업은 요청 기록과 함께 확인합니다.",
            "확정 사본 정보", _text_block(document["diagnostic_snapshot"])])
        for request in requests:
            if not isinstance(request, dict):
                continue
            lines.append("요청 " + _inline(request.get("request_id")) + " · " + _inline(request.get("kind"))
                + " · " + _inline(request.get("status")) + " · 단계 " + _inline(request.get("stage"))
                + " · 경과 " + _value(request.get("elapsed_seconds"), "초"))
            if request.get("error"):
                lines.extend(["오류", _text_block(request["error"])])
    lines.extend(["", "[전체 공개 대화]"])
    for index, message in enumerate(document["conversation"], 1):
        name = "플레이어" if message["role"] == "user" else message.get("speaker") or "게임 마스터"
        lines.extend([f"{index}. {_inline(name)}", _text_block(message["content"])])
    lines.extend(["", "[턴별 진단 요약]"])
    for index, turn in enumerate(document["turns"], 1):
        timing, tokens = turn["timing"], turn["tokens"]
        lines.extend([f"{index}. 턴 {_inline(turn['turn_id'])} · {_inline(turn['kind'])} · {_inline(turn['status'])}",
            "입력: " + _value(turn["input"].get("text")),
            "시간: 분류 " + _value(timing["classification_seconds"], "초")
            + " / 검색 " + _value(timing["search_seconds"], "초")
            + " / 본문 " + _value(timing["modelgen_seconds"], "초")
            + " / 교정 " + _value(timing["repair_seconds"], "초")
            + " / 추천 " + _value(timing["followup_seconds"], "초")
            + " / 전체 " + _value(timing["total_seconds"], "초"),
            "토큰: 입력 " + _value(tokens["input"]) + " / 출력 " + _value(tokens["output"])
            + " · 공개 근거 " + _value(turn["public_source_count"], "개")
            + " · 본문 요청 " + str(len(turn.get("stage_metrics") or [])) + "회"])
        if turn["output"].get("error"):
            lines.extend(["오류", _text_block(turn["output"]["error"])])
        guard = turn["output"].get("guard")
        if guard and (guard.get("hard_issues") or guard.get("errors") or guard.get("warnings")):
            lines.extend(["검사", _text_block(guard)])
    lines.extend(["", "[현재 공개 상태]", _text_block(document["current_state"]),
        "", "[기계 판독용 JSON]", "```json",
        json.dumps(compact_diagnostics_document(document), ensure_ascii=False,
                   separators=(",", ":"), allow_nan=False), "```"])
    return "\n".join(lines)
