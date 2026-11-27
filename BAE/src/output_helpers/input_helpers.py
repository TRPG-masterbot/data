# 출력 실행 보조 함수: 노트북과 같은 전역 공간에서 원본 본문을 실행합니다.
from diagnostics_live import current_stage

def diagnostics_metadata():
    """복사·비교 기록에 필요한 실행 정보만 반환합니다. 주소·환경 변수는 포함하지 않습니다."""
    backend = output_model
    return {"backend": "Ollama", "output_version": OUTPUT_VERSION, "model": getattr(backend, "model", MODEL_NAME),
        "model_digest": getattr(backend, "digest", None),
        "model_details": deepcopy(getattr(backend, "details", {})),
        "quantization": (getattr(backend, "details", {}) or {}).get("quantization_level"),
        "num_ctx": getattr(backend, "num_ctx", OLLAMA_NUM_CTX),
        "runtime_info": deepcopy(_OUTPUT_RUNTIME_INFO),
        "source_sha256": _SOURCE_SHA256,
        "system_prompt_sha256": hashlib.sha256(OUTPUT_SYSTEM.encode("utf-8")).hexdigest(),
        "conversation_prompt_sha256": hashlib.sha256((NPC_CONVERSATION_SYSTEM + NPC_CONVERSATION_FIRST_PASS + NPC_CONVERSATION_REFERENCE_PASS).encode("utf-8")).hexdigest(),
        "opening_lines_sha256": hashlib.sha256(_json_text(OPENING_LINES).encode("utf-8")).hexdigest(),
        "output_policy_sha256": hashlib.sha256(_json_text(OUTPUT_POLICY).encode("utf-8")).hexdigest()}


def _version_metadata():
    return {"output_code_sha256": _SOURCE_SHA256, "output_version": OUTPUT_VERSION,
            "system_prompt_sha256": hashlib.sha256(OUTPUT_SYSTEM.encode("utf-8")).hexdigest(),
            "conversation_prompt_sha256": hashlib.sha256((NPC_CONVERSATION_SYSTEM + NPC_CONVERSATION_FIRST_PASS + NPC_CONVERSATION_REFERENCE_PASS).encode("utf-8")).hexdigest(),
            "opening_lines_sha256": hashlib.sha256(_json_text(OPENING_LINES).encode("utf-8")).hexdigest(),
            "output_policy_sha256": hashlib.sha256(_json_text(OUTPUT_POLICY).encode("utf-8")).hexdigest()}


def canonical_npc_name(value):
    return value.strip() if isinstance(value, str) else ""


def read_handoff(value):
    """원본 I3 JSON과 기존 retrieval 형식을 받으며 필드 이름만 변환합니다."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("전달 자료는 JSON 객체 또는 dict여야 합니다.")
    payload = deepcopy(value)
    if not isinstance(payload.get("player_input"), str) or not payload["player_input"].strip():
        raise ValueError("player_input에는 플레이어 원문이 필요합니다.")
    if not isinstance(payload.get("game_state"), dict):
        raise ValueError("game_state에는 현재 상태 dict가 필요합니다.")
    if "retrieval" not in payload:
        context = payload.get("retrieved_context")
        if not isinstance(context, dict):
            raise ValueError("원본 retrieved_context 또는 retrieval이 필요합니다.")
        retrieval = {"rule_results": [], "scenario_results": [], "memory_results": []}
        for original, target in (("rules", "rule_results"), ("scenarios", "scenario_results")):
            records = context.get(original, [])
            if not isinstance(records, list):
                raise ValueError("검색 결과는 list여야 합니다.")
            for row in records:
                if not isinstance(row, dict):
                    raise ValueError("검색 결과 항목은 dict여야 합니다.")
                meta = {key: deepcopy(item) for key, item in row.items()
                        if key not in {"source", "text", "score"}}
                retrieval[target].append({"source": row.get("source", "rule" if original == "rules" else "scenario"),
                    "text": row.get("text"), "score": row.get("score"), "metadata": meta})
        payload["retrieval"] = retrieval
        payload["upstream_handoff"] = deepcopy(value)
    if not isinstance(payload["retrieval"], dict):
        raise ValueError("retrieval은 dict여야 합니다.")
    for key in ("rule_results", "scenario_results"):
        rows = payload["retrieval"].setdefault(key, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("text"), str) for row in rows):
            raise ValueError("검색 결과에는 문자열 text가 필요합니다: " + key)
        if any(not isinstance(row.get("metadata", {}), dict) for row in rows):
            raise ValueError("검색 결과 metadata는 dict여야 합니다.")
    payload["retrieval"].setdefault("memory_results", [])
    payload.setdefault("schema_version", "1.0")
    return payload


def validate_engine_result(result):
    if not isinstance(result, dict) or result.get("confirmed") is not True:
        raise ValueError("엔진이 확정한 행동 결과만 전달하세요.")
    return deepcopy(result)


def normalize_history(history):
    """실제 사용자·AI 발언만 복사합니다. 역할이 없는 임의 자료는 대화로 취급하지 않습니다."""
    return [deepcopy(row) for row in history or [] if isinstance(row, dict)
            and row.get("role") in {"user", "assistant"} and isinstance(row.get("content"), str)]
