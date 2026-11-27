# 출력 실행 보조 함수: 노트북과 같은 전역 공간에서 원본 본문을 실행합니다.

"""모든 응답에 같은 출력 형식을 적용합니다.

시나리오 사실이나 특정 인물의 관계를 문장 규칙으로 판정하지 않습니다.
본문 유무·한국어 출력·내부 대화 형식과 이번 턴의 화자·표현 종류를 확인합니다.
인물 관계나 사건 내용의 사실 정확도를 이 검사만으로 보장하지 않습니다.
"""
from html import unescape
import json
import re
_HAN = re.compile(r"[\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003347f]")
_PROTOCOL = re.compile(r"<\|(?:im_start|im_end|endoftext|start_header_id|end_header_id)\|>|</?think>")
_HANGUL = re.compile(r"[\uac00-\ud7a3]")


class ResponseValidationError(ValueError):
    """잘못된 블록의 위치와 이유를 진단에 남기는 형식 오류입니다."""
    def __init__(self, report):
        self.guard_report = report
        super().__init__(report["hard_issues"][0]["message"])


def _rejected_response(code, message, **detail):
    issue = {"code": code, "message": message, "severity": "error", **detail}
    return ResponseValidationError({"status": "rejected", "issues": [issue],
        "warnings": [], "hard_issues": [issue], "scope": "format_and_language_only",
        "semantic_accuracy_evaluated": False})


def _language_issue(text):
    """대부분 한국어인 긴 본문의 소수 한자는 경고로 구별합니다.

    글자를 임의로 지우거나 바꾸지 않습니다. 이 검사는 문장 의미나 사실을
    보장하지 않으며, 한자가 중심인 응답은 이전처럼 표시를 막습니다.
    """
    decoded = unescape(text)
    matches = list(_HAN.finditer(decoded))
    if not matches:
        return None
    hangul = len(_HANGUL.findall(decoded))
    isolated = len(matches) <= 2 and hangul >= 20 and len(matches) / (hangul + len(matches)) <= .03
    return {"code": "language_mixed_characters" if isolated else "language",
        "message": "한국어 본문에 소수의 한자가 섞여 있습니다." if isolated else "본문을 한국어로 작성해야 합니다.",
        "severity": "warning" if isolated else "error", "hangul_count": hangul,
        "han_count": len(matches), "characters": [{"character": match.group(),
            "codepoint": "U+%04X" % ord(match.group()), "decoded_text_offset": match.start()}
            for match in matches]}


def first_han_index(text):
    """공개 본문의 한자 문자 위치를 반환합니다. 없으면 -1입니다."""
    found = _HAN.search(text) if isinstance(text, str) else None
    return found.start() if found else -1


def contains_han(text):
    """한자 포함 여부만 확인합니다. 표시 차단 여부는 inspect_response로 정합니다."""
    return isinstance(text, str) and first_han_index(unescape(text)) >= 0


def _contains_response_json(text):
    """대사 안에 다시 들어간 출력 계약 객체를 찾습니다.

    모델의 전체 응답은 JSON이지만 각 text 필드는 사람이 읽을 본문입니다.
    본문 중간에 blocks 또는 speech 객체가 섞여도 원시 출력 구조로 판단합니다.
    인물 이름·사건 내용·일반 JSON의 의미는 검사하지 않습니다.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("blocks"), list):
            return True
        if {"kind", "speaker_id", "text"} <= value.keys() and value.get("kind") in {"speech", "narration"}:
            return True
    return False


def inspect_response(text, payload=None, npc_name=None, engine_result=None):
    """본문 형식의 문제 목록을 반환합니다. 사실 정확도나 게임 상태를 판정하지 않습니다."""
    issues = []
    if not isinstance(text, str) or not text.strip():
        issues.append({"code": "empty_response", "message": "표시할 본문이 비어 있습니다."})
    else:
        language = _language_issue(text)
        if language:
            issues.append(language)
    if isinstance(text, str) and _PROTOCOL.search(text):
        issues.append({"code": "protocol", "message": "내부 모델 형식을 본문에 표시할 수 없습니다."})
    if isinstance(text, str) and _contains_response_json(text):
        issues.append({"code": "protocol", "message": "응답 구조 대신 대화 본문만 필요합니다."})
    if isinstance(text, str) and text.lstrip().startswith("```"):
        issues.append({"code": "format", "message": "코드 블록 대신 대화 본문이 필요합니다."})
    if isinstance(text, str) and text.lstrip().startswith(("{", "[")):
        try:
            structured = json.loads(text)
        except ValueError:
            structured = None
        if isinstance(structured, (dict, list)):
            issues.append({"code": "format", "message": "JSON 대신 대화 본문이 필요합니다."})
    warnings = [issue for issue in issues if issue.get("severity") == "warning"]
    hard = [issue for issue in issues if issue.get("severity") != "warning"]
    return {"status": "rejected" if hard else "passed_with_warnings" if warnings else "passed",
            "issues": issues, "warnings": warnings, "hard_issues": hard,
            "scope": "format_and_language_only", "semantic_accuracy_evaluated": False}


def allowed_speaker_map(payload):
    """엔진이 공개한 현장 화자의 ID와 표시 이름만 읽습니다."""
    rows = (payload or {}).get("allowed_speakers", [])
    if not isinstance(rows, list):
        raise ValueError("allowed_speakers는 목록이어야 합니다.")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("허용 화자는 ID와 이름이 있는 객체여야 합니다.")
        identifier, name = row.get("speaker_id") or row.get("card_id"), row.get("name")
        if not isinstance(identifier, str) or not identifier or not isinstance(name, str) or not name.strip():
            raise ValueError("허용 화자에는 실제 ID와 표시 이름이 필요합니다.")
        if identifier in result and result[identifier] != name:
            raise ValueError("동일한 화자 ID의 표시 이름이 다릅니다.")
        result[identifier] = name
    return result


def response_requirements(payload):
    """현재 턴의 표현 계약을 엔진이 제공한 역할 정보로 정합니다.

    대화 중인 NPC의 이름·ID는 허용 화자 목록과 일치해야 합니다. 이름이
    없거나 서로 다른 인물 정보가 섞이면 임의의 인물을 골라 보완하지 않습니다.
    GM 장면에는 공개된 여러 화자가 등장할 수 있고, 규칙 설명은 GM 서술입니다.
    """
    data = {} if payload is None else payload
    if not isinstance(data, dict):
        raise ValueError("응답 자료는 dict여야 합니다.")
    allowed = allowed_speaker_map(data)
    actor = data.get("actor_context") or {}
    state = data.get("dialogue_state") or {}
    if not isinstance(actor, dict) or not isinstance(state, dict):
        raise ValueError("현재 인물과 대화 단계는 객체여야 합니다.")
    current = state.get("current_speaker") or {}
    if not isinstance(current, dict):
        raise ValueError("현재 화자는 객체여야 합니다.")

    # 명시된 현재 NPC를 찾습니다. 허용 목록에 있는 인물 전부를 현재 화자로
    # 취급하지 않으므로, 같은 장소의 다른 NPC가 대신 대답하지 않습니다.
    names = [data.get("npc_name")]
    identifiers = [data.get("speaker_id")]
    if actor.get("kind") in (None, "npc"):
        names.append(actor.get("name"))
        identifiers.extend((actor.get("speaker_id"), actor.get("card_id")))
    if current.get("kind") in (None, "npc"):
        names.append(current.get("name"))
        identifiers.extend((current.get("speaker_id"), current.get("card_id")))
    if any(value is not None and not isinstance(value, str) for value in names + identifiers):
        raise ValueError("현재 NPC의 이름과 ID는 문자열이어야 합니다.")
    names = {name.strip() for name in names if isinstance(name, str) and name.strip()}
    identifiers = {identifier for identifier in identifiers if isinstance(identifier, str) and identifier}
    mode = data.get("response_mode")
    if mode in (None, ""):
        mode = "npc_dialogue" if names or identifiers else "scene_dialogue"
    if mode not in {"npc_dialogue", "scene_dialogue", "rule_explanation"}:
        raise ValueError("응답 방식은 NPC 대화·GM 장면·규칙 설명 중 하나여야 합니다.")
    phase = state.get("phase") or "continue"
    if not isinstance(phase, str):
        raise ValueError("대화 단계는 문자열이어야 합니다.")

    target_id, target_name = None, None
    if mode == "npc_dialogue":
        if actor.get("kind") not in (None, "npc") or current.get("kind") not in (None, "npc"):
            raise ValueError("NPC 대화의 현재 화자는 NPC여야 합니다.")
        if len(names) > 1 or len(identifiers) > 1:
            raise ValueError("현재 NPC의 이름 또는 ID가 서로 다릅니다.")
        if identifiers:
            target_id = next(iter(identifiers))
            if target_id not in allowed:
                raise ValueError("현재 NPC의 ID가 허용 화자 목록에 없습니다.")
            if names and next(iter(names)) != allowed[target_id].strip():
                raise ValueError("현재 NPC의 이름과 허용 ID가 다른 인물을 가리킵니다.")
        elif names:
            name = next(iter(names))
            matches = [identifier for identifier, display in allowed.items() if display.strip() == name]
            if len(matches) == 1:
                target_id = matches[0]
        if target_id is None:
            raise ValueError("NPC 대화에는 현재 인물의 고유한 허용 ID가 필요합니다.")
        target_name = allowed[target_id]

    return {"mode": mode, "phase": phase, "target_speaker_id": target_id, "target_name": target_name,
        "speech_required": mode == "npc_dialogue",
        "narration_allowed": mode != "npc_dialogue",
        "max_blocks": 3 if mode == "rule_explanation" else 2 if mode == "npc_dialogue" else 8}


def response_blocks_schema(payload):
    """현재 턴의 화자·표현 종류·블록 수를 제약하는 Ollama용 JSON 스키마입니다."""
    def block_schema(kind, speaker):
        return {"type": "object", "properties": {
            "kind": {"type": "string", "enum": [kind]},
            "speaker_id": speaker, "text": {"type": "string", "minLength": 1}},
            "required": ["kind", "speaker_id", "text"], "additionalProperties": False}
    # GM 설명은 이름 없는 narration, NPC 직접 대사는 실제 ID가 있는 speech입니다.
    # 화면은 이 ID로 초상화를 고르므로 모델이 표시 이름을 임의로 쓰지 않습니다.
    requirements = response_requirements(payload)
    variants = []
    if requirements["narration_allowed"]:
        variants.append(block_schema("narration", {"type": "null"}))
    identifiers = ([requirements["target_speaker_id"]] if requirements["mode"] == "npc_dialogue"
                   else list(allowed_speaker_map(payload)) if requirements["mode"] == "scene_dialogue" else [])
    if identifiers:
        variants.append(block_schema("speech", {"type": "string", "enum": identifiers}))
    item = variants[0] if len(variants) == 1 else {"anyOf": variants}
    return {"type": "object", "properties": {"blocks": {"type": "array", "minItems": 1,
            "maxItems": requirements["max_blocks"], "items": item}}, "required": ["blocks"], "additionalProperties": False}


def validate_response_block(row, allowed, payload=None, engine_result=None):
    """화자 식별과 표현 형식만 검사하며 대사 의미의 사실 여부는 판정하지 않습니다."""
    if not isinstance(row, dict) or set(row) != {"kind", "speaker_id", "text"}:
        raise _rejected_response("block_fields", "각 블록에는 kind·speaker_id·text만 필요합니다.")
    kind, identifier, text = row["kind"], row["speaker_id"], row["text"]
    requirements = response_requirements(payload)
    # payload가 있는 실제 생성에서는 엔진의 허용 목록을 다시 기준으로 씁니다.
    # allowed 인자는 기존 단일 블록 검사 호출과의 연결 형식을 유지합니다.
    names = allowed_speaker_map(payload) if payload is not None else allowed
    if kind == "narration":
        if not requirements["narration_allowed"]:
            raise _rejected_response("turn_contract", "이번 NPC 대화는 장면 반복 없이 현재 인물의 직접 대사로 답해야 합니다.")
        if identifier not in (None, ""):
            raise _rejected_response("speaker_id", "GM 서술의 speaker_id는 null이어야 합니다.")
        identifier, name = None, ""
    elif kind == "speech":
        if requirements["mode"] == "rule_explanation":
            raise _rejected_response("turn_contract", "규칙 설명은 NPC 대사가 아닌 GM 서술로 작성해야 합니다.")
        if not isinstance(identifier, str) or identifier not in names:
            raise _rejected_response("speaker_id", "대사의 화자는 현재 허용된 speaker_id여야 합니다.")
        if requirements["mode"] == "npc_dialogue" and identifier != requirements["target_speaker_id"]:
            raise _rejected_response("speaker_id", "이번 대화는 현재 NPC의 허용 ID로 답해야 합니다.")
        name = names[identifier]
    else:
        raise _rejected_response("block_kind", "블록 종류는 narration 또는 speech여야 합니다.")
    report = inspect_response(text, payload, name, engine_result)
    if report["hard_issues"]:
        # 교정 요청에는 실제 실패 이유를 전달합니다. 잘못된 초안 문장 자체를
        # 다시 입력할 필요 없이 언어 또는 응답 구조의 문제를 구분할 수 있습니다.
        raise ResponseValidationError(report)
    return {"kind": kind, "speaker_id": identifier, "speaker": name, "text": text.strip()}


def unique_json_keys(pairs):
    """같은 키가 두 번 온 JSON은 마지막 값으로 조용히 덮어쓰지 않고 거절합니다."""
    value = {}
    for key, item in pairs:
        if key in value:
            raise _rejected_response("duplicate_json_key", "JSON 필드가 중복됩니다: " + key)
        value[key] = item
    return value


def parse_response_blocks(raw, payload, engine_result=None):
    """모델의 블록 JSON을 확인하고 표시 이름은 엔진 목록에서만 연결합니다."""
    try:
        value = json.loads(raw, object_pairs_hook=unique_json_keys)
    except ResponseValidationError:
        raise
    except (ValueError, TypeError) as error:
        raise _rejected_response("invalid_json", "완성된 응답 JSON을 읽을 수 없습니다.") from error
    if not isinstance(value, dict) or set(value) != {"blocks"}:
        raise _rejected_response("response_fields", "최상위 JSON에는 blocks만 필요합니다.")
    rows = value["blocks"]
    requirements = response_requirements(payload)
    if not isinstance(rows, list) or not 1 <= len(rows) <= requirements["max_blocks"]:
        raise _rejected_response("block_count", f"이번 응답은 1~{requirements['max_blocks']}개의 허용된 블록이어야 합니다.")
    allowed = allowed_speaker_map(payload)
    blocks, warnings = [], []
    for index, row in enumerate(rows):
        try:
            blocks.append(validate_response_block(row, allowed, payload, engine_result))
        except ResponseValidationError as error:
            # 블록 오류만 기록합니다. 오류 본문을 설정 사실로 다시 주입하지 않습니다.
            for issue in error.guard_report["issues"]:
                issue["block_index"] = index
            raise
        for issue in inspect_response(row["text"])["warnings"]:
            warnings.append({**issue, "block_index": index})
    if requirements["speech_required"] and not any(row["kind"] == "speech"
            and row["speaker_id"] == requirements["target_speaker_id"] for row in blocks):
        raise _rejected_response("missing_current_speech", "이번 응답에는 현재 NPC의 직접 대사가 필요합니다.")
    return blocks, {"status": "passed_with_warnings" if warnings else "passed",
                    "issues": warnings, "warnings": warnings, "hard_issues": [],
                    "scope": "block_structure_speaker_id_language_and_turn_contract",
                    "semantic_accuracy_evaluated": False}
