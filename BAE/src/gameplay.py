"""게임 상태·행동·AI 대화를 연결합니다.

규칙 판정은 game.Engine, 대사 생성은 전달받은 출력 모듈이 담당합니다.
자연어로 제안된 행동은 확인을 받고, 버튼이나 **행동** 선언은 직접 실행합니다.
"""
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path
import re
from secrets import token_urlsafe
from threading import Event as CancellationEvent, RLock
import time

from diagnostics_live import current_request_id, current_stage
from campaign_flow import (CONVERSATION_MILESTONES, EARLY_AMBIENT_ACTIONS, INTRODUCTION_SOURCES,
                           NPC_STAGE, ROOM_STAGE, pending_stage, reached)
from game.content import Content, npc_name
from game.dice import Dice, DICE_EXPR
from game.engine import Engine, fmt_money
from game.events import Event, ChoiceView
from game.model import Character, GameState, ABILITY_KO
from game.startup import apply_starting_equipment
from action_requirements import equipment_rows
from query_contract import availability, focused_information, public_services
from navigation_contract import action_destination, exact_action
from state_projection import (new_ledger, before_action, record_action, restore_ledger,
    player_party, npc_party, player_snapshot, model_state,
    native_conversation_history)


TRANSACTION_BINDINGS = json.loads(Path(__file__).with_name("transaction_bindings.json").read_text(encoding="utf-8"))

# 원본 엔진의 한 지역 안에서 실제로 드나드는 공간만 화면·상호작용 층으로 나눕니다.
MARKET_SPACES = {
    "village_market:forge": {"name": "대장간", "npc_id": "scn-m-npc-smith", "service": "shop"},
    "village_market:inn": {"name": "여관", "npc_id": "scn-m-npc-innkeeper", "service": "inn"},
}


def _question_form(message):
    """문장 끝의 질문 표지는 실제 행동 실행보다 우선합니다."""
    text = message.strip().strip('"“”‘’')
    if text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()
    if re.search(r"[?？]\s*$|(?:까요|나요|습니까|입니까|인가요|는지요|은지요|인지요|(?<!니)까|느냐|냐)[.!。！]?\s*$", text):
        return True
    formal = re.search(r"([가-힣])니까[.!。！]?\s*$", text)
    # 격식 질문의 받침 ㅂ + 니까와 원인 설명의 ~하니까를 구별합니다.
    return bool(formal and (ord(formal[1]) - ord("가")) % 28 == 17)


def _information_question(message):
    """정보를 묻는 질문이 인사와 섞였다는 이유로 근거 없이 처리되지 않게 합니다."""
    explanation = re.search(r"(?:알려|설명해|말씀해)\s*(?:주실|주시겠|줄\s*수|주세요|주십시오|줘)", message)
    if explanation:
        return True
    if not _question_form(message):
        return False
    # '어떻게 지내세요' 같은 안부의 어떻게만으로는 정보 질문을 강제하지 않습니다.
    return bool(re.search(r"누구|누군|누가|어디|어딘|무엇|무슨|언제|얼마|몇|왜|어떤", message))




class GameEngine(Engine):
    """게임 규칙에 개발자 조작을 명시적으로 추가한 엔진입니다."""

    def __init__(self, *args, **kwargs):
        self.developer = False
        self.next_success = False
        self.pending_roll_override = None
        super().__init__(*args, **kwargs)
        apply_starting_equipment(self)
        self._save_checkpoint("게임 시작")

    def _level_up(self, char, to_level):
        """고정 HP 방식에 드워프의 레벨당 추가 HP도 적용합니다."""
        events = []
        while char.level < to_level:
            char.level += 1
            dwarf_bonus = 1 if "dwarven_toughness" in char.traits else 0
            gain = max(1, self.c.rules["fixed_hp_per_level"][char.cls] + char.mod("con") + dwarf_bonus)
            char.hp_max += gain
            char.hp += gain
            char.hd_left += 1
            self.init_resources(char)
            events.append(Event("status", f"{char.name}가 {char.level}레벨이 되었다. 최대 HP +{gain} → {char.max_hp}"))
        return events

    def _force(self, event):
        if not self.developer:
            return None
        actor = event.data.get("char", event.data.get("actor"))
        override = self.pending_roll_override
        if override is not None and actor == override["actor_id"] and event.data.get("side") != "enemy":
            success = override["success"]
            self.pending_roll_override = None
        elif self.next_success and actor == self.hero.id:
            success, self.next_success = True, False
        else:
            return None
        self.s.vars["developer_used"] = True
        event.data["developer_override"] = True
        event.data["developer_forced_outcome"] = "success" if success else "failure"
        event.data["natural_success"] = event.data.get("success", event.data.get("hit"))
        if "hit" in event.data:
            event.data["hit"] = success
            if not success:
                event.data["crit"] = False
        else:
            event.data["success"] = success
        # 실제 주사위 값은 남기고, 테스트용 결과 변경을 별도로 표시합니다.
        label = "성공" if success else "실패"
        event.text = event.text.split("→", 1)[0] + f"→ {label} [개발자: 결과 강제 {label}]"
        return success

    def roll_check(self, char, chk, method_index=None):
        ok, event = super().roll_check(char, chk, method_index)
        forced = self._force(event)
        return (ok if forced is None else forced), event

    def roll_save(self, char, ability, dc, advantage=False, avoids=None):
        ok, event = super().roll_save(char, ability, dc, advantage, avoids)
        forced = self._force(event)
        return (ok if forced is None else forced), event

    def _attack_roll(self, monster, bonus, label):
        hit, critical, mode, event = super()._attack_roll(monster, bonus, label)
        forced = self._force(event)
        if forced is not None:
            hit, critical = forced, critical if forced else False
        return hit, critical, mode, event

    def require_developer(self):
        if not self.developer:
            raise ValueError("개발자 모드를 먼저 켜세요.")

    def arm_success(self):
        self.require_developer()
        self.next_success = True
        return [Event("developer", "다음 플레이어 능력 판정·구원·공격 1회를 성공시킵니다. 적의 굴림에는 적용되지 않습니다.")]

    def force_victory(self):
        self.require_developer()
        if not self.s.combat:
            raise ValueError("진행 중인 전투가 없습니다.")
        for enemy in self.living_enemies():
            enemy["hp"], enemy["out"] = 0, "dead"
            self.s.combat["xp"] += self.c.monsters[enemy["kind"]].get("xp", 0)
        self.s.vars["developer_used"] = True
        return [Event("developer", "개발자 조작으로 전투를 승리 처리했습니다.")] + self._end_combat("victory")

    def full_heal(self):
        self.require_developer()
        self.s.ended = ""
        self.hero.conditions.clear()
        self.hero.hp_max_loss = 0
        self.hero.ability_loss.clear()
        self.hero.hp = self.hero.max_hp
        self.init_resources(self.hero)
        self.s.vars["developer_used"] = True
        return [Event("developer", "개발자 조작으로 HP·상태·직업 자원을 회복했습니다.")]


def _jsonable(value):
    if isinstance(value, set):
        return [_jsonable(v) for v in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


class PlaySession:
    """접속자 한 명의 게임과 대화. 다른 접속자의 상태를 공유하지 않습니다."""

    def __init__(self, content, api, character, seed=None):
        self.content, self.api = content, api
        self.engine = GameEngine(content=content, char_id=character, seed=seed)
        self.messages, self.histories, self.audit = [], {}, []
        # 채팅 원문과 별도로 턴별 측정값을 보존합니다. 미측정 값은 None입니다.
        self.turn_records = []
        self._active_route_diagnostics = None
        self._failed_dialogue = None
        self.diagnostics_meta = {"schema": "trpg_output_diagnostics_1",
            "model": api.get("MODEL_NAME") if api else None, "code_sha256": None,
            "version_hash": api.get("_bae_loaded_runtime_fingerprint") if api else None,
            "model_digest": None, "quantization": None, "options": None, "output_version": None,
            "runtime_info": {}, "gpu": None}
        self._refresh_diagnostics_meta()
        self.npc = ""
        self.scene_focus_explicit = False  # 장면/게임 마스터를 직접 고른 상태
        self.pending = None
        self.revision = 0
        self.last_result = None
        self.debug = {}
        self.last_status = "게임을 시작했습니다."
        self.rule_mode = "available"
        self.suggestions = []
        self._recommendation_epoch = 0
        self._recommendation_cancel = CancellationEvent()
        self._pending_recommendation = None
        self.last_generation_seconds = 0.0
        self.action_generation_calls = 0
        self.turn_intent = None
        self.transaction_ledger = new_ledger()
        self.turn_needs_information = False
        self.turn_needs_state = False
        self.last_input_kind = None
        self._travel_transit = False
        self._developer_snapshots = []
        self._developer_snapshot_serial = 0
        self._restoring_developer_snapshot = False
        self._last_record_minutes = self.engine.s.minutes
        self.memory_session = api["create_memory_session"]() if api else None
        self._record(self.engine.opening, player_input="게임을 시작한다", opening_scene=True)
        auto = self.content.rooms[self.engine.s.room].get("auto_meet")
        if auto and auto in self.content.cards:
            card = self.content.cards[auto]
            approach = self._approach(card)
            if approach in {choice.id for choice in self.choices()}:
                self._record(self.engine.choose(approach), choice_id=approach,
                             player_input=self.content.choices[approach]["label"])
            else:
                self.talk_to(npc_name(card))
        self.refresh_suggestions()
        self._capture_developer_checkpoint("게임 시작")

    def message_scene_context(self):
        """지금 생성하는 메시지의 실제 장면·방·공간을 한 번 캡처합니다."""
        place = self.current_place()
        return {"scene_id": self.content.rooms[self.engine.s.room]["scene"], "room_id": self.engine.s.room,
                "place_id": place["id"], "room_name": place["name"]}

    def append_message(self, message, *, scene_context=None):
        """새 메시지에 생성 당시 위치를 저장합니다. 과거 메시지는 보정하지 않습니다.

        안내 카드의 room_id는 목적지일 수 있습니다. 화면 배경은 이 중첩 기록만
        사용하므로 현재 장소와 안내 목적지가 서로 섞이지 않습니다.
        """
        if not isinstance(message, dict):
            raise TypeError("메시지는 사전이어야 합니다.")
        context = self.message_scene_context() if scene_context is None else deepcopy(scene_context)
        if (not isinstance(context, dict) or type(context.get("scene_id")) is not int
                or any(not isinstance(context.get(key), str) or not context[key]
                       for key in ("room_id", "place_id", "room_name"))):
            raise ValueError("새 메시지의 장면 정보를 확인할 수 없습니다.")
        record = deepcopy(message)
        record["scene_context"] = {key: context[key] for key in
                                   ("scene_id", "room_id", "place_id", "room_name")}
        list.append(self.messages, record)
        return record

    def _visible_error(self, message, error, *, stage, engine_committed=False):
        """실패를 채팅에도 남깁니다. 이미 실행한 행동을 다시 실행하지 않습니다."""
        previous = self.messages[-1] if self.messages else {}
        if (previous.get("kind") == "error" and previous.get("failed_input") == message
                and previous.get("error_type") == type(error).__name__):
            return previous["content"]
        text = ("입력을 처리하지 못했습니다. 현재 위치와 가방은 그대로입니다. 다시 입력해 주세요."
                if stage == "routing" else
                "답변을 만들지 못했습니다. 같은 질문의 답변만 다시 받을 수 있습니다.")
        if engine_committed:
            text = "행동 결과는 반영했습니다. 이어지는 대사 생성에 실패했으며 행동을 다시 실행하지 않습니다."
        self.append_message({"role": "assistant", "speaker": "", "kind": "error",
                             "content": text, "failed_input": message,
                             "error_type": type(error).__name__, "failure_stage": stage})
        self.last_status = text
        self.debug["최근 오류"] = {"stage": stage, "type": type(error).__name__,
                                  "message": str(error), "engine_committed": engine_committed}
        return text

    def _refresh_diagnostics_meta(self, backend=None):
        """출력 연결이 실제로 확인한 모델·GPU 정보를 복사 기록에도 보존합니다."""
        metadata = {}
        read_metadata = self.api.get("diagnostics_metadata") if self.api else None
        if callable(read_metadata):
            try:
                value = read_metadata()
                if isinstance(value, dict):
                    metadata = deepcopy(value)
            except Exception:
                pass  # 진단 정보 조회 실패 때문에 기존 게임 처리를 중단하지 않습니다.
        if isinstance(backend, dict):
            runtime = metadata.get("runtime_info")
            request_runtime = backend.get("runtime_info")
            metadata.update(deepcopy(backend))
            if isinstance(runtime, dict) and isinstance(request_runtime, dict):
                metadata["runtime_info"] = {**deepcopy(runtime), **deepcopy(request_runtime)}
        details = metadata.get("model_details")
        details = details if isinstance(details, dict) else {}
        for key, value in (("model", metadata.get("model")),
            ("code_sha256", metadata.get("source_sha256")),
            ("output_version", metadata.get("output_version")),
            ("model_digest", metadata.get("model_digest", metadata.get("digest"))),
            ("quantization", metadata.get("quantization") or details.get("quantization_level"))):
            if isinstance(value, str):
                self.diagnostics_meta[key] = value
        runtime = metadata.get("runtime_info")
        if isinstance(runtime, dict):
            previous = self.diagnostics_meta.get("runtime_info")
            self.diagnostics_meta["runtime_info"] = {
                **(deepcopy(previous) if isinstance(previous, dict) else {}), **deepcopy(runtime)}
            if isinstance(runtime.get("gpu"), str):
                self.diagnostics_meta["gpu"] = runtime["gpu"]
        return metadata

    def _record_turn_diagnostics(self, kind, message, payload, result, started, *,
                                 route_debug=None, blocks=None, error=None, history=None):
        """실제 전달·생성·엔진 결과의 공개 필드만 한 턴에 한 번 복사합니다.

        카드 원문이나 전체 입력 payload를 다시 수집하지 않습니다. 검색 근거는
        팀 I3가 반환한 목록만 사용하고, 미측정 시간·토큰을 0으로 추정하지 않습니다.
        """
        payload = payload if isinstance(payload, dict) else {}
        result = result if isinstance(result, dict) else {}
        handoff = payload.get("input_handoff", payload)
        handoff = handoff if isinstance(handoff, dict) else {}
        route = route_debug if isinstance(route_debug, dict) else self._active_route_diagnostics or {}
        measured = handoff.get("input_metrics") or {}
        measured = measured if isinstance(measured, dict) else {}
        number = lambda value: value if type(value) in (int, float) and math.isfinite(value) else None
        stages = result.get("stage_metrics", getattr(error, "stage_metrics", None))
        stages = stages if isinstance(stages, list) else []
        stage_rows = [{key: deepcopy(row[key]) for key in
            ("seconds", "load_seconds", "prompt_seconds", "decode_seconds", "input_tokens",
             "output_tokens", "done_reason", "generation_limit", "stage")
            if key in row and isinstance(row[key], (str, int, float, bool, type(None)))}
            for row in stages if isinstance(row, dict)]
        for copied, original in zip(stage_rows, (row for row in stages if isinstance(row, dict))):
            if isinstance(original.get("request_options"), dict):
                copied["request_options"] = {key: value for key, value in original["request_options"].items()
                    if key in {"num_ctx", "num_predict", "sample", "temperature", "top_p", "top_k", "seed", "thinking", "structured_response"}
                    and isinstance(value, (str, int, float, bool, type(None)))}
            if isinstance(original.get("validation"), dict):
                copied["validation"] = deepcopy(original["validation"])
            # 받은 검색 결과 전체와 각 요청에서 실제로 사용한 자료를 구분합니다.
            # 자유 대화의 첫 요청은 새 검색 원문을 사용하지 않아 빈 목록입니다.
            if isinstance(original.get("text"), str):
                copied["text"] = original["text"]
            if isinstance(original.get("messages"), list):
                copied["messages"] = [{"role": row["role"], "content": row["content"]}
                    for row in original["messages"] if isinstance(row, dict)
                    and row.get("role") in {"system", "user", "assistant"}
                    and isinstance(row.get("content"), str)]
            if isinstance(original.get("provided_sources"), list):
                copied["provided_sources"] = [{key: deepcopy(row.get(key, (row.get("metadata") or {}).get(key)))
                    for key in ("label", "source", "card_id", "title", "page", "score", "text")
                    if isinstance(row.get(key, (row.get("metadata") or {}).get(key)), (str, int, float, bool, type(None)))}
                    for row in original["provided_sources"] if isinstance(row, dict)]
        durations = [number(row.get("seconds")) for row in stage_rows]
        # 형식 재작성과 의미 대조는 다른 작업입니다. 두 번째 요청이라는 이유만으로
        # 모두 형식 교정 시간으로 합치지 않습니다.
        repairs = [number(row.get("seconds")) for row in stage_rows
                   if row.get("stage") in {"format_repair", "source_rebuild"}]
        meaning_checks = [number(row.get("seconds")) for row in stage_rows
                          if row.get("stage") == "semantic_review"]
        generation = number(result.get("generation_seconds"))
        if generation is None and durations and all(value is not None for value in durations):
            generation = sum(durations)
        repair = (sum(repairs) if repairs and all(value is not None for value in repairs)
                  else number(result.get("format_repair_seconds")))
        meaning_seconds = (sum(meaning_checks) if meaning_checks and all(value is not None for value in meaning_checks)
                           else number(result.get("meaning_review_seconds")))
        sources, counts = [], {}
        context = handoff.get("retrieved_context")
        groups = (("rules", "rules"), ("scenarios", "scenarios"))
        if not isinstance(context, dict):
            context = handoff.get("retrieval") or {}
            groups = (("rule_results", "rules"), ("scenario_results", "scenarios"))
        for key, label in groups:
            rows = context.get(key) if isinstance(context, dict) else None
            counts[label] = len(rows) if isinstance(rows, list) else None
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                sources.append({name: deepcopy(row.get(name, meta.get(name))) for name in
                    ("source", "card_id", "title", "page", "score", "text")
                    if isinstance(row.get(name, meta.get(name)), (str, int, float, bool, type(None)))})
        versions = result.get("versions", getattr(error, "versions", None)) or {}
        backend = result.get("backend", getattr(error, "backend", None)) or {}
        versions = versions if isinstance(versions, dict) else {}
        backend = self._refresh_diagnostics_meta(backend if isinstance(backend, dict) else {})
        for key, value in (("model", backend.get("model")),
            ("code_sha256", versions.get("output_code_sha256")),
            ("output_version", versions.get("output_version", backend.get("output_version"))),
            ("model_digest", backend.get("model_digest", backend.get("digest"))),
            ("quantization", backend.get("quantization") or
                (backend.get("model_details") or {}).get("quantization_level"))):
            if isinstance(value, str):
                self.diagnostics_meta[key] = value
        if isinstance(backend.get("options"), dict):
            self.diagnostics_meta["options"] = {key: value for key, value in backend["options"].items()
                if key in {"temperature", "top_p", "top_k", "num_ctx", "num_predict", "seed", "repeat_penalty", "sample", "thinking"}
                and isinstance(value, (str, int, float, bool, type(None)))}
        state = self.rag_state()
        public_state = {key: deepcopy(state[key]) for key in
            ("scene", "current_room", "inventory", "gold_cp", "money", "hero", "combat") if key in state}
        controller_event = payload.get("controller_event")
        event = (controller_event if isinstance(controller_event, dict) else self.last_result or {}) if kind == "action" else {}
        delta = event.get("engine_delta") or {}
        public_delta = {key: deepcopy(delta[key]) for key in
            ("executed_action_id", "executed_action_ids", "room_before", "room_after", "method_index",
             "gold_cp_change", "inventory_changes", "hp_change", "minutes_before", "minutes_after",
             "elapsed_minutes", "engine_pending") if key in delta}
        raw_guard = result.get("response_guard", getattr(error, "guard_report", None))
        guard = ({key: deepcopy(raw_guard[key]) for key in
            ("status", "valid", "errors", "warnings", "repaired", "checks", "reason", "issues", "hard_issues", "scope") if key in raw_guard}
            if isinstance(raw_guard, dict) else None)
        turn_started = number(route.get("_turn_started"))
        if turn_started is None:
            turn_started = started
        def token_total(key):
            value = number(result.get(key))
            values = [number(row.get(key)) for row in stage_rows]
            return value if value is not None else (sum(values) if values
                and all(item is not None for item in values) else None)
        def prompt_messages(value):
            if not isinstance(value, list):
                return None
            return [{"role": row["role"], "content": row["content"]} for row in value
                if isinstance(row, dict) and row.get("role") in {"system", "user", "assistant", "tool"}
                and isinstance(row.get("content"), str)]
        input_trace = handoff.get("input_trace") or {}
        input_trace = input_trace if isinstance(input_trace, dict) else {}
        upstream = handoff.get("upstream_handoff") or {}
        upstream = upstream if isinstance(upstream, dict) else {}
        query = result.get("retrieval_query", getattr(error, "retrieval_query", input_trace.get("retrieval_query")))
        reply_to = result.get("reply_to", getattr(error, "reply_to", None))
        if isinstance(reply_to, dict):
            reply_to = {key: value for key, value in reply_to.items()
                if key in {"role", "text", "content", "speaker_id", "speaker"}
                and isinstance(value, (str, type(None)))}
        elif not isinstance(reply_to, str):
            reply_to = None
        actor = payload.get("actor_context") or {}
        active_npc = ({key: actor[key] for key in ("name", "card_id", "speaker_id")
                       if isinstance(actor.get(key), str)} if isinstance(actor, dict) else None)
        actual_options = (stage_rows[-1].get("request_options") if stage_rows else None)
        attempted_options = result.get("request_options", getattr(error, "request_options", None))
        if isinstance(attempted_options, dict):
            actual_options = {key: value for key, value in attempted_options.items()
                if key in {"num_ctx", "num_predict", "sample", "temperature", "top_p", "top_k", "seed",
                           "thinking", "think", "structured_response", "repeat_penalty"}
                and isinstance(value, (str, int, float, bool, type(None)))}
        if actual_options is None and isinstance(backend.get("options"), dict):
            actual_options = deepcopy(self.diagnostics_meta.get("options"))
        next_id = self.turn_records[-1]["turn_id"] + 1 if self.turn_records else 1
        self.turn_records.append({"turn_id": next_id, "request_id": current_request_id(), "kind": kind,
            "input": {"text": route.get("_player_input", message), "request": message,
                      "kind": "action" if kind == "action" else self.last_input_kind or "speech",
                      "npc_name": payload.get("npc_name", self.npc if kind == "dialogue" else ""),
                      "original_player_input": input_trace.get("original_player_input", handoff.get("player_input")),
                      "retrieval_query": query if isinstance(query, str) else None,
                      "upstream_player_input": upstream.get("player_input"),
                      "retry_of_turn_id": route.get("retry_of_turn_id")},
            "status": "error" if error is not None else "completed",
            "timing": {"search_seconds": number(measured.get("search_seconds")),
                "handoff_seconds": number(measured.get("handoff_seconds")),
                "input_seconds": number(route.get("seconds")),
                "input_pipeline_seconds": number(measured.get("total_seconds")),
                "classification_seconds": number(route.get("classification_seconds")),
                "mapper_seconds": number(route.get("mapper_seconds")),
                "generation_seconds": generation, "modelgen_seconds": durations[0] if durations else None,
                "repair_seconds": repair, "meaning_review_seconds": meaning_seconds,
                "total_seconds": round(time.perf_counter() - turn_started, 6)},
            "tokens": {"input": token_total("input_tokens"), "output": token_total("output_tokens")},
            "stage_metrics": stage_rows, "sources_count": counts, "sources_public": sources,
            "prompt_trace": {"messages": prompt_messages(result.get("messages", getattr(error, "messages", None))),
                "context_messages": prompt_messages(result.get("context_messages", getattr(error, "context_messages", None))),
                "draft_messages": prompt_messages(result.get("draft_messages", getattr(error, "draft_messages", None))),
                "review_messages": prompt_messages(result.get("review_messages", getattr(error, "review_messages", None))),
                "reply_to": reply_to, "retrieval_query": query if isinstance(query, str) else None,
                "active_npc": active_npc, "phase": (payload.get("dialogue_state") or {}).get("phase"),
                "history_roles": [row["role"] for row in history if isinstance(row, dict)
                    and row.get("role") in {"user", "assistant"}] if isinstance(history, list) else None,
                "model": backend.get("model"), "options": deepcopy(actual_options)},
            "versions": {key: versions[key] for key in ("output_code_sha256", "system_prompt_sha256", "output_version",
                                                       "opening_lines_sha256", "output_policy_sha256", "conversation_prompt_sha256")
                         if isinstance(versions.get(key), str)},
            "output": {"raw": result.get("raw_text", getattr(error, "raw_response", None)),
                "draft": result.get("draft_text", getattr(error, "draft_response", None)),
                "shown": result.get("text") if error is None or blocks is not None else None,
                "blocks": deepcopy(blocks if blocks is not None else result.get("blocks", []))
                          if error is None or blocks is not None else [],
                "guard": guard, "response_origin": result.get("response_origin"),
                "meaning_review": deepcopy(result.get("meaning_review", getattr(error, "meaning_review", None))),
                "error": {"code": type(error).__name__, "message": str(error),
                          "validation": deepcopy(getattr(error, "guard_report", None)),
                          "attempts": deepcopy(getattr(error, "validation_attempts", None))} if error is not None else None},
            "engine": {"event_id": event.get("event_id"),
                "executed_action_ids": deepcopy(event.get("executed_action_ids", [])),
                "outcome": deepcopy(event.get("outcome")),
                "delta": public_delta or None, "state_after": public_state,
                "rolls": [{key: deepcopy(row["data"][key]) for key in
                    ("actor", "char", "side", "roll", "rolls", "mod", "total", "dc", "ac", "success", "hit", "crit", "critical", "mode",
                     "ability", "skill", "tool", "developer_override", "developer_forced_outcome", "natural_success")
                    if key in row["data"]} for row in event.get("public_event_records", [])
                    if row.get("kind") == "check" and isinstance(row.get("data"), dict)]}})
        self._capture_developer_checkpoint(f"{next_id} · " + str(message)[:65])

    def close(self):
        self.cancel_recommendations()
        if self.api and self.memory_session:
            self.api["close_memory_session"](self.memory_session)

    def cancel_recommendations(self):
        """새 입력·이동 뒤에는 이전 턴의 추천을 생성하거나 적용하지 않습니다."""
        self._recommendation_cancel.set()
        self._pending_recommendation = None
        self._recommendation_epoch += 1

    def take_recommendation_job(self):
        """본문을 표시한 뒤 UI가 실행할 선택적 추천 작업을 한 번만 가져갑니다.

        호출과 결과 반영은 세션 잠금 안에서, generate_recommendations는 잠금
        밖에서 실행합니다. 작업은 생성에 필요한 사본만 가지며 게임을 바꾸지 않습니다.
        """
        job, self._pending_recommendation = self._pending_recommendation, None
        return job if job is not None and not job["cancel_event"].is_set() else None

    def generate_recommendations(self, job):
        """저장된 본문에 붙일 추천을 생성하며 세션 상태에는 쓰지 않습니다."""
        cancelled = job["cancel_event"]
        if cancelled.is_set():
            return {"called": False, "reason": "cancelled", "wall_seconds": 0.0}
        started = time.perf_counter()
        result = {"called": True, "reason": "generated", "seconds": 0.0}
        current_stage("recommendations", detail="다음 선택지 생성", npc_name=job.get("speaker"))
        try:
            value = self.api["generate_followup_question"](
                job["payload"], job["reply"], history=job["history"], cancel_event=cancelled)
            if not isinstance(value, dict):
                raise ValueError("추천 생성 결과 형식이 올바르지 않습니다.")
            result.update(value)
        except Exception as error:
            if cancelled.is_set() or isinstance(error, self.api.get("OutputStreamCancelled", ())):
                result.update(reason="cancelled")
            else:
                result.update(reason="error", error=repr(error))
        result["wall_seconds"] = time.perf_counter() - started
        if cancelled.is_set():
            result["reason"] = "cancelled"
        current_stage("recommendations", detail="다음 선택지 생성 종료", status=result["reason"],
                      elapsed_seconds=result["wall_seconds"], error=result.get("error"))
        return result

    def apply_recommendations(self, job, result):
        """같은 턴·인물일 때만 늦게 도착한 추천을 화면과 저장 이력에 반영합니다."""
        record = next((row for row in reversed(self.turn_records)
                       if row["turn_id"] == job.get("diagnostic_turn_id")), None)
        if record is not None:
            for source, target in (("seconds", "followup_seconds"), ("wall_seconds", "followup_wall_seconds")):
                value = result.get(source)
                if type(value) in (int, float) and math.isfinite(value):
                    record["timing"][target] = value
            record["recommendations"] = {key: deepcopy(result[key]) for key in
                ("called", "reason", "suggestions", "recommendation_options", "text", "input_tokens", "output_tokens", "error")
                if key in result}
        if (job.get("applied") or job["cancel_event"].is_set() or job["epoch"] != self._recommendation_epoch
                or job["revision"] != self.revision or job["speaker"] != self.npc
                or job["room"] != self.engine.s.room):
            return False
        job["applied"] = True
        rows, seen = [], set()
        values = result.get("recommendation_options")
        if not isinstance(values, list):
            values = result.get("suggestions")
        if not isinstance(values, list):
            values = [result.get("text", "")]
        offered = {row.get("id") for row in (job.get("payload") or {}).get("available_actions", [])
                   if isinstance(row, dict) and isinstance(row.get("id"), str)}
        available = {choice.id: choice for choice in self.choices()}
        if result.get("reason") not in {"error", "cancelled"}:
            for value in values:
                text = value.get("text", "") if isinstance(value, dict) else value
                if not isinstance(text, str) or not 1 <= len(text.strip()) <= 90:
                    continue
                text = text.strip()
                key = re.sub(r"[\W_]+", "", text).casefold()
                if not key or key in seen or self._asked_recently(text, job["player_message"]):
                    continue
                option = {"label": text, "text": text, "source": "model"}
                if isinstance(value, dict):
                    option["kind"] = value.get("kind", "dialogue")
                    action_id = value.get("action_id")
                    if action_id:
                        # 모델이 본 선택지와 지금 가능한 엔진 행동이 모두 일치해야 실행합니다.
                        if not isinstance(action_id, str) or action_id not in offered or action_id not in available:
                            continue
                        # 실제 실행과 다른 모델 문구를 버튼에 표시하지 않습니다.
                        option.update(action_id=action_id, model_text=text,
                                      label=available[action_id].label, text=available[action_id].label)
                        key = action_id
                        if key in seen:
                            continue
                rows.append(option)
                seen.add(key)
                if len(rows) >= job["count"]:
                    break
            if not rows:
                result = {**result, "reason": "invalid_or_repeated_question"}
        self.refresh_suggestions(rows, player_message=job["player_message"])
        history = self.histories.get(job["speaker"], [])
        for record in reversed(history):
            if record.get("role") == "assistant" and record.get("memory_turn") == job["memory_turn"]:
                record["suggestions"] = [row["text"] for row in rows]
                break
        try:
            self.api["save_dialogue_history"](self.memory_session, job["speaker"], history)
        except Exception as error:
            result = {**result, "history_save_error": repr(error)}
        self.debug["후속 질문"] = deepcopy(result)
        self.debug["추천"] = deepcopy(rows)
        self.debug.setdefault("호출 수", {})["generate_followup_question"] = int(bool(result.get("called")))
        timings = self.debug.setdefault("처리 시간", {})
        timings["followup_seconds"] = float(result.get("seconds", 0))
        timings["followup_wall_seconds"] = round(float(result.get("wall_seconds", 0)), 3)
        timings["recommendation_finished_after_body_seconds"] = round(time.perf_counter() - job["queued_at"], 3)
        return True

    def npc_cards(self):
        """원본 카드의 표시 이름으로 NPC 카드를 찾습니다. 카드 내용은 바꾸지 않습니다."""
        return {npc_name(card): card for card in self.content.cards.values() if npc_name(card)}

    def current_place(self):
        """엔진 지역 ID는 유지하고 현재 들어가 있는 대장간·여관만 구분합니다."""
        native = self.engine.s.room
        place = self.engine.s.vars.get("presentation_place", "village_market:forge") if native == "village_market" else native
        if place not in MARKET_SPACES and native == "village_market":
            place = "village_market:forge"
        return {"id": place, "native_id": native,
                "name": MARKET_SPACES[place]["name"] if place in MARKET_SPACES else self.content.room_name(native)}

    def _person_place(self, card):
        return next((key for key, value in MARKET_SPACES.items() if value["npc_id"] == card.get("id")), card.get("room"))

    def _person_present(self, card):
        return card.get("room") == self.engine.s.room and self._person_place(card) == self.current_place()["id"]

    def _place_allowed(self, place):
        if place not in MARKET_SPACES:
            return self._story_room_allowed(place)
        if not self._story_room_allowed("village_market"):
            return False
        card = self.content.cards[MARKET_SPACES[place]["npc_id"]]
        return self._story_person_allowed(card)

    def _request_candidates(self, speaker=None):
        """현재 실제로 마주한 NPC의 원본 수락 선언만 찾습니다."""
        speaker = self.npc if speaker is None else speaker
        card = self.npc_cards().get(speaker, {})
        if not card or not self._person_present(card) or card["id"] not in self.engine.s.shown:
            return []
        legal = {row.id for row in self.engine.available_choices()}
        return [row for row in self.content.choices.values() if row.get("interaction") == "accept"
                and row.get("target_npc") == speaker and row.get("room") == self.engine.s.room
                and (row["id"] in legal or row["id"] in self.engine.s.done)]

    def _request_quest(self, definition):
        changed = {effect["var"] for effect in definition.get("effects", []) if "var" in effect}
        def referenced(value):
            if isinstance(value, dict):
                return ({value["var"]} if "var" in value else set()) | set().union(*(referenced(x) for x in value.values()))
            if isinstance(value, list):
                return set().union(*(referenced(x) for x in value))
            return set()
        return next((quest for scene in self.content.scenes for quest in scene.get("quest_log", {}).values()
                     if changed & referenced(quest.get("accepted_when"))), {})

    def request_view(self):
        """화면에 실제로 표시한 부탁만 제안·진행·완료 상태로 반환합니다."""
        records = self.engine.s.vars.get("request_disclosures", {})
        result = []
        for identifier, record in records.items():
            definition = self.content.choices.get(identifier, {})
            if definition.get("interaction") != "accept" or not isinstance(record, dict):
                continue
            row = deepcopy(record)
            quest = self._request_quest(definition)
            accepted = identifier in self.engine.s.done
            row["status"] = ("completed" if accepted and quest.get("completed_when")
                             and self.engine.cond(quest["completed_when"]) else "accepted" if accepted else "offered")
            result.append(row)
        return result

    def _confirmed_offer(self, speaker=None):
        name = self.npc if speaker is None else speaker
        return next((row for row in self.request_view() if row["issuer"]["name"] == name), None)

    def _request_record(self, definition):
        speaker = definition["target_npc"]
        card = self.npc_cards()[speaker]
        quest = self._request_quest(definition)
        effects = definition.get("effects", [])
        grants = [{"kind": "money", "amount_cp": row["gold_cp"]} for row in effects
                  if type(row.get("gold_cp")) is int and row["gold_cp"] > 0]
        return {"event_id": "offer:" + token_urlsafe(8), "request_id": definition["id"],
                "source_choice_id": definition["id"], "issuer": {"name": speaker, "speaker_id": card["id"]},
                "status": "offered", "title": quest.get("title") or speaker + "의 부탁",
                "purpose": quest.get("summary") or definition.get("purpose") or definition["label"],
                "acceptance_label": definition["label"], "on_acceptance": grants}

    def present_request(self, identifier):
        """제안 내용을 실제로 표시한 뒤에만 수락을 엽니다. 원본 효과는 실행하지 않습니다."""
        definition = next((row for row in self._request_candidates() if row["id"] == identifier), None)
        if not definition or identifier in self.engine.s.done or self.pending or self.engine.s.pending or self.engine.s.combat:
            raise ValueError("지금 들을 수 있는 부탁이 아닙니다.")
        existing = self.engine.s.vars.setdefault("request_disclosures", {})
        if identifier in existing:
            return
        record = self._request_record(definition)
        # 제안과 수락은 별도 사건입니다. 여기서는 돈·물품 효과를 실행하지 않습니다.
        text = "**의뢰 제안 · " + record["title"] + "**\n\n" + record["purpose"]
        if record["on_acceptance"]:
            text += "\n\n수락 시 준비금: " + " · ".join(fmt_money(row["amount_cp"]) for row in record["on_acceptance"])
        self.append_message({"role": "assistant", "speaker": "", "kind": "narration", "content": text,
                              "offer_event_id": record["event_id"]})
        existing[identifier] = record
        self.audit.append({"kind": "offer", "text": text, "data": deepcopy(record)})
        self.revision += 1
        display_event = {"event_id": record["event_id"], "executed_action_ids": [],
                         "outcome": {"status": "offered", "action_id": definition["id"]}}
        self._record_turn_diagnostics("action", "부탁의 내용을 듣는다",
            {"confirmed_offer": record, "controller_event": display_event, "npc_name": self.npc},
            {"text": text, "model_generated": False, "response_origin": "controller_offer_display"}, time.perf_counter())
        if self.api and callable(self.api.get("generate_from_handoff")):
            # 의뢰 카드는 생성 실패에도 남습니다. 자세한 부탁은 현재 NPC가 설명합니다.
            self.turn_intent, self.turn_needs_information = "listen", True
            self._generate_dialogue("부탁하시는 일과 그 이유, 준비 조건을 말씀해 주세요.", "TALK",
                {"kind": "offer", "intent": "listen", "needs_information": True, "needs_state": False,
                 "_turn_started": time.perf_counter()})
        self.refresh_suggestions()
        self.last_status = "부탁의 내용을 확인했습니다. 수락 여부를 선택할 수 있습니다."

    def dialogue_affordances(self):
        """대사와 화면이 같은 현재 서비스·공개 장비 조건을 참고합니다."""
        services = public_services(self)
        # 미발견 장소의 하강 조건을 미리 읽어 오지 않습니다. 화면과 같은 공개 범위입니다.
        checks = [{"label": row["label"], "requirements": row["requirement"],
                   "available": row["satisfied"], "missing": [] if row["satisfied"] else [row["requirement"]]}
                  for row in equipment_rows(self)]
        return {"services": services, "services_complete": True, "equipment_checks": checks}

    def confirmed_information(self, scope, message=""):
        """게임이 답을 가진 조회는 생성 문장으로 다시 추측하지 않고 표시합니다.

        NPC의 성격·사건 대화는 이 경로에 들어오지 않습니다. 조회만 하므로 구매,
        장비 사용·회복·의뢰 수락 효과를 실행하지 않으며, 미공개 조건도 읽지 않습니다.
        """
        if scope == "directions":
            guides = self.delivered_guidance(self.npc)
            if not guides:
                return "아직 이 상대에게 다음 방문지를 안내받지 않았습니다. 현재 이야기를 먼저 들어보세요."
            # 이미 전달한 방문 기록의 조회입니다. 새 길을 열거나 NPC 발언을 만들어내지 않습니다.
            guide = guides[-1]
            return ("**" + guide["source_npc_name"] + "이 알려준 방문지**\n\n"
                    + guide["room_name"] + " · " + guide["target_npc_name"]
                    + "\n\n" + guide["purpose"])
        focused = focused_information(self, scope, message)
        if focused is not None:
            return focused
        conditions = self.dialogue_affordances()
        if scope == "services":
            rows = ["**현재 이용할 수 있는 도움 · " + self.current_place()["name"] + "**"]
            rows += ["- " + row["label"] for row in conditions["services"]]
            rows.append("목록에 없는 서비스는 현재 이용할 수 없습니다." if conditions["services"] else
                        "이곳에서 지금 이용할 수 있는 판매·휴식 항목은 없습니다.")
            return "\n\n".join(rows)
        owned = " · ".join(self.content.items[key]["name"] + " " + str(count) + "개"
                            for key, count in self.engine.s.inventory.items() if count > 0)
        if scope == "requirements":
            rows = ["**현재 확인한 장비 조건**", "현재 가방: " + (owned or "없음")]
            rows += [row["label"] + "\n" + row["requirements"] for row in conditions["equipment_checks"]]
            if not conditions["equipment_checks"]:
                rows.append("아직 이 장소에서 확인한 장비 조건이 없습니다. 필요한 물건이 없다는 뜻은 아닙니다. "
                            "이동할 곳을 살펴본 뒤 그 행동의 장비 조건을 확인하세요.")
            return "\n\n".join(rows)
        return ("**현재 상태**\n\n현재 장소: " + self.current_place()["name"]
                + f"\n\nHP {self.engine.hero.hp}/{self.engine.hero.max_hp} · 소지금 " + fmt_money(self.engine.s.gold_cp)
                + "\n\n현재 가방: " + (owned or "없음"))

    def _capture_developer_checkpoint(self, label):
        """최근 20개 기록에 현재 접속자의 상태·대화·판정 대기·난수 상태를 함께 보관합니다."""
        if self._restoring_developer_snapshot:
            return
        names = ("messages", "histories", "audit", "revision", "turn_records", "diagnostics_meta",
                 "transaction_ledger", "npc", "pending", "last_result", "debug", "last_status", "suggestions",
                 "scene_focus_explicit", "last_generation_seconds", "action_generation_calls", "_failed_dialogue",
                 "turn_intent", "turn_needs_information", "turn_needs_state", "last_input_kind")
        self._developer_snapshot_serial += 1
        self._developer_snapshots.append({"id": str(self._developer_snapshot_serial), "label": str(label)[:100],
            "state": deepcopy(self.engine.s), "rng": self.engine.dice.rng.getstate(),
            "engine_checkpoint": deepcopy(self.engine.checkpoint), "engine_checkpoint_label": self.engine.checkpoint_label,
            "session": {key: deepcopy(getattr(self, key)) for key in names}})
        self._developer_snapshots = self._developer_snapshots[-20:]

    def developer_checkpoints(self):
        self.engine.require_developer()
        return [{"id": row["id"], "label": row["label"]} for row in self._developer_snapshots]

    def developer_restore_checkpoint(self, identifier):
        self.engine.require_developer()
        row = next((item for item in self._developer_snapshots if item["id"] == str(identifier)), None)
        if row is None:
            raise ValueError("되돌릴 기록을 다시 선택하세요.")
        self.cancel_recommendations()
        dev_audit = [deepcopy(item) for item in self.audit if item.get("kind") == "developer"]
        self._restoring_developer_snapshot = True
        try:
            self.engine.s = deepcopy(row["state"])
            self.engine.dice.rng.setstate(row["rng"])
            self.engine.checkpoint = deepcopy(row["engine_checkpoint"])
            self.engine.checkpoint_label = row["engine_checkpoint_label"]
            for key, value in row["session"].items():
                setattr(self, key, deepcopy(value))
            self.engine.developer, self.engine.next_success = True, False
            self.engine.pending_roll_override = None
            self.engine.s.vars["developer_used"] = True
            self.revision += 1
            if self.pending:
                self.pending["revision"] = self.revision
            if self._failed_dialogue:
                self._failed_dialogue["revision"] = self.revision
            self.audit.extend(item for item in dev_audit if item not in self.audit)
            self.audit.append({"kind": "developer", "text": "개발자 조작으로 선택한 기록을 복구했습니다.",
                               "data": {"checkpoint_id": row["id"], "label": row["label"]}})
            self._developer_snapshots = [item for item in self._developer_snapshots if int(item["id"]) <= int(row["id"])]
            if self.api and callable(self.api.get("create_memory_session")):
                if callable(self.api.get("close_memory_session")):
                    self.api["close_memory_session"](self.memory_session)
                self.memory_session = self.api["create_memory_session"]()
            if self.api and callable(self.api.get("save_dialogue_history")):
                for speaker, history in self.histories.items():
                    self.api["save_dialogue_history"](self.memory_session, speaker, history)
            self._active_route_diagnostics = None
            self.last_status = "선택한 기록으로 되돌렸습니다. 개발자 사용 기록은 유지됩니다."
        finally:
            self._restoring_developer_snapshot = False
        self._capture_developer_checkpoint("복구 · " + row["label"])

    def _campaign_dialogues(self):
        """공개 카드와 실제 표시된 NPC 직접 발언을 구분합니다. 이전 저장본도 호환합니다."""
        completed = set(self.engine.s.vars.get("campaign_dialogues", [])) & CONVERSATION_MILESTONES
        for history in self.histories.values():
            for row in history:
                if (row.get("role") != "assistant" or row.get("knowledge_only") is True
                        or row.get("generation_status", "completed") != "completed"):
                    continue
                for block in row.get("blocks") or [row]:
                    if (isinstance(block, dict) and block.get("kind") == "speech"
                            and block.get("speaker_id") in CONVERSATION_MILESTONES
                            and str(block.get("text", block.get("content", ""))).strip()):
                        completed.add(block["speaker_id"])
        return completed

    def progression_context(self):
        """수락·인형 전달·성공한 직접 대화로 사용자 지정 방문 순서를 안내합니다."""
        s = self.engine.s
        stage = pending_stage(s, self._campaign_dialogues())
        details = {
            "sera_request": ("세라의 가족 수색", "세라의 사정을 듣고 수색을 도울지 결정하세요.", "세라", "sq_accept_help"),
            "sera_doll": ("세라가 맡기는 인형", "수락한 수색을 위해 세라가 맡기는 인형과 용도를 확인하세요.", "세라", "sq_take_doll"),
            "halvik_visit": ("촌장의 이야기", "인형을 맡긴 세라가 알려준 촌장 할빅을 찾아가세요.", "할빅", "meet:scn-s1-npc-halvik"),
            "halvik_request": ("마을의 사정과 탐사 부탁", "할빅에게 우물 조사 부탁을 듣고 수락 여부를 결정하세요.", "할빅", "hh_accept"),
            "smith_visit": ("대장간에서 탐사 준비", "대장장이 그란과 이야기하세요. 장비 구매는 선택입니다.", "그란", "meet:scn-m-npc-smith"),
            "inn_visit": ("여관에서 준비 마무리", "여관 주인 오단과 이야기하세요. 숙박과 휴식은 선택입니다.", "오단", "meet:scn-m-npc-innkeeper"),
            "mera_visit": ("우물지기 메라의 이야기", "우물가의 메라에게 이야기를 들은 뒤 입구를 확인하세요.", "메라", "meet:scn-s1-npc-mera"),
            "well_exploration": ("우물 아래 조사", "메라의 이야기를 들었다면 우물 입구를 확인하고 아래를 조사하세요.", "", "ws_open_well"),
            "exploration": ("실종자와 물길 조사", "현재 장소에서 확인한 길과 단서를 따라 실종자와 물길을 조사하세요.", "", ""),
        }
        title, summary, name, action = details[stage]
        if stage == "sera_request" and not any(row["request_id"] == "sq_accept_help" for row in self.request_view()):
            title, summary, action = "세라와 이야기", "눈앞의 세라에게 이야기를 들어보세요.", "offer:sq_accept_help"
        cards = self.npc_cards()
        next_card = cards.get(name, {})
        next_room = self._person_place(next_card) if next_card else (s.room if s.scene > 1 else "well_side")
        briefed = any(row["request_id"] == "hh_accept" for row in self.request_view())
        if stage == "halvik_request" and not briefed:
            title, summary, action = "촌장의 사정 듣기", "할빅의 부탁을 먼저 확인하세요.", "offer:hh_accept"
        goal = "현재 질문에 먼저 답하고 필요한 다음 조사 대상을 자연스럽게 안내한다."
        if stage == "sera_request":
            goal = "가족 수색을 부탁하는 현재 사정을 설명한다. 수락 전 인형이 전달됐다고 말하지 않는다."
        elif stage == "sera_doll":
            goal = "수락한 수색을 위한 인형의 전달 조건과 용도를 설명한다. 실제 전달은 엔진 기록을 따른다."
        elif stage == "halvik_visit":
            goal = "실제 인형 전달과 용도를 설명하고 마을의 사정을 들으러 촌장 할빅에게 안내한다."
        elif stage == "halvik_request":
            goal = "촌장으로서 마을 사정과 우물 조사 부탁을 설명한다. 수락·선금 지급은 실제 완료 기록을 따른다."
        elif stage == "smith_visit":
            goal = ("대장간의 장비·수리·거래 역할로 응대하고 다음으로 여관 주인 오단과 이야기하도록 안내한다. 구매는 선택이며 새 의뢰를 만들지 않는다."
                    if self.npc == "그란" else "탐사 수락에 반응하고 대장장이 그란에게 준비에 관해 물어보도록 안내한다.")
        elif stage == "inn_visit":
            goal = ("숙박과 휴식에 관해 응대하고 다음으로 우물지기 메라와 이야기하도록 안내한다. 숙박·휴식은 선택이며 새 의뢰를 만들지 않는다."
                    if self.npc == "오단" else "준비를 도와줄 여관 주인 오단에게 안내한다. 구매·숙박·휴식을 완료했다고 가정하지 않는다.")
        elif stage == "mera_visit":
            goal = ("우물지기로서 아는 범위에서 응대하고 이야기를 마치면 우물 입구 확인과 아래 조사로 안내한다. 입구 개방·하강을 이미 실행했다고 말하지 않는다."
                    if self.npc == "메라" else "우물가의 메라에게 이야기를 듣도록 안내한다. 우물 안으로 먼저 내려가도록 안내하지 않는다.")
        current = cards.get(self.npc, {})
        first = bool(current) and self.dialogue_state(self.npc).get("phase") == "first_contact"
        guidance_name, guidance_room = name, next_room
        if (stage, self.npc) in {("smith_visit", "그란"), ("inn_visit", "오단")}:
            guidance_name = "오단" if stage == "smith_visit" else "메라"
            guidance_room = self._person_place(cards[guidance_name])
        elif stage == "mera_visit" and self.npc == "메라":
            guidance_name, guidance_room = "", "well_side"
        guidance_place = MARKET_SPACES[guidance_room]["name"] if guidance_room in MARKET_SPACES else self.content.room_name(guidance_room)
        guidance = [{"room_id": guidance_room, "room_name": guidance_place,
                     "purpose": "preparation" if stage in {"smith_visit", "inn_visit"} else "investigation"}]
        if guidance_name:
            guidance[0]["npc_name"] = guidance_name
        introduction_pending = bool(next_card and not self._introduced(next_card["id"])
                                    and not self._person_present(next_card))
        if introduction_pending:
            title = "다음 조사와 준비 이야기"
            summary = "현재 상대에게 다음에 누구의 도움을 받으면 좋을지 물어보세요."
        return {"stage": stage, "title": title, "summary": summary, "next_step": summary,
                "next_npc": name, "next_action_id": action, "next_room_id": next_room,
                "response_goal": goal, "allowed_guidance": guidance, "reference_required": False,
                "introduction_pending": introduction_pending,
                "first_contact_guidance": first and self.npc in {"그란", "오단", "메라"},
                "halvik_briefed": briefed, "optional_npcs": []}

    def _story_room_allowed(self, room_id):
        if self.engine.s.scene != 1 or room_id == self.engine.s.room:
            return True
        minimum = ROOM_STAGE.get(room_id)
        if minimum is None:
            return True
        if not reached(pending_stage(self.engine.s, self._campaign_dialogues()), minimum):
            return False
        target = next((cid for cid, stage in NPC_STAGE.items() if stage == minimum), None)
        return target is None or self._introduced(target)

    def _story_person_allowed(self, card):
        if self.engine.s.scene != 1:
            return True
        # 명시적 개발자 이동을 한 현장은 조사할 수 있습니다. 모드를 켰다는
        # 이유만으로 다른 장소의 인물까지 일반 선택지에 노출하지는 않습니다.
        if self.engine.developer and self._person_present(card):
            return True
        required = NPC_STAGE.get(card.get("id"))
        return (reached(pending_stage(self.engine.s, self._campaign_dialogues()), required)
                and self._introduced(card["id"])) if required else True

    def _introduced(self, identifier):
        """원본 인물을 이미 만났거나, 현재 진행에서 실제로 소개를 들었는지 읽습니다."""
        return (identifier not in INTRODUCTION_SOURCES or identifier in self.engine.s.shown
                or identifier in self.engine.s.vars.get("campaign_introductions", {}))

    def _publish_guidance(self, source_npc_id):
        """완료한 만남 다음의 방문 안내를 별도 사건으로 표시하고 경로를 엽니다.

        모델 문장에 이름이 들어갔다는 이유로 해금하지 않습니다. 안내 자체를
        화면에 전달한 뒤 그 메시지와 출처 ID를 저장하므로 부정·추측과 구별됩니다.
        """
        source = self.content.cards.get(source_npc_id, {})
        if not source or not self._can_address(source):
            return
        if source_npc_id in CONVERSATION_MILESTONES and source_npc_id not in self._campaign_dialogues():
            return
        for destination in self.progression_context()["allowed_guidance"]:
            target = self.npc_cards().get(destination.get("npc_name"), {})
            target_id = target.get("id")
            if (not target_id or INTRODUCTION_SOURCES.get(target_id) != source_npc_id
                    or self._introduced(target_id)):
                continue
            purpose = ("탐사에 필요한 준비와 시설 이용에 관해 이야기를 들어보세요."
                       if destination["purpose"] == "preparation"
                       else "마을의 사정과 우물 조사에 관해 이야기를 들어보세요.")
            identifier = f"guidance:{source_npc_id}:{target_id}"
            receipt = {"guidance_id": identifier, "source_npc_id": source_npc_id,
                       "target_npc_id": target_id, "room_id": destination["room_id"],
                       "message_index": len(self.messages)}
            text = ("**새 방문 안내 · " + npc_name(source) + "**\n\n"
                    + destination["room_name"] + " · " + npc_name(target) + "\n\n" + purpose)
            self.append_message({"role": "assistant", "kind": "guidance", "speaker": "",
                                  "content": text, **deepcopy(receipt)})
            self.engine.s.vars.setdefault("campaign_introductions", {})[target_id] = receipt
            self.audit.append({"kind": "guidance", "text": text, "data": deepcopy(receipt)})
            self._update_discovery()

    def delivered_guidance(self, speaker):
        """현재 NPC가 이미 화면으로 전달한 안내만 다시 물어볼 근거로 돌려줍니다."""
        source = self.npc_cards().get(speaker, {})
        result = []
        for receipt in self.engine.s.vars.get("campaign_introductions", {}).values():
            if receipt.get("source_npc_id") != source.get("id"):
                continue
            index = receipt.get("message_index")
            if type(index) is not int or not 0 <= index < len(self.messages):
                continue
            message = self.messages[index]
            if (message.get("kind") != "guidance" or not receipt.get("guidance_id")
                    or any(message.get(key) != receipt.get(key) for key in
                           ("guidance_id", "source_npc_id", "target_npc_id", "room_id"))):
                continue
            target = self.content.cards.get(receipt["target_npc_id"], {})
            room_id = receipt["room_id"]
            if not npc_name(target) or self._person_place(target) != room_id:
                continue
            room_name = (MARKET_SPACES[room_id]["name"] if room_id in MARKET_SPACES
                         else self.content.room_name(room_id))
            result.append({**deepcopy(receipt), "source_npc_name": npc_name(source),
                           "target_npc_name": npc_name(target), "room_name": room_name,
                           "purpose": message["content"].rsplit("\n\n", 1)[-1]})
        return result

    def _story_choice_allowed(self, identifier):
        if identifier.startswith("npc:attack:"):
            card = self.content.cards.get(identifier[len("npc:attack:"):], {})
            if card and (not self._person_present(card) or not self._story_person_allowed(card)):
                return False
        definition = self.content.choices.get(identifier, {})
        owner = self.npc_cards().get(definition.get("target_npc")) or next((person for person in self.npc_cards().values()
            if identifier in (person.get("dialogue") or {}).get("action_ids", [])), None)
        if owner and (not self._person_present(owner) or not self._story_person_allowed(owner)):
            return False
        place = self.current_place()["id"]
        if self.engine.s.room == "village_market":
            if (identifier in {"mk_smith", "mk_goods"} or identifier.startswith("buy:")) and place != "village_market:forge":
                return False
            if identifier in {"mk_innkeeper", "mk_rest"} and place != "village_market:inn":
                return False
        card = next((card for card in self.npc_cards().values()
                     if identifier in {self._approach(card), "talk:" + card["id"], "meet:" + card["id"]}), None)
        if card and not identifier.startswith("meet:") and not self._person_present(card):
            return False
        if self.engine.s.scene != 1:
            return True
        if identifier in EARLY_AMBIENT_ACTIONS:
            return False
        if identifier.startswith("travel:"):
            return self._place_allowed(identifier[7:])
        if identifier.startswith("move:"):
            return self._story_room_allowed(identifier[5:])
        if identifier in {"ws_open_well", "ws_descend"} and self.progression_context()["stage"] != "well_exploration":
            return False
        definition = self.content.choices.get(identifier, {})
        if definition.get("interaction") == "accept" and identifier not in self.engine.s.vars.get("request_disclosures", {}):
            return False
        return not card or (self._story_person_allowed(card)
            and (identifier.startswith("meet:") or self._person_present(card)))

    def discovery_view(self):
        """발견한 장소·인물·현재 행동을 화면과 모델이 같은 기준으로 읽습니다."""
        routes = self.map_locations()
        people = [{"name": row["name"], "card_id": row["card_id"], "here": place["current"],
                   "room_id": place["id"], "room_name": place["name"], "can_meet": row["can_meet"]}
                  for place in routes for row in place["npcs"]
                  if row["card_id"] in self._known_npcs()]
        clues = []
        for identifier in self.engine.player_known_cards(self.engine.hero.id):
            card = self.content.cards.get(identifier, {})
            if card.get("kind") in {"clue", "fact", "rumor"} and isinstance(card.get("player_text"), str):
                title = card.get("title") or card["player_text"].split("\n", 1)[0]
                clues.append({"id": identifier, "title": self._presentation_text(title[:100])})
        return {"current_room": self.current_place(),
                "people": people, "routes": routes,
                "actions": [{"id": row.id, "label": row.label, "check": row.check_text,
                             "requires": bool(self.content.choices.get(row.id, {}).get("requires"))}
                            for row in self.choices() if not row.id.startswith("npc:attack:")],
                "objective": self.progression_context(), "requests": self.request_view(), "clues": clues}

    def allowed_speakers(self):
        """현재 현장에서 공개되고 대화 가능한 원본 인물 ID만 허용합니다."""
        return [{"speaker_id": card["id"], "name": name} for name, card in self.npc_cards().items()
                if self._person_present(card) and card["id"] in self.engine.s.shown
                and self._story_person_allowed(card)
                and self.engine.eligible(card["id"]) and self.engine.npc_can_talk(card["id"])]

    def dialogue_state(self, speaker="", input_kind=None):
        """실제 발화 기록으로 대화 단계를 구분합니다. 인물 설정은 입력 자료가 제공합니다."""
        card = self.npc_cards().get(speaker, {})
        player = {"kind": "player", "id": self.engine.hero.id, "name": self.engine.hero.name}
        spoken = bool(card) and any(row.get("role") == "assistant" and row.get("kind") != "narration"
            and (row.get("speaker_id") == card["id"] or row.get("speaker") == speaker)
            and isinstance(row.get("content"), str) and row["content"].strip() for row in self.messages)
        phase = ("first_contact" if card and not spoken else
                 "action_result" if input_kind == "action" else "continue")
        player_spoken = any(row.get("role") == "user" and row.get("dialogue_turn") is True
                            for row in self.histories.get(speaker, []))
        return {"current_speaker": {"kind": "npc" if card else "gm", "speaker_id": card.get("id"), "name": speaker},
                "listener": player, "performer": deepcopy(player),
                "phase": phase,
                "npc_has_spoken": spoken, "player_has_spoken": player_spoken,
                "contact_state": ("awaiting_player" if card and spoken and not player_spoken else
                                  "talking" if card and spoken else "first_contact" if card else "scene"),
                "performer_scope": "player_input_only",
                "last_input_kind": input_kind or self.last_input_kind,
                "last_confirmed_event_id": (self.last_result or {}).get("event_id"),
                "completed_item_events": [{key: deepcopy(transfer[key]) for key in
                    ("item_id", "quantity", "from", "to") if key in transfer}
                    for receipt in (self.last_result or {}).get("transaction_records", [])
                    if receipt.get("confirmed") is True for transfer in receipt.get("transfers", [])
                    if transfer.get("transfer_completed") is True]}

    def interaction_context(self, speaker):
        """공개된 현재 요청과 조건부 전달의 역할을 엔진 선언에서 읽습니다.

        새 인물 설정이나 RAG 근거를 만들지 않습니다. 실행 가능한 현재 요청 또는
        이미 수락한 요청만 다루고, 숨은 효과·후반 설명은 내보내지 않습니다.
        전달 예정과 전달 완료는 다릅니다. 완료는 현재 확정 영수증이 있을 때만 표시합니다.
        """
        card = self.npc_cards().get(speaker, {})
        if (not card or card.get("id") not in self.engine.s.shown
                or not self._person_present(card)):
            return {"requests": []}
        available = {row.id for row in self.engine.available_choices()}
        giver = {"kind": "npc", "speaker_id": card["id"], "name": speaker}
        recipient = {"kind": "player"}
        receipts = (self.last_result or {}).get("item_handovers", [])
        requests = []
        for identifier, parent in self.content.choices.items():
            accepted = identifier in self.engine.s.done
            if (parent.get("interaction") != "accept" or parent.get("target_npc") != speaker
                    or parent.get("room") != self.engine.s.room
                    or not (identifier in available or accepted)
                    or not accepted and identifier not in self.engine.s.vars.get("request_disclosures", {})):
                continue
            transfers = []
            for child_id in parent.get("auto_followups", [])[:4]:
                child = self.content.choices.get(child_id, {})
                declared = child.get("item_handover") or {}
                if (child.get("interaction") != "handover" or child.get("target_npc") != speaker
                        or child.get("room") != self.engine.s.room or child.get("once") is not True
                        or "check" in child or declared.get("source_card_id") != card["id"]):
                    continue
                item_id = declared.get("item_id")
                if item_id not in self.content.items:
                    continue
                gives = [effect for effect in child.get("effects", []) if effect.get("give") == item_id]
                quantity = gives[0].get("n", 1) if len(gives) == 1 else None
                if type(quantity) is not int or quantity < 1:
                    continue
                receipt = next((row for row in receipts if isinstance(row, dict) and row.get("confirmed") is True
                    and row.get("action_id") == child_id and row.get("item_id") == item_id), None)
                transfer = {"action_id": child_id, "relation": "conditional_handover_to_player",
                    "item_id": item_id, "item_name": self.content.items[item_id].get("name", item_id),
                    "expected_giver": deepcopy(giver), "recipient": deepcopy(recipient),
                    "planned_quantity": quantity, "player_current_quantity": self.engine.s.inventory.get(item_id, 0),
                    "status": "confirmed_completed" if receipt else
                        "current_receipt_unavailable" if accepted else "planned_after_acceptance",
                    "requires_engine_conditions": True}
                if receipt is not None:
                    transfer["confirmed_receipt"] = deepcopy(receipt)
                transfers.append(transfer)
            requests.append({"request_id": identifier, "request_issuer": deepcopy(giver),
                "request_acceptor": deepcopy(recipient), "purpose": parent.get("purpose"),
                "status": "accepted" if accepted else "offered", "conditional_transfers": transfers})
        return {"origin": "public_engine_interaction_declarations", "requests": requests}

    def _append_model_response(self, result, default_npc=""):
        """모델 블록의 ID를 서버 원본으로 다시 확인하고 화자별 메시지를 기록합니다."""
        current_stage("validation", detail="표시할 문장과 화자 확인", npc_name=default_npc)
        allowed = {row["speaker_id"]: row["name"] for row in self.allowed_speakers()}
        blocks = result.get("blocks")
        if blocks is None:
            card = self.npc_cards().get(default_npc, {})
            blocks = [{"kind": "speech" if card else "narration", "speaker_id": card.get("id"),
                       "text": result.get("text", "")}]
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("표시할 대화 블록이 없습니다.")
        checked = []
        for block in blocks:
            if not isinstance(block, dict) or not isinstance(block.get("text"), str) or not block["text"].strip():
                raise ValueError("대화 블록의 본문을 확인할 수 없습니다.")
            kind, identifier = block.get("kind"), block.get("speaker_id")
            if kind == "narration" and identifier in (None, ""):
                identifier, name = None, ""
            elif kind == "speech" and identifier in allowed:
                name = allowed[identifier]
            else:
                raise ValueError("현재 장면에서 허용되지 않은 화자입니다.")
            checked.append({"kind": kind, "speaker_id": identifier, "speaker": name, "text": block["text"].strip()})
        current_stage("render", detail="확인된 대화 블록 반영", block_count=len(checked))
        for block in checked:
            self.append_message({"role": "assistant", "content": block["text"],
                                  **{key: block[key] for key in ("kind", "speaker_id", "speaker")}})
        completed = self._campaign_dialogues()
        for block in checked:
            if (block["kind"] == "speech" and block["speaker_id"] in CONVERSATION_MILESTONES
                    and self._can_address(self.content.cards[block["speaker_id"]])):
                completed.add(block["speaker_id"])
        self.engine.s.vars["campaign_dialogues"] = sorted(completed)
        self._update_discovery()
        return checked

    def _approach(self, card):
        """현재 장면에 실제로 정의된 인물 공개 행동을 찾습니다."""
        return next((row["id"] for row in self.content.choices.values()
            if row.get("room") == card.get("room") and row.get("interaction") == "talk"
            and any(effect.get("show") == card.get("id") for effect in row.get("effects", []))), None)

    def _choice_npc_owner(self, identifier):
        """행동의 상대를 명시된 target_npc 또는 공개 인물 카드의 행동 연결에서 읽습니다."""
        definition = self.content.choices.get(identifier, {})
        declared = self.npc_cards().get(definition.get("target_npc"))
        if declared:
            return declared
        owners = [card for card in self.npc_cards().values()
                  if identifier in (card.get("dialogue") or {}).get("action_ids", [])]
        return owners[0] if len(owners) == 1 else None

    def _conversation_action_owner(self, identifier):
        """판정·거래·열람과 구별되는 실제 대화 행동의 현장 상대만 반환합니다."""
        definition = self.content.choices.get(identifier, {})
        owner = self._choice_npc_owner(identifier)
        if not owner or not self._can_address(owner) or definition.get("check"):
            return None
        if definition.get("interaction") in {"accept", "handover"}:
            return None
        effects = [effect for branch in ("effects", "before", "success", "failure")
                   for effect in definition.get(branch, [])]
        if any(not isinstance(effect, dict) or not set(effect) <= {"say", "show", "flag", "var", "value"}
               for effect in effects):
            return None
        shown_kinds = [self.content.cards.get(effect["show"], {}).get("kind")
                       for effect in effects if "show" in effect]
        # 판매 목록·물품·퍼즐 가사는 정확한 열람/판정 결과로 유지합니다.
        if any(kind not in {"npc", "rumor"} for kind in shown_kinds):
            return None
        speaks = any("say" in effect for effect in effects) or "rumor" in shown_kinds
        return owner if speaks or definition.get("interaction") == "talk" else None

    def nearby_npcs(self):
        return [(person["label"], person["card_id"]) for person in self.visible_people()]

    def _discovery(self):
        """발견은 저장된 사실이며 모델 발언이나 플레이어의 추측으로 추가하지 않습니다."""
        return self.engine.s.vars.setdefault("discovery", {"rooms": [], "npcs": [], "paths": {}})

    def _update_discovery(self, choice_id=None):
        """방문·공개 카드·실행된 시나리오 주석만 발견 기록에 반영합니다."""
        record = self._discovery()
        rooms, people = set(record.get("rooms", [])), set(record.get("npcs", []))
        rooms.update(self.engine.s.visited)
        public = self.engine.player_known_cards(self.engine.hero.id)
        sources = [self.content.cards[cid] for cid in public if cid in self.content.cards]
        sources.extend(self.content.choices[cid] for cid in self.engine.s.done
                       if cid in self.content.choices and self.content.choices[cid].get("discover"))
        if choice_id in self.content.choices:
            sources.append(self.content.choices[choice_id])
        for source in sources:
            if npc_name(source):
                people.add(source["id"])
            annotation = source.get("discover", {})
            if isinstance(annotation, dict):
                rooms.update(rid for rid in annotation.get("rooms", []) if rid in self.content.rooms)
                for cid in annotation.get("npcs", []):
                    card = self.content.cards.get(cid, {})
                    if not self._story_person_allowed(card):
                        continue
                    if (npc_name(card) and card.get("owner") == "all"
                            and card.get("reveal_after", 1) <= self.engine.s.scene
                            and (not card.get("req_flag") or card["req_flag"] in self.engine.s.flags)):
                        people.add(cid)
                        rooms.add(card["room"])
        current = self.content.rooms[self.engine.s.room]
        paths = record.setdefault("paths", {})
        for number, destination in enumerate(self.content.exits(self.engine.s.room), 1):
            if self.engine.cond(current.get("exit_requires", {}).get(destination)):
                name = self.content.room_name(destination)
                paths[destination] = current.get("exit_labels", {}).get(destination) or (
                    name if name in current.get("look_text", "") else f"눈앞에 이어진 길 {number}")
        # 완료된 이전 단계가 안내한 다음 인물과 목적지만 해금합니다.
        progress = self.progression_context()
        target = self.npc_cards().get(progress["next_npc"], {})
        if target and self._story_person_allowed(target):
            people.add(target["id"])
            rooms.add(target["room"])
        people = {cid for cid in people if cid in self.content.cards
                  and self._story_person_allowed(self.content.cards[cid])}
        rooms = {rid for rid in rooms if self._story_room_allowed(rid)}
        record["paths"] = {rid: label for rid, label in paths.items() if self._story_room_allowed(rid)}
        record.update(rooms=sorted(rooms), npcs=sorted(people))

    def _known_npcs(self):
        return {cid for cid in set(self._discovery().get("npcs", [])) | set(self.engine.s.shown)
                if npc_name(self.content.cards.get(cid, {}))
                and self._story_person_allowed(self.content.cards[cid])}

    def _room_label(self, room_id):
        record = self._discovery()
        if room_id in record.get("rooms", []) or room_id in self.engine.s.visited:
            return self.content.room_name(room_id)
        return record.get("paths", {}).get(room_id, "아직 알지 못하는 곳")

    def _person_label(self, card):
        return npc_name(card) if card["id"] in self._known_npcs() else "낯선 사람"

    def _public_text(self, text):
        """화면·행동 연결에 전달할 라벨에서 미발견 고유명을 제외합니다."""
        if not isinstance(text, str):
            return text
        for name, card in self.npc_cards().items():
            if card["id"] not in self._known_npcs() and name in text:
                if card["room"] != self.engine.s.room:
                    return None
                text = text.replace(name, self._person_label(card))
        known, _ = self._public_paths()
        for room_id in self.content.rooms:
            name = self.content.room_name(room_id)
            if name in text and room_id not in self.engine.s.visited and room_id not in self._discovery().get("rooms", []):
                if room_id not in known:
                    return None
                text = text.replace(name, self._room_label(room_id))
        return text

    def _presentation_text(self, text):
        """생성용 사본에서만 미발견 고유명을 가립니다. 사건·수치·원문 기록은 보존합니다."""
        if not isinstance(text, str):
            return text
        public = self._public_text(text)
        if public is not None:
            return public
        for name, card in self.npc_cards().items():
            if card["id"] not in self._known_npcs():
                text = text.replace(name, "낯선 사람")
        discovered = set(self._discovery().get("rooms", [])) | set(self.engine.s.visited)
        for room_id in self.content.rooms:
            if room_id not in discovered:
                text = text.replace(self.content.room_name(room_id), self._room_label(room_id))
        return text

    def progression_options(self):
        """조건을 만족한 공개 시나리오의 다음 행동만 대화의 근거로 제공합니다."""
        available = {choice.id: choice for choice in self.choices()}
        result = []
        for option in self._guided_suggestions():
            identifier = option.get("action_id")
            if identifier:
                result.append({"id": identifier, "label": option["text"],
                    "source": {"kind": "progression", "room_id": self.engine.s.room}})
        for row in self.content.rooms[self.engine.s.room].get("suggestions", []):
            identifier = row.get("action_id")
            if (identifier in available and self.engine.cond(row.get("requires"))
                    and (not row.get("npc_name") or row["npc_name"] == self.npc)
                    and self._story_choice_allowed(identifier)):
                result.append({"id": identifier, "label": available[identifier].label,
                    "source": {"kind": "authored_scene_option", "room_id": self.engine.s.room,
                               "trigger": row.get("trigger", "")}})
        return result

    def _guided_suggestions(self):
        """완료 기록에 맞는 다음 단계만 추천합니다. 대사와 거래 결과는 만들지 않습니다."""
        progress = self.progression_context()
        if self.engine.s.scene != 1:
            return []
        if progress.get("introduction_pending"):
            # 다음 인물을 버튼으로 먼저 알려 주지 않고 현재 상대에게 물어봅니다.
            return [{"text": "다음에는 누구에게 이야기를 들어보면 좋을까요?",
                     "kind": "dialogue", "source": "progression"}]
        stage, name = progress["stage"], progress["next_npc"]
        available = {row.id for row in self.choices()}
        action = progress["next_action_id"]
        if action.startswith("offer:"):
            if action not in available:
                return []
            text = "부탁의 내용을 듣는다"
        elif action.startswith("meet:"):
            card = self.npc_cards().get(name, {})
            if self._person_present(card):
                action = self._approach(card) if self._approach(card) in available else "talk:" + card.get("id", "")
            if action not in available:
                return []
            text = name + ("에게 말을 건다" if self._person_present(card) else "을 찾아간다")
        elif stage == "well_exploration":
            if self.engine.s.room != "well_side":
                action, text = "travel:well_side", "준비를 마치고 우물가로 간다"
                if not any(row["id"] == "well_side" and row["reachable"] for row in self.map_locations()):
                    return []
            else:
                action = ("ws_descend" if "well_entrance_known" in self.engine.s.flags
                          else "ws_open_well")
                text = "우물 입구를 확인한다" if action == "ws_open_well" else "우물 아래로 내려간다"
                if action not in available:
                    return []
        elif action in available:
            text = ("세라에게 가족을 찾아보겠다고 약속한다" if stage == "sera_request"
                    else "우물의 물길을 조사하겠다고 할빅에게 답한다")
        else:
            return []
        return [{"text": text, "action_id": action, "source": "progression"}]

    def _public_paths(self):
        """방문·출처 안내·직접 관찰한 출구만 경로 탐색에 사용합니다."""
        record = self._discovery()
        known = {rid for rid in set(self.engine.s.visited) | set(record.get("rooms", []))
                 | set(record.get("paths", {})) | {self.engine.s.room} if self._story_room_allowed(rid)}
        # 전체 지형을 여는 조작은 보라색 개발자 이동 도구에만 둡니다.
        paths, queue = {self.engine.s.room: []}, deque([self.engine.s.room])
        while queue:
            origin = queue.popleft()
            room = self.content.rooms[origin]
            for destination in self.content.exits(origin):
                if (destination in known and destination not in paths
                        and self.engine.cond(room.get("exit_requires", {}).get(destination))):
                    paths[destination] = paths[origin] + [destination]
                    queue.append(destination)
        return known, paths

    def _travel_blocked(self):
        return bool(self.pending or self.engine.s.pending or self.engine.s.combat or self.engine.s.ended)

    def map_locations(self):
        """공개 장소와 만날 수 있는 인물만 반환하며 실제 이동은 실행하지 않습니다."""
        known, paths = self._public_paths()
        blocked = self._travel_blocked()
        locations = []
        for room_id in self.content.rooms:
            if room_id not in known:
                continue
            reachable = room_id in paths and not blocked
            people = []
            for name, card in self.npc_cards().items():
                if (card["room"] != room_id or not self._story_person_allowed(card) or not self.engine.developer and (
                        card.get("owner") != "all" or card.get("reveal_after", 1) > self.engine.s.scene
                        or card.get("req_flag") and card["req_flag"] not in self.engine.s.flags
                        or card["id"] not in self._known_npcs() and room_id != self.engine.s.room)):
                    continue
                people.append({"name": self._person_label(card), "card_id": card["id"],
                               "can_meet": reachable and self.engine.npc_can_talk(card["id"])})
            locations.append({"id": room_id, "name": self._room_label(room_id),
                              "current": room_id == self.engine.s.room, "reachable": reachable, "npcs": people})
        projected = []
        for location in locations:
            if location["id"] != "village_market":
                projected.append({**location, "native_id": location["id"]})
                continue
            for place, info in MARKET_SPACES.items():
                if not self._place_allowed(place):
                    continue
                projected.append({"id": place, "native_id": "village_market", "name": info["name"],
                    "current": self.current_place()["id"] == place,
                    "reachable": location["reachable"],
                    "npcs": [row for row in location["npcs"] if row["card_id"] == info["npc_id"]]})
        return projected

    def _move_local_place(self, place):
        if place not in MARKET_SPACES or self.engine.s.room != "village_market" or not self._place_allowed(place):
            raise ValueError("지금 들어갈 수 있는 장소가 아닙니다.")
        if self._travel_blocked():
            raise ValueError("현재 판정이나 전투를 먼저 마쳐 주세요.")
        if place == self.current_place()["id"]:
            return
        before = self.current_place()["id"]
        self.engine.s.vars["presentation_place"] = place
        self.npc = ""
        self._record([Event("move", "== " + MARKET_SPACES[place]["name"] + " ==",
                            {"from_place": before, "to_place": place, "native_room": "village_market"})],
                     player_input=MARKET_SPACES[place]["name"] + "으로 이동한다")
        self._auto_meet_target()

    def _auto_meet_target(self):
        """목적지에서 현재 안내 대상과 첫 만남만 연결합니다. 수락·지급은 하지 않습니다."""
        if self._travel_transit or self._travel_blocked() or any(row.get("kind") in {"check", "damage", "trap", "combat", "end"}
                for row in (self.last_result or {}).get("public_event_records", [])):
            return
        name = self.progression_context().get("next_npc")
        card = self.npc_cards().get(name, {})
        if not card or card["id"] in self.engine.s.shown or not self._can_address(card):
            return
        approach = self._approach(card)
        if approach in {row.id for row in self.choices()}:
            self.propose(approach, direct=True)

    def travel_to(self, room_id, npc_name=None):
        """공개 경로를 한 칸씩 실제 실행하고, 요청한 인물은 도착 후 접근합니다."""
        if self._travel_blocked():
            raise ValueError("현재 전투·판정·확인 대기를 먼저 마쳐 주세요.")
        room_id = "village_market:forge" if room_id == "village_market" else room_id
        locations = {row["id"]: row for row in self.map_locations()}
        destination = locations.get(room_id)
        if not destination or not destination["reachable"]:
            raise ValueError("현재 공개된 경로로 갈 수 없는 장소입니다.")
        card = next((card for card in self.npc_cards().values() if npc_name in
                     {card["id"], self._person_label(card)}), None) if npc_name else None
        if npc_name and (not card or not any(person["card_id"] == card["id"] and person["can_meet"] for person in destination["npcs"])):
            raise ValueError("그 장소에서 만날 수 있는 인물이 아닙니다.")
        native = destination.get("native_id", room_id)
        _, paths = self._public_paths()
        executed = []
        steps = paths[native]
        try:
            for index, step in enumerate(steps):
                action_id = "move:" + step
                if self._travel_blocked() or action_id not in {choice.id for choice in self.choices()}:
                    self.last_status = "이동 중 상황이 바뀌어 멈췄습니다. 현재 행동을 확인해 주세요."
                    return False
                self._travel_transit = index < len(steps) - 1
                if step == "village_market" and room_id in MARKET_SPACES:
                    self.engine.s.vars["presentation_place"] = room_id
                self.propose(action_id, direct=True)
                executed.append(action_id)
                if self._travel_blocked() or any(row.get("kind") in {"check", "damage", "trap", "combat", "end"}
                        for row in (self.last_result or {}).get("public_event_records", [])):
                    self.last_status = "이동 중 사건이 발생해 현재 장소에서 멈췄습니다."
                    return False
        finally:
            self._travel_transit = False
            self.debug["지도 이동"] = {"실행한 행동": executed, "도착": self.engine.s.room == native}
        if room_id in MARKET_SPACES and self.current_place()["id"] != room_id:
            self._move_local_place(room_id)
        if npc_name:
            if (card.get("room") != self.engine.s.room or not self.engine.eligible(card["id"])
                    or not self.engine.npc_can_talk(card["id"])):
                raise ValueError("도착 후 대화할 수 없는 상태로 바뀌었습니다.")
            choices = {choice.id for choice in self.choices()}
            approach = self._approach(card)
            talk = "talk:" + card["id"]
            action_id = approach if approach in choices else talk if talk in choices else None
            if self.npc_cards().get(self.npc, {}).get("id") == card["id"] and card["id"] in self.engine.s.shown:
                action_id = None
            elif action_id:
                self.propose(action_id, direct=True)
                executed.append(action_id)
            else:
                self.talk_to(card["id"])
            if self._travel_blocked():
                self.debug["지도 이동"] = {"실행한 행동": executed, "도착": True}
                return False
        self.debug["지도 이동"] = {"실행한 행동": executed, "도착": True, "대화 상대": self.npc}
        self.refresh_suggestions()
        self._auto_meet_target()
        return True

    def visit_npc(self, name):
        """공개 지도에서 인물을 다시 확인한 뒤 이동과 만남을 연결합니다."""
        for room in self.map_locations():
            person = next((person for person in room["npcs"] if name in {person["name"], person["card_id"]} and person["can_meet"]), None)
            if person:
                return self.travel_to(room["id"], npc_name=person["card_id"])
        raise ValueError("현재 공개된 경로로 만날 수 없는 인물입니다.")

    def visible_people(self):
        """현재 장소의 공개 조건을 통과한 사람과 실제 접근 행동만 노출합니다."""
        available = {choice.id for choice in self.choices()}
        people = []
        for name, card in self.npc_cards().items():
            if (not self._person_present(card) or not self._story_person_allowed(card)
                    or not self.engine.eligible(card["id"]) or not self.engine.npc_can_talk(card["id"])):
                continue
            approach = self._approach(card)
            people.append({"card_id": card["id"], "name": self._person_label(card), "label": self._person_label(card),
                "presence": self._person_label(card) + "이(가) 있다.",
                "approach_action": (approach if approach in available
                                    else "talk:" + card["id"] if "talk:" + card["id"] in available else None),
                "met": card["id"] in self.engine.s.shown})
        return people

    def scene_context(self):
        """엔진의 위치·관찰·인물·출구만 전달합니다. 카드 본문은 팀 입력을 통해 받습니다."""
        room = self.content.rooms[self.engine.s.room]
        aftermath = []
        for name, card in self.npc_cards().items():
            if self._person_present(card):
                status = self.engine.npc_state(card["id"])
                if not status["can_talk"]:
                    aftermath.append({"name": self._person_label(card), "status": status["status"], "hp": status["hp"]})
        place = self.current_place()
        return {"room_id": self.engine.s.room, "place_id": place["id"], "room_name": place["name"],
                "observation": self._public_text(room.get("look_text", "")) or "", "people": self.visible_people(),
                "aftermath": aftermath,
                "exits": [{"id": choice.id, "destination": self._room_label(choice.id[5:])}
                          for choice in self.choices() if choice.id.startswith("move:")]}

    def scene_opportunities(self, exclude_card_ids=()):
        context = self.scene_context()
        parts = [person["presence"] for person in context["people"]
                 if person["presence"] and person["card_id"] not in exclude_card_ids]
        for person in context["aftermath"]:
            label = {"dead": "숨진 채 쓰러져 있다", "unconscious": "의식을 잃고 쓰러져 있다"}.get(person["status"], "전투 중이다")
            parts.append(person["name"] + "는 " + label + ".")
        if context["exits"]:
            parts.append("이어진 길: " + " · ".join(item["destination"] for item in context["exits"]) + ".")
        return "\n\n".join(parts)

    def _asked_recently(self, text, player_message=None):
        """현재 인물에게 이미 한 말은 띄어쓰기·문장부호가 달라도 다시 추천하지 않습니다."""
        normalize = lambda value: re.sub(r"[\W_]+", "", re.sub(r"^\s*\[(?:대화|행동|이동)\]\s*", "", value)).casefold()
        previous = [row["content"] for row in self.histories.get(self.npc, [])[-12:]
                    if row.get("role") == "user" and isinstance(row.get("content"), str)]
        if player_message:
            previous.append(player_message)
        if self.messages and self.messages[-1]["role"] == "user":
            previous.append(self.messages[-1]["content"])
        target = normalize(text)
        return bool(target) and target in {normalize(value) for value in previous}

    def _suggestion_option(self, value):
        """표시 종류와 실제 전송할 원문을 나누고 행동 종류는 엔진 정의를 따릅니다."""
        option = dict(value) if isinstance(value, dict) else {"text": value, "source": "model"}
        text = re.sub(r"^\s*\[(?:대화|행동|이동)\]\s*", "", option.get("text", "")).strip()
        action_id, kind = option.get("action_id", ""), option.get("kind", "dialogue")
        if action_id:
            definition = self.content.choices.get(action_id, {})
            if action_id.startswith("move:"):
                kind = "move"
            elif action_id.startswith("meet:"):
                card = self.content.cards.get(action_id[5:], {})
                kind = "dialogue" if self._person_present(card) else "move"
            elif action_id.startswith("travel:"):
                kind = "move"
            else:
                kind = "dialogue" if action_id.startswith("talk:") or definition.get("interaction") == "talk" else "action"
        if not isinstance(kind, str) or kind not in {"dialogue", "action", "move"}:
            kind = "dialogue"
        label = option.get("label") or text
        label = re.sub(r"^\s*\[(?:대화|행동|이동|판정)\]\s*", "", label).strip()
        is_check = bool(action_id and self.content.choices.get(action_id, {}).get("check"))
        tag = "판정" if is_check else {"dialogue": "대화", "action": "행동", "move": "이동"}[kind]
        return {**option, "kind": kind, "text": text, "is_check": is_check,
                "label": "[" + tag + "] " + label}

    def refresh_suggestions(self, generated=None, player_message=None):
        """첫 만남 선택지와 현재 턴의 새 추천을 구분하고 유효한 진행 선택지를 섞습니다."""
        if self.pending or self.engine.s.combat or self.engine.s.ended:
            self.suggestions = []
            return
        choices = self.choices()
        available = {choice.id for choice in choices}
        available.update("meet:" + person["card_id"] for room in self.map_locations()
                         for person in room["npcs"] if person["can_meet"] and person["name"] != self.npc)
        used = self.engine.s.vars.get("used_suggestion_groups", [])
        rows = []
        group = None
        # 실패·생성 중인 질문은 첫 만남을 완료한 것으로 세지 않습니다.
        # 이전 저장 파일은 성공 뒤에만 사용자 행을 쌓았으므로 상태가 없으면 완료로 봅니다.
        discussed = any(row.get("role") == "user" and (row.get("dialogue_turn") or "memory_turn" in row)
                        and row.get("generation_status", "completed") == "completed"
                        for row in self.histories.get(self.npc, []))
        for option in self.content.rooms[self.engine.s.room].get("suggestions", []):
            if any(row["id"] not in self.engine.s.vars.get("request_disclosures", {})
                   for row in self._request_candidates()):
                continue
            if not self._campaign_suggestion_allowed(option):
                continue
            if option.get("npc_name") and option["npc_name"] != self.npc:
                continue
            if option.get("once_key") in used or not self.engine.cond(option.get("requires")):
                continue
            if option.get("action_id") and option["action_id"] not in available:
                continue
            if discussed and option.get("trigger") in {"first_contact", "first_offer"}:
                continue
            if group is not None and option.get("once_key") != group:
                continue
            group = option.get("once_key")
            rows.append({**option, "source": "scene"})
        fresh = list(generated or [])
        target_count = (1 if self.last_generation_seconds > 6 else 3) if fresh else 2
        candidates = self._guided_suggestions() + fresh + rows
        latest_action = ((self.last_result or {}).get("engine_delta") or {}).get("executed_action_id")
        selected, seen = [], set()
        for value in candidates:
            option = dict(value) if isinstance(value, dict) else {"label": value, "text": value, "source": "model"}
            action_id, text = option.get("action_id"), option.get("text", "")
            if not self._campaign_suggestion_allowed(option):
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            if self._public_text(text) != text:
                continue
            if action_id and action_id not in available:
                continue
            if action_id and action_id == latest_action:
                continue
            if not action_id and self._asked_recently(text, player_message):
                continue
            key = action_id or re.sub(r"[\W_]+", "", text).casefold()
            if key not in seen:
                selected.append(self._suggestion_option(option))
                seen.add(key)
        # 실행하지 않은 물리 행동을 만들어 빈 칸을 채우지는 않습니다.
        self.suggestions = selected[:target_count]

    def _campaign_suggestion_allowed(self, option):
        """익명 주민 소문 고정 행동을 추천·저장된 버튼에서 제외합니다."""
        if self.engine.s.scene != 1:
            return True
        identifier = option.get("action_id", "") if isinstance(option, dict) else ""
        if identifier and not self._story_choice_allowed(identifier):
            return False
        text = option.get("text", option.get("label", "")) if isinstance(option, dict) else str(option)
        if not isinstance(text, str):
            return False
        if self._public_text(text) != text:
            return False
        named = any(name in text for name in self.npc_cards())
        return named or not (re.search(r"(?:마을\s*사람|주민들?).*(?:묻|물어|소문)", text)
                             or re.search(r"소문(?:을|\s)*\s*(?:듣|들어|들으|모으|모은|수집|알아)", text))

    def _consume_suggestions(self, selected=None):
        # 자유 입력/거절만으로 게임 상태에 빈 사용 기록을 새로 만들지 않습니다.
        if selected and selected.get("once_key"):
            groups = self.engine.s.vars.setdefault("used_suggestion_groups", [])
            if selected["once_key"] not in groups:
                groups.append(selected["once_key"])
        self.suggestions = []

    def choose_suggestion(self, index):
        """추천 버튼도 같은 사용자 입력·게임 행동 경로로 보냅니다."""
        if type(index) is not int or not 0 <= index < len(self.suggestions):
            raise ValueError("현재 표시된 선택지를 다시 골라 주세요.")
        option = dict(self.suggestions[index])
        if not self._campaign_suggestion_allowed(option):
            raise ValueError("현재 진행 순서에 맞는 선택지를 다시 골라 주세요.")
        revision_before = self.revision
        once_key = option.get("once_key")
        already_used = once_key in self.engine.s.vars.get("used_suggestion_groups", [])
        self._consume_suggestions(option)
        if option.get("action_id"):
            self.last_input_kind = "action"
            self.append_message({"role": "user", "content": option["text"], "input_kind": "action",
                                  "speaker_id": self.engine.hero.id, "speaker": self.engine.hero.name})
            if option["action_id"].startswith("meet:"):
                self.visit_npc(option["action_id"][5:])
            else:
                self.propose(option["action_id"], direct=True)
            self.refresh_suggestions()
        else:
            try:
                self.send(option["text"], input_kind="action" if option.get("kind") in {"action", "move"} else "speech")
            except Exception:
                # 대답이 나오기 전에 실패한 순수 대화만 선택 그룹을 되돌립니다.
                # 이동·거래 등 이미 확정된 엔진 행동과 다른 그룹의 기록은 유지합니다.
                if (option.get("kind", "dialogue") == "dialogue" and once_key and not already_used
                        and self.revision == revision_before and self.retryable_dialogue_turn_id() is not None):
                    groups = self.engine.s.vars.get("used_suggestion_groups", [])
                    self.engine.s.vars["used_suggestion_groups"] = [key for key in groups if key != once_key]
                    self._failed_dialogue["suggestion_once_key"] = once_key
                raise

    def quest_journal(self):
        """시나리오에 정의된 실제 조건을 읽어 목표와 현재 단계를 보여 줍니다."""
        lines = []
        disclosed = {row["title"] for row in self.request_view()}
        for scene in self.content.scenes:
            for quest in scene.get("quest_log", {}).values():
                condition = quest.get("accepted_when")
                accepted = bool(condition) and self.engine.cond(condition)
                done_condition = quest.get("completed_when")
                done = bool(done_condition) and self.engine.cond(done_condition)
                if not self.engine.developer:
                    if quest["title"] not in disclosed:
                        continue
                    visible = quest.get("visible_when")
                    if (visible is not None and not self.engine.cond(visible)
                            or visible is None and not (accepted or done)):
                        continue
                lines.append(("✓ " if done else "진행 · " if accepted else "의뢰 · ") + quest["title"])
                lines.append(quest.get("summary", ""))
                for step in quest.get("steps", []):
                    if step.get("when") and self.engine.cond(step["when"]):
                        lines.append("  ✓ " + step["label"])
        return "\n\n".join(lines)

    def _story_say_texts(self):
        """장면 효과에 선언된 문장만 구분해 주사위·함정의 계산 안내와 나눕니다."""
        if not hasattr(self, "_declared_story_says"):
            declared = set()

            def collect(value):
                if isinstance(value, dict):
                    for field in ("say", "end_scene"):
                        if isinstance(value.get(field), str):
                            declared.add(value[field])
                    for child in value.values():
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)

            collect(self.content.scenes)
            collect(self.content.effect_sets)
            self._declared_story_says = declared
        return self._declared_story_says

    def _record(self, events, transaction_record=None, choice_id=None, player_input="", engine_delta=None,
                opening_scene=False):
        """실제 사건·공개 지식을 기록하고 NPC 반응과 확정 결과를 구분해 표시합니다."""
        started, message_start = time.perf_counter(), len(self.messages)
        delta = deepcopy(engine_delta or {})
        elapsed = delta.get("elapsed_minutes")
        if type(elapsed) is not int:
            elapsed = self.engine.s.minutes - getattr(self, "_last_record_minutes", self.engine.s.minutes)
        delta["elapsed_minutes"] = max(0, elapsed)
        self._last_record_minutes = self.engine.s.minutes
        self.cancel_recommendations()
        events = list(events)
        self._update_discovery(choice_id)
        movement_scene = any(event.kind == "move" for event in events)
        event_context = (deepcopy(delta.get("message_scene_before"))
                         or deepcopy(getattr(self, "_last_record_scene_context", None))
                         or self.message_scene_context()) if movement_scene else self.message_scene_context()
        conversation_owner = self._conversation_action_owner(choice_id) if not movement_scene else None
        if conversation_owner:
            self.npc = npc_name(conversation_owner)
            self.scene_focus_explicit = False
        quiet_transit = self._travel_transit and movement_scene and not any(
            event.kind in {"check", "damage", "trap", "combat", "end"} for event in events)
        opening_text = ""
        if opening_scene and movement_scene:
            room = self.content.rooms[self.engine.s.room]
            intro = room.get("opening_narration")
            if isinstance(intro, str) and intro.strip():
                # 시작 도입은 재방문·일반 대화와 분리해 원문 순서대로 한 번 표시합니다.
                opening_text = intro
                arrival = room.get("arrival_hook")
                if isinstance(arrival, str) and arrival.strip():
                    opening_text += "\n\n" + arrival
        opening_inserted = False
        if movement_scene and not self.engine.s.combat and not quiet_transit:
            room = self.content.rooms[self.engine.s.room]
            key = "arrival_hook:" + self.engine.s.room
            parts = []
            introduced = set()
            if room.get("arrival_hook") and key not in self.engine.s.once_done:
                self.engine.s.once_done.add(key)
                parts.append(room["arrival_hook"])
                if room.get("auto_meet"):
                    introduced.add(room["auto_meet"])
            opportunities = self.scene_opportunities(exclude_card_ids=introduced)
            if opportunities:
                parts.append(opportunities)
            if parts:
                events.append(Event("scene_cue", "\n\n".join(parts)))
        public, public_records, narratives, knowledge = [], [], [], []
        known_cards = set(self.engine.player_known_cards(self.engine.hero.id))
        for event in events:
            if event.kind == "move":
                # 도착 사건 전의 판정·피해와 도착 이후의 발언을 다른 위치로 기록합니다.
                event_context = self.message_scene_context()
            card_id = event.data.get("id") if event.kind == "card" else None
            card = self.content.cards.get(card_id, {}) if card_id else {}
            text = event.text
            displayed = False
            if (card_id in known_cards and card.get("kind") in {"clue", "fact", "rumor"}
                    and not quiet_transit):
                # 카드 전문을 NPC 대사로 만들지 않습니다. 엔진이 현재 플레이어에게
                # 공개한 조건부 뷰의 실제 문장만 별도 지식 메시지로 전달합니다.
                public_text, _ = self.engine._view_text(card_id)
                if isinstance(public_text, str) and public_text.strip():
                    item = {"card_id": card_id, "title": card.get("title") or card.get("label") or "단서",
                            "kind": card["kind"], "text": public_text}
                    if card.get("kind") == "rumor":
                        item.update(epistemic_status="rumor", authority="reported_rumor")
                    if item not in knowledge:
                        knowledge.append(item)
            if npc_name(card):
                self.npc = npc_name(card)
                self.scene_focus_explicit = False
            story = (event.kind in {"card", "scene_cue", "interaction"}
                     or event.kind in {"say", "end"} and event.text in self._story_say_texts())
            if opening_text and story:
                # 최초 배경은 위 도입으로 표시하고, 공개 사건·감사 기록은 그대로 남깁니다.
                pass
            elif story and not quiet_transit:
                reference = {"kind": event.kind,
                             "source_id": choice_id or "", "knowledge_only": event.kind == "card"}
                if card_id:
                    reference.update(source_card_id=card_id,
                                     known_by=deepcopy(event.data.get("known_by", [])))
                else:
                    reference["text"] = event.text
                narratives.append(reference)
            elif event.kind != "flag" and not quiet_transit:
                self.append_message({"role": "assistant", "content": text,
                                     "speaker": ""}, scene_context=event_context)
                displayed = True
                if event.kind == "move" and opening_text and not opening_inserted:
                    self.append_message({"role": "assistant", "content": opening_text, "speaker": ""},
                                        scene_context=event_context)
                    opening_inserted = True
            self.audit.append({"kind": event.kind, "text": text, "data": deepcopy(event.data)})
            if event.kind not in {"flag", "card"}:
                public.append(event.text)
                public_records.append({"kind": event.kind, "text": event.text,
                                       "data": deepcopy(event.data), "displayed": displayed})
        self._last_record_scene_context = self.message_scene_context()
        self.revision += 1
        if self.npc and (not self._person_present(self.npc_cards()[self.npc])
                         or not self.engine.npc_can_talk(self.npc_cards()[self.npc]["id"])):
            self.npc = ""
        if events:
            witness = self.npc if (self.npc and self.npc_cards()[self.npc]["room"] == self.engine.s.room
                and self.engine.npc_can_talk(self.npc_cards()[self.npc]["id"])) else ""
            self.last_result = {"confirmed": True, "event_id": f"game:{self.revision}",
                "public_events": public, "public_event_records": public_records,
                "state_after": self.rag_state(), "witness_npc": witness}
            self.last_result["revealed_card_ids"] = [event.data["id"] for event in events
                if event.kind == "card" and event.data.get("id") in self.content.cards]
            self.last_result["engine_delta"] = delta
            self.last_result["revealed_knowledge"] = knowledge
            records = (transaction_record if isinstance(transaction_record, list) else
                       [transaction_record] if transaction_record is not None else [])
            if records:
                self.last_result["transaction_records"] = deepcopy(records)
                parties = {name for record in records for name in record.get("knowledge_npcs", [])}
                self.last_result["witness_npc"] = next(iter(parties)) if len(parties) == 1 else ""
            self.last_result["executed_action_ids"] = list(delta.get(
                "executed_action_ids", [choice_id] if choice_id else []))
            self.last_result["item_handovers"] = [deepcopy(record["item_handover"])
                for record in records if record.get("confirmed") is True and record.get("item_handover")]
            reply_ids = set()
            if not movement_scene:
                if conversation_owner:
                    reply_ids.add(conversation_owner["id"])
                reply_ids.update(identifier for identifier in self.last_result["revealed_card_ids"]
                                 if npc_name(self.content.cards[identifier]))
                target = self.content.choices.get(choice_id, {}).get("target_npc")
                if target in self.npc_cards():
                    reply_ids.add(self.npc_cards()[target]["id"])
                if isinstance(choice_id, str) and choice_id.startswith("talk:"):
                    reply_ids.add(choice_id[5:])
                for record in records:
                    reply_ids.update(self.npc_cards()[name]["id"] for name in record.get("knowledge_npcs", [])
                                     if name in self.npc_cards())
            self.last_result["reply_speakers"] = [{"speaker_id": identifier, "name": npc_name(self.content.cards[identifier])}
                for identifier in sorted(reply_ids) if identifier in self.content.cards
                and self._can_address(self.content.cards[identifier])]
            action = (self.content.choices.get(choice_id, {}).get("label") or player_input
                      or "현재 장소를 확인한다")
            ordered = [deepcopy(row) for row in public_records
                       if row["kind"] not in {"say", "interaction", "scene_cue", "guidance"}]
            if not ordered and choice_id in self.content.choices:
                # 숫자 결과가 없는 실제 완료 행동도 선택한 범위만 표시합니다.
                ordered.append({"kind": "action", "text": action,
                                "data": {"action_id": choice_id, "completed": True},
                                "displayed": False})
            explicit_look = bool(player_input and not choice_id and events
                                 and all(event.kind == "scene_cue" for event in events))
            show_observation = (movement_scene and not opening_text or explicit_look) and not quiet_transit
            observation = self.scene_context().get("observation", "") if show_observation else ""
            self.last_result["outcome_facts"] = {
                "action_id": choice_id, "action_label": action,
                "room_before": delta.get("room_before"), "room_after": self.engine.s.room,
                "elapsed_minutes": delta["elapsed_minutes"], "ordered_events": ordered,
                "changes": deepcopy(delta), "observation": observation,
            }
        # 이번 호출의 자료만 사용하며 다음 행동으로 표시 대기를 넘기지 않습니다.
        if not quiet_transit and events and (narratives or choice_id or player_input):
            action = player_input or self.content.choices.get(choice_id, {}).get("label") or "현재 장소를 확인한다"
            return self._render_action_result(narratives, action, choice_id, movement_scene=movement_scene)
        if events and (choice_id or player_input):
            shown = "\n\n".join(row["content"] for row in self.messages[message_start:])
            self._record_turn_diagnostics("action", player_input, {"npc_name": ""},
                {"text": shown, "model_generated": False}, started)
        return {"generated": False, "error": None, "narrative_count": 0}

    def _render_action_result(self, narratives, player_input, choice_id, movement_scene=False):
        """NPC 반응·확정 GM 결과를 표시한 뒤 실제 공개 단서와 방문 안내를 처리합니다."""
        npc = ""
        current = self.npc_cards().get(self.npc, {})
        handovers = (self.last_result or {}).get("item_handovers", [])
        allowed_cards = {current.get("id")} | {identifier for row in handovers
            if row.get("source_npc") == self.npc for identifier in row.get("knowledge_card_ids", [])}
        if (current and not movement_scene and self._person_present(current)
                and self.engine.npc_can_talk(current["id"])
                and (self.last_result or {}).get("witness_npc") == self.npc
                and all(row["kind"] in {"say", "interaction"} or row["kind"] == "card"
                        and row.get("knowledge_only") is True
                        and row.get("source_card_id") in allowed_cards for row in narratives)):
            definition = self.content.choices.get(choice_id, {})
            source = json.dumps(definition.get("source", {}), ensure_ascii=False)
            if (definition.get("target_npc") == self.npc or choice_id == "talk:" + current["id"]
                    or current["id"] in {row.get("source_card_id") for row in narratives}
                    or current["id"] in source or any(row.get("source_npc") == self.npc for row in handovers)):
                npc = self.npc
        replies = (self.last_result or {}).get("reply_speakers", [])
        if len(replies) == 1:
            npc = replies[0]["name"]
            current = self.npc_cards().get(npc, {})
        definition = self.content.choices.get(choice_id, {})
        confirmation_only = (
            definition.get("interaction") in {"accept", "handover"}
            or any(row.get("confirmed") is True
                   for row in (self.last_result or {}).get("transaction_records", []))
            or any("long_rest" in effect for effect in definition.get("effects", []))
        )
        # 완료된 의뢰 수락·전달·구매·휴식은 엔진의 확인입니다. 이 일을 NPC가
        # 새로 약속하거나 주고받은 역할을 뒤집는 생성 문장으로 재서술하지 않습니다.
        generated_speaker = "" if confirmation_only else npc
        actor_context = {"name": npc, "card_id": current["id"]} if generated_speaker else {}
        player_role = {"kind": "player", "name": self.engine.hero.name}
        participant_roles = {"performer": player_role}
        if npc:
            participant_roles["respondent"] = {"kind": "npc", "name": npc}
        if definition.get("interaction") == "accept" and definition.get("target_npc"):
            participant_roles.update(request_issuer={"kind": "npc", "name": definition["target_npc"]},
                                     request_acceptor=deepcopy(player_role))
        payload = {"player_input": player_input, "npc_name": generated_speaker, "actor_context": actor_context,
                    "allowed_speakers": self.allowed_speakers(),
                    "dialogue_state": ({"phase": "action_result", "input_kind": "action"}
                                       if confirmation_only else self.dialogue_state(npc, "action")),
                   "response_mode": "npc_dialogue" if generated_speaker else "scene_dialogue",
                    "interaction_context": self.interaction_context(npc),
                    "action_kind": ("conversation" if self._conversation_action_owner(choice_id) else
                                    definition.get("interaction") or definition.get("operation")),
                    "participant_roles": participant_roles,
                    "movement_scene": movement_scene,
                   "public_narratives": deepcopy(narratives), "engine_result": deepcopy(self.last_result),
                    "item_handovers": deepcopy(handovers),
                    "executed_action_ids": deepcopy((self.last_result or {}).get("executed_action_ids", [])),
                    "action_subject": {"kind": "player", "name": self.engine.hero.name},
                    "engine_delta": deepcopy((self.last_result or {}).get("engine_delta")),
                   "player_snapshot": player_snapshot(self.engine), "scene_context": self.scene_context(),
                   "dialogue_affordances": self.dialogue_affordances(),
                   "progression_options": self.progression_options(),
                    "outcome_facts": deepcopy((self.last_result or {}).get("outcome_facts", {})),
                    "revealed_knowledge": deepcopy((self.last_result or {}).get("revealed_knowledge", [])),
                    "guidance_rendered_separately": True,
                    "delivered_guidance": self.delivered_guidance(npc),
                    # 현재 화자의 실제 발화만 넘기며 카드 열람 기록을 대사 이력으로 바꾸지 않습니다.
                    "recent_conversation": deepcopy(self.histories.get(npc, []))}
        payload["story_progression"] = self.progression_context()
        payload["reference_required"] = bool(payload["story_progression"]["reference_required"])
        payload["confirmed_offer"] = self._confirmed_offer(npc)
        payload["revealed_card_ids"] = deepcopy((self.last_result or {}).get("revealed_card_ids", []))
        # 화면에 아직 공개하지 않은 이름을 모델의 장면 묘사로 먼저 노출하지 않습니다.
        # 원본 RAG, 감사 기록, 영수증, 엔진 상태와 식별자는 그대로 유지합니다.
        for row in payload["public_narratives"]:
            if isinstance(row.get("text"), str):
                row["text"] = self._presentation_text(row["text"])
        event_copy = payload["engine_result"] or {}
        event_copy["public_events"] = [self._presentation_text(text)
                                       for text in event_copy.get("public_events", [])]
        for row in event_copy.get("public_event_records", []):
            if isinstance(row.get("text"), str):
                row["text"] = self._presentation_text(row["text"])
        result = None
        started = time.perf_counter()
        try:
            if generated_speaker:
                get_handoff = self.api.get("get_handoff") if self.api else None
                if not callable(get_handoff):
                    raise RuntimeError("NPC 반응에는 팀 입력 JSON 연결이 필요합니다.")
                # 실제 NPC 발언만 기존 I2→I3 근거를 받습니다. 확정된 GM 결과는
                # 검색·본문 모델을 다시 부르지 않고 엔진 사건을 그대로 표시합니다.
                payload["input_handoff"] = deepcopy(get_handoff(
                    player_input, self.rag_state(), npc_name=npc,
                    history=payload["recent_conversation"]))
            generate = self.api.get("generate_action_result") if self.api else None
            if not callable(generate):
                raise RuntimeError("행동 결과 출력 모델이 준비되지 않았습니다.")
            result = generate(deepcopy(payload))
            if not isinstance(result, dict) or not isinstance(result.get("text"), str) or not result["text"].strip():
                raise ValueError("행동 결과의 표시할 문장이 비었습니다.")
            self.action_generation_calls += int(result.get("actual_model_calls", 0))
            blocks = (self._append_model_response(result, npc)
                      if result.get("display_required") is not False else [])
            if (npc == "세라" and payload["dialogue_state"]["phase"] == "first_contact"
                    and str(result.get("response_origin", "")).startswith("prepared_opening")
                    and any(row.get("speaker_id") == current.get("id") for row in blocks)):
                # 원본에 묶인 첫 인사는 이미 실제 수색 부탁을 말한 경우입니다.
                definition = next((row for row in self._request_candidates(npc) if row["id"] == "sq_accept_help"), None)
                if definition:
                    self.engine.s.vars.setdefault("request_disclosures", {}).setdefault(definition["id"], self._request_record(definition))
            speech = [row for row in blocks if row["kind"] == "speech"
                      and row.get("speaker_id") == current.get("id")]
            if npc and speech:
                # GM·단서·안내를 NPC가 말한 기억으로 저장하지 않습니다.
                history = self.histories.setdefault(npc, [])
                if (not history or history[-1].get("role") != "user"
                        or history[-1].get("content") != player_input):
                    history.append({"role": "user", "content": player_input, "input_kind": "action",
                                    "speaker_id": self.engine.hero.id, "speaker": self.engine.hero.name})
                history.append({"role": "assistant", "content": "\n\n".join(row["text"] for row in speech),
                                "blocks": deepcopy(speech), "speaker_id": current["id"], "speaker": npc})
                save_history = self.api.get("save_dialogue_history") if self.api else None
                if callable(save_history) and getattr(self, "memory_session", None):
                    save_history(self.memory_session, npc, history)
            self.last_generation_seconds = float(result.get("generation_seconds", 0))
            self.debug["행동 결과 생성"] = {"전달 자료": payload, "출력 결과": deepcopy(result),
                "완료된 본문 모델 요청": len(result.get("stage_metrics", [])),
                "누적 행동 출력 요청": self.action_generation_calls,
                "전체 시간(초)": round(time.perf_counter() - started, 3)}
            self.last_status = (f"행동을 반영했습니다. 이야기 생성 {self.last_generation_seconds:.2f}초."
                                if result.get("model_generated") else "확정된 행동 결과를 표시했습니다.")
            self._record_turn_diagnostics("action", player_input, payload, result, started, blocks=blocks,
                                          history=payload.get("recent_conversation"))
            return {"generated": bool(result.get("model_generated")), "error": None,
                    "displayed": bool(blocks), "narrative_count": len(narratives)}
        except Exception as error:
            if self.api and isinstance(error, self.api.get("OutputStreamCancelled", ())):
                self._record_turn_diagnostics("action", player_input, payload, result, started, error=error,
                                              history=payload.get("recent_conversation"))
                raise
            self.debug["행동 결과 생성"] = {"전달 자료": payload, "출력 결과": deepcopy(result),
                "생성 오류": repr(error), "guard_report": deepcopy(getattr(error, "guard_report", None)),
                "raw_response": getattr(error, "raw_response", None),
                "완료된 본문 모델 요청": len(getattr(error, "stage_metrics", [])),
                "누적 행동 출력 요청": self.action_generation_calls,
                "전체 시간(초)": round(time.perf_counter() - started, 3)}
            # 표현 실패로 이미 끝난 판정·구매를 반복하지 않습니다. 근거 문장 전문 대신
            # 엔진이 확정한 실행과 현재 상태만 짧게 남겨 다음 선택의 맥락을 보존합니다.
            action_label = self.content.choices.get(choice_id, {}).get("label") or player_input.strip("*")
            facts = ["행동 반영: " + action_label, "현재 장소: " + self._room_label(self.engine.s.room)]
            delta = (self.last_result or {}).get("engine_delta") or {}
            if delta.get("gold_cp_change"):
                facts.append("현재 소지금: " + fmt_money(self.engine.s.gold_cp))
            if delta.get("hp_change"):
                facts.append(f"현재 HP: {self.engine.hero.hp}/{self.engine.hero.max_hp}")
            self.append_message({"role": "assistant", "speaker": "", "kind": "narration",
                "content": "\n\n".join(facts), "response_origin": "confirmed_action_summary"})
            fallback_result = {**(result or {}), "text": "\n\n".join(facts),
                               "response_origin": "confirmed_action_summary"}
            self._record_turn_diagnostics("action", player_input, payload, fallback_result, started, error=error,
                blocks=[{"kind": "narration", "speaker_id": None, "text": "\n\n".join(facts)}],
                history=payload.get("recent_conversation"))
            self.last_status = "행동 결과를 요약해 표시했습니다. 자세한 이야기 생성은 실패했지만 다음 선택을 할 수 있습니다."
            self._visible_error(player_input, error, stage="generation", engine_committed=True)
            return {"generated": False, "error": repr(error), "narrative_count": len(narratives)}
        finally:
            # 성공·실패·본문 생략 어느 경로에서도 이미 획득한 단서를 잃지 않습니다.
            # 단서의 현재 공개 뷰는 NPC 생성과 별개이며, 아직 모르는 ID는 표시하지 않습니다.
            known = set(self.engine.player_known_cards(self.engine.hero.id))
            for item in payload["revealed_knowledge"]:
                card_id, text = item.get("card_id"), item.get("text")
                card = self.content.cards.get(card_id, {})
                if (card_id not in known or card.get("kind") not in {"clue", "fact", "rumor"}
                        or not isinstance(text, str) or not text.strip()):
                    continue
                if any(row.get("kind") == "knowledge" and row.get("card_id") == card_id
                       and row.get("content") == text for row in self.messages):
                    continue
                self.append_message({"role": "assistant", "kind": "knowledge", "speaker": "",
                                      "card_id": card_id, "knowledge_only": True,
                                      "epistemic_status": "rumor" if card.get("kind") == "rumor" else "public_record",
                                      "content": text})
            if npc and current.get("id"):
                self._publish_guidance(current["id"])

    def rag_state(self):
        s, h = self.engine.s, self.engine.hero
        return {"scene": s.scene, "cls": h.cls, "flags": sorted(s.flags), "visited": sorted(s.visited),
                # 팀 시나리오 v5는 room으로 현재 방을 필터링합니다. 화면의 가상 대장간·
                # 여관 ID가 아닌 원본 엔진 방 ID를 전달하고 기존 current_room도 유지합니다.
                "room": s.room, "current_room": s.room, "inventory": dict(s.inventory), "gold_cp": s.gold_cp,
                "money": fmt_money(s.gold_cp), "vars": deepcopy(s.vars),
                "hero": {"name": h.name, "hp": h.hp, "max_hp": h.max_hp, "ac": self.engine.armor_class(h),
                         "level": h.level, "conditions": list(h.conditions), "abilities": dict(h.abilities),
                         "proficiency_bonus": h.proficiency_bonus, "resources": deepcopy(h.resources),
                         "hit_dice": {"remaining": h.hd_left, "max": h.hd_max, "die": h.hit_die}},
                "rule_search_mode": self.rule_mode,
                "combat": {"encounter": s.combat["enc"], "round": s.combat["round"],
                           "turn": deepcopy(s.combat.get("turn", {}))} if s.combat else None}

    def talk_to(self, name):
        """처음 인물은 만남을 처리하고, 이미 만난 인물은 대화 상대만 전환합니다."""
        self.cancel_recommendations()
        if not name:
            self.npc = ""
            self.scene_focus_explicit = True
            self.last_status = "장면을 대상으로 이야기합니다."
            self.refresh_suggestions()
            return
        cards = self.npc_cards()
        name = next((actual for actual, card in cards.items()
                     if name in {card["id"], self._person_label(card)} and card["room"] == self.engine.s.room), name)
        if (name not in cards or not self._person_present(cards[name]) or not self._story_person_allowed(cards[name])
                or not self.engine.eligible(cards[name]["id"]) or not self.engine.npc_can_talk(cards[name]["id"])):
            raise ValueError("현재 장소에서 만날 수 없는 인물입니다.")
        if self.engine.developer:
            self.engine.s.vars["developer_used"] = True
            self.audit.append({"kind": "developer", "text": "개발자 모드에서 인물을 만났습니다.",
                               "data": {"card_id": cards[name]["id"]}})
        self.npc = name
        self.scene_focus_explicit = False
        if cards[name]["id"] in self.engine.s.shown:
            # 상대 선택은 질문이나 NPC 발언이 아닙니다. 빈 대사를 생성하지 않습니다.
            self.last_status = f"{name}에게 할 말을 입력하세요."
            self.refresh_suggestions()
            return
        self._record(self.engine._show(cards[name]["id"]), choice_id="talk:" + cards[name]["id"],
                     player_input=name + "에게 말을 건다")
        self.last_status = f"대화 상대: {name}. 기존 대화 이력을 유지합니다."
        self.refresh_suggestions()

    def choices(self):
        views = [row for row in self.engine.available_choices() if self._story_choice_allowed(row.id)]
        if self.engine.s.combat or self.engine.s.pending or self.engine.s.ended:
            return views
        ids = {view.id for view in views}
        for definition in self._request_candidates():
            if definition["id"] not in self.engine.s.vars.get("request_disclosures", {}) and definition["id"] not in self.engine.s.done:
                identifier = "offer:" + definition["id"]
                views.append(ChoiceView(identifier, "부탁의 내용을 듣는다", [], ""))
                ids.add(identifier)
        if self.engine.s.room == "village_market":
            for place, info in MARKET_SPACES.items():
                if place != self.current_place()["id"] and self._place_allowed(place):
                    views.append(ChoiceView("travel:" + place, info["name"] + "으로 이동한다", [], ""))
        # 이미 대화 중인 상대에게 빈 '말 걸기'를 반복하지 않습니다. 현장의 다른
        # 사람 선택만 제공하고, 실제 질문은 입력칸이나 구체적인 대화 선택지로 보냅니다.
        for name, card in self.npc_cards().items():
            approach = self._approach(card)
            if (name != self.npc and self._person_present(card) and card["id"] in self.engine.s.shown
                    and self.engine.eligible(card["id"]) and self.engine.npc_can_talk(card["id"])
                    and approach not in ids):
                views.append(ChoiceView("talk:" + card["id"], name + "와 대화하기", [], ""))
        _, paths = self._public_paths()
        # 다음으로 안내된 인물은 원본 장면 행동과 동일한 실제 이동·만남으로 연결합니다.
        progress = self.progression_context()
        target = self.npc_cards().get(progress["next_npc"], {})
        next_meet = "meet:" + target.get("id", "")
        if (target and target["room"] in paths and target["id"] in self._known_npcs()
                and self._story_person_allowed(target) and self.engine.npc_can_talk(target["id"])
                and not self._person_present(target) and next_meet not in ids):
            views.append(ChoiceView(next_meet, npc_name(target) + "을 찾아간다", [], ""))
            ids.add(next_meet)
        for row in self.content.rooms[self.engine.s.room].get("suggestions", []):
            identifier = row.get("action_id", "")
            card = self.content.cards.get(identifier[5:], {}) if identifier.startswith("meet:") else {}
            if (npc_name(card) and card["id"] in self._known_npcs()
                    and card["room"] in paths and self.engine.cond(row.get("requires"))
                    and self.engine.npc_can_talk(card["id"]) and identifier not in ids):
                views.append(ChoiceView(identifier, row.get("label", npc_name(card) + "을 찾아간다"), [], ""))
                ids.add(identifier)
        if not self.engine.developer:
            public = []
            for view in views:
                definition = self.content.choices.get(view.id, {})
                target = definition.get("target_npc")
                card = self.npc_cards().get(target, {})
                if (card and card["id"] not in self._known_npcs()
                        and view.id != self._approach(card)):
                    continue
                approach = next((card for card in self.npc_cards().values()
                    if view.id == self._approach(card) and card["id"] not in self._known_npcs()), None)
                label = (self._person_label(approach) + "에게 말을 건다" if approach else
                         "이동: " + self._room_label(view.id[5:]) if view.id.startswith("move:") else self._public_text(view.label))
                if label is not None:
                    copied = deepcopy(view)
                    copied.label = label
                    public.append(copied)
            views = public
        return [view for view in views
                if self._story_choice_allowed(view.id) and availability(self, view.id)["available"]]

    def _needs_roll(self, choice_id):
        ch = self.content.choices.get(choice_id, {})
        return bool(ch.get("check")) or choice_id.startswith(("cmb:attack:", "cmb:cast:", "cmb:holy:", "cmb:escape", "cmb:unarmed:", "cmb:subdue:"))

    def propose(self, choice_id, direct=False):
        if choice_id.startswith("talk:"):
            # 표시 중 상대가 바뀐 오래된 버튼도 엔진 행동이나 가짜 발언으로 실행하지 않습니다.
            self.talk_to(choice_id[5:])
            return
        condition = availability(self, choice_id)
        if not condition["available"]:
            raise ValueError(condition["reason"])
        available = {c.id: c for c in self.choices()}
        if choice_id not in available:
            raise ValueError("현재 실행할 수 없는 행동입니다.")
        if choice_id.startswith("offer:"):
            self.present_request(choice_id[6:])
            return
        if choice_id.startswith("travel:"):
            self.travel_to(choice_id[7:])
            return
        if self.engine.developer:
            self.engine.s.vars["developer_used"] = True
            self.audit.append({"kind": "developer", "text": "개발자 모드에서 행동을 실행했습니다.",
                               "data": {"action_id": choice_id}})
        if choice_id.startswith("meet:"):
            self.visit_npc(choice_id[5:])
            return
        self.pending = {"type": "choice", "choice": choice_id, "label": available[choice_id].label,
                        "revision": self.revision, "roll": self._needs_roll(choice_id),
                        "actor_id": available[choice_id].recommended or self.engine.hero.id,
                        "check_text": available[choice_id].check_text,
                        "request": self.messages[-1]["content"] if self.messages and self.messages[-1]["role"] == "user" else ""}
        methods = self.pending_check_options()
        if methods:
            self.pending["method_index"] = next(row["index"] for row in methods if row["recommended"])
        if direct and not self.pending["roll"]:
            self.resolve_pending()
        else:
            text = available[choice_id].check_text or "선택한 행동을 실행합니다."
            self.append_message({"role": "assistant", "content": f"**행동 확인** · {available[choice_id].label}\n{text}\n아래 버튼으로 실행하거나 ‘굴린다’라고 입력하세요."})
            self.last_status = "행동을 확인한 뒤 실행할 수 있습니다."
            self._capture_developer_checkpoint("판정 대기 · " + available[choice_id].label)

    def pending_check_options(self):
        """현재 가능한 판정의 원본 수단과 자기 수정치만 보여 줍니다."""
        pending = self.pending
        if not pending or pending.get("type") != "choice":
            return []
        if pending.get("revision") != self.revision:
            raise ValueError("게임 상태가 바뀌었습니다. 판정을 다시 선택하세요.")
        available = {row.id: row for row in self.choices()}
        view = available.get(pending["choice"])
        if view is None:
            raise ValueError("현재 실행할 수 없는 판정입니다.")
        check = self.content.choices.get(pending["choice"], {}).get("check")
        if not check:
            return []
        actor_id = pending.get("actor_id", self.engine.hero.id)
        if actor_id not in view.actors:
            raise ValueError("이 캐릭터는 현재 판정에 참여할 수 없습니다.")
        rows = self.engine.check_options(self.engine.s.member(actor_id), check)
        best = max(rows, key=lambda row: row["modifier"])["index"]
        selected = pending.get("method_index", best)
        self.engine.selected_check(self.engine.s.member(actor_id), check, selected)
        return [{key: row[key] for key in ("index", "label", "modifier", "dc", "formula")}
                | {"recommended": row["index"] == best, "selected": row["index"] == selected}
                for row in rows]

    def select_check_method(self, index, *, choice_id=None, revision=None):
        """굴림 전 선택만 바꾸며 게임 상태·방법 번호를 실행 직전 다시 확인합니다."""
        rows = self.pending_check_options()
        if (not rows or type(index) is not int or index not in {row["index"] for row in rows}
                or choice_id is not None and choice_id != self.pending["choice"]
                or revision is not None and revision != self.revision):
            raise ValueError("현재 판정에 없는 방법이거나 이전 화면의 선택입니다.")
        self.pending["method_index"] = index
        self.last_status = "판정 방법을 골랐습니다. 굴리면 이 방법으로 한 번 판정합니다."

    def _action_receipt(self, before, events):
        """구매 장부는 기존대로 기록하고, 고유 물품 전달은 이번 출력용으로 검증합니다."""
        purchase = record_action(self.transaction_ledger, before, self.engine, events)
        if purchase is not None:
            return purchase
        action_id = before["action_id"]
        definition = self.content.choices.get(action_id, {})
        declared = definition.get("item_handover") or {}
        card = self.content.cards.get(declared.get("source_card_id"), {})
        item_id = declared.get("item_id")
        source_npc = definition.get("target_npc")
        index = before["log_length"]
        if (definition.get("interaction") != "handover" or not isinstance(item_id, str)
                or not source_npc or npc_name(card) != source_npc
                or before["active_npc"] != source_npc or not before["available"]
                or card.get("room") != before["room_id"] or self.engine.s.room != before["room_id"]
                or len(self.engine.s.log) != index + 1 or self.engine.s.log[index].get("choice") != action_id
                or not isinstance(declared.get("explanation"), str) or not declared["explanation"].strip()):
            return None
        gives = [effect for effect in definition.get("effects", []) if effect.get("give") == item_id]
        quantity = gives[0].get("n", 1) if len(gives) == 1 else None
        if (type(quantity) is not int or quantity < 1 or definition.get("once") is not True
                or action_id not in self.engine.s.done
                or self.engine.s.inventory.get(item_id, 0) - before["inventory"].get(item_id, 0) != quantity
                or not any(event.kind == "item" and event.data.get("item") == item_id for event in events)):
            return None
        transfer = {"item_id": item_id, "item_name": self.content.items.get(item_id, {}).get("name", item_id),
            "quantity": quantity, "player_quantity": self.engine.s.inventory.get(item_id, 0),
            "transfer_completed": True, "npc_current_quantity": None, "npc_custody_status": "not_tracked",
            "from": npc_party(source_npc, card["id"]), "to": player_party(self.engine)}
        shown = {effect.get("show") for effect in definition.get("effects", []) if "show" in effect}
        clue_ids = [event.data["id"] for event in events if event.kind == "card"
                    and event.data.get("id") in shown
                    and self.content.cards.get(event.data.get("id"), {}).get("kind") == "clue"]
        handover = {**deepcopy(declared), "action_id": action_id, "source_npc": source_npc,
                    "item_name": transfer["item_name"], "quantity": transfer["quantity"],
                    "from": deepcopy(transfer["from"]), "to": deepcopy(transfer["to"]),
                    "knowledge_card_ids": clue_ids, "confirmed": True}
        return {"kind": "handover", "entry_id": f"handover:{index}", "source_log_index": index,
                "action_id": action_id, "confirmed": True, "knowledge_npcs": [source_npc],
                "transfers": [deepcopy(transfer)], "payments": [], "item_handover": handover}

    def _automatic_handovers(self, parent_id):
        """장면이 선언한 같은 상대의 무판정 물품 전달만 기존 엔진 선택으로 이어갑니다."""
        parent = self.content.choices.get(parent_id, {})
        if parent.get("interaction") != "accept" or parent.get("target_npc") != self.npc:
            return [], [], []
        events, receipts, executed = [], [], []
        for identifier in parent.get("auto_followups", [])[:4]:
            child = self.content.choices.get(identifier, {})
            effects = child.get("effects", [])
            if (identifier in executed or identifier not in {row.id for row in self.engine.available_choices()}
                    or child.get("interaction") != "handover" or child.get("target_npc") != self.npc
                    or child.get("room") != self.engine.s.room or "check" in child
                    or not child.get("item_handover") or self.engine.s.pending or self.engine.s.combat
                    or any(len(effect) != 1 or not set(effect) <= {"flag", "give", "show", "say"}
                           for effect in effects)):
                continue
            before = before_action(self.engine, self.npc, identifier, TRANSACTION_BINDINGS)
            child_events = self.engine.choose(identifier)
            receipt = self._action_receipt(before, child_events)
            events.extend(child_events)
            executed.append(identifier)
            if receipt is not None:
                receipts.append(receipt)
        return events, receipts, executed

    def can_force_pending_roll(self):
        """개발자 버튼은 실제 플레이어 판정·명중 굴림이 있는 행동에만 보입니다."""
        pending = self.pending
        if (not self.engine.developer or not pending or not pending.get("roll")
                or pending.get("revision") != self.revision):
            return False
        if pending.get("type") == "free_check":
            return pending.get("ability") in ABILITY_KO
        if pending.get("type") != "choice":
            return False
        identifier = pending.get("choice", "")
        if identifier not in {row.id for row in self.choices()}:
            return False
        if self.content.choices.get(identifier, {}).get("check"):
            try:
                return any(row["selected"] for row in self.pending_check_options())
            except ValueError:
                return False
        if identifier.startswith(("cmb:attack:", "cmb:unarmed:", "cmb:subdue:", "cmb:escape")):
            return True
        if identifier.startswith("cmb:cast:"):
            spell = self.content.rules.get("spells", {}).get(identifier.split(":")[2], {})
            return spell.get("kind") == "attack"
        return False

    def resolve_pending_developer(self, success):
        """일반 실행 경로의 다음 대상 굴림 결과만 명시적으로 바꾸고 즉시 해제합니다."""
        self.engine.require_developer()
        if type(success) is not bool or not self.can_force_pending_roll():
            raise ValueError("현재 선택한 판정·공격 굴림에서만 성공·실패를 시험할 수 있습니다.")
        pending = self.pending
        original_success = self.engine.next_success
        self.engine.next_success = False
        self.engine.pending_roll_override = {"success": success,
            "actor_id": pending.get("actor_id", self.engine.hero.id)}
        self.audit.append({"kind": "developer", "text": "현재 굴림의 테스트 결과를 지정했습니다.",
            "data": {"forced_success": success, "choice_id": pending.get("choice"),
                     "revision": self.revision}})
        try:
            return self.resolve_pending()
        finally:
            self.engine.pending_roll_override = None
            self.engine.next_success = original_success

    def resolve_pending(self):
        if not self.pending:
            raise ValueError("대기 중인 행동이나 주사위 판정이 없습니다.")
        pending = self.pending
        if pending["revision"] != self.revision:
            raise ValueError("게임 상태가 바뀌었습니다. 현재 행동을 다시 선택하세요.")
        if pending.get("type") == "choice" and pending.get("choice", "").startswith("talk:"):
            # 이전 버전에서 저장한 대화 상대 확인 대기도 새 대사를 생성하지 않습니다.
            self.pending = None
            self.talk_to(pending["choice"][5:])
            return
        if pending.get("type") == "choice":
            methods = self.pending_check_options()
            if methods and not any(row["selected"] for row in methods):
                raise ValueError("판정 방법을 먼저 선택하세요.")
        self._capture_developer_checkpoint("실행 전 · " + pending.get("label", "현재 행동"))
        self.pending = None
        before_values = {"room": self.engine.s.room, "gold_cp": self.engine.s.gold_cp,
                         "inventory": dict(self.engine.s.inventory), "hp": self.engine.hero.hp,
                         "minutes": self.engine.s.minutes,
                         "message_scene": self.message_scene_context()}
        transaction_records, executed_action_ids = [], []
        if pending["type"] == "choice":
            definition = self.content.choices.get(pending["choice"], {})
            target = definition.get("target_npc")
            if target and definition.get("interaction") in {"accept", "handover"}:
                card = self.npc_cards().get(target, {})
                if (not card or not self._person_present(card)
                        or card["id"] not in self.engine.s.shown
                        or not self.engine.eligible(card["id"]) or not self.engine.npc_can_talk(card["id"])):
                    raise ValueError("현재 장소에서 부탁이나 물품을 주고받을 상대를 확인할 수 없습니다.")
                # 명시한 상대에게 하는 행동은 이전 대화창 선택과 관계없이 그 상대에 연결합니다.
                self.npc, self.scene_focus_explicit = target, False
            action_snapshot = before_action(self.engine, self.npc, pending["choice"], TRANSACTION_BINDINGS)
            events = self.engine.choose(pending["choice"], actor_id=pending.get("actor_id"),
                                        method_index=pending.get("method_index"))
            for name, card in self.npc_cards().items():
                if self._approach(card) == pending["choice"]:
                    self.npc = name
            # 부모 기록을 자식 실행 전에 확인하여 각 log 위치와 실제 수량 차이를 보존합니다.
            receipt = self._action_receipt(action_snapshot, events)
            if receipt is not None:
                transaction_records.append(receipt)
            executed_action_ids.append(pending["choice"])
            follow_events, follow_receipts, follow_ids = self._automatic_handovers(pending["choice"])
            events.extend(follow_events)
            transaction_records.extend(follow_receipts)
            executed_action_ids.extend(follow_ids)
            for event in events:
                card = self.content.cards.get(event.data.get("id"), {}) if event.kind == "card" else {}
                if npc_name(card) and pending.get("request"):
                    self.histories.setdefault(npc_name(card), []).append({"role": "user", "content": pending["request"]})
        else:
            _, event = self.engine.roll_check(self.engine.hero, {"ability": pending["ability"], "dc": 15})
            events = [event, Event("status", "자유 판정 결과를 기록했습니다. 시나리오 보상·이동·단서는 해당 행동으로 확인합니다.")]
        inventory = self.engine.s.inventory
        changes = [{"item_id": item_id, "item_name": self.content.items.get(item_id, {}).get("name", item_id),
                    "quantity_change": inventory.get(item_id, 0) - before_values["inventory"].get(item_id, 0)}
                   for item_id in sorted(set(inventory) | set(before_values["inventory"]))
                   if inventory.get(item_id, 0) != before_values["inventory"].get(item_id, 0)]
        delta = {"executed_action_id": pending.get("choice"), "room_before": before_values["room"],
                 "message_scene_before": before_values["message_scene"],
                 "executed_action_ids": executed_action_ids,
                 "method_index": pending.get("method_index"),
                 "room_after": self.engine.s.room, "gold_cp_change": self.engine.s.gold_cp - before_values["gold_cp"],
                 "inventory_changes": changes, "hp_change": self.engine.hero.hp - before_values["hp"],
                 "minutes_before": before_values["minutes"], "minutes_after": self.engine.s.minutes,
                 "elapsed_minutes": self.engine.s.minutes - before_values["minutes"],
                 "engine_pending": bool(self.engine.s.pending)}
        outcome = self._record(events, transaction_records, choice_id=pending.get("choice"),
                               player_input=pending.get("request") or pending.get("label", "행동을 실행한다"),
                               engine_delta=delta)
        if not outcome.get("error") and not outcome.get("generated"):
            self.last_status = "행동 결과를 게임 상태에 반영했습니다."
        if pending.get("choice", "").startswith("npc:attack:") and self.engine.s.combat:
            unarmed = next((choice.id for choice in self.choices() if choice.id.startswith("cmb:unarmed:")), None)
            if unarmed:
                self.propose(unarmed, direct=False)
        self.refresh_suggestions()
        if any(event.kind == "move" for event in events):
            self._auto_meet_target()

    def free_check(self, ability):
        self.engine.require_developer()
        if ability not in ABILITY_KO:
            raise ValueError("알 수 없는 능력치입니다.")
        if self.engine.s.combat:
            raise ValueError("전투 중에는 현재 전투 행동을 선택하세요.")
        self.pending = {"type": "free_check", "ability": ability, "revision": self.revision,
                        "label": f"{ABILITY_KO[ability]} 자유 판정", "roll": True,
                        "check_text": "테스트용 보통 난이도 DC 15 · 캐릭터 능력 수정치 적용"}
        self.append_message({"role": "assistant", "content": f"{ABILITY_KO[ability]} 판정이 준비됐습니다. DC 15입니다. 🎲 굴린다를 누르세요."})
        self.last_status = "자유 판정은 결과만 기록하며 시나리오 사건을 대신 확정하지 않습니다."

    def utility_dice(self, expr):
        self.engine.require_developer()
        match = DICE_EXPR.fullmatch(expr)
        if not match or not 1 <= int(match[1] or 1) <= 20 or not 2 <= int(match[2]) <= 100:
            raise ValueError("주사위는 1~20개, 2~100면 범위로 입력하세요. 예: 1d20, 2d6+3")
        total, rolls = self.engine.dice.roll(expr)
        self._record([Event("check", f"🎲 {expr}: {rolls} → 합계 {total}", {"expression": expr, "rolls": rolls, "total": total})])
        self.last_status = "일반 주사위 결과입니다. 별도의 행동 성공·실패 판정은 아닙니다."

    def set_developer(self, enabled):
        self.cancel_recommendations()
        self.engine.developer = bool(enabled)
        if not enabled:
            self.engine.next_success = False
            self.engine.pending_roll_override = None
        self.refresh_suggestions()
        self.last_status = "개발자 모드를 켰습니다." if enabled else "개발자 모드를 껐습니다. 성공 보장을 해제했습니다."

    def developer_jump(self, room, npc=""):
        self.engine.require_developer()
        if self.engine.s.combat:
            raise ValueError("전투를 끝낸 뒤 이동하세요. 즉시 승리 버튼을 사용할 수 있습니다.")
        self.cancel_recommendations()
        self.pending = self.engine.s.pending = None
        self.engine.pending_roll_override = None
        if room == "village_market" and npc in self.npc_cards():
            self.engine.s.vars["presentation_place"] = self._person_place(self.npc_cards()[npc])
        self.engine.s.vars["developer_used"] = True
        self._record([Event("developer", "개발자 바로 이동을 사용했습니다.")] + self.engine._enter(room))
        if npc:
            self.talk_to(npc)
        self.refresh_suggestions()
        self.last_status = "개발자 이동 완료. 테스트 조작을 기록했습니다."

    def developer_encounter(self, encounter):
        self.engine.require_developer()
        if self.engine.s.combat:
            raise ValueError("이미 전투 중입니다.")
        if not self.engine.hero.can_act():
            raise ValueError("먼저 HP·자원 회복을 누르세요.")
        if encounter not in self.content.encounters:
            raise ValueError("알 수 없는 조우입니다.")
        self.engine.s.vars["developer_used"] = True
        self.cancel_recommendations()
        self.pending = self.engine.s.pending = None
        self.engine.pending_roll_override = None
        self._record([Event("developer", "개발자 시험 전투를 시작합니다.")] + self.engine.start_combat(encounter))
        self.refresh_suggestions()
        self.last_status = "시험 전투입니다. 정상 플레이의 조우 조건을 건너뛰었습니다."

    def _explicit_public_navigation(self, message, locations):
        """명확한 만남·이동 선언의 공개 이름을 현재 지도 ID로 연결합니다.

        이름과 ID는 공개 지도에서만 가져옵니다. 질문·복합 행동·모호한 대상은
        이 경로가 결정하지 않으며 기존 의도 해석과 엔진 조건 검사를 유지합니다.
        """
        if self._travel_blocked() or _question_form(message):
            return None
        text = message.strip()
        if text.startswith("**") and text.endswith("**"):
            text = text[2:-2].strip()
        text = text.rstrip(".!。！").strip()
        lead = r"(?:(?:나는|내가|저는|제가|우리는|우리가|일행은|일행이|그럼|그러면|이제|먼저|일단|다시|직접)\s+)*"
        go = r"(?:간다|갑니다|가요|가겠다|가겠습니다|갈게요|가본다|가봅니다|가볼게요|가보겠습니다)"
        travel = r"(?:" + go + r"|이동한다|이동합니다|이동하겠다|이동하겠습니다|이동할게요|향한다|향합니다|향하겠습니다)"
        meet = r"(?:만난다|만납니다|만나요|만나겠다|만나겠습니다|만나볼게요|이야기한다|이야기합니다|대화한다|대화합니다|말을\s*건다|말을\s*겁니다)"
        meeting = r"(?:" + meet + r"|(?:만나러|찾으러|이야기하러|대화하러|말을\s*걸러)\s*" + travel + r"|찾아\s*" + go + r"|" + travel + r")"
        matches = []
        for room in locations:
            if not room["reachable"]:
                continue
            for person in room["npcs"]:
                if not person["can_meet"] or person["name"] == self.npc:
                    continue
                role = ""
                titles = {role, role.split()[-1] if role else ""} - {""}
                title = (r"(?:(?:" + "|".join(re.escape(value) for value in sorted(titles)) + r")\s+)?") if titles else ""
                pattern = (lead + title + re.escape(person["name"]) + r"(?:님)?\s*(?:을|를|에게|한테|께|와|과)?\s*" + meeting)
                if re.fullmatch(pattern, text):
                    matches.append("meet:" + person["card_id"])
            if not room["current"]:
                pattern = lead + re.escape(room["name"]) + r"\s*(?:으로|로|에|까지|쪽으로)?\s*" + travel
                if re.fullmatch(pattern, text):
                    matches.append("travel:" + room["id"])
        matches = list(dict.fromkeys(matches))
        return {"action_id": matches[0], "intent": "action", "offer_id": "", "blocked_action_id": ""} if len(matches) == 1 else None

    def _addressed_person(self, message):
        """다른 인물을 화제로 삼는 말과, 그 인물에게 직접 말 거는 선언을 구별합니다."""
        text = message.strip().strip("*").strip()
        if re.search(r"(?:만나러|찾아|이야기하러|대화하러|말을\s*걸러).*"
                     r"(?:간다|갑니다|가요|가겠|갈게|가본|가봅|가볼|이동|향한|향합)", text):
            return None  # 찾아가는 선언은 발견된 경로를 실제로 이동한 뒤 만납니다.
        for name, card in self.npc_cards().items():
            target = re.escape(name) + r"(?:\s*(?:님|씨))?"
            # 조사 뒤의 목적어도 허용합니다. '할빅에 대해'라는 화제와 직접 수신자는 구분합니다.
            address = (target + r"\s*(?:에게|한테|께|와|과)\s*(?:[^.!?。！？\n]{0,120}?)"
                       r"(?:말을\s*(?:건|걸|겁)|대화(?:한|하겠|할)|이야기(?:한|하겠|할)|질문(?:한|하겠|할)|물어(?:본|보겠|볼)|묻(?:는|겠|습니다))")
            vocative = r"^" + target + r"\s*[,，!！:]"
            if re.search(address, text) or re.search(vocative, text):
                return name, card
        return None

    def _can_address(self, card):
        """모델 호출 전에 현재 장소·공개·생존·대화 조건을 함께 확인합니다."""
        return bool(self._person_present(card) and self._story_person_allowed(card)
                    and self.engine.eligible(card["id"]) and self.engine.npc_can_talk(card["id"]))

    def _declared_npc_target(self, message):
        """명시한 공격 선언의 상대만 읽습니다. 질문이나 다른 인물의 이야기는 실행하지 않습니다."""
        if _question_form(message):
            return None
        text = message.strip().strip("*").strip()
        attack = r"(?:공격(?:한다|합니다|하겠다|하겠습니다|할게요)|때린다|때리겠다|찌른다|벤다)[.!。！]?\s*$"
        matches = []
        for name, card in self.npc_cards().items():
            found = re.search(re.escape(name) + r"(?:님|씨)?(?:을|를|에게|한테)?[^.!?\n]{0,40}" + attack, text)
            if found:
                matches.append((found.start(), name, card))
        if matches:
            _, name, card = max(matches, key=lambda value: value[0])
            return name, card
        return None

    def _purchase_route(self, message, *, structured_intent=False):
        """명시한 구매를 실제 buy ID 또는 분명한 선행 조건으로 연결합니다.

        판매 목록 열람을 구매 성공으로 대신 실행하지 않습니다. 물품 이름·가격·
        재고·장소는 원본에 등록된 값만 사용하며, 질문·가정은 조회 경로에 남깁니다.
        """
        if _question_form(message):
            return None
        text = message.strip().strip("*").strip().rstrip(".!。！")
        if re.search(r"(?:만약|만일|가정|상상|척하|(?:산다|구입한다|구매한다)면)", text):
            return None
        declaration = re.search(
            r"(?:구매|구입)(?:한다|합니다|하겠다|하겠습니다|할게요|할게|하고\s*싶습니다|하고\s*싶어요)$"
            r"|(?:산다|사겠다|사겠습니다|살게요|살게|살래요|사고\s*싶습니다|사고\s*싶어요)$", text)
        if not declaration and not structured_intent:
            return None
        items = []
        for identifier, item in self.content.items.items():
            if type(item.get("price_cp")) is not int:
                continue
            aliases = {item["name"], item["name"].split("(", 1)[0].strip()}
            if any(alias and alias.casefold() in text.casefold() for alias in aliases):
                items.append(identifier)
        count_match = re.search(r"(\d+)\s*(?:개|병|묶음)", text)
        count = int(count_match[1]) if count_match else 1
        if not count_match:
            numbers = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3,
                       "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7,
                       "여덟": 8, "아홉": 9, "열": 10}
            word = re.search(r"(" + "|".join(numbers) + r")\s*(?:개|병|묶음)", text)
            if word:
                count = numbers[word[1]]
            elif re.search(r"(?:몇|여러|많은|수십)\s*(?:개|병|묶음)", text):
                count = 0
        reason, next_id = "", None
        if len(items) != 1:
            reason = "구입할 물품 하나를 현재 판매 목록에서 선택해 주세요. 구매는 실행하지 않았습니다."
        elif count != 1:
            reason = "현재 구매 선택지는 한 번에 1개씩 처리합니다. 요청한 여러 개를 임의로 1개만 사지는 않았습니다."
        elif self.current_place()["id"] != "village_market:forge":
            reason = "현재 장소에서는 그 물품을 구입할 수 없습니다. 판매하는 곳으로 이동한 뒤 선택해 주세요."
        else:
            identifier = "buy:" + items[0]
            condition = availability(self, identifier)
            available = {row.id: row for row in self.choices()}
            if not condition["available"]:
                reason = condition["reason"]
            elif identifier in available:
                return identifier, {"blocked_action": None, "operation": "purchase",
                                    "item_id": items[0], "quantity": 1}
            else:
                catalogue = next((row for row in available.values()
                    if any(self.content.cards.get(effect.get("show"), {}).get("kind") == "shop"
                           for effect in self.content.choices.get(row.id, {}).get("effects", []))), None)
                reason = "먼저 판매 목록을 확인해야 합니다. 목록 열람을 구매로 대신 실행하지 않았습니다."
                if catalogue:
                    next_id = catalogue.id
        blocked = {"blocked_reason": reason}
        if next_id:
            blocked.update(next_action_id=next_id, next_action_label=self.content.choices[next_id]["label"])
        return "CLARIFY", {"blocked_action": blocked, "operation": "purchase",
                           "item_id": items[0] if len(items) == 1 else None, "quantity": count}

    def route_message(self, message, input_kind=None):
        """자료·행동 목록 없이 말의 종류를 고르고, 행동일 때만 실제 ID를 연결합니다."""
        started = time.perf_counter()
        current_stage("routing", detail="입력 종류와 실행 경로 확인", player_input=message,
                      input_kind=input_kind, npc_name=self.npc)
        self.turn_needs_information = self.turn_needs_state = False
        if input_kind == "speech" and not (_question_form(message) or _information_question(message)):
            self.turn_intent, self.turn_needs_information = "say", _information_question(message)
            return "TALK", {"route": "TALK", "kind": "speech", "intent": "say", "state_topics": [],
                            "needs_information": self.turn_needs_information,
                            "model_calls": 0, "classification_calls": 0, "mapper_calls": 0,
                            "classification_seconds": 0.0, "mapper_seconds": 0.0, "seconds": 0.0}
        purchase = self._purchase_route(message) if input_kind != "speech" else None
        if purchase is not None:
            selected, details = purchase
            self.turn_intent = "action"
            return selected, {"route": selected, "kind": "action", "intent": "action", **details,
                "state_topics": [], "model_calls": 0, "classification_calls": 0, "mapper_calls": 0,
                "classification_seconds": 0.0, "mapper_seconds": 0.0, "seconds": 0.0,
                "navigation_resolution": {"source": "declared_purchase", "action_id": selected}}
        exact = None if _question_form(message) else exact_action(self, message)
        if exact:
            self.turn_intent = "action"
            selected = exact["action_id"]
            return selected, {"route": selected, "kind": "action", "intent": "action",
                "blocked_action": exact["blocked_action"], "state_topics": [],
                "model_calls": 0, "classification_calls": 0, "mapper_calls": 0,
                "classification_seconds": 0.0, "mapper_seconds": 0.0, "seconds": 0.0,
                "navigation_resolution": {"source": "exact_public_action", "action_id": selected}}
        explicit = self._explicit_public_navigation(message, self.map_locations())
        if explicit:
            self.turn_intent = "action"
            return explicit["action_id"], {"route": explicit["action_id"], "kind": "action", "intent": "action",
                "state_topics": [], "model_calls": 0, "classification_calls": 0, "mapper_calls": 0,
                "classification_seconds": 0.0, "mapper_seconds": 0.0, "seconds": 0.0,
                "navigation_resolution": {"source": "public_entity", "action_id": explicit["action_id"]}}
        recent = [{"role": row["role"], "content": row["content"]}
                  for row in native_conversation_history(self.histories.get(self.npc, []))[-2:]]
        system = (
            '플레이어의 현재 발언을 분류한다. JSON으로 반환한다: '
            '{"kind":"social|listen|accept|decline|farewell|knowledge|state|action|look|rule|clarify","needs_information":false,"fact_scope":"world|state|services|requirements|directions|contracts|conversation","operation":"purchase|none","state_topics":[]}. '
            'social=일상 대화, decline=거절, farewell=작별, knowledge=설정·사건 질문, '
            'listen=현재 상대의 이야기나 부탁 설명을 들어보겠다는 응답, accept=현재 상대가 부탁한 일을 맡겠다는 약속, '
            'state=실제 상태 확인, action=행동 실행, '
            'look=주변 관찰, rule=규칙 질문, clarify=대상 확인. '
            'needs_information은 말투나 의도와 별개로 설정·사건·이유·조건의 사실이 필요한지다. 설명을 요청하면 listen이어도 true다. '
            'fact_scope는 답에 필요한 사실의 종류다: services=현재 가게·시설의 판매·숙박·휴식 서비스와 이용 조건·가격, '
            'directions=다음에 어디로 또는 누구를 찾아가야 하는지 방문 안내를 확인하는 질문, '
            'contracts=현재 들은 의뢰의 목적·수락·진행·완료·선금·보수·지급 조건 확인, '
            'requirements=행동에 필요한 장비·자원·능력치의 선행조건, state=자기 소지품/수치, '
            'world=인물·사건·장소·의뢰 내용·조사 대상·위험 설명, conversation=사실 확인 없는 잡담. '
            'operation은 지금 실행하려는 조작이며, 실제 물품 구매 선언일 때만 purchase다. '
            '질문·가정·구매 의향 설명은 실행과 구별한다. '
            'state_topics는 직접 확인하는 엔진 상태 범위다: money, items, hp, rest, resources. 해당 없으면 빈 배열이다.\n'
            + '현재 NPC: ' + self.npc + '\n이전 대화: ' + json.dumps(recent, ensure_ascii=False))
        if not self.api:
            return "TALK", {"route": "TALK", "no_model": True, "model_calls": 0,
                            "classification_calls": 0, "mapper_calls": 0, "mapper_seconds": 0.0, "state_topics": []}
        preparation_seconds = time.perf_counter() - started
        classification_started = time.perf_counter()
        raw = ("" if input_kind == "action" else self.api["generate_answer"](
            [{"role": "system", "content": system}, {"role": "user", "content": message}],
            max_new_tokens=64, sample=False).strip())
        classification_seconds = time.perf_counter() - classification_started

        def read_object(value):
            try:
                result = json.loads(value.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
                return result if isinstance(result, dict) else {}
            except ValueError:
                return {}

        classified = {"kind": "action", "state_topics": []} if input_kind == "action" else read_object(raw)
        kind = classified.get("kind")
        known_state_topics = ("money", "items", "hp", "rest", "resources")
        raw_topics = classified.get("state_topics")
        if isinstance(raw_topics, list) and all(isinstance(topic, str) and topic in known_state_topics for topic in raw_topics):
            state_topics = list(dict.fromkeys(raw_topics))
        else:
            state_topics = []
        routes = {"social": ("TALK", "inform", False, False), "decline": ("TALK", "decline", False, False),
            "listen": ("TALK", "listen", False, False),
            "accept": ("TALK", "accept", False, False),
            "farewell": ("TALK", "farewell", False, False), "knowledge": ("TALK", "ask", True, False),
            "state": ("TALK", "ask", True, True), "action": ("CLARIFY", "action", True, True),
            "look": ("LOOK", "action", False, False), "rule": ("RULE", "ask", True, False),
            "clarify": ("CLARIFY", "action", False, False)}
        classification_valid = isinstance(kind, str) and kind in routes
        if not classification_valid:
            raise ValueError("입력 분류 결과를 읽을 수 없습니다. 같은 입력을 다시 보내 주세요.")
        kind = kind if classification_valid else "clarify"
        classified_kind, question_form = kind, _question_form(message)
        information_override = kind in {"social", "clarify"} and _information_question(message)
        if information_override:
            kind = "knowledge"
        if question_form and kind in {"action", "look"} and input_kind != "action":
            kind = "knowledge"
        selected, intent, info, state = routes[kind]
        if input_kind == "speech" and kind in {"action", "look", "accept", "listen"}:
            # 대화 선택지는 조건·의뢰 조회도 분류하지만 행동 실행으로 바뀌지는 않습니다.
            kind, selected, intent = "knowledge", "TALK", "ask"
        if kind == "action" and classified.get("operation") == "purchase":
            purchase = self._purchase_route(message, structured_intent=True)
            if purchase is not None:
                selected, details = purchase
                self.turn_intent = "action"
                return selected, {"route": selected, "kind": "action", "intent": "action", **details,
                    "state_topics": [], "model_calls": int(input_kind != "action"),
                    "classification_calls": int(input_kind != "action"), "mapper_calls": 0,
                    "classification_seconds": round(classification_seconds, 3), "mapper_seconds": 0.0,
                    "seconds": round(time.perf_counter() - started, 3)}
        # '듣겠다'와 '사실 설명이 필요하다'는 동시에 성립할 수 있습니다.
        info = info or classified.get("needs_information") is True or _information_question(message)
        mapper_raw, mapper_seconds, mapper_calls = "", 0.0, 0
        navigation_resolution = None
        blocked_action = None
        if kind in {"listen", "accept"}:
            offers = [row.id for row in self.choices() if self.content.choices.get(row.id, {}).get("intent") == "accept"
                      and self.content.choices[row.id].get("target_npc") == self.npc]
            undisclosed = [row["id"] for row in self._request_candidates()
                           if row["id"] not in self.engine.s.vars.get("request_disclosures", {})]
            if len(undisclosed) == 1:
                selected, intent = "offer:" + undisclosed[0], "listen"
            elif kind == "accept" and len(offers) == 1:
                selected = offers[0]
            else:
                # 설명을 아직 듣지 않은 수락은 대화를 이어 가며 보상·물품을 미리 지급하지 않습니다.
                selected = "TALK"
        if kind in {"action", "look"}:
            choices = self.choices()
            if self.npc:
                current = self.npc_cards()[self.npc]
                repeated = {self._approach(current), "talk:" + current["id"]}
                choices = [choice for choice in choices if choice.id not in repeated]
            locations = self.map_locations()
            meetings = [{"id": "meet:" + person["card_id"], "label": person["name"] + "을 찾아가 이야기한다",
                         "destination": room["name"], "check": ""}
                        for room in locations for person in room["npcs"]
                        if person["can_meet"] and person["name"] != self.npc]
            travels = [{"id": "travel:" + room["id"], "label": room["name"] + "까지 이동한다",
                        "destination": room["name"], "target": room["name"], "operation": "travel", "check": ""}
                       for room in locations if room["reachable"] and not room["current"]]
            explicit_action = message.startswith("**") and message.endswith("**") and bool(message[2:-2].strip())
            action_rows = []
            for choice in choices:
                definition = self.content.choices.get(choice.id, {})
                action_rows.append({"id": choice.id, "label": choice.label, "check": choice.check_text,
                    "destination_id": action_destination(definition),
                    **{key: public for key in ("interaction", "intent", "target_npc", "purpose", "target", "operation")
                       if key in definition and (public := self._public_text(definition[key])) is not None}})
            offers = [row for row in action_rows if row.get("intent") == "accept" and row.get("target_npc") == self.npc]
            blocked = []
            available_ids = {row["id"] for row in action_rows}
            _, public_paths = self._public_paths()
            for definition in self.content.choices.values():
                room_id = definition["room"]
                if (definition["id"] in available_ids or not definition.get("target")
                        or not definition.get("operation") or room_id not in public_paths):
                    continue
                if room_id == self.engine.s.room and not definition.get("blocked_reason"):
                    continue
                public = {key: self._public_text(definition[key]) for key in
                    ("id", "label", "purpose", "target", "operation", "blocked_reason") if key in definition}
                if any(value is None for value in public.values()):
                    continue
                blocked.append(public)
                if room_id != self.engine.s.room:
                    blocked[-1]["blocked_reason"] = "이 행동은 " + self._room_label(room_id) + "에서 할 수 있습니다. 먼저 그 장소로 이동해야 합니다."
                    next_id = "move:" + public_paths[room_id][0]
                else:
                    next_id = definition.get("next_action_id")
                if next_id in available_ids:
                    blocked[-1].update(next_action_id=next_id,
                        next_action_label=next(row["label"] for row in action_rows if row["id"] == next_id))
            surrounding = {"id": "look:surroundings", "label": "현재 장소의 주변 전체를 둘러본다",
                           "target": self.content.room_name(self.engine.s.room), "operation": "observe_surroundings"}
            data = {"npc": self.npc, "recent_dialogue": recent, "explicit_action": explicit_action,
                "available_actions": action_rows + meetings + travels + [surrounding], "current_offers": offers,
                "blocked_actions": blocked}
            mapper_system = (
                '현재 플레이어의 의도를 실제 가능한 행동 목록에 연결한다. JSON으로 반환한다: '
                '{"action_id":"현재 목록의 정확한 ID 또는 빈 문자열","intent":"action|accept|ask|hypothetical|joke|decline",'
                '"offer_id":"수락한 current_offers의 ID 또는 빈 문자열","blocked_action_id":"원하는 blocked_actions의 ID 또는 빈 문자열",'
                '"operation":"purchase|none"}. '
                'intent는 실행(action), 의뢰 수락(accept), 질문(ask), 가정(hypothetical), 농담(joke), 거절(decline)을 구분한다. '
                '대상·목적이 맞는 현재 목록의 ID를 선택하고, 수락한 의뢰는 offer_id, 조건이 막힌 행동은 blocked_action_id에 연결한다. '
                '물품을 구입하려는 실행은 operation=purchase이며, 판매 목록만 보는 행동으로 대신하지 않는다. '
                '일치하는 행동이나 의뢰가 없으면 해당 ID는 빈 문자열이다.\n현재 자료:\n'
                + json.dumps(data, ensure_ascii=False))
            mapped = self._explicit_public_navigation(message, locations) if kind == "action" else None
            if mapped is not None:
                navigation_resolution = {"source": "public_entity", "action_id": mapped["action_id"]}
            else:
                mapper_started = time.perf_counter()
                mapper_raw = self.api["generate_answer"]([{"role": "system", "content": mapper_system},
                    {"role": "user", "content": message}], max_new_tokens=160, sample=False).strip()
                mapper_seconds, mapper_calls = time.perf_counter() - mapper_started, 1
                mapped = read_object(mapper_raw)
            identifier, intent = mapped.get("action_id"), mapped.get("intent")
            allowed = ({choice.id for choice in choices} | {meeting["id"] for meeting in meetings}
                       | {travel["id"] for travel in travels} | {surrounding["id"]})
            selected = identifier if isinstance(identifier, str) and identifier in allowed else "CLARIFY"
            # 찾아가서 묻겠다는 선언의 내포 질문은 이동 의도를 취소하지 않습니다.
            if (kind == "action" and not question_form and intent == "ask"
                    and selected.startswith(("meet:", "travel:", "move:"))):
                intent = "action"
            if intent in {"action", "accept"}:
                matched_offer = [row["id"] for row in offers if row["id"] == mapped.get("offer_id")]
                if matched_offer or selected == "CLARIFY" and intent == "accept" and len(offers) == 1:
                    selected = matched_offer[0] if matched_offer else offers[0]["id"]
                    intent = "accept"
                if selected in {"CLARIFY", "look:surroundings"}:
                    blocked_action = next((row for row in blocked if row["id"] == mapped.get("blocked_action_id")), None)
                    if blocked_action is None:
                        blocked_action = next((row for row in blocked if row["id"] == identifier), None)
                    if blocked_action is not None:
                        selected = "CLARIFY"
            if not isinstance(intent, str) or intent not in {"action", "accept", "ask", "hypothetical", "joke", "decline"}:
                selected, intent = "CLARIFY", "action"
            elif intent not in {"action", "accept"}:
                selected = ("CLARIFY" if explicit_action and selected == "CLARIFY"
                            else "TALK" if self.npc and input_kind != "action" else "CLARIFY")
            if intent == "joke":
                info = state = False
            if selected == "look:surroundings":
                selected = "LOOK"
            if intent == "action" and mapped.get("operation") == "purchase":
                purchase = self._purchase_route(message, structured_intent=True)
                if purchase is not None:
                    selected, details = purchase
                    blocked_action = details.get("blocked_action")
                    navigation_resolution = {"source": "purchase_operation", "action_id": selected,
                                             "item_id": details.get("item_id")}
        uncertain = selected == "CLARIFY" and bool(self.npc) and kind not in {"action", "look"}
        if uncertain:
            # 분류만 불확실한 질문은 현재 인물과 확인합니다. 이 경로는 행동을 실행하지 않습니다.
            selected, intent = "TALK", "clarify"
            if classification_valid:
                info = state = True
        state = state or bool(state_topics)
        self.turn_intent, self.turn_needs_information, self.turn_needs_state = intent, info, state
        return selected, {"route": selected, "kind": kind, "raw_route": raw, "raw_mapper": mapper_raw,
            "classified_kind": classified_kind, "question_form": question_form,
            "information_request_override": information_override,
            "intent": intent, "needs_information": info, "needs_state": state, "state_topics": state_topics,
            "needs_services": classified.get("fact_scope") == "services" or classified.get("needs_services") is True,
            "fact_scope": classified.get("fact_scope") if classified.get("fact_scope") in
                {"world", "state", "services", "requirements", "directions", "contracts", "conversation"} else "world",
            "operation": classified.get("operation", "none"),
            "blocked_action": blocked_action,
            "navigation_resolution": navigation_resolution,
            "input_uncertain": uncertain,
            "model_calls": int(input_kind != "action") + mapper_calls,
            "classification_calls": int(input_kind != "action"), "mapper_calls": mapper_calls,
            "preparation_seconds": round(preparation_seconds, 3),
            "classification_seconds": round(classification_seconds, 3), "mapper_seconds": round(mapper_seconds, 3),
            "seconds": round(time.perf_counter() - started, 3)}

    def send_speech(self, message):
        return self.send(message, input_kind="speech")

    def send_action(self, message):
        return self.send(message, input_kind="action")

    def send(self, message, input_kind=None):
        """현재 입력의 분류 기록이 다음 버튼 실행에 섞이지 않게 호출 동안만 연결합니다."""
        started = time.perf_counter()
        diagnostic_count = len(self.turn_records)
        log_count = len(self.engine.s.log)
        try:
            return self._send_body(message, input_kind)
        except Exception as error:
            if self.api and isinstance(error, self.api.get("OutputStreamCancelled", ())):
                raise
            stage = "generation" if self._active_route_diagnostics else "routing"
            text = self._visible_error(message, error, stage=stage,
                                       engine_committed=len(self.engine.s.log) != log_count)
            if len(self.turn_records) == diagnostic_count:
                self._record_turn_diagnostics("action" if input_kind == "action" else "dialogue", message,
                    {"npc_name": self.npc}, {"text": text, "model_generated": False,
                    "response_origin": "visible_error"}, started, error=error,
                    route_debug=self._active_route_diagnostics)
            raise
        finally:
            self._active_route_diagnostics = None

    def _send_body(self, message, input_kind=None):
        if input_kind not in (None, "speech", "action"):
            raise ValueError("입력 종류는 말하기 또는 행동이어야 합니다.")
        if not isinstance(message, str):
            raise TypeError("입력은 문자열이어야 합니다.")
        message = message.strip()
        if not message or len(message) > 1200:
            raise ValueError("1~1200자로 말이나 행동을 입력하세요.")
        current_stage("routing", detail="플레이어 입력 접수", player_input=message,
                      input_kind=input_kind, npc_name=self.npc)
        self.cancel_recommendations()
        turn_started = time.perf_counter()
        self._consume_suggestions()
        self.turn_intent = None
        self.turn_needs_information = self.turn_needs_state = False
        if message.startswith("**") and message.endswith("**") and message[2:-2].strip():
            input_kind = "action"
        self.last_input_kind = input_kind or "speech"
        self.append_message({"role": "user", "content": message, "input_kind": self.last_input_kind,
                              "speaker_id": self.engine.hero.id, "speaker": self.engine.hero.name})
        if self.pending and input_kind != "speech" and message in {"굴린다", "굴려", "주사위 굴려", "실행", "확인", "네", "응"}:
            self.resolve_pending()
            return
        self.pending = None
        if message.startswith("/굴림 "):
            self.utility_dice(message[4:].strip())
            return
        if message.startswith("/행동 "):
            self.propose(message[4:].strip(), direct=True)
            return
        addressed = self._addressed_person(message) or self._declared_npc_target(message)
        if addressed and not self._can_address(addressed[1]):
            text = "지금 이곳에서는 그 상대와 대화할 수 없습니다. 발견한 장소와 이동 경로를 확인해 주세요."
            self.append_message({"role": "assistant", "content": text, "speaker": "", "kind": "narration"})
            self.last_status = "현재 장소의 대화 가능 여부를 확인했습니다."
            route_debug = {"kind": "locality_guard", "intent": "address", "model_calls": 0,
                           "target_card_id": addressed[1]["id"], "current_room": self.engine.s.room}
            self.debug = {"의도 해석": route_debug}
            self._record_turn_diagnostics("dialogue", message, {"npc_name": ""},
                {"text": text, "model_generated": False}, turn_started, route_debug=route_debug)
            self.refresh_suggestions()
            return
        if self.npc and not self._can_address(self.npc_cards().get(self.npc, {})):
            self.npc = ""  # 이전 장소의 대화 상대를 현재 화자로 이어 붙이지 않습니다.
        if addressed and addressed[0] != self.npc:
            self.talk_to(addressed[1]["id"])
        if message.startswith("/규칙 "):
            route, route_debug = "RULE", {"route": "RULE", "command": True}
        else:
            route, route_debug = self.route_message(message, input_kind=input_kind)
        route_debug.update(_turn_started=turn_started, _player_input=message)
        current_stage("routing", detail="입력 경로 확인 완료", route=route,
                      elapsed_seconds=time.perf_counter() - turn_started,
                      routing_metadata={key: deepcopy(value) for key, value in route_debug.items()
                          if key in {"kind", "intent", "needs_information", "classification_calls", "mapper_calls",
                                     "classification_seconds", "mapper_seconds", "model_calls", "seconds"}})
        self._active_route_diagnostics = route_debug
        if route_debug.get("blocked_action"):
            blocked = route_debug["blocked_action"]
            text = blocked["blocked_reason"]
            if blocked.get("next_action_id") in {choice.id for choice in self.choices()}:
                text += "\n\n다음 가능한 행동: " + blocked["next_action_label"]
                next_steps = [{"label": blocked["next_action_label"], "text": blocked["next_action_label"],
                               "action_id": blocked["next_action_id"], "source": "prerequisite"}]
            else:
                next_steps = []
            self.append_message({"role": "assistant", "content": text})
            self.debug = {"의도 해석": route_debug}
            self.last_status = "선행 조건을 안내했습니다. 행동은 실행하지 않았습니다."
            self._record_turn_diagnostics("action", message, {"npc_name": ""},
                {"text": text, "model_generated": False}, turn_started, route_debug=route_debug)
            self.refresh_suggestions(next_steps)
            if next_steps:
                # 이번에 요청한 행동의 선행 단계는 일반 장면 추천보다 우선합니다.
                self.suggestions = [self._suggestion_option(option) for option in next_steps]
            return
        if route.startswith("travel:"):
            self.debug = {"의도 해석": route_debug}
            self.travel_to(route[7:])
            return
        if route.startswith("meet:"):
            self.debug = {"의도 해석": route_debug}
            self.visit_npc(route[5:])
            return
        if route not in {"TALK", "LOOK", "RULE", "CLARIFY"}:
            self.debug = {"의도 해석": route_debug}
            # 목적지가 명확한 이동·대화 시작은 바로 진행합니다. 거래·판정은 확인을 유지합니다.
            harmless = route.startswith(("move:", "talk:", "npc:attack:")) or self.content.choices.get(route, {}).get("interaction") == "talk"
            definition = self.content.choices.get(route, {})
            acceptance = (route_debug.get("intent") == "accept" and definition.get("intent") == "accept"
                          and definition.get("target_npc") == self.npc)
            self.propose(route, direct=harmless or acceptance or (message.startswith("**") and message.endswith("**")))
            self.refresh_suggestions()
            return
        if route == "CLARIFY":
            context = self.scene_context()
            options = [person["label"] + "에게 말을 걸기" for person in context["people"]]
            options += [item["destination"] + " 쪽으로 가기" for item in context["exits"]]
            text = ("현재 가능한 행동과 연결되지 않아 실행하지 않았습니다. 지도나 가능한 행동에서 대상을 골라 주세요."
                    if route_debug.get("kind") in {"action", "look"} else
                    "어느 쪽으로 움직일까요?" if context["exits"] else "어떤 대상을 살피거나 말을 걸까요?")
            # 이미 실패한 같은 판정은 대상 불명 입력과 구별해 이유를 설명합니다.
            request_text = re.sub(r"[\W_]+", "", message).casefold()
            attempted = next((row for row in self.content.choices.values()
                if row["room"] == self.engine.s.room
                and re.sub(r"[\W_]+", "", row["label"]).casefold() == request_text), None)
            if attempted and (attempted["id"], self.engine.hero.id) in self.engine.s.failed:
                text = "이미 실패한 판정입니다. 같은 조건에서 다시 굴릴 수 없습니다. 다른 방법을 선택해 주세요."
            if options:
                text += "\n\n" + " · ".join(options)
            self.append_message({"role": "assistant", "content": text})
            self.debug = {"의도 해석": route_debug, "현재 장소": context}
            self.last_status = "대상을 확인합니다. 위치·물건·시간은 바꾸지 않았습니다."
            self._record_turn_diagnostics("action", message, {"npc_name": ""},
                {"text": text, "model_generated": False}, turn_started, route_debug=route_debug)
            self.refresh_suggestions()
            return
        if route == "LOOK":
            context = self.scene_context()
            texts = [context["observation"], self.scene_opportunities()]
            self.debug = {"의도 해석": route_debug, "표시": "현재 장소의 공개 관찰 자료"}
            reference = "\n\n".join(value for value in texts if isinstance(value, str) and value.strip())
            if reference:
                # 관찰 자료도 같은 행동 결과 출력 경로에서 이번 차례에 한 번만 표현합니다.
                self._record([Event("scene_cue", reference)], player_input=message)
            else:
                self.last_status = "현재 장소에서 새로 표현할 공개 관찰 자료가 없습니다."
            self.refresh_suggestions()
            return
        if route == "TALK" and not self.npc and not self.scene_focus_explicit:
            people = self.visible_people()
            if len(people) == 1:
                # 한 사람이 눈앞에 있으면 대화 선택 상자를 요구하지 않고 실제 만남부터 기록합니다.
                self.talk_to(people[0]["name"])
                route_debug["automatic_contact"] = people[0]["name"]
            elif len(people) > 1:
                self.append_message({"role": "assistant", "content": "누구에게 말을 걸까요?\n\n" + " · ".join(person["label"] for person in people)})
                self.last_status = "현재 보이는 사람 중 대화할 상대를 확인합니다."
                self.debug = {"의도 해석": route_debug, "현재 장소": self.scene_context()}
                return
        route_debug["_turn_started"] = turn_started
        self._generate_dialogue(message, route, route_debug)

    def retryable_dialogue_turn_id(self):
        """실패 뒤 같은 대화·장소에 머물러 있을 때만 재시도 대상을 반환합니다."""
        failed = self._failed_dialogue
        if (not failed or self.pending or not self.turn_records
                or self.turn_records[-1]["turn_id"] != failed["turn_id"]
                or self.turn_records[-1].get("status") != "error"
                or failed["revision"] != self.revision or failed["room"] != self.engine.s.room
                or failed["npc"] != self.npc
                or failed["user_count"] != sum(row.get("role") == "user" for row in self.messages)):
            return None
        history = self.histories.get(failed["speaker"], [])
        if (not history or history[-1].get("role") != "user"
                or history[-1].get("content") != failed["message"]
                or history[-1].get("generation_status") != "failed"):
            return None
        return failed["turn_id"]

    def retry_failed_dialogue(self, expected_turn_id):
        """사용자 질문을 다시 추가하거나 엔진 행동을 재실행하지 않고 답변만 재시도합니다."""
        if type(expected_turn_id) is not int or self.retryable_dialogue_turn_id() != expected_turn_id:
            raise ValueError("현재 대화에서 실패한 답변만 다시 받을 수 있습니다.")
        failed = deepcopy(self._failed_dialogue)
        self.cancel_recommendations()
        self.last_input_kind = failed["input_kind"]
        route_debug = failed["route_debug"]
        route_debug.update(_turn_started=time.perf_counter(), _player_input=failed["message"],
            retry_of_turn_id=expected_turn_id, seconds=0.0, classification_seconds=0.0,
            mapper_seconds=0.0, model_calls=0, classification_calls=0, mapper_calls=0)
        once_key = failed.get("suggestion_once_key")
        try:
            result = self._generate_dialogue(failed["message"], failed["route"], route_debug)
        except Exception:
            if once_key and self._failed_dialogue is not None:
                self._failed_dialogue["suggestion_once_key"] = once_key
            raise
        if once_key:
            groups = self.engine.s.vars.setdefault("used_suggestion_groups", [])
            if once_key not in groups:
                groups.append(once_key)
        return result

    def _generate_dialogue(self, message, route, route_debug):
        """질문은 생성 전에 보존하고, 성공한 답변만 대화 이력에 추가합니다."""
        started = time.perf_counter()
        speaker = "" if route == "RULE" else self.npc
        history = self.histories.setdefault(speaker, [])
        retry = route_debug.get("retry_of_turn_id") is not None
        reuse = (retry and history and history[-1].get("role") == "user"
            and history[-1].get("content") == message and history[-1].get("generation_status") == "failed")
        if reuse:
            user_record = history[-1]
        else:
            user_record = {"role": "user", "content": message, "input_kind": self.last_input_kind or "speech",
                "speaker_id": self.engine.hero.id, "speaker": self.engine.hero.name, "dialogue_turn": True}
            history.append(user_record)
        user_record["generation_status"] = "pending"
        # 현재 질문은 payload.player_input으로 한 번 전달하고, 이력에는 그 이전 발언만 둡니다.
        captured = {"user_record": user_record, "history_before": deepcopy(history[:-1])}
        try:
            self._generate_dialogue_body(message, route, route_debug, captured)
        except Exception as error:
            user_record["generation_status"] = "completed" if captured.get("blocks") else "failed"
            if not (self.api and isinstance(error, self.api.get("OutputStreamCancelled", ()))):
                self._visible_error(message, error, stage="generation")
            self._record_turn_diagnostics("dialogue", message, captured.get("payload"),
                captured.get("result"), started, route_debug=route_debug,
                blocks=captured.get("blocks"), error=error, history=captured["history_before"])
            self._failed_dialogue = ({"turn_id": self.turn_records[-1]["turn_id"], "message": message,
                "speaker": speaker, "npc": self.npc, "room": self.engine.s.room, "revision": self.revision,
                "input_kind": self.last_input_kind or "speech", "route": route, "route_debug": deepcopy(route_debug),
                "user_count": sum(row.get("role") == "user" for row in self.messages)}
                if user_record["generation_status"] == "failed" else None)
            raise
        user_record["generation_status"] = "completed"
        self._failed_dialogue = None
        self._record_turn_diagnostics("dialogue", message, captured.get("payload"),
            captured.get("result"), started, route_debug=route_debug, blocks=captured.get("blocks"),
            history=captured["history_before"])
        if self._pending_recommendation is not None:
            self._pending_recommendation["diagnostic_turn_id"] = self.turn_records[-1]["turn_id"]

    def _generate_dialogue_body(self, message, route, route_debug, captured):
        if not self.api:
            raise RuntimeError("대화 모델이 연결되지 않았습니다.")
        started = time.perf_counter()
        speaker = "" if route == "RULE" else self.npc
        if speaker and not self._can_address(self.npc_cards().get(speaker, {})):
            raise ValueError("현재 장소에서 대화할 수 없는 상대입니다.")
        history = self.histories.setdefault(speaker, [])
        model_history = native_conversation_history(captured["history_before"])
        recommendation_history = deepcopy(model_history)
        scope = route_debug.get("fact_scope", "world")
        handoff_used = scope not in {"services", "requirements", "state", "directions", "contracts"}
        if handoff_used:
            payload = deepcopy(self.api["get_handoff"](
                message, self.rag_state(), npc_name=speaker, history=model_history))
        else:
            # 수량·가격 조회는 검색 결과나 NPC 발언이 필요하지 않은 읽기 전용 처리입니다.
            payload = {"player_input": message, "game_state": self.rag_state(),
                       "retrieval": {"rule_results": [], "scenario_results": [], "memory_results": []}}
        captured["payload"] = payload
        payload.update(response_contract="game_turn", recommendations_in_router=True, state_snapshot_complete=True,
            suggestion_count=1, turn_intent=self.turn_intent, needs_state=self.turn_needs_state,
            needs_information=bool(self.turn_needs_information))
        payload["needs_services"] = route_debug.get("needs_services") is True
        payload["fact_scope"] = route_debug.get("fact_scope", "world")
        payload["available_actions"] = [{"id": choice.id, "label": choice.label, "check": choice.check_text}
            for choice in self.choices() if not choice.id.startswith("npc:attack:")]
        payload["scene_context"] = self.scene_context()
        payload["progression_options"] = self.progression_options()
        payload["story_progression"] = self.progression_context()
        payload["reference_required"] = bool(self.turn_needs_information
            or payload["story_progression"]["reference_required"])
        payload["confirmed_offer"] = self._confirmed_offer(speaker)
        payload["player_snapshot"] = player_snapshot(self.engine)
        payload["model_state"] = model_state(self.engine, payload["player_snapshot"])
        payload["actor_context"] = ({"name": speaker, "card_id": self.npc_cards()[speaker]["id"]}
                                    if speaker in self.npc_cards() else {})
        payload["allowed_speakers"] = self.allowed_speakers()
        payload["dialogue_state"] = self.dialogue_state(speaker)
        payload["available_checks"] = [{"id": row.id, "label": row.label, "check": row.check_text}
            for row in self.choices() if self.content.choices.get(row.id, {}).get("check")]
        payload["response_mode"] = "rule_explanation" if route == "RULE" else "npc_dialogue" if speaker else "scene_dialogue"
        payload["interaction_context"] = self.interaction_context(speaker)
        payload["guidance_rendered_separately"] = True
        payload["delivered_guidance"] = self.delivered_guidance(speaker)
        payload["dialogue_affordances"] = self.dialogue_affordances()
        payload["input_uncertain"] = bool(route_debug.get("input_uncertain"))
        payload["state_topics"] = list(route_debug.get("state_topics", []))
        payload["reply_format"] = "natural_text"
        # 팀 입력·RAG의 원문, 순위와 출처는 그대로 전달합니다.
        memory_hits = 0
        context_seconds = time.perf_counter() - started
        scope = payload.get("fact_scope")
        if scope in {"services", "requirements", "state", "directions", "contracts"}:
            # 가능 여부·가격·수량·실행 조건은 엔진의 조회 결과입니다. 모델의
            # 그럴듯한 약속으로 바뀌지 않게 기존 시스템 정보 표시 경로를 사용합니다.
            result = {"text": self.confirmed_information(scope, message), "response_format": "engine_state",
                "response_origin": "confirmed_current_information", "model_generated": False,
                "generation_seconds": 0.0, "input_tokens": 0, "output_tokens": 0,
                "engine_state_topics": [scope], "stage_metrics": [], "payload": payload}
        else:
            result = self.api["generate_from_handoff"](payload, history=model_history, npc_name=speaker,
                                                      engine_result=self.last_result)
        captured["result"] = result
        current_stage("validation", detail="생성 결과와 확정 상태 표시 확인",
                      response_origin=result.get("response_origin"))
        engine_only = result.get("response_format") == "engine_state"
        state_display = ""
        explicit_state = (route_debug.get("kind") == "state" and self.turn_needs_state
            and bool(payload["state_topics"]) and all(isinstance(topic, str)
                and topic in {"money", "items", "hp", "rest", "resources"}
                for topic in payload["state_topics"]))
        if explicit_state and not engine_only and callable(self.api.get("format_confirmed_state")):
            state_display = self.api["format_confirmed_state"](payload, engine_result=self.last_result, npc_name=speaker)
        self.last_generation_seconds = result["generation_seconds"]
        slow, ending = self.last_generation_seconds > 6, self.turn_intent in {"decline", "farewell"}
        has_undisclosed = any(row["id"] not in self.engine.s.vars.get("request_disclosures", {})
                             for row in self._request_candidates(speaker))
        can_recommend = (speaker and not engine_only and not self.turn_needs_state and not ending and not has_undisclosed
                         and callable(self.api.get("generate_followup_question")))
        followup = {"called": False, "seconds": 0.0, "wall_seconds": 0.0,
                    "reason": "deferred" if can_recommend else "state_uses_engine_choices" if engine_only or self.turn_needs_state else "ending" if ending else "not_npc" if not speaker else "api_unavailable"}
        recommendations = []
        result["suggestions"] = [row["text"] for row in recommendations if not row.get("action_id")]
        turn = sum(row.get("role") == "user" for row in history)
        if not engine_only:
            blocks = self._append_model_response(result, speaker)
            captured["blocks"] = blocks
            history.append({"role": "assistant", "content": "\n\n".join(row["text"] for row in blocks),
                            "blocks": deepcopy(blocks), "suggestions": result["suggestions"], "memory_turn": turn,
                            "speaker_id": blocks[0]["speaker_id"] if len(blocks) == 1 else None,
                            "speaker": blocks[0]["speaker"] if len(blocks) == 1 else ""})
        if state_display:
            # 확정 상태는 게임 시스템의 표시이며 NPC가 했던 발언으로 저장하지 않습니다.
            self.append_message({"role": "assistant", "content": state_display, "speaker": ""})
        if engine_only:
            self.append_message({"role": "assistant", "content": result["text"],
                                  "kind": "narration", "speaker_id": None, "speaker": ""})
            # 이미 답한 상태 질문을 다음 NPC 대화에 미답변 질문으로 다시 넣지 않습니다.
            captured["user_record"]["answered_by"] = "game_state"
        elif speaker:
            self._publish_guidance(self.npc_cards()[speaker]["id"])
        captured["user_record"]["generation_status"] = "completed"
        save_history = self.api.get("save_dialogue_history")
        if callable(save_history):
            save_history(self.memory_session, speaker, history)
        route_debug = dict(route_debug)
        route_seconds = route_debug.get("seconds", 0)
        turn_started = route_debug.pop("_turn_started", started - route_seconds)
        elapsed, total = time.perf_counter() - started, time.perf_counter() - turn_started
        timings = {"input_seconds": route_seconds, "context_seconds": round(context_seconds, 3),
            "scope_classification_seconds": route_debug.get("classification_seconds", 0),
            "action_mapper_seconds": route_debug.get("mapper_seconds", 0),
            "body_generation_seconds": result["generation_seconds"], "followup_seconds": followup.get("seconds", 0),
            "followup_wall_seconds": round(followup["wall_seconds"], 3), "dialogue_seconds": round(elapsed, 3),
            "total_seconds": round(total, 3)}
        calls = {"generate_answer": route_debug.get("model_calls", 0), "get_handoff": int(handoff_used),
                 "scope_classification": route_debug.get("classification_calls", 0),
                 "action_mapper": route_debug.get("mapper_calls", 0),
                 "add_memory_context": 0, "generate_from_handoff": int(not engine_only),
                 "body_model_generation": len(result.get("stage_metrics", [])), "generate_followup_question": int(followup["called"])}
        self.debug = {"의도 해석": route_debug, "입력·검색": payload,
            "생성": json.loads(self.api["generation_for_display"](result)), "후속 질문": followup,
            "처리 시간": timings, "호출 수": calls, "사건": self.last_result, "추천": recommendations}
        if state_display:
            self.debug["현재 상태 표시"] = state_display
        self.debug["생성"]["표시된 본문 종류"] = "엔진 상태" if engine_only else "NPC 대사"
        self.debug["생성"]["모델 초안 대체 범위"] = result.get("engine_state_topics", [])
        self.refresh_suggestions(recommendations, player_message=message)
        self.last_status = (f"전체 {total:.2f}초 / 입력 분류 {route_debug.get('classification_seconds', 0):.2f}초 / "
            f"행동 연결 {route_debug.get('mapper_seconds', 0):.2f}초 / 검색·자료 {context_seconds:.2f}초 / "
            f"답변 생성 {result['generation_seconds']:.2f}초 / 후속 질문 {followup.get('seconds', 0):.2f}초 / 기억 {memory_hits}개")
        self.revision += 1
        if can_recommend:
            self._recommendation_cancel = CancellationEvent()
            count = 1 if slow else 2
            job_payload = deepcopy(result["payload"])
            job_payload["suggestion_count"] = count
            self._pending_recommendation = {
                "epoch": self._recommendation_epoch, "revision": self.revision,
                "speaker": speaker, "room": self.engine.s.room,
                "player_message": message, "payload": job_payload, "reply": result["text"],
                "history": recommendation_history, "memory_turn": turn, "count": count,
                "queued_at": time.perf_counter(), "cancel_event": self._recommendation_cancel}

    def save_data(self):
        """JSON 상태·다음 주사위·사망 복구 지점을 함께 저장합니다."""
        return {"schema": "trpg_game_save_1", "state": _jsonable(asdict(self.engine.s)),
                "rng": _jsonable(self.engine.dice.rng.getstate()), "npc": self.npc,
                "scene_focus_explicit": self.scene_focus_explicit,
                "histories": deepcopy(self.histories), "messages": deepcopy(self.messages),
                "audit": self.audit[-200:], "revision": self.revision,
                "pending_choice": deepcopy(self.pending) if self.pending and self.pending.get("type") == "choice" else None,
                "turn_records": deepcopy(self.turn_records), "diagnostics_meta": deepcopy(self.diagnostics_meta),
                "transaction_ledger": deepcopy(self.transaction_ledger),
                "engine_checkpoint": ({"state": _jsonable(asdict(self.engine.checkpoint)),
                    "label": self.engine.checkpoint_label} if self.engine.checkpoint is not None else None)}

    def restore_data(self, data):
        """임시 게임에서 검사·화면 준비가 끝난 뒤에만 현재 게임을 한 번에 바꿉니다.

        손상된 저장, 잘못된 전투, 대화 기억 연결 실패는 기존 게임과 기억을 그대로
        둡니다. 후보는 api=None이므로 검사 중 검색·대사 생성·외부 효과가 없습니다.
        """
        from save_contract import validate_save_payload
        from play_view import (action_entries, discovery_data, discovery_overview, discovery_samples,
            people_panel, discovery_actions, inventory_panel, dice_panel, encounter_panel, journey_panel)
        from portrait_view import present_messages

        validate_save_payload(self.content, data)
        incoming = deepcopy(data)
        # v20의 이름 언급 영수증은 안내 사건이 아닙니다. 아직 실제로 만나지 않은
        # 대상의 옛 해금은 후보에서만 거두고 현재 NPC에게 다시 안내받게 합니다.
        for raw in [incoming["state"], (incoming.get("engine_checkpoint") or {}).get("state", {})]:
            values = raw.get("vars", {})
            ledger = values.get("campaign_introductions", {})
            for target, receipt in list(ledger.items()):
                if not receipt.get("guidance_id") and target not in raw.get("shown", {}):
                    del ledger[target]
        candidate = PlaySession(self.content, None, incoming["state"]["party"][0]["id"])
        candidate.rule_mode = self.rule_mode
        candidate._restore_data_unchecked(incoming)
        # 대기 행동은 출력의 현재 공간·소개 조건까지 만족해야 합니다. 원본 엔진의
        # 가능 목록만 확인하면 여관 안에서 대장간 행동을 복원할 수 있습니다.
        if candidate.pending:
            views = {row.id: row for row in candidate.choices()}
            view = views.get(candidate.pending["choice"])
            actor = candidate.pending.get("actor_id", candidate.engine.hero.id)
            if view is None or (view.actors and actor not in view.actors):
                raise ValueError("저장된 행동을 현재 장소와 진행 상태에서 실행할 수 없습니다.")
            candidate.pending_check_options()
        # 상태 사전뿐 아니라 실제 화면이 읽는 모든 표를 후보에서 준비합니다.
        public = discovery_data(candidate)
        fields_to_check = {"rag": candidate.rag_state(), "scene": candidate.scene_context(),
            "quest": candidate.quest_journal(), "actions": action_entries(candidate),
            "discovery": public, "people": people_panel(candidate, public),
            "overview": discovery_overview(candidate, public), "samples": discovery_samples(candidate, public),
            "navigation": discovery_actions(candidate, public), "inventory": inventory_panel(candidate),
            "dice": dice_panel(candidate), "combat": encounter_panel(candidate),
            "journey": journey_panel(candidate, public), "messages": present_messages(candidate.messages)}
        json.dumps(fields_to_check, ensure_ascii=False, allow_nan=False)

        # 새 기억을 먼저 준비합니다. 여기서 실패해도 이전 기억을 닫지 않습니다.
        api, old_memory, prepared_memory = self.api, self.memory_session, None
        if api:
            create, save, close = (api.get(name) for name in
                ("create_memory_session", "save_dialogue_history", "close_memory_session"))
            if not callable(create) or not callable(save) or not callable(close):
                raise ValueError("대화 기억 연결 함수를 확인할 수 없습니다.")
            try:
                prepared_memory = create()
                if not isinstance(prepared_memory, str) or not prepared_memory:
                    raise ValueError("새 대화 기억 ID를 확인할 수 없습니다.")
                for speaker, history in candidate.histories.items():
                    save(prepared_memory, speaker, deepcopy(history))
            except Exception:
                if prepared_memory:
                    try:
                        close(prepared_memory)
                    except Exception:
                        pass
                raise
        # 현재 접속자의 취소 세대와 개발자 되돌리기 기록을 보존합니다. 후보의
        # 새 게임 시작 스냅샷이 이 접속자의 기존 기록을 덮어쓰지 않습니다.
        protected = {"content", "api", "memory_session", "_developer_snapshots", "_developer_snapshot_serial",
            "_restoring_developer_snapshot", "_recommendation_epoch", "_recommendation_cancel", "_pending_recommendation"}
        prepared = {key: value for key, value in candidate.__dict__.items() if key not in protected}
        self.cancel_recommendations()
        self.__dict__.update(prepared)
        self.memory_session = prepared_memory
        self._last_record_minutes = self.engine.s.minutes
        self._capture_developer_checkpoint("저장 파일 불러오기")
        # 교체 후의 이전 기억 정리는 게임 복원 성공과 분리합니다.
        if api and old_memory and old_memory != prepared_memory:
            try:
                api["close_memory_session"](old_memory)
            except Exception:
                self.audit.append({"kind": "warning", "text": "이전 대화 기억 정리를 완료하지 못했습니다.", "data": {}})

    def restart_checkpoint(self):
        """사망 복구는 원본 엔진의 지점으로 이동하고 새 대화 가지를 시작합니다."""
        if self.engine.s.ended != "game_over":
            raise ValueError("게임 오버 후 체크포인트에서 다시 시작할 수 있습니다.")
        if self.engine.checkpoint is None:
            raise ValueError("이전 저장에는 복구 지점이 없습니다. 새 여정이나 별도 저장을 선택하세요.")
        candidate = PlaySession(self.content, None, self.engine.hero.id)
        candidate._restore_data_unchecked(self.save_data())
        events = candidate.engine.restart_from_checkpoint()  # 난수는 되감지 않는 원본 동작입니다.
        candidate.pending, candidate.engine.s.pending = None, None
        candidate.histories, candidate.turn_records = {}, []
        candidate.transaction_ledger = restore_ledger(None, candidate.engine.s.log)
        # 미래 대화는 지우되 소개 영수증이 참조하는 최소한의 실제 안내는 보존합니다.
        provenance, indices = [], {}
        for receipt in candidate.engine.s.vars.get("campaign_introductions", {}).values():
            old_index = receipt["message_index"]
            if old_index not in indices:
                indices[old_index] = len(provenance)
                provenance.append(deepcopy(candidate.messages[old_index]))
            receipt["message_index"] = indices[old_index]
        candidate.messages = provenance + [{"role": "assistant", "kind": "narration", "speaker_id": None,
            "speaker": "", "content": "체크포인트에서 다시 시작했습니다. 이후 대화는 새 진행으로 구분합니다."}]
        candidate.npc, candidate.scene_focus_explicit = "", True
        candidate.engine.checkpoint = deepcopy(candidate.engine.s)
        candidate.last_result, candidate.debug = None, {}
        candidate._record(events, player_input="체크포인트에서 다시 시작한다")
        self.restore_data(candidate.save_data())
        self.last_status = "체크포인트에서 다시 시작했습니다. 대화 기억과 이후 진행을 새로 시작합니다."

    def _restore_data_unchecked(self, data):
        """검사를 통과한 자료를 api=None인 후보에 적용합니다. 직접 UI에 연결하지 않습니다."""
        if not isinstance(data, dict) or data.get("schema") != "trpg_game_save_1":
            raise ValueError("지원하지 않는 저장 파일입니다.")
        # 값을 모두 확인한 뒤에만 현재 게임을 바꿉니다. int 변환을 나중에 하면
        # 손상된 revision을 거절하면서 위치·대화만 이미 적용되는 문제가 생깁니다.
        revision = data.get("revision", 0)
        if type(revision) is not int or revision < 0:
            raise ValueError("저장된 변경 번호가 올바르지 않습니다.")
        saved_npc, saved_audit = data.get("npc", ""), data.get("audit", [])
        if not isinstance(saved_npc, str) or not isinstance(saved_audit, list):
            raise ValueError("대화 상대·활동 기록이 올바르지 않습니다.")
        scene_focus = data.get("scene_focus_explicit", False)
        if type(scene_focus) is not bool:
            raise ValueError("대화 대상 저장 형식이 올바르지 않습니다.")
        raw = deepcopy(data["state"])
        if raw.get("room") not in self.content.rooms or raw.get("gold_cp", -1) < 0:
            raise ValueError("장소·금액이 올바르지 않습니다.")
        if not isinstance(raw.get("vars", {}), dict):
            raise ValueError("진행 기록이 올바르지 않습니다.")
        campaign_dialogues = raw.get("vars", {}).get("campaign_dialogues", [])
        if (not isinstance(campaign_dialogues, list)
                or any(not isinstance(value, str) or value not in CONVERSATION_MILESTONES
                       for value in campaign_dialogues)):
            raise ValueError("초반 대화 진행 기록이 올바르지 않습니다.")
        introductions = raw.get("vars", {}).get("campaign_introductions", {})
        if not isinstance(introductions, dict):
            raise ValueError("소개 완료 기록이 올바르지 않습니다.")
        for target_id, receipt in introductions.items():
            if (target_id not in INTRODUCTION_SOURCES or not isinstance(receipt, dict)
                    or receipt.get("target_npc_id") != target_id
                    or receipt.get("source_npc_id") != INTRODUCTION_SOURCES[target_id]
                    or receipt.get("room_id") not in {*self.content.rooms, *MARKET_SPACES}
                    or type(receipt.get("message_index")) is not int
                    or receipt["message_index"] < 0):
                raise ValueError("소개 완료 항목이 올바르지 않습니다.")
        conversations = raw.get("vars", {}).get("campaign_dialogues", [])
        if (not isinstance(conversations, list) or any(not isinstance(value, str)
                or value not in CONVERSATION_MILESTONES for value in conversations)):
            raise ValueError("초반 대화 진행 기록이 올바르지 않습니다.")
        discovery = raw.get("vars", {}).get("discovery") if isinstance(raw.get("vars"), dict) else None
        if discovery is not None and (not isinstance(discovery, dict)
                or any(not isinstance(discovery.get(key, []), list) or any(
                    not isinstance(value, str) or value not in allowed for value in discovery.get(key, []))
                    for key, allowed in (("rooms", self.content.rooms),
                        ("npcs", {card["id"] for card in self.npc_cards().values()})))
                or not isinstance(discovery.get("paths", {}), dict)
                or any(room not in self.content.rooms or not isinstance(label, str) or len(label) > 200
                       for room, label in discovery.get("paths", {}).items())):
            raise ValueError("발견 기록이 올바르지 않습니다.")
        if not isinstance(raw.get("party"), list) or len(raw["party"]) != 1:
            raise ValueError("저장 파일에는 플레이어 캐릭터 한 명이 필요합니다.")
        char = raw["party"][0]
        if char.get("id") not in {c[0] for c in self.content.characters()}:
            raise ValueError("알 수 없는 캐릭터입니다.")
        if char.get("cls") not in {"fighter", "rogue", "cleric", "wizard"}:
            raise ValueError("알 수 없는 직업입니다.")
        if not isinstance(char.get("abilities"), dict) or set(char["abilities"]) != set(ABILITY_KO):
            raise ValueError("능력치가 올바르지 않습니다.")
        if not 0 <= char.get("hp", -1) <= char.get("hp_max", 0) or char.get("hp_max", 0) > 10000:
            raise ValueError("HP가 올바르지 않습니다.")
        inventory = raw.get("inventory", {})
        if not isinstance(inventory, dict) or any(k not in self.content.items or type(v) is not int or not 0 <= v <= 9999 for k, v in inventory.items()):
            raise ValueError("소지품이 올바르지 않습니다.")
        allowed = {f.name for f in fields(GameState)}
        if set(raw)-allowed:
            raise ValueError("알 수 없는 저장 상태 필드가 있습니다.")
        raw["party"] = [Character(**char)]
        for key in ["flags", "visited", "done", "once_done"]:
            raw[key] = set(raw.get(key, []))
        raw["failed"] = {tuple(x) for x in raw.get("failed", [])}
        state = GameState(**raw)
        if state.pending and state.pending.get("type") == "inspire":
            check = self.content.choices.get(state.pending.get("choice"), {}).get("check")
            actor = next((member for member in state.party if member.id == state.pending.get("actor")), None)
            if not check or actor is None:
                raise ValueError("저장된 재굴림 대상을 확인할 수 없습니다.")
            self.engine.selected_check(actor, check, state.pending.get("method_index"))
        # 저장 파일을 일부만 적용하지 않도록 먼저 구조와 RNG를 검사합니다.
        dice = Dice()
        dice.rng.setstate(_tuples(data["rng"]))
        histories = data.get("histories", {})
        if not isinstance(histories, dict) or any(not isinstance(v, list) for v in histories.values()):
            raise ValueError("대화 이력이 올바르지 않습니다.")
        for records in [*histories.values(), data.get("messages", [])]:
            if not isinstance(records, list) or any(not isinstance(m, dict) or m.get("role") not in {"user", "assistant"}
                or not isinstance(m.get("content"), str) or len(m["content"]) > 20000 for m in records):
                raise ValueError("대화 메시지가 올바르지 않습니다.")
        transaction_ledger = restore_ledger(data.get("transaction_ledger"), state.log)
        # 예전 저장 파일에는 진단 필드가 없습니다. 새 기록은 개수 제한 없이 복원합니다.
        turn_records = deepcopy(data.get("turn_records", []))
        diagnostics_meta = deepcopy(data.get("diagnostics_meta", self.diagnostics_meta))
        if not isinstance(turn_records, list) or not isinstance(diagnostics_meta, dict):
            raise ValueError("턴별 진단 기록이 올바르지 않습니다.")
        previous_id = 0
        for record in turn_records:
            if (not isinstance(record, dict) or type(record.get("turn_id")) is not int
                    or record["turn_id"] <= previous_id or record.get("kind") not in {"action", "dialogue"}
                    or record.get("status") not in {"completed", "error"}
                    or any(not isinstance(record.get(key), dict) for key in
                           ("input", "timing", "tokens", "output", "engine"))
                    or not isinstance(record.get("input", {}).get("text"), str)
                    or not isinstance(record.get("sources_public"), list)):
                raise ValueError("턴별 진단 항목이 올바르지 않습니다.")
            previous_id = record["turn_id"]
        try:
            json.dumps({"records": turn_records, "meta": diagnostics_meta}, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("진단 기록은 유효한 JSON 값이어야 합니다.") from error
        # 형식만 여기서 읽습니다. 원본 엔진에는 talk: 같은 출력 전용 행동이
        # 없으므로 최종 실행 가능 여부는 바깥 후보의 실제 공개 choices로 확인합니다.
        pending = deepcopy(data.get("pending_choice"))
        if pending is not None:
            if not isinstance(pending, dict) or pending.get("type") != "choice":
                raise ValueError("저장된 행동 확인 형식이 올바르지 않습니다.")
            actor_id = pending.get("actor_id", state.party[0].id)
            if actor_id != state.party[0].id or state.pending:
                raise ValueError("저장된 판정을 현재 상태에서 실행할 수 없습니다.")
            pending["revision"] = revision + 1
        self.cancel_recommendations()
        self.engine.s, self.engine.dice = state, dice
        self.transaction_ledger = transaction_ledger
        self.engine.next_success = False
        self.engine.pending_roll_override = None
        self.engine.developer = False
        self.pending = pending
        self._failed_dialogue = None
        self.histories = deepcopy(histories)
        self.messages = deepcopy(data.get("messages", []))
        self.turn_records, self.diagnostics_meta = turn_records, diagnostics_meta
        self.audit = deepcopy(saved_audit[-200:])
        self.revision = revision + 1
        self.npc = saved_npc
        self.scene_focus_explicit = scene_focus and not self.npc
        if self.npc not in self.npc_cards() or not self._person_present(self.npc_cards()[self.npc]):
            self.npc = ""
        if self.pending and self.pending.get("choice", "").startswith("talk:"):
            card = self.content.cards.get(self.pending["choice"][5:], {})
            if card.get("id") in self.engine.s.shown and self._can_address(card):
                # 구버전의 단순 상대 선택은 현재의 focus 전환으로 복원합니다.
                # 현장·발견·대화 가능 조건을 만족하지 않으면 후보 검사가 거절합니다.
                self.npc, self.scene_focus_explicit, self.pending = npc_name(card), False, None
        checkpoint = data.get("engine_checkpoint")
        if checkpoint is not None:
            saved = deepcopy(checkpoint["state"])
            saved["party"] = [Character(**member) for member in saved["party"]]
            for key in ("flags", "visited", "done", "once_done"):
                saved[key] = set(saved.get(key, []))
            saved["failed"] = {tuple(pair) for pair in saved.get("failed", [])}
            self.engine.checkpoint = GameState(**saved)
            self.engine.checkpoint_label = checkpoint["label"]
        else:
            # 파일에 없는 지점을 새 세션의 '게임 시작'으로 위장하지 않습니다.
            self.engine.checkpoint, self.engine.checkpoint_label = None, ""
        self.append_message({"role": "assistant", "kind": "narration", "speaker_id": None, "speaker": "",
            "content": "저장한 게임을 불러왔습니다. 개발자 성공 보장은 해제했습니다."})
        if self.pending:
            self.pending["revision"] = self.revision
        self.refresh_suggestions()
        self.last_status = "저장 파일 불러오기 완료."


class PlaySessions:
    """브라우저에는 토큰만 두고 실제 게임 객체는 서버에 보관합니다."""

    def __init__(self, api=None, content=None):
        self.api, self.content = api, content or Content()
        self.games = OrderedDict()
        self.lock = RLock()

    def start(self, character, previous=None, seed=None):
        with self.lock:
            self.close(previous)
            if len(self.games) >= 32:
                self.close(next(iter(self.games)))
            token = token_urlsafe(24)
            self.games[token] = PlaySession(self.content, self.api, character, seed)
            return token

    def get(self, token):
        if token not in self.games:
            raise ValueError("캐릭터를 선택하고 게임을 시작하세요.")
        self.games.move_to_end(token)
        return self.games[token]

    def close(self, token):
        game = self.games.pop(token, None)
        if game:
            game.close()
