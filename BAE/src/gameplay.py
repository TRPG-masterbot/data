"""게임 상태·행동·AI 대화를 연결합니다.

규칙 판정은 game.Engine, 대사 생성은 전달받은 출력 모듈이 담당합니다.
자연어로 제안된 행동은 확인을 받고, 버튼이나 **행동** 선언은 직접 실행합니다.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import re
from secrets import token_urlsafe
from threading import RLock
import time

from game.content import Content
from game.dice import Dice, DICE_EXPR
from game.engine import Engine, fmt_money
from game.events import Event, ChoiceView
from game.model import Character, GameState, ABILITY_KO
from game.startup import apply_starting_equipment
from dialogue_projection import eligible_facts
from state_projection import (new_ledger, before_action, record_action, restore_ledger,
    directed_transactions, shop_state, player_snapshot, model_state, mentioned_context_items,
    native_conversation_history)


DIALOGUE_CARDS = json.loads(Path(__file__).with_name("context_cards.json").read_text(encoding="utf-8"))
TRANSACTION_BINDINGS = json.loads(Path(__file__).with_name("transaction_bindings.json").read_text(encoding="utf-8"))




class GameEngine(Engine):
    """게임 규칙에 개발자 조작을 명시적으로 추가한 엔진입니다."""

    def __init__(self, *args, **kwargs):
        self.developer = False
        self.next_success = False
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
        if not self.developer or not self.next_success:
            return False
        self.next_success = False
        self.s.vars["developer_used"] = True
        event.data["developer_override"] = True
        event.data["success"] = True
        # 실제 주사위 값은 남기고, 테스트용 결과 변경을 별도로 표시합니다.
        event.text = event.text.split("→", 1)[0] + "→ 성공 [개발자: 성공 보장 사용]"
        return True

    def roll_check(self, char, chk):
        ok, event = super().roll_check(char, chk)
        return (True if char.id == self.hero.id and self._force(event) else ok), event

    def roll_save(self, char, ability, dc, advantage=False, avoids=None):
        ok, event = super().roll_save(char, ability, dc, advantage, avoids)
        return (True if char.id == self.hero.id and self._force(event) else ok), event

    def _attack_roll(self, monster, bonus, label):
        hit, critical, mode, event = super()._attack_roll(monster, bonus, label)
        if self._force(event):
            event.data["hit"] = True
            hit = True
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
        self.npc = ""
        self.scene_focus_explicit = False  # 장면/게임 마스터를 직접 고른 상태
        self.pending = None
        self.revision = 0
        self.last_result = None
        self.debug = {}
        self.last_status = "게임을 시작했습니다."
        self.rule_mode = "available"
        self.suggestions = []
        self.last_generation_seconds = 0.0
        self.action_generation_calls = 0
        self.turn_intent = None
        self.transaction_ledger = new_ledger()
        self.turn_needs_information = False
        self.turn_needs_state = False
        self.memory_session = api["create_memory_session"]() if api else None
        self._record(self.engine.opening, player_input="게임을 시작한다")
        auto = self.content.rooms[self.engine.s.room].get("auto_meet")
        if auto and auto in self.content.cards:
            profile = self.content.cards[auto]["dialogue"]
            approach = profile.get("approach_action")
            if approach in {choice.id for choice in self.choices()}:
                self._record(self.engine.choose(approach), choice_id=approach,
                             player_input=self.content.choices[approach]["label"])
            else:
                self.talk_to(profile["name"])
        self.refresh_suggestions()

    def close(self):
        if self.api and self.memory_session:
            self.api["close_memory_session"](self.memory_session)

    def profiles(self):
        return {c["dialogue"]["name"]: c for c in self.content.cards.values() if c.get("dialogue")}

    def nearby_npcs(self):
        return [(person["label"], person["name"]) for person in self.visible_people()]

    def visible_people(self):
        """현재 장소의 공개 조건을 통과한 사람과 실제 접근 행동만 노출합니다."""
        available = {choice.id for choice in self.choices()}
        people = []
        for name, card in self.profiles().items():
            if card["room"] != self.engine.s.room or not self.engine.eligible(card["id"]) or not self.engine.npc_can_talk(card["id"]):
                continue
            profile = card["dialogue"]
            people.append({"card_id": card["id"], "name": name, "label": name if card["id"] in self.engine.s.shown else profile.get("appearance", name),
                "presence": profile.get("presence_text", ""),
                "approach_action": (profile.get("approach_action") if profile.get("approach_action") in available
                                    else "talk:" + card["id"] if "talk:" + card["id"] in available else None),
                "met": card["id"] in self.engine.s.shown})
        return people

    def scene_context(self):
        """모델과 화면이 함께 사용하는 현재 장소·사람·출구 정보입니다."""
        room = self.content.rooms[self.engine.s.room]
        aftermath = []
        for name, card in self.profiles().items():
            if card["room"] == self.engine.s.room:
                status = self.engine.npc_state(card["id"])
                if not status["can_talk"]:
                    aftermath.append({"name": name, "status": status["status"], "hp": status["hp"]})
        return {"room_id": self.engine.s.room, "room_name": self.content.room_name(self.engine.s.room),
                "observation": room.get("look_text", ""), "people": self.visible_people(),
                "aftermath": aftermath,
                "exits": [{"id": choice.id, "destination": self.content.room_name(choice.id[5:])}
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
        normalize = lambda value: re.sub(r"[\W_]+", "", value).casefold()
        previous = [row["content"] for row in self.histories.get(self.npc, [])[-12:]
                    if row.get("role") == "user" and isinstance(row.get("content"), str)]
        if player_message:
            previous.append(player_message)
        if self.messages and self.messages[-1]["role"] == "user":
            previous.append(self.messages[-1]["content"])
        target = normalize(text)
        return bool(target) and target in {normalize(value) for value in previous}

    def refresh_suggestions(self, generated=None, player_message=None):
        """장면에 준비된 선택지가 있으면 사용하고, 그 외에는 이번 생성 추천을 씁니다."""
        if self.pending or self.engine.s.combat or self.engine.s.ended:
            self.suggestions = []
            return
        choices = self.choices()
        available = {choice.id for choice in choices}
        used = self.engine.s.vars.get("used_suggestion_groups", [])
        rows = []
        group = None
        for option in self.content.rooms[self.engine.s.room].get("suggestions", []):
            if option.get("npc_name") and option["npc_name"] != self.npc:
                continue
            if option.get("once_key") in used or not self.engine.cond(option.get("requires")):
                continue
            if option.get("action_id") and option["action_id"] not in available:
                continue
            if group is not None and option.get("once_key") != group:
                continue
            group = option.get("once_key")
            rows.append({**option, "source": "scene"})
        target_count = min(2, len(rows)) if rows else (1 if self.last_generation_seconds > 6 else 2)
        candidates = rows or generated or []
        selected, seen = [], set()
        for value in candidates:
            option = dict(value) if isinstance(value, dict) else {"label": value, "text": value, "source": "model"}
            action_id, text = option.get("action_id"), option.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            if action_id and action_id not in available:
                continue
            if not action_id and self._asked_recently(text, player_message):
                continue
            key = action_id or re.sub(r"[\W_]+", "", text).casefold()
            if key not in seen:
                selected.append(option)
                seen.add(key)
        # 추천 질문을 제외한 뒤에도 선택할 수 있는 실제 행동을 남깁니다.
        for choice in choices:
            if len(selected) >= target_count:
                break
            if choice.id in seen or choice.id.startswith(("npc:attack:", "buy:", "use:", "talk:")):
                continue
            if generated is not None and self.turn_intent in {"decline", "farewell"} and not choice.id.startswith("move:"):
                continue
            selected.append({"label": choice.label, "text": choice.label, "action_id": choice.id, "source": "engine"})
            seen.add(choice.id)
        self.suggestions = selected[:target_count]

    def _consume_suggestions(self):
        groups = self.engine.s.vars.setdefault("used_suggestion_groups", [])
        for option in self.suggestions:
            if option.get("once_key") and option["once_key"] not in groups:
                groups.append(option["once_key"])
        self.suggestions = []

    def choose_suggestion(self, index):
        """추천 버튼도 같은 사용자 입력·게임 행동 경로로 보냅니다."""
        if type(index) is not int or not 0 <= index < len(self.suggestions):
            raise ValueError("현재 표시된 선택지를 다시 골라 주세요.")
        option = dict(self.suggestions[index])
        if option.get("action_id"):
            self._consume_suggestions()
            self.messages.append({"role": "user", "content": option["text"]})
            self.propose(option["action_id"], direct=True)
            self.refresh_suggestions()
        else:
            self.send(option["text"])

    def quest_journal(self):
        """시나리오에 정의된 실제 조건을 읽어 목표와 현재 단계를 보여 줍니다."""
        lines = []
        for scene in self.content.scenes:
            for quest in scene.get("quest_log", {}).values():
                condition = quest.get("accepted_when")
                accepted = bool(condition) and self.engine.cond(condition)
                done_condition = quest.get("completed_when")
                done = bool(done_condition) and self.engine.cond(done_condition)
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

    def _record(self, events, transaction_record=None, choice_id=None, player_input=""):
        """엔진 사건과 공개 지식을 기록한 뒤 이야기 표현만 출력 모델에 맡깁니다.

        첫 만남의 opening_line은 준비된 대사로 표시합니다. 다른 카드·장면 문장은
        감사 기록과 참고 자료에 남기며 채팅에 직접 붙이지 않습니다. 수치·주사위·
        거래·함정의 계산 안내는 엔진의 확정 결과로 계속 표시합니다.
        """
        events = list(events)
        if any(event.kind == "move" for event in events) and not self.engine.s.combat:
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
        public, public_records, narratives = [], [], []
        for event in events:
            card_id = event.data.get("id") if event.kind == "card" else None
            card = self.content.cards.get(card_id, {}) if card_id else {}
            prepared_opening = False
            text = event.text
            if card.get("dialogue"):
                profile = card["dialogue"]
                self.npc = profile["name"]
                self.scene_focus_explicit = False
                history = self.histories.setdefault(self.npc, [])
                previously_seen = any(row.get("kind") == "card" and row.get("data", {}).get("id") == card_id
                                      for row in self.audit)
                prepared_opening = not previously_seen and not any(row.get("role") == "assistant" for row in history)
                if prepared_opening:
                    line = profile["opening_line"]
                    history.append({"role": "assistant", "content": line})
                    text = "**" + self.npc + "**\n" + line
            story = (event.kind in {"card", "scene_cue", "interaction"}
                     or event.kind in {"say", "end"} and event.text in self._story_say_texts())
            if story and not prepared_opening:
                reference = {"kind": event.kind, "text": event.text,
                             "source_id": choice_id or "", "knowledge_only": event.kind == "card"}
                if card_id:
                    reference.update(source_card_id=card_id,
                                     known_by=deepcopy(event.data.get("known_by", [])))
                narratives.append(reference)
            elif event.kind != "flag":
                self.messages.append({"role": "assistant", "content": text})
            self.audit.append({"kind": event.kind, "text": text, "data": deepcopy(event.data)})
            if event.kind not in {"flag", "card"}:
                public.append(event.text)
                public_records.append({"kind": event.kind, "text": event.text, "data": deepcopy(event.data)})
        self.revision += 1
        if self.npc and (self.profiles()[self.npc]["room"] != self.engine.s.room
                         or not self.engine.npc_can_talk(self.profiles()[self.npc]["id"])):
            self.npc = ""
        if events:
            witness = self.npc if (self.npc and self.profiles()[self.npc]["room"] == self.engine.s.room
                and self.engine.npc_can_talk(self.profiles()[self.npc]["id"])) else ""
            self.last_result = {"confirmed": True, "event_id": f"game:{self.revision}",
                "public_events": public, "public_event_records": public_records,
                "state_after": self.rag_state(), "witness_npc": witness}
            if transaction_record is not None:
                self.last_result["transaction_records"] = [deepcopy(transaction_record)]
                parties = transaction_record.get("knowledge_npcs", [])
                self.last_result["witness_npc"] = parties[0] if len(parties) == 1 else ""
        # 이번 호출의 자료만 사용하며 다음 행동으로 표시 대기를 넘기지 않습니다.
        if narratives:
            action = player_input or self.content.choices.get(choice_id, {}).get("label") or "현재 장소를 확인한다"
            return self._render_action_result(narratives, action, choice_id)
        return {"generated": False, "error": None, "narrative_count": 0}

    def _render_action_result(self, narratives, player_input, choice_id):
        """현재 행동의 공개 자료를 한 번만 표현하며 실패 시 원문으로 대신하지 않습니다."""
        npc = ""
        profile = None
        # 공개 카드·소문·이동 장면은 GM의 서술이며, 선택된 NPC의 개인 기억으로 바꾸지 않습니다.
        if self.npc and all(row["kind"] in {"say", "interaction"} for row in narratives):
            card = self.profiles().get(self.npc)
            if (card and card["room"] == self.engine.s.room and self.engine.npc_can_talk(card["id"])
                    and (self.last_result or {}).get("witness_npc") == self.npc):
                source = json.dumps(self.content.choices.get(choice_id, {}).get("source", {}), ensure_ascii=False)
                owned = (choice_id == "talk:" + card["id"]
                         or re.search(r"(?<![A-Za-z0-9_-])" + re.escape(card["id"]) + r"(?![A-Za-z0-9_-])", source))
                parties = [name for record in (self.last_result or {}).get("transaction_records", [])
                           for name in record.get("knowledge_npcs", [])]
                if owned or self.npc in parties:
                    npc, profile = self.npc, card["dialogue"]
        speaker_context = {}
        if npc:
            actor = DIALOGUE_CARDS.get("actors", {}).get(npc, {})
            speaker_context = {"name": npc, "role": actor.get("role", profile.get("role", "")),
                               "voice": actor.get("voice", profile.get("style", "")),
                               "relationships": [{key: relation[key] for key in ("person", "relation_to_me")}
                                   for relation in profile.get("relationships", [])
                                   if isinstance(relation, dict) and all(isinstance(relation.get(key), str)
                                       for key in ("person", "relation_to_me"))]}
        recent = native_conversation_history(self.histories.get(npc, []))[-12:]
        payload = {"player_input": player_input, "npc_name": npc, "speaker_context": speaker_context,
                   "response_mode": "npc_dialogue" if npc else "scene_dialogue",
                   "public_narratives": deepcopy(narratives), "engine_result": deepcopy(self.last_result),
                   "player_snapshot": player_snapshot(self.engine), "scene_context": self.scene_context(),
                   "recent_conversation": [{"role": row["role"], "content": row["content"]} for row in recent]}
        result = None
        called = False
        started = time.perf_counter()
        try:
            generate = self.api.get("generate_action_result") if self.api else None
            if not callable(generate):
                raise RuntimeError("행동 결과 출력 모델이 준비되지 않았습니다.")
            called = True
            self.action_generation_calls += 1
            result = generate(deepcopy(payload))
            if not isinstance(result, dict) or not isinstance(result.get("text"), str) or not result["text"].strip():
                raise ValueError("행동 결과의 표시할 문장이 비었습니다.")
            self.messages.append({"role": "assistant", "content": result["text"]})
            history = self.histories.setdefault(npc, [])
            if not history or history[-1].get("role") != "user" or history[-1].get("content") != player_input:
                history.append({"role": "user", "content": player_input})
            history.append({"role": "assistant", "content": result.get("raw_text") or result["text"]})
            self.last_generation_seconds = float(result.get("generation_seconds", 0))
            self.debug["행동 결과 생성"] = {"전달 자료": payload, "출력 결과": deepcopy(result),
                "현재 묶음 모델 호출": 1, "누적 행동 결과 모델 호출": self.action_generation_calls,
                "전체 시간(초)": round(time.perf_counter() - started, 3)}
            self.last_status = f"행동을 반영했습니다. 이야기 생성 {self.last_generation_seconds:.2f}초."
            return {"generated": True, "error": None, "narrative_count": len(narratives)}
        except Exception as error:
            self.debug["행동 결과 생성"] = {"전달 자료": payload, "출력 결과": deepcopy(result),
                "생성 오류": repr(error), "현재 묶음 모델 호출": int(called),
                "누적 행동 결과 모델 호출": self.action_generation_calls,
                "전체 시간(초)": round(time.perf_counter() - started, 3)}
            self.last_status = "이야기 생성에 실패했습니다. 행동과 게임 상태는 이미 반영되었습니다."
            return {"generated": False, "error": repr(error), "narrative_count": len(narratives)}

    def rag_state(self):
        s, h = self.engine.s, self.engine.hero
        return {"scene": s.scene, "cls": h.cls, "flags": sorted(s.flags), "visited": sorted(s.visited),
                "current_room": s.room, "inventory": dict(s.inventory), "gold_cp": s.gold_cp,
                "money": fmt_money(s.gold_cp), "vars": deepcopy(s.vars),
                "hero": {"name": h.name, "hp": h.hp, "max_hp": h.max_hp, "ac": self.engine.armor_class(h),
                         "level": h.level, "conditions": list(h.conditions), "abilities": dict(h.abilities),
                         "proficiency_bonus": h.proficiency_bonus, "resources": deepcopy(h.resources),
                         "hit_dice": {"remaining": h.hd_left, "max": h.hd_max, "die": h.hit_die}},
                "rule_search_mode": self.rule_mode,
                "combat": {"encounter": s.combat["enc"], "round": s.combat["round"],
                           "turn": deepcopy(s.combat.get("turn", {}))} if s.combat else None}

    def talk_to(self, name):
        if not name:
            self.npc = ""
            self.scene_focus_explicit = True
            self.last_status = "장면을 대상으로 이야기합니다."
            self.refresh_suggestions()
            return
        profiles = self.profiles()
        if (name not in profiles or profiles[name]["room"] != self.engine.s.room
                or not self.engine.eligible(profiles[name]["id"]) or not self.engine.npc_can_talk(profiles[name]["id"])):
            raise ValueError("현재 장소에서 만날 수 없는 인물입니다. 개발자 바로 만나기를 사용할 수 있습니다.")
        self.npc = name
        self.scene_focus_explicit = False
        self._record(self.engine._show(profiles[name]["id"]), choice_id="talk:" + profiles[name]["id"],
                     player_input=name + "에게 말을 건다")
        self.last_status = f"대화 상대: {name}. 기존 대화 이력을 유지합니다."
        self.refresh_suggestions()

    def choices(self):
        views = self.engine.available_choices()
        if self.engine.s.combat or self.engine.s.pending or self.engine.s.ended:
            return views
        ids = {view.id for view in views}
        # 최초 만남의 일회성 선택지가 끝나도 같은 장소의 인물에게 다시 말을 걸 수 있습니다.
        for name, card in self.profiles().items():
            profile = card["dialogue"]
            if (card["room"] == self.engine.s.room and card["id"] in self.engine.s.shown
                    and self.engine.eligible(card["id"]) and self.engine.npc_can_talk(card["id"])
                    and profile.get("approach_action") not in ids):
                views.append(ChoiceView("talk:" + card["id"], name + "에게 다시 말을 건다", [self.engine.hero.id], self.engine.hero.id))
        return views

    def _needs_roll(self, choice_id):
        ch = self.content.choices.get(choice_id, {})
        return bool(ch.get("check")) or choice_id.startswith(("cmb:attack:", "cmb:cast:", "cmb:holy:", "cmb:escape", "cmb:unarmed:", "cmb:subdue:"))

    def propose(self, choice_id, direct=False):
        available = {c.id: c for c in self.choices()}
        if choice_id not in available:
            raise ValueError("현재 실행할 수 없는 행동입니다.")
        self.pending = {"type": "choice", "choice": choice_id, "label": available[choice_id].label,
                        "revision": self.revision, "roll": self._needs_roll(choice_id),
                        "check_text": available[choice_id].check_text,
                        "request": self.messages[-1]["content"] if self.messages and self.messages[-1]["role"] == "user" else ""}
        if direct and not self.pending["roll"]:
            self.resolve_pending()
        else:
            text = available[choice_id].check_text or "선택한 행동을 실행합니다."
            self.messages.append({"role": "assistant", "content": f"**행동 확인** · {available[choice_id].label}\n{text}\n아래 버튼으로 실행하거나 ‘굴린다’라고 입력하세요."})
            self.last_status = "행동을 확인한 뒤 실행할 수 있습니다."

    def resolve_pending(self):
        if not self.pending:
            raise ValueError("대기 중인 행동이나 주사위 판정이 없습니다.")
        pending, self.pending = self.pending, None
        if pending["revision"] != self.revision:
            raise ValueError("게임 상태가 바뀌었습니다. 현재 행동을 다시 선택하세요.")
        if pending["type"] == "choice":
            action_snapshot = before_action(self.engine, self.npc, pending["choice"], TRANSACTION_BINDINGS)
            if pending["choice"].startswith("talk:"):
                card = self.content.cards.get(pending["choice"][5:], {})
                if not card.get("dialogue"):
                    raise ValueError("대화할 인물을 확인할 수 없습니다.")
                name = card["dialogue"]["name"]
                self.talk_to(name)
                if pending.get("request"):
                    self.histories.setdefault(name, []).append({"role": "user", "content": pending["request"]})
                events = [Event("interaction", name + "와 다시 마주 보고 이야기를 나눈다.")]
            else:
                events = self.engine.choose(pending["choice"])
                for name, card in self.profiles().items():
                    if card["dialogue"].get("approach_action") == pending["choice"]:
                        self.npc = name
            for event in events:
                card = self.content.cards.get(event.data.get("id"), {}) if event.kind == "card" else {}
                if card.get("dialogue") and pending.get("request"):
                    self.histories.setdefault(card["dialogue"]["name"], []).append({"role": "user", "content": pending["request"]})
        else:
            _, event = self.engine.roll_check(self.engine.hero, {"ability": pending["ability"], "dc": 15})
            events = [event, Event("status", "자유 판정 결과를 기록했습니다. 시나리오 보상·이동·단서는 해당 행동으로 확인합니다.")]
        transaction_record = (record_action(self.transaction_ledger, action_snapshot, self.engine, events)
                              if pending["type"] == "choice" else None)
        outcome = self._record(events, transaction_record, choice_id=pending.get("choice"),
                               player_input=pending.get("request") or pending.get("label", "행동을 실행한다"))
        if not outcome.get("error") and not outcome.get("generated"):
            self.last_status = "행동 결과를 게임 상태에 반영했습니다."
        if pending.get("choice", "").startswith("npc:attack:") and self.engine.s.combat:
            unarmed = next((choice.id for choice in self.choices() if choice.id.startswith("cmb:unarmed:")), None)
            if unarmed:
                self.propose(unarmed, direct=False)
        self.refresh_suggestions()

    def free_check(self, ability):
        self.engine.require_developer()
        if ability not in ABILITY_KO:
            raise ValueError("알 수 없는 능력치입니다.")
        if self.engine.s.combat:
            raise ValueError("전투 중에는 현재 전투 행동을 선택하세요.")
        self.pending = {"type": "free_check", "ability": ability, "revision": self.revision,
                        "label": f"{ABILITY_KO[ability]} 자유 판정", "roll": True,
                        "check_text": "테스트용 보통 난이도 DC 15 · 캐릭터 능력 수정치 적용"}
        self.messages.append({"role": "assistant", "content": f"{ABILITY_KO[ability]} 판정이 준비됐습니다. DC 15입니다. 🎲 굴린다를 누르세요."})
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
        self.engine.developer = bool(enabled)
        if not enabled:
            self.engine.next_success = False
        self.last_status = "개발자 모드를 켰습니다." if enabled else "개발자 모드를 껐습니다. 성공 보장을 해제했습니다."

    def developer_jump(self, room, npc=""):
        self.engine.require_developer()
        if self.engine.s.combat:
            raise ValueError("전투를 끝낸 뒤 이동하세요. 즉시 승리 버튼을 사용할 수 있습니다.")
        self.pending = None
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
        self.pending = None
        self._record([Event("developer", "개발자 시험 전투를 시작합니다.")] + self.engine.start_combat(encounter))
        self.refresh_suggestions()
        self.last_status = "시험 전투입니다. 정상 플레이의 조우 조건을 건너뛰었습니다."

    def route_message(self, message):
        """자료·행동 목록 없이 말의 종류를 고르고, 행동일 때만 실제 ID를 연결합니다."""
        started = time.perf_counter()
        self.turn_needs_information = self.turn_needs_state = False
        recent = [{"role": row["role"], "content": row["content"]}
                  for row in native_conversation_history(self.histories.get(self.npc, []))[-2:]]
        system = (
            '마지막 플레이어 발언의 종류와 확인할 실제 상태의 범위를 각각 독립적으로 고른다. JSON 하나만 반환한다: '
            '{"kind":"social|decline|farewell|knowledge|state|action|look|rule|clarify","state_topics":[]}. '
            'social: 인사·감정 표현·공감·현재 기분과 일상적인 몸 상태의 안부·대화 속도·조용히 함께 있기·일상적인 허락·농담처럼 일상 반응만 필요한 말. '
            'decline: 부탁이나 의뢰를 거절하는 말. farewell: 대화를 마치거나 작별하는 말. '
            'knowledge: 인물의 관계·직업·과거·전해들은 이야기·사건 원인·단서·인지 방법을 묻거나 그런 사실을 주장하는 말. '
            '지속적인 감각·인지 제약이나 상대를 어떻게 알아챘는지에 관한 질문도 knowledge다. 현재의 가벼운 안부와 설정상 제약을 구별한다. '
            'state: 현재 물건의 소유·보관·수령·지급·구매·HP나 실제 행동 완료 기록을 확인하거나 그런 상태를 주장하는 말. '
            'action: 지금 하려는 명확한 물리적 행동·이동·접근·공격 또는 명확한 의뢰 수락. '
            '질문·과거 보고·가정·위협은 실행이나 수락이 아니다. '
            'look: 주변을 관찰하겠다는 선언. rule: 게임 밖에서도 똑같이 답할 수 있는 일반 규칙 질문. '
            'clarify: 대상이나 뜻이 불명확해 먼저 확인해야 하는 행동. '
            '질문형 문장이라는 이유만으로 knowledge나 state가 되지 않는다. '
            '직전 NPC가 꺼낸 의뢰와 마지막 플레이어가 지금 원하는 것을 구별한다. '
            'state_topics는 마지막 발언이 확인하거나 주장·전제하는 현재 상태나 실제 완료 기록의 범위만 고르는 배열이다. '
            '허용 값은 money(잔액·지급·수령·구매 대금·반환), items(물건의 소유·보관·수령), '
            'hp(게임상의 현재 HP·피해·회복), rest(실제 휴식·숙박·예약의 완료나 그에 따른 회복), resources(실제 직업·레벨·사용 가능한 주문·슬롯·능력 사용 횟수와 사용 완료 기록)다. '
            '발언 종류와 상태 범위는 독립적이므로 거절·작별·감정 표현 속에도 상태 확인이나 전제가 있으면 해당 범위를 넣는다. '
            '상태 주장을 포함하지 않은 일상 대화·거절이나 단순한 감정·신체 감각·말하는 속도·잠깐 함께 쉬자는 허락에는 빈 배열을 쓴다. '
            '아직 실행하지 않은 행동 의도나 일반 배경 설명만으로 완료 상태를 전제하지 않는다. '
            '플레이어와 NPC의 감정·부탁·결정을 구별한다. 대사나 추천은 만들지 않는다.\n'
            + '현재 NPC: ' + self.npc + '\n이전 대화: ' + json.dumps(recent, ensure_ascii=False))
        if not self.api:
            return "TALK", {"route": "TALK", "no_model": True, "model_calls": 0,
                            "classification_calls": 0, "mapper_calls": 0, "mapper_seconds": 0.0, "state_topics": []}
        preparation_seconds = time.perf_counter() - started
        classification_started = time.perf_counter()
        raw = self.api["generate_answer"]([{"role": "system", "content": system},
            {"role": "user", "content": message}], max_new_tokens=64, sample=False).strip()
        classification_seconds = time.perf_counter() - classification_started

        def read_object(value):
            try:
                result = json.loads(value.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
                return result if isinstance(result, dict) else {}
            except ValueError:
                return {}

        classified = read_object(raw)
        kind = classified.get("kind")
        known_state_topics = ("money", "items", "hp", "rest", "resources")
        raw_topics = classified.get("state_topics")
        if "state_topics" not in classified and kind == "state":
            # 이전 단일 필드 응답은 기존의 전체 상태 확인 범위를 유지합니다.
            state_topics = list(known_state_topics)
        elif isinstance(raw_topics, list) and all(isinstance(topic, str) and topic in known_state_topics for topic in raw_topics):
            state_topics = list(dict.fromkeys(raw_topics))
            if not state_topics and kind == "state":
                state_topics = list(known_state_topics)
        else:
            state_topics = []
        routes = {"social": ("TALK", "inform", False, False), "decline": ("TALK", "decline", False, False),
            "farewell": ("TALK", "farewell", False, False), "knowledge": ("TALK", "ask", True, False),
            "state": ("TALK", "ask", True, True), "action": ("CLARIFY", "action", True, True),
            "look": ("LOOK", "action", False, False), "rule": ("RULE", "ask", True, False),
            "clarify": ("CLARIFY", "action", False, False)}
        classification_valid = isinstance(kind, str) and kind in routes
        kind = kind if classification_valid else "clarify"
        selected, intent, info, state = routes[kind]
        mapper_raw, mapper_seconds, mapper_calls = "", 0.0, 0
        if kind == "action":
            choices = self.choices()
            if self.npc:
                current = self.profiles()[self.npc]
                repeated = {current["dialogue"].get("approach_action"), "talk:" + current["id"]}
                choices = [choice for choice in choices if choice.id not in repeated]
            data = {"npc": self.npc, "recent_dialogue": recent,
                "available_actions": [{"id": choice.id, "label": choice.label, "check": choice.check_text} for choice in choices]}
            mapper_system = (
                '현재 플레이어 발언을 실제 가능한 행동 목록과 대조한다. JSON 하나만 반환한다: '
                '{"action_id":"현재 목록의 정확한 ID 또는 빈 문자열","intent":"action|accept|ask|hypothetical|joke"}. '
                '지금 행동을 실행하는 선언은 action, 명확한 의뢰 수락은 accept다. '
                '질문은 ask, 가정·위협은 hypothetical, 농담은 joke이며 실제 실행하지 않는다. '
                '단순 언급·질문·과거 보고를 수락이나 실행으로 바꾸지 않는다. '
                '대상이 불명확하거나 현재 가능한 ID가 없으면 빈 문자열을 반환한다. ID를 만들지 않는다.\n현재 자료:\n'
                + json.dumps(data, ensure_ascii=False))
            mapper_started = time.perf_counter()
            mapper_raw = self.api["generate_answer"]([{"role": "system", "content": mapper_system},
                {"role": "user", "content": message}], max_new_tokens=96, sample=False).strip()
            mapper_seconds, mapper_calls = time.perf_counter() - mapper_started, 1
            mapped = read_object(mapper_raw)
            identifier, intent = mapped.get("action_id"), mapped.get("intent")
            selected = identifier if isinstance(identifier, str) and identifier in {choice.id for choice in choices} else "CLARIFY"
            if not isinstance(intent, str) or intent not in {"action", "accept", "ask", "hypothetical", "joke"}:
                selected, intent = "CLARIFY", "action"
            elif intent not in {"action", "accept"}:
                selected = "TALK" if self.npc else "CLARIFY"
            if intent == "joke":
                info = state = False
        uncertain = selected == "CLARIFY" and bool(self.npc)
        if uncertain:
            # 분류만 불확실한 질문은 현재 인물과 확인합니다. 이 경로는 행동을 실행하지 않습니다.
            selected, intent = "TALK", "clarify"
            if classification_valid:
                info = state = True
        state = state or bool(state_topics)
        self.turn_intent, self.turn_needs_information, self.turn_needs_state = intent, info, state
        return selected, {"route": selected, "kind": kind, "raw_route": raw, "raw_mapper": mapper_raw,
            "intent": intent, "needs_information": info, "needs_state": state, "state_topics": state_topics,
            "input_uncertain": uncertain,
            "model_calls": 1 + mapper_calls, "classification_calls": 1, "mapper_calls": mapper_calls,
            "preparation_seconds": round(preparation_seconds, 3),
            "classification_seconds": round(classification_seconds, 3), "mapper_seconds": round(mapper_seconds, 3),
            "seconds": round(time.perf_counter() - started, 3)}

    def send(self, message):
        message = message.strip()
        if not message or len(message) > 1200:
            raise ValueError("1~1200자로 말이나 행동을 입력하세요.")
        turn_started = time.perf_counter()
        self._consume_suggestions()
        self.turn_intent = None
        self.turn_needs_information = self.turn_needs_state = False
        self.messages.append({"role": "user", "content": message})
        if self.pending and message in {"굴린다", "굴려", "주사위 굴려", "실행", "확인", "네", "응"}:
            self.resolve_pending()
            return
        self.pending = None
        if message.startswith("/굴림 "):
            self.utility_dice(message[4:].strip())
            return
        if message.startswith("/행동 "):
            self.propose(message[4:].strip(), direct=True)
            return
        if message.startswith("/규칙 "):
            route, route_debug = "RULE", {"route": "RULE", "command": True}
        else:
            route, route_debug = self.route_message(message)
        if route not in {"TALK", "LOOK", "RULE", "CLARIFY"}:
            self.debug = {"의도 해석": route_debug}
            # 목적지가 명확한 이동·대화 시작은 바로 진행합니다. 거래·판정은 확인을 유지합니다.
            harmless = route.startswith(("move:", "talk:", "npc:attack:")) or self.content.choices.get(route, {}).get("interaction") == "talk"
            self.propose(route, direct=harmless or (message.startswith("**") and message.endswith("**")))
            self.refresh_suggestions()
            return
        if route == "CLARIFY":
            context = self.scene_context()
            options = [person["label"] + "에게 말을 걸기" for person in context["people"]]
            options += [item["destination"] + " 쪽으로 가기" for item in context["exits"]]
            text = "어느 쪽으로 움직일까요?" if context["exits"] else "어떤 대상을 살피거나 말을 걸까요?"
            if options:
                text += "\n\n" + " · ".join(options)
            self.messages.append({"role": "assistant", "content": text})
            self.debug = {"의도 해석": route_debug, "현재 장소": context}
            self.last_status = "대상을 확인합니다. 위치·물건·시간은 바꾸지 않았습니다."
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
                self.messages.append({"role": "assistant", "content": "누구에게 말을 걸까요?\n\n" + " · ".join(person["label"] for person in people)})
                self.last_status = "현재 보이는 사람 중 대화할 상대를 확인합니다."
                self.debug = {"의도 해석": route_debug, "현재 장소": self.scene_context()}
                return
        route_debug["_turn_started"] = turn_started
        self._generate_dialogue(message, route, route_debug)

    def _generate_dialogue(self, message, route, route_debug):
        if not self.api:
            raise RuntimeError("대화 모델이 연결되지 않았습니다.")
        started = time.perf_counter()
        speaker = "" if route == "RULE" else self.npc
        history = self.histories.setdefault(speaker, [])
        payload = deepcopy(self.api["get_handoff"](message, self.rag_state(), npc_name=speaker, history=history))
        payload.update(response_contract="game_turn", recommendations_in_router=True, state_snapshot_complete=True,
            suggestion_count=1, turn_intent=self.turn_intent, needs_state=self.turn_needs_state)
        payload["available_actions"] = [{"id": choice.id, "label": choice.label, "check": choice.check_text}
            for choice in self.choices() if not choice.id.startswith("npc:attack:")]
        payload["scene_context"] = self.scene_context()
        payload["player_snapshot"] = player_snapshot(self.engine)
        payload["model_state"] = model_state(self.engine, payload["player_snapshot"])
        if route == "RULE":
            payload["retrieval"]["scenario_results"] = []
            payload["speaker_context"], payload["known_characters"] = None, []
        else:
            payload["retrieval"]["rule_results"] = []
            if speaker:
                payload["retrieval"]["scenario_results"] = [record for record in payload["retrieval"]["scenario_results"]
                    if record.get("metadata", {}).get("kind") != "desc"]
        for record in payload["retrieval"]["scenario_results"]:
            metadata = record.setdefault("metadata", {})
            card_id = record.get("card_id") or metadata.get("card_id")
            evidence = self.content.cards.get(card_id, {}).get("evidence")
            if evidence:
                metadata["evidence"] = deepcopy(evidence)
        payload["response_mode"] = "rule_explanation" if route == "RULE" else "npc_dialogue" if speaker else "scene_dialogue"
        payload["input_uncertain"] = bool(route_debug.get("input_uncertain"))
        payload["state_topics"] = list(route_debug.get("state_topics", []))
        if speaker in DIALOGUE_CARDS["actors"]:
            actor, facts = DIALOGUE_CARDS["actors"][speaker], []
            compact = {key: actor[key] for key in ("name", "role", "voice")}
            if actor.get("public_concern"):
                compact["public_concern"] = actor["public_concern"]
            if actor.get("sensory_limits"):
                compact["sensory_limits"] = actor["sensory_limits"]
            # 인사·거절에는 최근 대화를 우선하고, 정보가 필요한 말에 배경 지식을 보탭니다.
            # 분류가 불확실하면 이미 공개가 허용된 지식을 유지합니다.
            if self.turn_needs_information or payload["input_uncertain"]:
                retrieved = payload["retrieval"]["scenario_results"]
                ids = [row.get("card_id") or row.get("metadata", {}).get("card_id") for row in retrieved]
                facts = eligible_facts(self.engine, speaker, DIALOGUE_CARDS, ids)
                # 공개된 이름·정체를 참고하며 다른 인물의 개인 기억과는 구분합니다.
                for person in payload.get("known_characters", []):
                    if (person.get("name") != speaker and isinstance(person.get("identity"), str)
                            and person.get("identity") and isinstance(person.get("card_id"), str)):
                        facts.append({"id": "person:" + person["card_id"], "category": "public_identity",
                            "text": person["identity"], "direct_or_heard": "public_identity",
                            "source_card_id": person["card_id"], "name": person.get("name", "")})
            current_profile = self.profiles()[speaker]
            profile = current_profile["dialogue"]
            # 현재 인물 자신의 직접 관계만 참고하며, 타인 관계나 비밀 정보는 옮기지 않습니다.
            compact["relationships"] = [
                {key: relation[key] for key in ("person", "relation_to_me")}
                for relation in profile.get("relationships", [])
                if isinstance(relation, dict) and all(
                    isinstance(relation.get(key), str) and relation[key].strip()
                    for key in ("person", "relation_to_me"))]
            payload["dialogue_context"] = {"compact_speaker": compact, "selected_facts": facts,
                "transactions": directed_transactions(self.engine, profile, speaker, TRANSACTION_BINDINGS),
                "selected_memories": []}
            payload["dialogue_context"]["shop"] = shop_state(self.engine, speaker, self.transaction_ledger, TRANSACTION_BINDINGS)
            payload["dialogue_context"]["contract"] = deepcopy(profile.get("quest_offer"))
            payload["dialogue_context"]["npc_state"] = self.engine.npc_state(current_profile["id"])
            payload.update(recommendations_in_router=False, reply_format="natural_text")
        elif speaker:
            # 분류 자료가 없는 인물은 기존 출력 모듈이 읽는 프로필 형식을 유지합니다.
            profile = self.profiles()[speaker]["dialogue"]
            payload["dialogue_context"] = {"persona": {"own_facts": profile.get("first_person_facts", []),
                "relationships": profile.get("relationships", []), "goals": profile.get("goals", []),
                "style": profile.get("style", ""), "knowledge": profile.get("knowledge", {}),
                "knowledge_limits": profile.get("knowledge", {}).get("unknown", []), "quest_offer": profile.get("quest_offer")},
                "npc_state": self.engine.npc_state(self.profiles()[speaker]["id"]), "active_npc": speaker}
        payload, memory_hits = self.api["add_memory_context"](payload, self.memory_session, speaker, history)
        if speaker in DIALOGUE_CARDS["actors"] and (self.turn_needs_information or payload["input_uncertain"]):
            memories = payload["dialogue_context"]["selected_memories"]
            for record in payload.get("retrieval", {}).get("memory_results", []):
                if not isinstance(record, dict) or not isinstance(record.get("text"), str) or not record["text"].strip():
                    continue
                metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                memories.append({"text": record["text"], "metadata": {"speaker": metadata.get("speaker", "unknown"),
                    "trust": metadata.get("trust", "unverified_statement")}})
                if len(memories) == 2:
                    break
        context_seconds = time.perf_counter() - started
        result = self.api["generate_from_handoff"](payload, history=history, npc_name=speaker,
                                                  engine_result=self.last_result)
        engine_only = result.get("response_format") == "engine_state"
        state_display = ""
        if self.turn_needs_state and not engine_only and callable(self.api.get("format_confirmed_state")):
            state_display = self.api["format_confirmed_state"](payload, engine_result=self.last_result, npc_name=speaker)
        elif (speaker and not engine_only and not payload.get("state_topics")
                and callable(self.api.get("format_confirmed_state"))
                and mentioned_context_items(message, payload, speaker)):
            # UI-only copy: keep normal NPC generation, classification and memory intact.
            display_payload = deepcopy(payload)
            display_payload.update(needs_state=True, state_topics=["items"])
            state_display = self.api["format_confirmed_state"](display_payload,
                engine_result=self.last_result, npc_name=speaker)
        self.last_generation_seconds = result["generation_seconds"]
        slow, ending = self.last_generation_seconds > 6, self.turn_intent in {"decline", "farewell"}
        followup = {"called": False, "seconds": 0.0, "wall_seconds": 0.0,
                    "reason": "state_uses_engine_choices" if engine_only or self.turn_needs_state else "ending" if ending else "slow_reply" if slow else "not_npc" if not speaker else "api_unavailable"}
        recommendations = []
        if speaker and not engine_only and not self.turn_needs_state and not slow and not ending and callable(self.api.get("generate_followup_question")):
            followup_started = time.perf_counter()
            followup.update(called=True, reason="generated")
            try:
                value = self.api["generate_followup_question"](payload, result["text"], history=history)
                followup.update(value)
                question = value.get("text", "")
                if (isinstance(question, str) and 1 <= len(question.strip()) <= 35
                        and question.strip().endswith("?") and not self._asked_recently(question, message)):
                    recommendations.append({"label": question.strip(), "text": question.strip(), "source": "model_question"})
                else:
                    followup["reason"] = "invalid_or_repeated_question"
            except Exception as error:
                # 부가 추천 실패가 이미 생성한 본문과 대화 저장을 막지 않도록 따로 기록합니다.
                followup.update(reason="error", error=repr(error))
            followup["wall_seconds"] = time.perf_counter() - followup_started
        # 질문 추천에 이어 붙이는 행동은 실제 엔진 목록에서 고릅니다. 거절·작별 뒤에는 이동을 우선합니다.
        repeated = ({self.profiles()[speaker]["dialogue"].get("approach_action"), "talk:" + self.profiles()[speaker]["id"]}
                    if speaker else set())
        choices = [choice for choice in self.choices() if choice.id not in repeated
                   and not choice.id.startswith(("npc:attack:", "buy:", "use:", "talk:"))]
        if ending:
            choices = [choice for choice in choices if choice.id.startswith("move:")]
        target = 1 if slow else 2
        for choice in choices:
            if len(recommendations) >= target:
                break
            recommendations.append({"label": choice.label, "text": choice.label, "action_id": choice.id, "source": "engine"})
        result["suggestions"] = [row["text"] for row in recommendations if not row.get("action_id")]
        self.api["remember_engine_result"](result["payload"], self.memory_session, speaker,
                                          self.api["input_namespace"]["memory_embedding_model"])
        embedder = self.api["input_namespace"]["memory_embedding_model"]
        turn = self.api["remember_player_message"](self.memory_session, speaker, message, embedder)
        user_record = {"role": "user", "content": message, "memory_turn": turn}
        if engine_only:
            user_record["answered_by"] = "game_state"
        history.append(user_record)
        if not engine_only:
            # NPC가 실제로 생성한 대사만 NPC 발언 기억에 저장합니다.
            self.api["remember_npc_message"](self.memory_session, speaker, result["text"], embedder, turn_id=turn,
                                             speech=result["speech"], narration=result["narration"])
            history.append({"role": "assistant", "content": result["text"], "memory_turn": turn,
                            "suggestions": result["suggestions"]})
        self.api["save_dialogue_history"](self.memory_session, speaker, history)
        if state_display:
            # 확정 상태는 게임 시스템의 표시이며 NPC가 했던 발언으로 저장하지 않습니다.
            self.messages.append({"role": "assistant", "content": state_display})
        heading = "**" + speaker + "**\n" if speaker and not engine_only else ""
        self.messages.append({"role": "assistant", "content": heading + result["text"]})
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
        calls = {"generate_answer": route_debug.get("model_calls", 0), "get_handoff": 1,
                 "scope_classification": route_debug.get("classification_calls", 0),
                 "action_mapper": route_debug.get("mapper_calls", 0),
                 "add_memory_context": 1, "generate_from_handoff": 1,
                 "body_model_generation": int(result.get("model_generated", not engine_only)), "generate_followup_question": int(followup["called"])}
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

    def save_data(self):
        """실행 코드 없이 JSON 값만 저장합니다. 주사위 RNG도 보존합니다."""
        return {"schema": "trpg_game_save_1", "state": _jsonable(asdict(self.engine.s)),
                "rng": _jsonable(self.engine.dice.rng.getstate()), "npc": self.npc,
                "scene_focus_explicit": self.scene_focus_explicit,
                "histories": deepcopy(self.histories), "messages": self.messages[-100:],
                "audit": self.audit[-200:], "revision": self.revision,
                "transaction_ledger": deepcopy(self.transaction_ledger)}

    def restore_data(self, data):
        if not isinstance(data, dict) or data.get("schema") != "trpg_game_save_1":
            raise ValueError("지원하지 않는 저장 파일입니다.")
        scene_focus = data.get("scene_focus_explicit", False)
        if type(scene_focus) is not bool:
            raise ValueError("대화 대상 저장 형식이 올바르지 않습니다.")
        raw = deepcopy(data["state"])
        if raw.get("room") not in self.content.rooms or raw.get("gold_cp", -1) < 0:
            raise ValueError("장소·금액이 올바르지 않습니다.")
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
        # 저장 파일을 일부만 적용하지 않도록 먼저 구조와 RNG를 검사합니다.
        dice = Dice()
        dice.rng.setstate(_tuples(data["rng"]))
        histories = data.get("histories", {})
        if not isinstance(histories, dict) or any(not isinstance(v, list) or len(v) > 300 for v in histories.values()):
            raise ValueError("대화 이력이 올바르지 않습니다.")
        for records in [*histories.values(), data.get("messages", [])]:
            if not isinstance(records, list) or any(not isinstance(m, dict) or m.get("role") not in {"user", "assistant"}
                or not isinstance(m.get("content"), str) or len(m["content"]) > 20000 for m in records):
                raise ValueError("대화 메시지가 올바르지 않습니다.")
        transaction_ledger = restore_ledger(data.get("transaction_ledger"), state.log)
        self.engine.s, self.engine.dice = state, dice
        self.transaction_ledger = transaction_ledger
        self.engine.next_success = False
        self.engine.developer = False
        self.pending = None
        self.histories = histories
        if self.api:
            self.close()
            self.memory_session = self.api["create_memory_session"]()
            embedder = self.api["input_namespace"]["memory_embedding_model"]
            for speaker, records in self.histories.items():
                turn = None
                for record in records[-100:]:
                    if record["role"] == "user":
                        turn = self.api["remember_player_message"](self.memory_session, speaker, record["content"], embedder)
                    elif turn is not None:
                        self.api["remember_npc_message"](self.memory_session, speaker, record["content"], embedder, turn_id=turn)
                    if turn is not None:
                        record["memory_turn"] = turn
        self.messages = data.get("messages", [])[-100:]
        self.audit = data.get("audit", [])[-200:]
        self.revision = int(data.get("revision", 0)) + 1
        self.npc = data.get("npc", "")
        self.scene_focus_explicit = scene_focus and not self.npc
        if self.npc not in self.profiles() or self.profiles()[self.npc]["room"] != state.room:
            self.npc = ""
        self._record([Event("status", "저장한 게임을 불러왔습니다. 개발자 성공 보장은 해제했습니다.")])
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
