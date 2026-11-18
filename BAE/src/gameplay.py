"""게임 상태·행동·AI 대화를 연결합니다.

규칙 판정은 game.Engine, 대사 생성은 전달받은 출력 모듈이 담당합니다.
자연어로 제안된 행동은 확인을 받고, 버튼이나 **행동** 선언은 직접 실행합니다.
"""
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import re
from secrets import token_urlsafe
from threading import Event as CancellationEvent, RLock
import time

from game.content import Content
from game.dice import Dice, DICE_EXPR
from game.engine import Engine, fmt_money
from game.events import Event, ChoiceView
from game.model import Character, GameState, ABILITY_KO
from game.startup import apply_starting_equipment
from dialogue_projection import eligible_facts
from state_projection import (new_ledger, before_action, record_action, restore_ledger,
    directed_transactions, shop_state, player_snapshot, model_state,
    native_conversation_history)


DIALOGUE_CARDS = json.loads(Path(__file__).with_name("context_cards.json").read_text(encoding="utf-8"))
TRANSACTION_BINDINGS = json.loads(Path(__file__).with_name("transaction_bindings.json").read_text(encoding="utf-8"))


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
    if not _question_form(message):
        return False
    # '어떻게 지내세요' 같은 안부의 어떻게만으로는 정보 질문을 강제하지 않습니다.
    return bool(re.search(r"누구|누군|어디|어딘|무엇|무슨|언제|얼마|몇", message)
                or re.search(r"(?:알려|설명해|말씀해)\s*(?:주실|주시겠|줄\s*수|주세요|주십시오|줘)", message))




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
        self._recommendation_epoch = 0
        self._recommendation_cancel = CancellationEvent()
        self._pending_recommendation = None
        self.last_generation_seconds = 0.0
        self.action_generation_calls = 0
        self.turn_intent = None
        self.transaction_ledger = new_ledger()
        self.turn_needs_information = False
        self.turn_needs_state = False
        self.memory_session = api["create_memory_session"]() if api else None
        self._record(self.engine.opening, player_input="게임을 시작한다", opening_scene=True)
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
        return result

    def apply_recommendations(self, job, result):
        """같은 턴·인물일 때만 늦게 도착한 추천을 화면과 저장 이력에 반영합니다."""
        if (job.get("applied") or job["cancel_event"].is_set() or job["epoch"] != self._recommendation_epoch
                or job["revision"] != self.revision or job["speaker"] != self.npc
                or job["room"] != self.engine.s.room):
            return False
        job["applied"] = True
        rows, seen = [], set()
        values = result.get("suggestions")
        if not isinstance(values, list):
            values = [result.get("text", "")]
        if result.get("reason") not in {"error", "cancelled"}:
            for value in values:
                text = value.get("text", "") if isinstance(value, dict) else value
                if not isinstance(text, str) or not 1 <= len(text.strip()) <= 90:
                    continue
                text = text.strip()
                key = re.sub(r"[\W_]+", "", text).casefold()
                if not key or key in seen or self._asked_recently(text, job["player_message"]):
                    continue
                # 생성 추천은 자유 입력과 같은 경로로 보내며 행동 ID를 만들지 않습니다.
                rows.append({"label": text, "text": text, "source": "model"})
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

    def profiles(self):
        return {c["dialogue"]["name"]: c for c in self.content.cards.values() if c.get("dialogue")}

    def nearby_npcs(self):
        return [(person["label"], person["name"]) for person in self.visible_people()]

    def _public_paths(self):
        """방문한 곳과 그곳에서 조건을 통과한 출구만 지도에 표시합니다."""
        known = set(self.engine.s.visited) | {self.engine.s.room}
        for origin in tuple(known):
            room = self.content.rooms[origin]
            for destination in self.content.exits(origin):
                if self.engine.cond(room.get("exit_requires", {}).get(destination)):
                    known.add(destination)
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
            for name, card in self.profiles().items():
                if (card["room"] != room_id or card.get("owner") != "all"
                        or card.get("reveal_after", 1) > self.engine.s.scene
                        or card.get("req_flag") and card["req_flag"] not in self.engine.s.flags):
                    continue
                people.append({"name": name, "card_id": card["id"],
                               "can_meet": reachable and self.engine.npc_can_talk(card["id"])})
            locations.append({"id": room_id, "name": self.content.room_name(room_id),
                              "current": room_id == self.engine.s.room, "reachable": reachable, "npcs": people})
        return locations

    def travel_to(self, room_id, npc_name=None):
        """공개 경로를 한 칸씩 실제 실행하고, 요청한 인물은 도착 후 접근합니다."""
        if self._travel_blocked():
            raise ValueError("현재 전투·판정·확인 대기를 먼저 마쳐 주세요.")
        locations = {row["id"]: row for row in self.map_locations()}
        destination = locations.get(room_id)
        if not destination or not destination["reachable"]:
            raise ValueError("현재 공개된 경로로 갈 수 없는 장소입니다.")
        if npc_name and not any(person["name"] == npc_name and person["can_meet"] for person in destination["npcs"]):
            raise ValueError("그 장소에서 만날 수 있는 인물이 아닙니다.")
        _, paths = self._public_paths()
        executed = []
        for step in paths[room_id]:
            action_id = "move:" + step
            if self._travel_blocked() or action_id not in {choice.id for choice in self.choices()}:
                self.last_status = "이동 중 상황이 바뀌어 멈췄습니다. 현재 행동을 확인해 주세요."
                self.debug["지도 이동"] = {"실행한 행동": executed, "도착": False}
                return False
            self.propose(action_id, direct=True)
            executed.append(action_id)
            if self._travel_blocked():
                self.last_status = "이동 중 전투나 판정이 발생해 멈췄습니다."
                self.debug["지도 이동"] = {"실행한 행동": executed, "도착": self.engine.s.room == room_id}
                return False
        if npc_name:
            card = self.profiles().get(npc_name, {})
            if (card.get("room") != self.engine.s.room or not self.engine.eligible(card["id"])
                    or not self.engine.npc_can_talk(card["id"])):
                raise ValueError("도착 후 대화할 수 없는 상태로 바뀌었습니다.")
            choices = {choice.id for choice in self.choices()}
            approach = card["dialogue"].get("approach_action")
            talk = "talk:" + card["id"]
            action_id = approach if approach in choices else talk if talk in choices else None
            if action_id:
                self.propose(action_id, direct=True)
                executed.append(action_id)
            else:
                self.talk_to(npc_name)
            if self._travel_blocked():
                self.debug["지도 이동"] = {"실행한 행동": executed, "도착": True}
                return False
        self.debug["지도 이동"] = {"실행한 행동": executed, "도착": True, "대화 상대": self.npc}
        self.refresh_suggestions()
        return True

    def visit_npc(self, name):
        """공개 지도에서 인물을 다시 확인한 뒤 이동과 만남을 연결합니다."""
        for room in self.map_locations():
            if any(person["name"] == name and person["can_meet"] for person in room["npcs"]):
                return self.travel_to(room["id"], npc_name=name)
        raise ValueError("현재 공개된 경로로 만날 수 없는 인물입니다.")

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
        """현재 관찰과 이미 공개된 장소 설정을 구분해 GM·NPC에게 전달합니다.

        observation은 방의 정식 관찰문입니다. conditional_background는 시간·조건이
        붙을 수 있는 공개 설명이며, 그 조건이 지금 성립한다는 뜻은 아닙니다.
        생성된 GM 문장이나 NPC 개인 기억은 현재 장면의 확정값으로 옮기지 않습니다.
        """
        room = self.content.rooms[self.engine.s.room]
        background = []
        for card_id in self.engine.player_known_cards(self.engine.hero.id):
            card = self.content.cards.get(card_id, {})
            if (card.get("room") != self.engine.s.room or card.get("kind") != "desc"
                    or card.get("owner") != "all" or not self.engine.eligible(card_id)):
                continue
            text, _ = self.engine._view_text(card_id)
            # 조건이 충족되어도 엔진이 아직 공개하지 않은 지연 문장은 전달하지 않습니다.
            for deferred in self.engine.s.deferred.values():
                if deferred.get("card") == card_id and isinstance(deferred.get("part"), str):
                    text = text.replace(deferred["part"], "")
            if text.strip():
                background.append({"source_card_id": card_id, "text": text.strip()})
        aftermath = []
        for name, card in self.profiles().items():
            if card["room"] == self.engine.s.room:
                status = self.engine.npc_state(card["id"])
                if not status["can_talk"]:
                    aftermath.append({"name": name, "status": status["status"], "hp": status["hp"]})
        return {"room_id": self.engine.s.room, "room_name": self.content.room_name(self.engine.s.room),
                "observation": room.get("look_text", ""), "people": self.visible_people(),
                "conditional_background": background,
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
        available.update("meet:" + person["card_id"] for room in self.map_locations()
                         for person in room["npcs"] if person["can_meet"] and person["name"] != self.npc)
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
        # 이동·인물 만남은 별도 지도에서 고르며, 빈 추천을 임의의 행동으로 채우지 않습니다.
        self.suggestions = selected[:target_count]

    def _consume_suggestions(self, selected=None):
        groups = self.engine.s.vars.setdefault("used_suggestion_groups", [])
        if selected and selected.get("once_key") and selected["once_key"] not in groups:
            groups.append(selected["once_key"])
        self.suggestions = []

    def choose_suggestion(self, index):
        """추천 버튼도 같은 사용자 입력·게임 행동 경로로 보냅니다."""
        if type(index) is not int or not 0 <= index < len(self.suggestions):
            raise ValueError("현재 표시된 선택지를 다시 골라 주세요.")
        option = dict(self.suggestions[index])
        self._consume_suggestions(option)
        if option.get("action_id"):
            self.messages.append({"role": "user", "content": option["text"]})
            if option["action_id"].startswith("meet:"):
                card = self.content.cards.get(option["action_id"][5:], {})
                self.visit_npc(card.get("dialogue", {}).get("name", ""))
            else:
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

    def _record(self, events, transaction_record=None, choice_id=None, player_input="", engine_delta=None,
                opening_scene=False):
        """엔진 사건과 공개 지식을 기록한 뒤 이야기 표현만 출력 모델에 맡깁니다.

        새 게임의 도입과 첫 만남의 opening_line은 준비된 문장으로 표시합니다. 다른 카드·장면 문장은
        감사 기록과 참고 자료에 남기며 채팅에 직접 붙이지 않습니다. 수치·주사위·
        거래·함정의 계산 안내는 엔진의 확정 결과로 계속 표시합니다.
        """
        self.cancel_recommendations()
        events = list(events)
        movement_scene = any(event.kind == "move" for event in events)
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
        if movement_scene and not self.engine.s.combat:
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
        opening_displayed = False
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
                    opening_displayed = True
                    line = profile["opening_line"]
                    history.append({"role": "assistant", "content": line})
                    text = "**" + self.npc + "**\n" + line
            story = (event.kind in {"card", "scene_cue", "interaction"}
                     or event.kind in {"say", "end"} and event.text in self._story_say_texts())
            if opening_text and story and not prepared_opening:
                # 최초 배경은 위 도입으로 표시하고, 공개 사건·감사 기록은 그대로 남깁니다.
                pass
            elif story and not prepared_opening:
                reference_text = event.text
                if movement_scene and card.get("dialogue"):
                    profile = card["dialogue"]
                    actor = DIALOGUE_CARDS.get("actors", {}).get(profile["name"], {})
                    reference_text = "\n".join(value for value in
                        (profile.get("appearance"), profile.get("presence_text"),
                         profile["name"] + " · " + actor.get("role", ""))
                        if isinstance(value, str) and value.strip())
                reference = {"kind": event.kind, "text": reference_text,
                             "source_id": choice_id or "", "knowledge_only": event.kind == "card"}
                if card_id:
                    reference.update(source_card_id=card_id,
                                     known_by=deepcopy(event.data.get("known_by", [])))
                narratives.append(reference)
            elif event.kind != "flag":
                self.messages.append({"role": "assistant", "content": text,
                                      "speaker": self.npc if prepared_opening else ""})
                if event.kind == "move" and opening_text and not opening_inserted:
                    self.messages.append({"role": "assistant", "content": opening_text, "speaker": ""})
                    opening_inserted = True
            self.audit.append({"kind": event.kind, "text": text, "data": deepcopy(event.data)})
            if event.kind not in {"flag", "card"}:
                public.append(event.text)
                public_records.append({"kind": event.kind, "text": event.text, "data": deepcopy(event.data)})
        if opening_displayed:
            # 첫 대사는 이미 표시했으므로 같은 만남의 인물 보충 설명을 다시 생성하지 않습니다.
            # 카드 공개와 감사 기록은 유지하여 이후 질문의 근거로 사용할 수 있습니다.
            narratives = [row for row in narratives
                if not (row["kind"] == "card"
                    and self.content.cards.get(row.get("source_card_id"), {}).get("kind") == "npc"
                    and not self.content.cards.get(row.get("source_card_id"), {}).get("dialogue"))]
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
            if engine_delta is not None:
                self.last_result["engine_delta"] = deepcopy(engine_delta)
            if transaction_record is not None:
                self.last_result["transaction_records"] = [deepcopy(transaction_record)]
                parties = transaction_record.get("knowledge_npcs", [])
                self.last_result["witness_npc"] = parties[0] if len(parties) == 1 else ""
        # 이번 호출의 자료만 사용하며 다음 행동으로 표시 대기를 넘기지 않습니다.
        if narratives:
            action = player_input or self.content.choices.get(choice_id, {}).get("label") or "현재 장소를 확인한다"
            return self._render_action_result(narratives, action, choice_id, movement_scene=movement_scene)
        return {"generated": False, "error": None, "narrative_count": 0}

    def _render_action_result(self, narratives, player_input, choice_id, movement_scene=False):
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
                                "style": deepcopy(actor.get("style") or profile.get("style", "")),
                                "speech_style": deepcopy(actor.get("speech_style", {})),
                               "relationships": [{key: relation[key] for key in ("person", "relation_to_me")}
                                   for relation in profile.get("relationships", [])
                                   if isinstance(relation, dict) and all(isinstance(relation.get(key), str)
                                       for key in ("person", "relation_to_me"))]}
        definition = self.content.choices.get(choice_id, {})
        player_role = {"kind": "player", "name": self.engine.hero.name}
        participant_roles = {"performer": player_role}
        if npc:
            participant_roles["respondent"] = {"kind": "npc", "name": npc}
        if definition.get("interaction") == "accept" and definition.get("target_npc"):
            participant_roles.update(request_issuer={"kind": "npc", "name": definition["target_npc"]},
                                     request_acceptor=deepcopy(player_role))
        payload = {"player_input": player_input, "npc_name": npc, "speaker_context": speaker_context,
                   "response_mode": "npc_dialogue" if npc else "scene_dialogue",
                    "action_kind": definition.get("interaction") or definition.get("operation"),
                    "participant_roles": participant_roles,
                    "movement_scene": movement_scene,
                   "public_narratives": deepcopy(narratives), "engine_result": deepcopy(self.last_result),
                    "action_subject": {"kind": "player", "name": self.engine.hero.name},
                    "engine_delta": deepcopy((self.last_result or {}).get("engine_delta")),
                   "player_snapshot": player_snapshot(self.engine), "scene_context": self.scene_context(),
                    # 행동 결과는 이번 실행 자료로만 표현하며 과거 대사를 재연하지 않습니다.
                    "recent_conversation": []}
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
            self.messages.append({"role": "assistant", "content": result["text"], "speaker": npc})
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
            if self.api and isinstance(error, self.api.get("OutputStreamCancelled", ())):
                raise
            self.debug["행동 결과 생성"] = {"전달 자료": payload, "출력 결과": deepcopy(result),
                "생성 오류": repr(error), "guard_report": deepcopy(getattr(error, "guard_report", None)),
                "raw_response": getattr(error, "raw_response", None), "현재 묶음 모델 호출": int(called),
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
        self.cancel_recommendations()
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
        before_values = {"room": self.engine.s.room, "gold_cp": self.engine.s.gold_cp,
                         "inventory": dict(self.engine.s.inventory), "hp": self.engine.hero.hp}
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
        inventory = self.engine.s.inventory
        changes = [{"item_id": item_id, "item_name": self.content.items.get(item_id, {}).get("name", item_id),
                    "quantity_change": inventory.get(item_id, 0) - before_values["inventory"].get(item_id, 0)}
                   for item_id in sorted(set(inventory) | set(before_values["inventory"]))
                   if inventory.get(item_id, 0) != before_values["inventory"].get(item_id, 0)]
        delta = {"executed_action_id": pending.get("choice"), "room_before": before_values["room"],
                 "room_after": self.engine.s.room, "gold_cp_change": self.engine.s.gold_cp - before_values["gold_cp"],
                 "inventory_changes": changes, "hp_change": self.engine.hero.hp - before_values["hp"],
                 "engine_pending": bool(self.engine.s.pending)}
        outcome = self._record(events, transaction_record, choice_id=pending.get("choice"),
                               player_input=pending.get("request") or pending.get("label", "행동을 실행한다"),
                               engine_delta=delta)
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
                role = DIALOGUE_CARDS.get("actors", {}).get(person["name"], {}).get("role", "")
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
            '인사와 정보 요청이 함께 있으면 인사로 전체를 덮지 말고 마지막 정보 요청에 맞춰 분류한다. '
            'decline: 부탁이나 의뢰를 거절하는 말. farewell: 대화를 마치거나 작별하는 말. '
            'knowledge: 인물의 관계·직업·과거·전해들은 이야기·사건 원인·단서·인지 방법을 묻거나 그런 사실을 주장하는 말. '
            '지속적인 감각·인지 제약이나 상대를 어떻게 알아챘는지에 관한 질문도 knowledge다. 현재의 가벼운 안부와 설정상 제약을 구별한다. '
            '주변에서 지금 무엇을 보거나 들을 수 있는지 묻는 질문도 knowledge다. 실제 관찰을 하겠다는 선언인 look과 구별한다. '
            'state: 현재 물건의 소유·보관·수령·지급·구매·HP나 실제 행동 완료 기록을 확인하거나 그런 상태를 주장하는 말. '
            'action: 지금 하려는 명확한 물리적 행동·이동·접근·공격 또는 명확한 의뢰 수락. '
            '현재 상대가 부탁한 일을 맡거나 돕겠다고 약속하는 말도 action이며, 추가 물리 행동을 말할 필요는 없다. '
            '질문·과거 보고·가정·위협은 실행이나 수락이 아니다. '
            'look: 특정 대상이나 진입 목적 없이 주변 전체를 둘러보겠다는 선언. '
            '대상 안쪽을 조사하거나 구조물을 열고 들어가려는 선언은 action이다. '
            'rule: 게임 밖에서도 똑같이 답할 수 있는 일반 규칙 질문. '
            'clarify: 대상이나 뜻이 불명확해 먼저 확인해야 하는 행동. '
            '질문형 문장이라는 이유만으로 knowledge나 state가 되지 않는다. '
            '직전 NPC가 꺼낸 의뢰와 마지막 플레이어가 지금 원하는 것을 구별한다. '
            'state_topics는 마지막 발언이 확인하거나 주장·전제하는 현재 상태나 실제 완료 기록의 범위만 고르는 배열이다. '
            '허용 값은 money(잔액·지급·수령·구매 대금·반환), items(물건의 소유·보관·수령), '
            'hp(게임상의 현재 HP·피해·회복), rest(실제 휴식·숙박·예약의 완료나 그에 따른 회복), resources(실제 직업·레벨·사용 가능한 주문·슬롯·능력 사용 횟수와 사용 완료 기록)다. '
            '발언 종류와 상태 범위는 독립적이므로 거절·작별·감정 표현 속에도 상태 확인이나 전제가 있으면 해당 범위를 넣는다. '
            '상태 주장을 포함하지 않은 일상 대화·거절이나 단순한 감정·신체 감각·말하는 속도·잠깐 함께 쉬자는 허락에는 빈 배열을 쓴다. '
            '장소·시간·환경에서 보고 듣는 내용은 위의 수치·소유·완료 범위에 해당하지 않으면 state_topics를 빈 배열로 둔다. '
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
        if isinstance(raw_topics, list) and all(isinstance(topic, str) and topic in known_state_topics for topic in raw_topics):
            state_topics = list(dict.fromkeys(raw_topics))
        else:
            state_topics = []
        routes = {"social": ("TALK", "inform", False, False), "decline": ("TALK", "decline", False, False),
            "farewell": ("TALK", "farewell", False, False), "knowledge": ("TALK", "ask", True, False),
            "state": ("TALK", "ask", True, True), "action": ("CLARIFY", "action", True, True),
            "look": ("LOOK", "action", False, False), "rule": ("RULE", "ask", True, False),
            "clarify": ("CLARIFY", "action", False, False)}
        classification_valid = isinstance(kind, str) and kind in routes
        kind = kind if classification_valid else "clarify"
        classified_kind, question_form = kind, _question_form(message)
        information_override = kind == "social" and _information_question(message)
        if information_override:
            kind = "knowledge"
        if question_form and kind in {"action", "look"}:
            kind = "knowledge"
        selected, intent, info, state = routes[kind]
        mapper_raw, mapper_seconds, mapper_calls = "", 0.0, 0
        navigation_resolution = None
        blocked_action = None
        if kind in {"action", "look"}:
            choices = self.choices()
            if self.npc:
                current = self.profiles()[self.npc]
                repeated = {current["dialogue"].get("approach_action"), "talk:" + current["id"]}
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
                    **{key: definition[key] for key in ("interaction", "intent", "target_npc", "purpose", "target", "operation") if key in definition}})
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
                blocked.append({key: definition[key] for key in
                    ("id", "label", "purpose", "target", "operation", "blocked_reason") if key in definition})
                if room_id != self.engine.s.room:
                    blocked[-1]["blocked_reason"] = "이 행동은 " + self.content.room_name(room_id) + "에서 할 수 있습니다. 먼저 그 장소로 이동해야 합니다."
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
                '현재 플레이어 발언을 실제 가능한 행동 목록과 대조한다. JSON 하나만 반환한다: '
                '{"action_id":"현재 목록의 정확한 ID 또는 빈 문자열","intent":"action|accept|ask|hypothetical|joke|decline",'
                '"offer_id":"수락한 current_offers의 ID 또는 빈 문자열","blocked_action_id":"원하는 blocked_actions의 ID 또는 빈 문자열"}. '
                '지금 행동을 실행하는 선언은 action, 명확한 의뢰 수락은 accept다. '
                '현재 상대의 부탁을 맡거나 돕겠다는 약속은 accept이며, target_npc와 purpose가 맞는 수락 행동에 연결한다. '
                'intent를 action으로 표현하더라도 현재 부탁을 맡겠다는 뜻이면 offer_id에 그 의뢰를 넣는다. '
                '질문·조건부 가정·위협·거절에는 offer_id를 넣지 않고 수락하지 않는다. '
                '질문은 ask, 가정·위협은 hypothetical, 농담은 joke이며 실제 실행하지 않는다. '
                '대상과 operation이 맞는 실제 행동을 고른다. 대상 내부 조사·진입을 주변 전체 관찰로 바꾸지 않는다. '
                '목적지로 이동하려는 선언은 해당 목적지의 travel 행동을 고른다. 중간 장소 이동이나 중간 인물 만남으로 대신하지 않는다. '
                '원하는 행동의 조건이 막혀 있으면 blocked_action_id를 고르며, 그 선행 행동을 대신 실행하지 않는다. '
                '단순 언급·질문·과거 보고를 수락이나 실행으로 바꾸지 않는다. '
                'explicit_action은 플레이어가 행동 표시로 감쌌다는 뜻이다. 표시된 의도를 실제 가능한 행동과 대조한다. '
                '대상이 불명확하거나 현재 가능한 ID가 없으면 빈 문자열을 반환한다. ID를 만들지 않는다.\n현재 자료:\n'
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
                            else "TALK" if self.npc else "CLARIFY")
            if intent == "joke":
                info = state = False
            if selected == "look:surroundings":
                selected = "LOOK"
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
            "blocked_action": blocked_action,
            "navigation_resolution": navigation_resolution,
            "input_uncertain": uncertain,
            "model_calls": 1 + mapper_calls, "classification_calls": 1, "mapper_calls": mapper_calls,
            "preparation_seconds": round(preparation_seconds, 3),
            "classification_seconds": round(classification_seconds, 3), "mapper_seconds": round(mapper_seconds, 3),
            "seconds": round(time.perf_counter() - started, 3)}

    def send(self, message):
        message = message.strip()
        if not message or len(message) > 1200:
            raise ValueError("1~1200자로 말이나 행동을 입력하세요.")
        self.cancel_recommendations()
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
        if route_debug.get("blocked_action"):
            blocked = route_debug["blocked_action"]
            text = blocked["blocked_reason"]
            if blocked.get("next_action_id") in {choice.id for choice in self.choices()}:
                text += "\n\n다음 가능한 행동: " + blocked["next_action_label"]
                next_steps = [{"label": blocked["next_action_label"], "text": blocked["next_action_label"],
                               "action_id": blocked["next_action_id"], "source": "prerequisite"}]
            else:
                next_steps = []
            self.messages.append({"role": "assistant", "content": text})
            self.debug = {"의도 해석": route_debug}
            self.last_status = "선행 조건을 안내했습니다. 행동은 실행하지 않았습니다."
            self.refresh_suggestions(next_steps)
            if next_steps:
                # 이번에 요청한 행동의 선행 단계는 일반 장면 추천보다 우선합니다.
                self.suggestions = next_steps
            return
        if route.startswith("travel:"):
            self.debug = {"의도 해석": route_debug}
            self.travel_to(route[7:])
            return
        if route.startswith("meet:"):
            self.debug = {"의도 해석": route_debug}
            card = self.content.cards.get(route[5:], {})
            self.visit_npc(card.get("dialogue", {}).get("name", ""))
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

    def _payment_prerequisites(self, npc_name, transactions):
        """실제 지급 행동의 물품 조건만 복사하며 비공개 진행 조건은 숨깁니다.

        all·any·not 구조를 보존합니다. complete는 물품 조건의 추출이 완전한지이며,
        지급 행동 자체가 지금 가능하다는 뜻은 아닙니다.
        """
        def project(condition):
            if condition is None:
                return None, False, True
            if not isinstance(condition, dict) or len(condition) != 1:
                return None, False, False
            operator, value = next(iter(condition.items()))
            if operator in {"all", "any"}:
                if not isinstance(value, list):
                    return None, False, False
                children = [project(child) for child in value]
                has_item = any(child[1] for child in children)
                tree = {operator: [child[0] if child[1] else {"other_condition": "omitted"}
                                   for child in children]} if has_item else None
                return tree, has_item, all(child[2] for child in children)
            if operator == "not":
                child, has_item, complete = project(value)
                return {"not": child} if has_item else None, has_item, complete
            if operator == "item":
                return ({"item": value}, True, True) if isinstance(value, str) and value else (None, False, False)
            known_non_items = {"flag", "var", "shown", "scene_at_least", "darkvision", "room_loaded", "trap_active", "condition"}
            return None, False, operator in known_non_items

        if not isinstance(transactions, dict) or transactions.get("npc_name") != npc_name:
            return []
        result = []
        for transaction in transactions.get("transactions", []):
            if not transaction.get("payments"):
                continue
            action_id = transaction.get("action_id")
            definition = self.content.choices.get(action_id)
            tree, has_item, complete = project(definition.get("requires")) if definition else (None, False, False)
            result.append({"action_id": action_id, "item_requirements": [tree] if has_item else [], "complete": complete})
        return result

    def _generate_dialogue(self, message, route, route_debug):
        if not self.api:
            raise RuntimeError("대화 모델이 연결되지 않았습니다.")
        started = time.perf_counter()
        speaker = "" if route == "RULE" else self.npc
        history = self.histories.setdefault(speaker, [])
        recommendation_history = deepcopy(history)
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
            compact["speech_style"] = deepcopy(actor.get("speech_style", {}))
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
            payload["dialogue_context"] = {"compact_speaker": compact, "fact_candidates": facts, "selected_facts": [],
                "transactions": directed_transactions(self.engine, profile, speaker, TRANSACTION_BINDINGS),
                "selected_memories": []}
            payload["dialogue_context"]["shop"] = shop_state(self.engine, speaker, self.transaction_ledger, TRANSACTION_BINDINGS)
            payload["dialogue_context"]["contract"] = deepcopy(profile.get("quest_offer"))
            payload["dialogue_context"]["request_participants"] = [
                {"requester": {"kind": "npc", "name": speaker},
                 "proposed_executor": {"kind": "player", "name": payload["player_snapshot"]["name"]},
                 "purpose": definition.get("purpose", ""),
                 "acceptance_completed": action_id in self.engine.s.done}
                for action_id in profile.get("action_ids", [])
                for definition in [self.content.choices.get(action_id, {})]
                if definition.get("intent") == "accept" and definition.get("target_npc") == speaker]
            payload["dialogue_context"]["npc_state"] = self.engine.npc_state(current_profile["id"])
            payload["dialogue_context"]["payment_prerequisites"] = self._payment_prerequisites(
                speaker, payload["dialogue_context"]["transactions"])
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
        explicit_state = (route_debug.get("kind") == "state" and self.turn_needs_state
            and bool(payload["state_topics"]) and all(isinstance(topic, str)
                and topic in {"money", "items", "hp", "rest", "resources"}
                for topic in payload["state_topics"]))
        if explicit_state and not engine_only and callable(self.api.get("format_confirmed_state")):
            state_display = self.api["format_confirmed_state"](payload, engine_result=self.last_result, npc_name=speaker)
        self.last_generation_seconds = result["generation_seconds"]
        slow, ending = self.last_generation_seconds > 6, self.turn_intent in {"decline", "farewell"}
        can_recommend = (speaker and not engine_only and not self.turn_needs_state and not ending
                         and callable(self.api.get("generate_followup_question")))
        followup = {"called": False, "seconds": 0.0, "wall_seconds": 0.0,
                    "reason": "deferred" if can_recommend else "state_uses_engine_choices" if engine_only or self.turn_needs_state else "ending" if ending else "not_npc" if not speaker else "api_unavailable"}
        recommendations = []
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
            self.messages.append({"role": "assistant", "content": state_display, "speaker": ""})
        heading = "**" + speaker + "**\n" if speaker and not engine_only else ""
        self.messages.append({"role": "assistant", "content": heading + result["text"],
                              "speaker": speaker if not engine_only else ""})
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
        if can_recommend and not any(row.get("source") == "scene" for row in self.suggestions):
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
        elif can_recommend:
            self.debug["후속 질문"]["reason"] = "scene_choices_available"

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
