"""NPC와의 충돌 및 전투 후 상태를 게임 한 판 안에 보관합니다.

입력: 인물 카드 ID와 현재 엔진의 장소·공개 조건·전투 상태입니다.
처리: 대화 가능 여부와 공격 선택지를 구하고, 공격 시 기존 전투로 연결합니다.
전투의 실제 HP·기절·사망은 GameState.vars의 인물별 기록에 반영합니다.
출력: NPC 상태 사본, 가능한 공격 목록, 상태 변화를 설명하는 Event 목록입니다.

Engine이 SocialMixin을 상속합니다. 인물의 정의는 Content에서 읽고 바뀌는 상태는
현재 게임에 저장하므로 서로 다른 사용자의 NPC 상태가 섞이지 않게 구성합니다."""
from copy import deepcopy

from .events import ChoiceView, Event


class SocialMixin:
    """Engine에 NPC 상태 조회·공격·전투 후 반영 기능을 보탭니다.

    self.c의 인물 정의와 self.s의 진행 상태를 함께 읽습니다.
    바뀌는 HP·태도·의식 상태는 s.vars['npcs']의 해당 카드 ID에 저장합니다."""

    def npc_state(self, card_id):
        """카드 정의와 저장된 인물 상태를 합쳐 화면·대화 처리용 사본을 반환합니다.

        해당 인물과 전투 중이거나 기절·사망한 경우를 반영해 can_talk를 구합니다.
        조회 결과를 바꾸는 것만으로 원래 저장 기록이 갱신되지는 않습니다."""
        card = self.c.cards.get(card_id, {})
        spec = getattr(self.c, "npc_combat", {}).get(card_id)
        profile = card.get("dialogue", {})
        default_hp = spec["hp"] if spec else None
        value = {"card_id": card_id, "name": profile.get("name", card_id),
                 "hp": default_hp, "hp_max": default_hp, "status": "alive", "attitude": "neutral"}
        value.update(deepcopy(self.s.vars.get("npcs", {}).get(card_id, {})))
        value["alive"] = value["status"] != "dead"
        active = bool(self.s.combat and self.c.encounters[self.s.combat["enc"]].get("npc_card_id") == card_id)
        value["can_talk"] = bool(profile) and value["status"] == "alive" and not active
        value["combat_supported"] = bool(spec)
        return value

    def npc_can_talk(self, card_id):
        """기절·사망·해당 인물과 전투 중인 경우 대화를 막습니다."""
        return self.npc_state(card_id)["can_talk"]

    def npc_for_choice(self, choice_id):
        """인물에게 직접 하는 대화·의뢰 행동만 연결합니다. 방 조사는 그대로 둡니다."""
        for card_id, card in self.c.cards.items():
            profile = card.get("dialogue", {})
            if choice_id == profile.get("approach_action") or choice_id in profile.get("action_ids", []):
                return card_id
        return None

    def npc_attack_choices(self):
        """현재 보이는 살아 있는 인물만 실제 공격 대상으로 고를 수 있습니다."""
        if self.s.combat or self.s.pending or self.s.ended or not self.hero.can_act():
            return []
        choices = []
        for card_id in getattr(self.c, "npc_combat", {}):
            card = self.c.cards[card_id]
            status = self.npc_state(card_id)
            if card["room"] != self.s.room or not self.eligible(card_id) or status["status"] != "alive":
                continue
            profile = card["dialogue"]
            name = profile["name"] if card_id in self.s.shown else profile.get("appearance", profile["name"])
            choices.append(ChoiceView("npc:attack:" + card_id, name + "에게 공격을 시도한다",
                                      [self.hero.id], self.hero.id, "우선권 후 공격 굴림"))
        return choices

    def npc_attack(self, card_id):
        """현재 가능한 공격인지 확인한 뒤 적대 기록을 남기고 기존 전투를 시작합니다.

        공격 시도 Event와 start_combat의 결과를 이어 반환합니다.
        선언만으로 명중·피해를 확정하지 않고 이후 전투의 굴림으로 처리합니다."""
        if "npc:attack:" + card_id not in {choice.id for choice in self.npc_attack_choices()}:
            raise ValueError("현재 공격할 수 없는 인물입니다.")
        status = self.npc_state(card_id)
        stored = {key: status[key] for key in ("card_id", "name", "hp", "hp_max", "status", "attitude")}
        stored["attitude"] = "hostile"
        self.s.vars.setdefault("npcs", {})[card_id] = stored
        return [Event("hostility", status["name"] + "에게 공격을 시도해 충돌이 시작됐다.",
                      {"npc_card_id": card_id, "attitude": "hostile"})] + self.start_combat("npc:" + card_id)

    def _sync_npc_combat(self, combat=None):
        """전투의 실제 적 HP·제압·사망을 해당 NPC의 지속 상태에 옮깁니다.

        이 기록은 전투가 끝난 뒤에도 남아 이후 대화 가능 여부와 재공격 여부에 쓰입니다."""
        combat = combat or self.s.combat
        if not combat:
            return
        card_id = self.c.encounters[combat["enc"]].get("npc_card_id")
        if not card_id:
            return
        enemy = next(iter(combat["enemies"].values()))
        profile = self.c.cards[card_id]["dialogue"]
        previous = self.s.vars.setdefault("npcs", {}).get(card_id, {})
        status = "dead" if enemy["hp"] <= 0 or enemy["out"] == "dead" else "unconscious" if enemy["out"] == "unconscious" else "alive"
        value = {"card_id": card_id, "name": profile["name"], "hp": enemy["hp"], "hp_max": enemy["hp_max"],
                 "status": status, "attitude": "hostile"}
        if status == "unconscious":
            value["unconscious_until"] = previous.get("unconscious_until", self.s.minutes + 60)
        self.s.vars["npcs"][card_id] = value
        if status == "dead":
            self.s.flags.add("npc_dead:" + card_id)

    def _npc_pass_time(self):
        """근접 제압으로 기절한 인물은 짧은 휴식이 끝나면 깨어납니다 [SRD p.17]."""
        events = []
        for card_id, value in self.s.vars.get("npcs", {}).items():
            if value.get("status") == "unconscious" and self.s.minutes >= value.get("unconscious_until", float("inf")):
                value["status"] = "alive"
                value.pop("unconscious_until", None)
                if self.c.cards.get(card_id, {}).get("room") == self.s.room:
                    events.append(Event("status", value["name"] + "이(가) 의식을 되찾았다.", {"npc_card_id": card_id}))
        return events
