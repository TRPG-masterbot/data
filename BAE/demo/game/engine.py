"""게임 엔진: 선택지 → 판정 → 효과(상태·flag·카드) → 다음 선택지.

규칙 근거 표기:
  [원본]  시나리오 카드        [SRD p.x] 실제로 읽어 확인한 SRD 쪽
  [보완]  개발 중 추가 규칙 (analysis/05_시스템설계/05-2a, 05-6)

모든 행동은 choose(choice_id, actor_id) 한 곳으로 들어온다. 나중에 자연어 입력을 붙일 때도
"자연어 → 현재 선택지 ID" 해석만 앞에 두면 이 경로를 그대로 쓴다.

한 번의 행동이 처리되는 순서 (화면·CLI·서버 모두 같다):
  1. available_choices()  지금 고를 수 있는 선택지(ChoiceView) 목록을 만든다.
  2. choose(id)           선택지 id의 앞부분(move:, buy:, cmb: …)을 보고 처리 함수로 보낸다.
                          장면 데이터의 선택지(접두어 없음)는 _run_choice → 판정 → _apply(효과 목록).
  3. 결과는 Event 목록으로 돌려준다. 상태(self.s)는 이미 바뀐 뒤이다.

이 파일이 쓰는 다른 파일:
  content.py (읽기 전용 데이터) · model.py (GameState·Character) · dice.py (주사위)
  events.py (Event·ChoiceView) · combat.py (CombatMixin: 전투 함수는 그쪽에 있고 Engine이 상속)
"""
import copy
import math
import re

from .combat import CombatMixin
from .content import Content
from .dice import Dice
from .events import ChoiceView, Event
from .model import ABILITY_KO, GameState

SKILL_KO = {
    "athletics": "운동", "acrobatics": "곡예", "sleight_of_hand": "손재주", "stealth": "은신",
    "arcana": "비전", "history": "역사", "investigation": "수사", "nature": "자연", "religion": "종교",
    "animal_handling": "동물 조련", "insight": "통찰", "medicine": "의학", "perception": "감지",
    "survival": "생존", "deception": "기만", "intimidation": "위협", "performance": "공연",
    "persuasion": "설득",
}
TOOL_KO = {"thieves_tools": "도둑 도구"}
LONG_REST_MIN = 8 * 60          # [SRD Rules Glossary: Long Rest] 최소 8시간
LONG_REST_GAP_MIN = 16 * 60     # [SRD Rules Glossary: Long Rest] 끝난 뒤 16시간 대기
SHORT_REST_MIN = 60             # [SRD Rules Glossary: Short Rest] 1시간
# 종족 특성 → 피해 저항 / 상태를 피하는 구원에 이점. Dwarven Resilience [SRD p.84]
TRAIT_RESIST = {"dwarven_resilience": {"poison"}}
TRAIT_SAVE_ADV = {"dwarven_resilience": {"poisoned"}}
# 지속 효과(조건) → 그 상태를 피하는 구원에 이점. Antitoxin [SRD p.94]
EFFECT_SAVE_ADV = {"antitoxin": {"poisoned"}}
EXPIRE_TEXT = {"poisoned": "{name}의 중독이 풀렸다.", "antitoxin": "{name}가 마신 해독제의 효과가 끝났다."}
DAMAGE_KO = {"poison": "독", "bludgeoning": "타격"}
SENTENCE_SPLIT = re.compile(r'(?<=[.!?。])\s+|(?<=[.!?]")\s+')
PERIAPT_WOUND = "periapt_wound_closure"   # 장면 2 전사 아이템 [원본]. 동조 절차는 생략(얻으면 바로 착용) [보완 W7]
MOVE_KEYS = "abcde"            # 이동 선택지의 고정 키 (s/k/q/h/r/w와 겹치지 않음)


class Engine(CombatMixin):
    """게임 한 판. 규칙 판정과 상태 변경은 모두 이 객체가 한다.

    self.c     Content    : 장면·카드·아이템 등 읽기 전용 데이터 (게임 중 바뀌지 않음)
    self.s     GameState  : 이번 판의 모든 바뀌는 값 (위치, flag, HP, 소지품 …). 체크포인트는 이것의 복사본
    self.dice  Dice       : 주사위 (테스트에서는 FixedDice로 값을 정해 넣는다)
    self.opening          : 게임 시작 때 첫 방에 들어가며 나온 Event 목록 (화면이 처음에 보여 준다)
    """

    def __init__(self, content=None, dice=None, seed=None, char_id=None):
        if char_id is None:
            raise ValueError("플레이할 캐릭터를 골라야 합니다 (char_id). 1인 1캐릭터 [05-6 ①]")
        self.c = content or Content()
        self.dice = dice or Dice(seed)
        start = self.c.start_room
        self.s = GameState(party=self.c.new_party(char_id), room=start, scene=self.c.start_scene)
        self.s.vars["last_long_rest_end"] = None
        self.s.stock = {k: v.get("stock") for k, v in self.c.items.items() if "price_cp" in v}
        for m in self.s.party:
            self.init_resources(m)    # 주문 슬롯·Second Wind 등 (W6)
        self.checkpoint = None
        self.checkpoint_label = ""
        self.opening = self._enter(start)
        self._save_checkpoint("게임 시작")

    @property
    def hero(self):
        """플레이어 캐릭터. 1인 1캐릭터라 파티는 항상 한 명이다 (party 목록 구조는 핫시트용으로 남겨 둠)."""
        return self.s.party[0]

    # ------------------------------------------------------------------ 카드 열람/표시 (05-2a 3장)
    def eligible(self, card_id, char=None):
        """열람 가능: 기획서의 2축 필터(장면·대상) + flag + 방. char가 주어지면 그 캐릭터 기준."""
        card = self.c.cards[card_id]
        s = self.s
        if s.scene < card["reveal_after"]:
            return False
        if card["req_flag"] and card["req_flag"] not in s.flags:
            return False
        if card["room"] and card["room"] not in s.visited:
            return False
        if card["owner"] != "all":
            members = [char] if char else s.party
            if not any(m.cls == card["owner"] for m in members):
                return False
        return True

    def _view_text(self, card_id):
        """플레이어용 문장. card_views(보완)가 있으면 조건부 반응 문장을 빼고, 조건이 맞는 문장만 붙인다.
        문장은 원본 player_text의 문장을 그대로 쓴다(새 문장을 만들지 않음)."""
        card = self.c.cards[card_id]
        view = self.c.card_views.get(card_id)
        if not view:
            return card["player_text"], []
        sents = [x for x in SENTENCE_SPLIT.split(card["player_text"]) if x.strip()]
        hide = view.get("drop", []) + [p["match"] for p in view.get("parts", [])]
        kept = [x for x in sents if not any(h in x for h in hide)]
        later = []
        for i, part in enumerate(view.get("parts", [])):
            text = " ".join(x for x in sents if part["match"] in x)
            if self.cond(part.get("when")):
                kept.append(text)
            else:
                later.append((f"{card_id}#{i}", text, part.get("when")))
        return " ".join(kept), later

    def _show(self, card_id, actor=None):
        """카드를 플레이어에게 보여 준다 ({"show": 카드 id} 효과, 방 진입 묘사).

        - 이미 보여 준 카드, player_text가 빈 GM 전용 카드는 무시한다.
        - 아직 열람 조건(장면·flag)이 안 되면 deferred에 적어 두었다가, 이 방에서 '다시 둘러보기'로 보여 준다.
        - 보여 준 카드는 s.shown[카드 id] = [알게 된 캐릭터 id]로 기록한다. 화면의 단서 탭과
          선택지 조건 {"shown": …}이 이 기록을 쓴다.
        """
        card = self.c.cards[card_id]
        if card_id in self.s.shown or not card["player_text"]:
            return []
        if not self.eligible(card_id):
            # 공개 시점·flag가 아직이면 나중에 이 방에서 다시 볼 수 있게 기록 (지연 공개)
            if card["owner"] == "all" or any(m.cls == card["owner"] for m in self.s.party):
                self.s.deferred[card_id] = {"room": self.s.room, "card": card_id}
            return []
        self.s.deferred.pop(card_id, None)
        if card["owner"] == "all":
            knowers = [m.id for m in self.s.party]
            who = ""
        else:  # 역할 전용 카드: 그 역할 캐릭터만 앎 (플레이어 축)
            knowers = [m.id for m in self.s.party if m.cls == card["owner"]]
            who = "·".join(self.s.member(k).name for k in knowers)
        text, later = self._view_text(card_id)
        for key, part_text, when in later:
            self.s.deferred[key] = {"room": self.s.room, "part": part_text, "when": when, "card": card_id}
        self.s.shown[card_id] = knowers
        head = f"[{card['label']}]" + (f" ({who}만 알아챘다)" if who else "")
        return [Event("card", f"{head} {text}", {"id": card_id, "known_by": knowers})]

    def _ready_deferred(self):
        """지연 공개 목록 중 '지금 이 방에서' 보여 줄 수 있게 된 것의 key 목록 ('r' 선택지를 띄울지 판단)."""
        out = []
        for key, d in self.s.deferred.items():
            if d["room"] != self.s.room:
                continue
            if "part" in d:
                if d["card"] in self.s.shown and self.cond(d.get("when")):
                    out.append(key)
            elif self.eligible(d["card"]):
                out.append(key)
        return out

    def _revisit(self):
        """'다시 둘러보며 새 소식을 듣는다'(revisit:방) — 준비된 지연 카드·문장을 한꺼번에 보여 준다."""
        events = []
        for key in self._ready_deferred():
            d = self.s.deferred.pop(key)
            if "part" in d:
                card = self.c.cards[d["card"]]
                events.append(Event("card", f"[{card['label']}] {d['part']}", {"id": d["card"], "part": key}))
            else:
                events += self._show(d["card"])
        return events

    def player_known_cards(self, char_id=None):
        """플레이어에게 이미 보여 준 카드 id 목록 (char_id를 주면 그 캐릭터가 아는 것만). 서버 단서 탭·CLI 'k'가 쓴다."""
        return [cid for cid, ks in self.s.shown.items() if char_id is None or char_id in ks]

    # ------------------------------------------------------------------ 조건식 (05-2 4장)
    def cond(self, c, actor=None):
        """장면 데이터의 조건식을 참/거짓으로 계산한다. None(조건 없음)은 항상 참.

        조건식은 키 하나짜리 dict이며 all/any/not으로 묶을 수 있다. 예:
          {"all": [{"item": "rope"}, {"not": {"var": "rope_anchored"}}]}  → 밧줄이 있고, 아직 고정하지 않았다
        쓰는 곳: 선택지 requires, 출구 exit_requires, 효과 {"if": …}, 방 진입 on_enter, card_views의 when.
        새 조건 종류를 추가하면 데이터에서 바로 쓸 수 있다 (모르는 키는 ValueError로 바로 드러난다).
        """
        if c is None:
            return True
        if "all" in c:
            return all(self.cond(x, actor) for x in c["all"])
        if "any" in c:
            return any(self.cond(x, actor) for x in c["any"])
        if "not" in c:
            return not self.cond(c["not"], actor)
        if "flag" in c:
            return c["flag"] in self.s.flags
        if "var" in c:
            return bool(self.s.vars.get(c["var"]))
        if "item" in c:
            return self.s.has_item(c["item"])
        if "shown" in c:
            return c["shown"] in self.s.shown
        if "scene_at_least" in c:
            return self.s.scene >= c["scene_at_least"]
        if "darkvision" in c:  # 행동하는 캐릭터(없으면 주인공)에게 암시야가 있는지 (W7)
            return (actor or self.hero).darkvision > 0
        if "room_loaded" in c:  # 그 방의 장면 파일이 불러와졌는지 (아직 없는 장면으로 가는 임시 선택지용)
            return c["room_loaded"] in self.c.rooms
        if "trap_active" in c:
            return self.trap_active(c["trap_active"])
        if "condition" in c:  # 행동하는 캐릭터(없으면 주인공)의 상태
            return c["condition"] in (actor or self.hero).conditions
        raise ValueError(f"알 수 없는 조건: {c}")

    # ------------------------------------------------------------------ 판정
    def check_mod(self, char, chk):
        """(수정치, 설명). 숙련이면 숙련 보너스 추가 [SRD p.8]."""
        ab = chk["ability"]
        mod = char.mod(ab)
        name = ABILITY_KO[ab]
        prof = False
        if "skill" in chk:
            prof = chk["skill"] in char.skill_profs
            name += f"({SKILL_KO[chk['skill']]})"
        elif "tool" in chk:
            prof = chk["tool"] in char.tool_profs
            name += f"·{TOOL_KO.get(chk['tool'], chk['tool'])}"
        if prof:
            mod += char.proficiency_bonus
        return mod, name

    def best_option(self, char, chk):
        """판정에 여러 방법이 있으면(options, 예: 운동 또는 곡예) 이 캐릭터에게 수정치가 가장 높은 것을 고른다."""
        if "options" in chk:
            opts = [dict(o, dc=chk["dc"]) for o in chk["options"]]
            return max(opts, key=lambda o: self.check_mod(char, o)[0])
        return chk

    def roll_check(self, char, chk):
        """능력 판정: d20 + 수정치 ≥ DC 이면 성공. (성공 여부, 화면용 check Event)를 돌려준다.
        chk 예: {"ability": "str", "skill": "athletics", "dc": 15}. 상태는 바꾸지 않는다 (결과 적용은 호출한 쪽)."""
        chk = self.best_option(char, chk)
        mod, name = self.check_mod(char, chk)
        # 중독: 능력 판정에 불리 [SRD Rules Glossary: Poisoned]
        mode = "disadvantage" if "poisoned" in char.conditions else "normal"
        d, rolls = self.dice.d20(mode)
        total = d + mod
        ok = total >= chk["dc"]
        tag = " (중독: 불리)" if mode == "disadvantage" else ""
        text = (f"{char.name}의 {name} 판정: d20 {rolls if len(rolls) > 1 else d} "
                f"{mod:+d} = {total} vs DC {chk['dc']} → {'성공' if ok else '실패'}{tag}")
        return ok, Event("check", text, {"char": char.id, "roll": d, "rolls": rolls, "mod": mod,
                                          "total": total, "dc": chk["dc"], "success": ok})

    def save_mode(self, char, ability, avoids=None, advantage=False):
        """구원 굴림의 이점/불리. 둘 다 있으면 상쇄 [SRD p.7]."""
        adv = advantage
        if avoids:
            adv = adv or any(avoids in TRAIT_SAVE_ADV.get(t, ()) for t in char.traits)
            adv = adv or any(avoids in EFFECT_SAVE_ADV.get(cn, ()) for cn in char.conditions)
        dis = ability == "dex" and "restrained" in char.conditions  # Restrained: 민첩 구원에 불리 [SRD Rules Glossary]
        if adv and dis:
            return "normal"
        return "advantage" if adv else "disadvantage" if dis else "normal"

    def roll_save(self, char, ability, dc, advantage=False, avoids=None):
        """구원 굴림 (함정·몬스터 효과·집중 유지). avoids = 이 구원으로 피하려는 상태 이름 (드워프·해독제 이점 판단용)."""
        mod = char.mod(ability) + (char.proficiency_bonus if ability in char.save_profs else 0)
        mode = self.save_mode(char, ability, avoids, advantage)
        d, rolls = self.dice.d20(mode)
        total = d + mod
        ok = total >= dc
        tag = {"advantage": " (이점)", "disadvantage": " (불리)"}.get(mode, "")
        text = (f"{char.name}의 {ABILITY_KO[ability]} 구원 굴림: d20 {rolls if len(rolls) > 1 else d} "
                f"{mod:+d} = {total} vs DC {dc} → {'성공' if ok else '실패'}{tag}")
        return ok, Event("check", text, {"char": char.id, "total": total, "dc": dc, "success": ok, "save": ability})

    # ------------------------------------------------------------------ 선택지 목록
    def _actor_pool(self, ch):
        """이 선택지를 할 수 있는 캐릭터 목록. 비어 있으면 선택지가 화면에 나오지 않는다.
        actor 값: any(기본) / party(누가 하든 상관없는 일행 행동) / cls:직업 / not_cls:직업 / tool:도구."""
        spec = ch.get("actor", "any")
        if spec == "party":
            return []
        pool = [m for m in self.s.party if m.can_act()]
        if spec.startswith("cls:"):
            pool = [m for m in pool if m.cls == spec[4:]]
        elif spec.startswith("not_cls:"):
            pool = [m for m in pool if m.cls != spec[8:]]
        elif spec.startswith("tool:"):
            tool = spec[5:]
            pool = [m for m in pool if tool in m.tools or self.s.has_item(tool)]
        # M3: 같은 캐릭터는 실패한 판정을 다시 할 수 없다 [보완]
        return [m for m in pool if (ch["id"], m.id) not in self.s.failed]

    def available_choices(self):
        """지금 고를 수 있는 선택지(ChoiceView) 목록. 화면·CLI·서버가 매 턴 이 함수 결과를 그대로 보여 준다.

        우선순위 (위에서 걸리면 그 목록만 돌려준다):
          게임 끝 → 없음 / 결정 대기(pending: 영감 재굴림·Hit Point Dice) → 전투 중(combat_choices)
          → 쓰러진 동료 안정시키기 → 평상시 목록
        평상시 목록의 순서와 키:
          방의 선택지(숫자 키, 데이터 순번이라 끝난 것이 빠져도 번호가 당겨지지 않음) → 다시 둘러보기(r)
          → 깨어나기(w) → 소모품(u숫자) → 짧은 휴식(t) → 상점(p숫자) → 이동(a~e)
        """
        if self.s.ended:
            return []
        p = self.s.pending
        if p and p["type"] == "inspire":  # 실패한 판정을 영웅적 영감으로 다시 굴릴지 먼저 정한다
            return [ChoiceView("inspire:yes", "영웅적 영감을 써서 다시 굴린다", [], key="i"),
                    ChoiceView("inspire:no", "결과를 받아들인다", [], key="n")]
        if p and p["type"] == "hit_dice":  # 짧은 휴식: Hit Point Die를 하나씩 굴리고 더 쓸지 정한다 [SRD Short Rest]
            h = self.hero
            out = []
            if h.hd_left > 0 and h.hp < h.max_hp:
                out.append(ChoiceView("hd:spend", f"Hit Point Die 1개를 쓴다 (d{h.hit_die}{h.mod('con'):+d}, 남은 {h.hd_left}개)", [], key="t"))
            out.append(ChoiceView("hd:done", "짧은 휴식을 끝낸다", [], key="n"))
            return out
        if self.s.combat:
            return self.combat_choices()
        dying = [m for m in self.s.party if m.down and m.conditions.get("dying") is not None]
        if dying:  # 동료가 있으면 안정시키기가 먼저 [SRD p.18] (1인에서는 자동 처리되어 여기에 오지 않음)
            return self._stabilize_choices(dying[0])
        out = []
        for idx, ch in enumerate(self.c.choices_in_room(self.s.room), 1):  # 키 = 그 방 선택지 목록에서의 고정 순번
            if ch["id"] in self.s.done or not self.cond(ch.get("requires")):
                continue
            pool = self._actor_pool(ch)
            if ch.get("actor", "any") != "party" and not pool:
                continue
            rec, chk_text = "", ""
            if pool:
                if "check" in ch:
                    best = max(pool, key=lambda m: self.check_mod(m, self.best_option(m, ch["check"]))[0])
                    rec = best.id
                    mod, name = self.check_mod(best, self.best_option(best, ch["check"]))
                    chk_text = f"{name} DC {ch['check']['dc']}, {best.name} {mod:+d}"
                else:
                    rec = pool[0].id
            out.append(ChoiceView(ch["id"], ch["label"], [m.id for m in pool], rec, chk_text, key=str(idx)))
        if self._ready_deferred():
            out.append(ChoiceView(f"revisit:{self.s.room}", "다시 둘러보며 새 소식을 듣는다", [], key="r"))
        for stable in [m for m in self.s.party if m.down and "stable" in m.conditions]:
            out.append(ChoiceView(f"wake:{stable.id}", f"안정된 {stable.name}가 깨어날 때까지 기다린다 (1d4시간)", [], key="w"))
        hero = self.hero
        for i, item_id in enumerate(self.c.use_order, 1):  # 소모품 사용 (W5)
            if hero.can_act() and self.s.has_item(item_id) and self._usable(hero, self.c.items[item_id]["use"]):
                it = self.c.items[item_id]
                out.append(ChoiceView(f"use:{item_id}", f"{it['use']['label']} (남은 {self.s.inventory[item_id]}개)",
                                      [hero.id], hero.id, key=f"u{i}"))
        if self.c.rooms[self.s.room].get("rest") == "short" and self._short_rest_useful():
            out.append(ChoiceView("rest:short", f"짧은 휴식을 한다 (1시간, Hit Point Dice 남은 {hero.hd_left}개)", [], key="t"))
        if self.c.rooms[self.s.room].get("shop") and "scn-m-shop-goods" in self.s.shown:
            for i, item_id in enumerate(self.c.shop_order, 1):
                it = self.c.items[item_id]
                if self.s.stock.get(item_id) == 0 or self.s.gold_cp < it["price_cp"]:
                    continue
                out.append(ChoiceView(f"buy:{item_id}", f"구입: {it['name']} ({fmt_money(it['price_cp'])})", [], key=f"p{i}"))
        gates = self.c.rooms[self.s.room].get("exit_requires", {})  # 조건이 붙은 출구 (W7, 예: 밧줄을 고정해야 내려감)
        for i, ex in enumerate(self.c.exits(self.s.room)):
            if self.cond(gates.get(ex)):
                out.append(ChoiceView(f"move:{ex}", f"이동: {self.c.room_name(ex)}", [], key=MOVE_KEYS[i]))
        return out

    def _stabilize_choices(self, target):
        """쓰러진 동료를 돕는 선택지 (Healer's Kit / 응급 처치). 1인 플레이에서는 도울 동료가 없어 나오지 않는다."""
        out = []
        helpers = [m for m in self.s.party if m.can_act()]
        if self.s.has_item("healers_kit") and self.s.vars.get("healers_kit_uses", 0) > 0 and helpers:
            out.append(ChoiceView(f"kit:{target.id}", f"Healer's Kit로 쓰러진 {target.name}를 안정시킨다",
                                  [h.id for h in helpers], helpers[0].id, key="x1"))
        if helpers:
            best = max(helpers, key=lambda m: self.check_mod(m, {"ability": "wis", "skill": "medicine"})[0])
            out.append(ChoiceView(f"aid:{target.id}", f"쓰러진 {target.name}에게 응급 처치를 한다 (지혜(의학) DC 10)",
                                  [h.id for h in helpers], best.id, "지혜(의학) DC 10", key="x2"))
        return out

    # ------------------------------------------------------------------ 선택 실행
    def choose(self, choice_id, actor_id=None):
        """선택지 하나를 실행하고 그 결과 Event 목록을 돌려준다. 모든 행동의 단일 입구.

        지금 목록에 없는 id는 ValueError (화면이 오래된 버튼을 눌러도 규칙을 건너뛸 수 없게).
        id 접두어 → 처리 함수:
          cmb: 전투(combat.py) · hd: Hit Point Die · move: 이동 · buy: 구입 · revisit: 다시 둘러보기
          kit:/aid: 안정시키기 · wake: 깨어나기 · use: 소모품 · rest: 짧은 휴식 · inspire: 영감 재굴림
          그 밖(접두어 없음) = 장면 JSON의 선택지 → _run_choice
        처리 기록은 s.log에 남는다.
        """
        views = {v.id: v for v in self.available_choices()}
        if choice_id not in views:
            raise ValueError(f"지금 고를 수 없는 선택지: {choice_id}")
        view = views[choice_id]
        actor = None
        if view.actors:
            actor_id = actor_id or view.recommended
            if actor_id not in view.actors:
                raise ValueError(f"{actor_id}는 이 행동을 할 수 없음")
            actor = self.s.member(actor_id)
        kind, _, arg = choice_id.partition(":")
        if kind == "cmb":
            events = self.combat_choose(choice_id)
        elif kind == "hd":
            events = self._spend_hit_die() if arg == "spend" else self._finish_short_rest()
        elif kind == "move":
            events = self._enter(arg)
        elif kind == "buy":
            events = self._buy(arg)
        elif kind == "revisit":
            events = self._revisit()
        elif kind in ("kit", "aid"):
            events = self._stabilize(kind, self.s.member(arg), actor)
        elif kind == "wake":
            events = self._wake(self.s.member(arg))
        elif kind == "use":
            events = self._use_item(arg, actor)
        elif kind == "rest":
            events = self._short_rest()
        elif kind == "inspire":
            events = self._inspire(arg == "yes")
        else:
            events = self._run_choice(self.c.choices[choice_id], actor)
        self.s.log.append({"choice": choice_id, "actor": actor.id if actor else None,
                           "events": [e.text for e in events]})
        return events

    def _run_choice(self, ch, actor):
        """장면 JSON 선택지 실행: before 효과 → (check가 있으면) 판정 → success/failure 효과, (없으면) effects 효과.
        판정에 실패했는데 영웅적 영감이 있으면 결과를 적용하지 않고 pending으로 멈춰 플레이어에게 묻는다."""
        events = self._apply(ch.get("before", []), actor)  # 판정 전에 항상 일어나는 일 (NPC의 대답 등)
        if "check" in ch:
            ok, ev = self.roll_check(actor, ch["check"])
            events.append(ev)
            if not ok and actor.heroic_inspiration:  # 결과를 적용하기 전에 다시 굴릴지 묻는다 [SRD Rules Glossary]
                self.s.pending = {"type": "inspire", "choice": ch["id"], "actor": actor.id}
                events.append(Event("say", f"{actor.name}에게 영웅적 영감이 있다. 이 판정을 다시 굴릴 수 있다."))
                return events
            events += self._resolve_check(ch, actor, ok)
        else:
            events += self._apply(ch.get("effects", []), actor)
            if ch.get("once"):
                self.s.done.add(ch["id"])
        return events

    def _resolve_check(self, ch, actor, ok):
        """판정 결과를 적용한다. 성공+once → 다시 안 나옴(done), 실패 → 같은 캐릭터 재시도 금지(failed, path 판정은 예외)."""
        events = self._apply(ch["success"] if ok else ch["failure"], actor)
        if ok and ch.get("once"):
            self.s.done.add(ch["id"])
        elif not ok and not ch.get("path"):  # 길을 가는 판정(path)은 다시 시도할 수 있다 [보완 M3]
            self.s.failed.add((ch["id"], actor.id))
        return events

    def _inspire(self, use):
        """Heroic Inspiration: 주사위를 굴린 직후 다시 굴리고, 새 결과를 써야 한다 [SRD Rules Glossary].
        W5에서는 선택지의 능력 판정에만 적용 (구원·공격 굴림은 전투 구현 때)."""
        p, self.s.pending = self.s.pending, None
        ch, actor = self.c.choices[p["choice"]], self.s.member(p["actor"])
        if not use:
            return self._resolve_check(ch, actor, False)
        actor.heroic_inspiration = False
        ok, ev = self.roll_check(actor, ch["check"])
        return [Event("say", f"{actor.name}가 영웅적 영감을 써서 다시 굴린다."), ev] + self._resolve_check(ch, actor, ok)

    # ------------------------------------------------------------------ 효과 (05-2 5장)
    def _apply(self, effects, actor):
        """효과 목록을 차례로 실행해 상태를 바꾸고 Event를 모은다. 장면 JSON의 effects/success/failure 등이 여기로 온다.

        효과는 키 하나로 종류가 정해지는 dict이다. 예: {"flag": "tracked_rats"}, {"give": "rope"},
        {"show": 카드 id}, {"combat": 조우 id, "on_victory": [...]}, {"if": 조건, "then": [...], "else": [...]}.
        새 효과 종류는 아래 elif에 추가한다 (모르는 키는 ValueError). 게임 오버가 되면 남은 효과는 실행하지 않는다.
        """
        events = []
        for e in effects:
            if self.s.ended == "game_over":
                break
            if "use" in e:
                events += self._apply(self.c.effect_sets[e["use"]], actor)
            elif "enter" in e:  # 다른 방(다른 장면 포함)으로 이동 [W3]
                events += self._enter(e["enter"])
            elif "if" in e:
                events += self._apply(e["then"] if self.cond(e["if"], actor) else e.get("else", []), actor)
            elif "show" in e:
                events += self._show(e["show"], actor)
            elif "flag" in e:
                if e["flag"] not in self.s.flags:
                    self.s.flags.add(e["flag"])
                    events.append(Event("flag", f"(상태: {e['flag']})", {"flag": e["flag"]}))
            elif "var" in e:
                self.s.vars[e["var"]] = e["value"]
            elif "give" in e:
                self.s.inventory[e["give"]] = self.s.inventory.get(e["give"], 0) + e.get("n", 1)
                events.append(Event("item", f"획득: {self.c.items[e['give']]['name']}", {"item": e["give"]}))
            elif "gold_cp" in e:
                self.s.gold_cp += e["gold_cp"]
                events.append(Event("gold", f"돈 {fmt_money(e['gold_cp'])} 획득 (소지금 {fmt_money(self.s.gold_cp)})"))
            elif "say" in e:
                events.append(Event("say", e["say"]))
            elif "trap" in e:
                events += self._trap(e["trap"], actor)
            elif "disarm_trap" in e:
                self.s.traps.setdefault(e["disarm_trap"], {"count": 0, "disarmed": False})["disarmed"] = True
            elif "add_condition" in e:
                (actor or self.hero).conditions[e["add_condition"]] = e.get("minutes")
            elif "remove_condition" in e:
                (actor or self.hero).conditions.pop(e["remove_condition"], None)
            elif "inspiration" in e:
                events += self._gain_inspiration(actor or self.hero)
            elif "combat" in e:  # 전투 시작 (W6): {"combat": 조우 id, "on_victory": [...], "on_retreat": [...]}
                on_end = {k[3:]: v for k, v in e.items() if k.startswith("on_")}
                events += self.start_combat(e["combat"], on_end)
            elif "level_up" in e:
                events += self._level_up(actor or self.hero, e["level_up"])
            elif "long_rest" in e:
                events += self._long_rest(e["long_rest"]["cost_cp_per_person"])
            elif "fall" in e:  # 낙하 (W7): d6 × 10ft, 이 구간의 높이까지만 [보완], 10ft당 1d6 [SRD Falling]
                events += self._fall(actor or self.hero, e["fall"])
            elif "end_scene" in e:
                self.s.ended = f"scene{self.s.scene}_complete"
                events.append(Event("end", e["end_scene"]))
            else:
                raise ValueError(f"알 수 없는 효과: {e}")
        return events

    # ------------------------------------------------------------------ 함정 (W5: traps.json 데이터로 처리)
    def trap_active(self, trap_id):
        """해제되지 않았고, 발동 횟수가 한도 미만이면 작동한다.
        한도: 지속 시간에 재장전이 없으면 1회 [SRD p.199], 독침 다트는 3회 [SRD p.200]."""
        st = self.s.traps.get(trap_id, {"count": 0, "disarmed": False})
        limit = self.c.traps[trap_id].get("max_triggers")
        return not st["disarmed"] and (limit is None or st["count"] < limit)

    def resists(self, char, damage_type):
        """종족 특성으로 이 피해 종류에 저항하는지 (드워프 → 독). 함정·몬스터 공격 피해에서 절반 처리에 쓴다."""
        return any(damage_type in TRAIT_RESIST.get(t, ()) for t in char.traits)

    def _trap(self, trap_id, actor):
        """함정 발동 ({"trap": id} 효과). 수치는 모두 traps.json에서 읽는다.
        순서: 발동 횟수 +1 → 발동 문장 → 구원 굴림 → 실패 시 상태 → 피해(성공 시 절반·저항 절반) → 뒤 문장·소음 기록."""
        t = self.c.traps[trap_id]
        victim = actor or self.hero
        if not self.trap_active(trap_id):
            return []
        self.s.traps.setdefault(trap_id, {"count": 0, "disarmed": False})["count"] += 1
        events = [Event("say", t["trigger_text"], {"trap": trap_id})]
        save = t.get("save")
        ok = False
        if save:
            ok, ev = self.roll_save(victim, save["ability"], save["dc"], avoids=save.get("avoids"))
            events.append(ev)
        fc = t.get("fail_condition")
        if fc and not ok:  # 상태는 맞는 순간 걸린다 (쓰러져 시간이 흐르면 그만큼 줄어듦)
            victim.conditions[fc["name"]] = fc["minutes"]
        if "damage" in t and not (ok and t.get("on_save_success") == "none"):
            n, why = 1, ""
            if "damage_count" in t:  # 예: 다트 1d3개, 개당 1d6 [원본 scn-g-darts-trap]
                n, _ = self.dice.roll(t["damage_count"])
                why = f"{n}개 × "
            rolls = []
            for _ in range(n):
                rolls += self.dice.roll(t["damage"])[1]
            dmg = sum(rolls)
            dtype = t.get("damage_type", "")
            why += f"{DAMAGE_KO.get(dtype, dtype)} 피해 {t['damage']}({'+'.join(map(str, rolls))})"
            if ok:  # 성공 시 절반, 내림 [SRD p.17]
                dmg //= 2
                why += " 절반"
            if self.resists(victim, dtype):
                dmg //= 2
                why += " · 저항"
            events += self._damage(victim, dmg, why)
        if fc and not ok and not victim.down and t.get("fail_condition_text") and fc["name"] in victim.conditions:
            events.append(Event("status", t["fail_condition_text"].format(name=victim.name)))
        if t.get("after_text") and self.s.ended != "game_over":
            events.append(Event("say", t["after_text"]))
        if t.get("noise"):
            self.s.vars[f"noise:{trap_id}"] = True  # 소음 → 몬스터 반응은 해당 장면에서 처리 [원본]
        return events

    def _fall(self, char, spec):
        """[원본 scn-s2-check-descend] 실패하면 d6 × 10피트 낙하, 10피트당 1d6 (최대 20d6) [SRD Falling].
        [보완 W7] 떨어지는 거리는 그 구간의 높이(max_ft)를 넘지 않는다."""
        n, _ = self.dice.roll("1d6")
        feet = min(n * 10, spec.get("max_ft", 200))
        dice = min(20, feet // 10)
        dmg, rolls = self.dice.roll(f"{dice}d6")
        events = [Event("say", f"손이 미끄러진다! {feet}피트 아래로 떨어졌다.", {"fall_ft": feet})]
        return events + self._damage(char, dmg, f"낙하 {feet}피트 {dice}d6({'+'.join(map(str, rolls))})")

    # ------------------------------------------------------------------ 소모품 (W5)
    def _usable(self, char, use):
        """소모품을 지금 쓰는 의미가 있는지 (HP가 가득이면 물약 선택지를 숨긴다 등). use = items.json의 "use" 값."""
        if "heal" in use:
            return char.hp < char.max_hp
        if "condition" in use:
            return use["condition"] not in char.conditions
        return True

    def _use_item(self, item_id, user):
        """전투 밖에서는 시간 비용 없이 바로 사용 (보너스 행동은 전투 구현 W6에서 행동 경제에 연결)."""
        it, use = self.c.items[item_id], self.c.items[item_id]["use"]
        self.s.inventory[item_id] -= 1
        events = [Event("item", f"{user.name}가 {it['name']}을(를) 사용했다. (남은 {self.s.inventory[item_id]}개)", {"item": item_id})]
        if "heal" in use:
            amount, rolls = self.dice.roll(use["heal"])
            before = user.hp
            user.hp = min(user.max_hp, user.hp + amount)
            events.append(Event("status", f"{use['heal']}({'+'.join(map(str, rolls))}) = {amount} 회복. "
                                          f"HP {before} → {user.hp}/{user.max_hp}", {"heal": user.hp - before}))
        if "condition" in use:
            user.conditions[use["condition"]] = use["minutes"]
            events.append(Event("status", f"{user.name}는 {use['minutes'] // 60}시간 동안 중독을 피하거나 끝내는 구원 굴림에 이점을 얻는다."))
        return events

    # ------------------------------------------------------------------ 짧은 휴식·Hit Point Dice (W5)
    def _hd_useful(self, m):
        """Hit Point Die를 쓸 수 있고 쓸 이유가 있는지 (의식이 있고, 다쳤고, 주사위가 남음)."""
        return m.hp >= 1 and m.hp < m.max_hp and m.hd_left > 0

    def _recharge_pending(self, m):
        """짧은 휴식으로 회복할 직업 기능이 남아 있는지 (Second Wind 1회, Action Surge, Channel Divinity 1회, Arcane Recovery)."""
        full = self.c.rules["class_levels"].get(m.cls, {}).get(str(m.level), {})
        for k in ("second_wind", "action_surge", "channel_divinity"):
            if k in full and m.resources.get(k, 0) < full[k]:
                return True
        slots = m.resources.get("slots")
        return bool(m.resources.get("arcane_recovery") and slots and slots != full.get("slots"))

    def _short_rest_useful(self):
        """짧은 휴식 선택지를 보여 줄지. 회복할 것이 없으면 숨긴다 (휴식 가능한 방 = 방 데이터의 "rest": "short")."""
        return any(m.hp >= 1 and (self._hd_useful(m) or self._recharge_pending(m)) for m in self.s.party)

    def _short_rest(self):
        """[SRD Rules Glossary: Short Rest] 1시간, HP 1 이상이어야 시작. 끝날 때 Hit Point Dice를 원하는 만큼 쓴다:
        주사위 하나마다 굴린 값 + 건강 수정치(최소 1), 한 개씩 굴리고 더 쓸지 정한다 (W6: 플레이어가 고름)."""
        events = self._pass_time(SHORT_REST_MIN)
        events.append(Event("rest", "잠시 숨을 돌리며 상처를 살핀다. (짧은 휴식 1시간)"))
        if self._hd_useful(self.hero):
            self.s.pending = {"type": "hit_dice"}
            events.append(Event("status", f"Hit Point Dice를 몇 개 쓸지 고른다. (남은 {self.hero.hd_left}개, d{self.hero.hit_die})"))
            return events
        return events + self._finish_short_rest()

    def _spend_hit_die(self):
        """'hd:spend' — Hit Point Die 1개를 굴려 회복. 더 쓸 수 없게 되면 휴식을 자동으로 끝낸다."""
        m = self.hero
        r, _ = self.dice.roll(f"1d{m.hit_die}")
        gain = max(1, r + m.mod("con"))
        twice = ""
        if self.s.has_item(PERIAPT_WOUND):  # [원본 scn-h-fighter-periapt] Hit Point Die 회복량 두 배
            gain, twice = gain * 2, " ×2 (Periapt)"
        m.hd_left -= 1
        before, m.hp = m.hp, min(m.max_hp, m.hp + gain)
        events = [Event("status", f"{m.name}가 Hit Point Die를 썼다: d{m.hit_die}({r}) {m.mod('con'):+d}{twice} = {gain} 회복. "
                                  f"HP {before} → {m.hp}/{m.max_hp} (남은 주사위 {m.hd_left}/{m.hd_max})")]
        if not self._hd_useful(m):  # 더 쓸 주사위가 없거나 HP가 가득 차면 휴식을 마친다
            events += self._finish_short_rest()
        return events

    def _finish_short_rest(self):
        """짧은 휴식이 끝날 때 회복되는 기능 [SRD p.37·47·78]."""
        self.s.pending = None
        events = [Event("rest", "짧은 휴식을 마쳤다.")]
        for m in self.s.party:
            full = self.c.rules["class_levels"].get(m.cls, {}).get(str(m.level), {})
            r = m.resources
            if "second_wind" in full and r.get("second_wind", 0) < full["second_wind"]:
                r["second_wind"] += 1                 # 1회 회복 [SRD p.47]
                events.append(Event("status", f"Second Wind 1회 회복 (남은 {r['second_wind']}회)"))
            if "action_surge" in full and r.get("action_surge", 0) < full["action_surge"]:
                r["action_surge"] = full["action_surge"]   # [SRD p.47]
                events.append(Event("status", "Action Surge 회복"))
            if "channel_divinity" in full and r.get("channel_divinity", 0) < full["channel_divinity"]:
                r["channel_divinity"] += 1            # 1회 회복 [SRD p.37]
                events.append(Event("status", f"Channel Divinity 1회 회복 (남은 {r['channel_divinity']}회)"))
            if r.get("arcane_recovery") and r.get("slots") and r["slots"] != full.get("slots"):
                # Arcane Recovery: 합계 레벨 ≤ 마법사 레벨의 절반(올림), 긴 휴식마다 1회 [SRD p.78]
                # 쓴 슬롯이 있으면 자동 사용, 높은 슬롯부터 [보완]
                budget = math.ceil(m.level / 2)
                got = []
                for lv in range(len(r["slots"]), 0, -1):
                    while budget >= lv and r["slots"][lv - 1] < full["slots"][lv - 1]:
                        r["slots"][lv - 1] += 1
                        budget -= lv
                        got.append(lv)
                if got:
                    r["arcane_recovery"] = 0
                    events.append(Event("status", f"Arcane Recovery: 주문 슬롯 회복 {got} (남은 슬롯 {r['slots']})"))
        return events

    # ------------------------------------------------------------------ 영웅적 영감 (W5)
    def _gain_inspiration(self, char):
        """이미 있으면 새로 얻은 것은 다른 PC에게 줄 수 있다. 1인이라 줄 사람이 없으므로 사라진다 [SRD Rules Glossary]."""
        if char.heroic_inspiration:
            return [Event("status", f"{char.name}는 이미 영웅적 영감을 가지고 있다. (새로 얻은 영감은 사라짐)")]
        char.heroic_inspiration = True
        return [Event("status", f"{char.name}가 영웅적 영감을 얻었다. 실패한 판정을 한 번 다시 굴릴 수 있다.")]

    # ------------------------------------------------------------------ 피해·0 HP [SRD p.17-18] + [보완 05-6 ②]
    def _damage(self, char, amount, why, critical=False):
        """캐릭터가 피해를 받는 유일한 통로 (함정·낙하·몬스터 공격 모두 여기로 온다).

        HP가 0이 되면: 남은 피해 ≥ 최대 HP → 즉사, 아니면 dying(빈사) 상태가 되고 _after_dropping으로 넘어간다.
        이미 0 HP일 때 맞으면 죽음 구원 실패가 1회(치명타 2회) 늘어난다.
        """
        events = []
        if "dead" in char.conditions:
            return events
        if char.hp == 0:  # 이미 0 HP에서 피해 [SRD p.18]
            events.append(Event("damage", f"{char.name}가 쓰러진 채로 {amount} 피해를 입었다 ({why})."))
            if amount >= char.max_hp:
                return events + self._die(char, "쓰러진 상태에서 최대 HP 이상의 피해")
            char.conditions.pop("stable", None)
            st = char.conditions.setdefault("dying", {"success": 0, "fail": 0})
            st["fail"] += 2 if critical else 1
            if st["fail"] >= 3:
                return events + self._die(char, "죽음 구원 실패 3회")
            return events + self._after_dropping(char)
        remainder = amount - char.hp
        char.hp = max(0, char.hp - amount)
        events.append(Event("damage", f"{char.name}가 {amount} 피해를 입었다 ({why}). HP {char.hp}/{char.max_hp}",
                            {"char": char.id, "amount": amount}))
        events += self._concentration_check(char, amount)  # 집중 유지 [SRD 용어집 Concentration]
        if char.hp > 0:
            return events
        if remainder >= char.max_hp:  # 즉사: 남은 피해 ≥ 최대 HP [SRD p.17]
            return events + self._die(char, f"초과 피해 {remainder} ≥ 최대 HP {char.max_hp}")
        char.conditions["dying"] = {"success": 0, "fail": 0}
        events.append(Event("status", f"{char.name}가 쓰러져 의식을 잃었다!"))
        return events + self._after_dropping(char)

    def _after_dropping(self, char):
        """도와줄 동료가 없으면 죽음 구원을 결과가 날 때까지 자동으로 굴린다 [보완 05-6 D1].
        (전투 밖이므로 '턴'이 없어 1회 = 1턴으로 본다. 전투 중에는 자기 턴마다 1회 — combat.py D2·D4)"""
        if self.s.combat:
            return []
        helpers = [m for m in self.s.party if m is not char and m.can_act()]
        if helpers:
            return []
        events = []
        while "dying" in char.conditions:
            events += self._death_save(char)
        if "stable" in char.conditions:
            events += self._wake(char)
        return events

    def _death_save(self, char):
        """죽음 구원 1회 [SRD p.17-18]: 10 이상 성공, 1은 실패 2회, 20은 1 HP 회복."""
        st = char.conditions["dying"]
        d, _ = self.dice.d20()
        if d == 20:
            char.conditions.pop("dying")
            char.hp = 1
            return [Event("check", f"{char.name}의 죽음 구원 굴림: d20 20 → 기적처럼 의식을 되찾았다! HP 1/{char.max_hp}")]
        note = ""
        if d <= 9 and self.s.has_item(PERIAPT_WOUND):
            # [원본 scn-h-fighter-periapt] 착용하면 죽음 구원 굴림에서 9 이하가 나와도 10으로 바꿀 수 있다
            note, d = f" (Periapt of Wound Closure: {d} → 10)", 10
        if d == 1:
            st["fail"] += 2
            result = "실패 2회 (1)"
        elif d >= 10:
            st["success"] += 1
            result = "성공"
        else:
            st["fail"] += 1
            result = "실패"
        events = [Event("check", f"{char.name}의 죽음 구원 굴림: d20 {d}{note} → {result} (성공 {st['success']} / 실패 {st['fail']})")]
        if st["fail"] >= 3:
            events += self._die(char, "죽음 구원 실패 3회")
        elif st["success"] >= 3:
            events += self._make_stable(char)
        return events

    def _die(self, char, why):
        """사망 처리. 파티 전원이 죽으면 s.ended = "game_over" (화면은 이 값을 보고 게임 오버 창을 띄운다)."""
        char.conditions.pop("dying", None)
        char.conditions.pop("stable", None)
        char.conditions["dead"] = None
        events = [Event("status", f"{char.name}가 숨을 거두었다. ({why})")]
        if all("dead" in m.conditions for m in self.s.party):
            self.s.ended = "game_over"
            events.append(Event("end", f"게임 오버. 체크포인트({self.checkpoint_label})에서 다시 시작할 수 있다."))
        return events

    def _make_stable(self, char):
        char.conditions.pop("dying", None)
        char.conditions["stable"] = None
        return [Event("status", f"{char.name}가 안정되었다. 의식은 없지만 더는 위험하지 않다.")]

    def _stabilize(self, kind, target, helper):
        """동료가 쓰러진 캐릭터를 돕는다 (kit: Healer's Kit / aid: 지혜(의학) DC 10). 먼저 그 턴의 죽음 구원을 굴린다."""
        events = self._death_save(target)
        if "dying" not in target.conditions:
            return events
        if kind == "kit":
            # [원본 scn-g-chest-loot] Healer's Kit: 판정 없이 안정
            self.s.vars["healers_kit_uses"] -= 1
            events.append(Event("say", f"{helper.name}가 Healer's Kit로 상처를 묶는다. (남은 사용 {self.s.vars['healers_kit_uses']}회)"))
            return events + self._make_stable(target)
        ok, ev = self.roll_check(helper, {"ability": "wis", "skill": "medicine", "dc": 10})
        events.append(ev)
        if ok:
            events += self._make_stable(target)
        return events

    def _wake(self, char):
        hours, _ = self.dice.roll("1d4")  # 치료받지 않은 안정 상태는 1d4시간 뒤 1 HP [SRD p.18]
        events = self._pass_time(hours * 60)
        char.conditions.pop("stable", None)
        char.hp = 1
        events.append(Event("status", f"{hours}시간이 지나 {char.name}가 깨어났다. HP 1/{char.max_hp}"))
        return events

    # ------------------------------------------------------------------ 체크포인트 [보완 05-6 D5]
    def _save_checkpoint(self, label):
        """장면 시작 또는 긴 휴식 직후. 둘 중 더 최근 것이 남는다 (덮어쓰기)."""
        self.checkpoint = copy.deepcopy(self.s)
        self.checkpoint_label = label

    def restart_from_checkpoint(self):
        """게임 오버 뒤 '다시 시작'. 저장해 둔 GameState 복사본으로 통째로 되돌린다 (주사위 상태는 되돌리지 않음)."""
        if self.checkpoint is None:
            raise RuntimeError("체크포인트가 없음")
        self.s = copy.deepcopy(self.checkpoint)
        return [Event("move", f"== 체크포인트({self.checkpoint_label})에서 다시 시작: {self.c.room_name(self.s.room)} ==",
                      {"room": self.s.room})]

    # ------------------------------------------------------------------ 시간·휴식·이동·상점
    def _pass_time(self, minutes):
        """게임 시간을 흘려보낸다. conditions 값이 정수(남은 분)인 상태만 줄어들고, 0 이하가 되면 풀린다.
        값이 None이나 dict인 상태(dying, stable, grappled 등)는 시간으로 끝나지 않는다."""
        self.s.minutes += minutes
        events = []
        for m in self.s.party:  # 시간이 정해진 상태·효과(분 단위 정수)만 줄어든다
            for name, left in list(m.conditions.items()):
                if type(left) is not int:
                    continue
                m.conditions[name] = left - minutes
                if m.conditions[name] <= 0:
                    del m.conditions[name]
                    if m.concentration == name:  # 지속 시간이 끝난 집중 주문
                        m.concentration = ""
                    events.append(Event("status", EXPIRE_TEXT.get(name, "{name}의 " + name + " 상태가 끝났다.").format(name=m.name)))
        return events

    def _long_rest(self, cost_per_person):
        """여관 긴 휴식 ({"long_rest": …} 효과, 장면 1). 숙박비 지불 → 8시간 → HP·Hit Point Dice·직업 자원 전부 회복
        → 체크포인트 저장. 16시간 안에 다시 하거나 돈이 모자라면 문장만 돌려주고 아무것도 바꾸지 않는다."""
        s = self.s
        last = s.vars.get("last_long_rest_end")
        if last is not None and s.minutes - last < LONG_REST_GAP_MIN:
            return [Event("say", "긴 휴식을 마친 지 16시간이 지나지 않아 다시 긴 휴식을 할 수 없다.")]
        cost = cost_per_person * len([m for m in s.party if "dead" not in m.conditions])
        if s.gold_cp < cost:
            return [Event("say", f"숙박비({fmt_money(cost)})가 부족하다.")]
        s.gold_cp -= cost
        events = [Event("gold", f"숙박비 {fmt_money(cost)} 지불 (소지금 {fmt_money(s.gold_cp)})")]
        events += self._pass_time(LONG_REST_MIN)
        for m in s.party:
            if m.hp >= 1:  # 긴 휴식은 HP 1 이상이어야 시작할 수 있다 [SRD Rules Glossary]
                m.ability_loss = {}   # 줄어든 능력치 복구 [SRD Rules Glossary: Long Rest]
                m.hp_max_loss = 0     # Specter 생명 흡수: 회복 시점은 SRD 블록에 없음 → 긴 휴식에서 복구 [보완]
                m.hp = m.hp_max
                m.hd_left = m.hd_max  # 쓴 Hit Point Dice를 모두 회복 [SRD Rules Glossary: Long Rest]
                self.init_resources(m)  # 주문 슬롯·Second Wind 등 모두 회복 [SRD 직업 기능]
        s.vars["last_long_rest_end"] = s.minutes
        events.append(Event("rest", "여관에서 하룻밤을 보냈다. (긴 휴식: HP와 Hit Point Dice를 모두 회복) 종소리는 밤새 이어졌다."))
        self._save_checkpoint("긴 휴식 직후")
        events.append(Event("status", "(체크포인트 저장: 긴 휴식 직후)"))
        return events

    def _enter(self, room_id):
        """방에 들어간다 (move: 선택지, {"enter": 방} 효과, 게임 시작). 다른 장면의 방이면 장면 전환도 여기서 일어난다.

        순서: 위치·방문 기록·장면 번호 갱신 → (새 장면이면) 장면 전환 Event → 방 이름 Event
        → 그 방의 장면 묘사(desc) 카드 표시 → 방 진입 반응(on_enter, 한 번만) → (새 장면이면) 체크포인트 저장.
        """
        s = self.s
        if room_id not in self.c.rooms:
            raise ValueError(f"불러온 장면에 없는 방: {room_id}")
        prev_scene = s.scene
        new_scene = self.c.rooms[room_id]["scene"] > s.scene
        s.room = room_id
        s.visited.add(room_id)
        s.scene = max(s.scene, self.c.rooms[room_id]["scene"])  # 장면 번호는 올라가기만 함 [원본 p.3]
        events = []
        if new_scene:
            events.append(Event("scene", f"=== 장면 {prev_scene} 완료 → 장면 {s.scene}: {self.c.scene_titles[s.scene]} ===",
                                {"from": prev_scene, "to": s.scene}))
        events.append(Event("move", f"== {self.c.room_name(room_id)} ==", {"room": room_id}))
        for cid, card in self.c.cards.items():  # 방에 들어가면 장면 묘사 카드 표시
            if card["room"] == room_id and card["kind"] == "desc":
                events += self._show(cid)
        for i, hook in enumerate(self.c.rooms[room_id].get("on_enter", [])):  # 방 진입 반응 [보완]
            key = f"enter:{room_id}:{i}"
            if key not in s.once_done and self.cond(hook.get("requires")):
                s.once_done.add(key)
                events += self._apply(hook["effects"], None)
        if new_scene:
            if self.c.scene_status.get(s.scene) == "skeleton":
                events.append(Event("say", f"(장면 {s.scene}의 콘텐츠는 아직 구현되지 않았습니다. 방 이동만 가능합니다.)"))
            self._save_checkpoint(f"장면 {s.scene} 시작")
            events.append(Event("status", f"(체크포인트 저장: 장면 {s.scene} 시작)"))
        return events

    def _buy(self, item_id):
        """상점 구입 (buy: 선택지). 살 수 있는지(돈·재고)는 available_choices에서 이미 걸렀으므로 여기서는 바로 처리한다."""
        it = self.c.items[item_id]
        self.s.gold_cp -= it["price_cp"]
        if self.s.stock.get(item_id) is not None:
            self.s.stock[item_id] -= 1
        self.s.inventory[item_id] = self.s.inventory.get(item_id, 0) + 1
        if item_id == "healers_kit":
            self.s.vars["healers_kit_uses"] = self.s.vars.get("healers_kit_uses", 0) + it["uses"]
        events = [Event("item", f"{it['name']}을(를) 샀다. (소지금 {fmt_money(self.s.gold_cp)})", {"item": item_id})]
        reaction = self.c.buy_reactions.get(item_id)  # 구매 반응 [보완]
        if reaction and f"buy:{reaction['id']}" not in self.s.once_done and self.cond(reaction.get("requires")):
            self.s.once_done.add(f"buy:{reaction['id']}")
            events += self._apply(reaction["effects"], None)
        return events


def fmt_money(cp):
    """cp 단위 정수를 화면용 글자로. 예: 1250 → '12 gp 5 sp'. 돈은 항상 cp 정수로 저장한다."""
    gp, rest = divmod(cp, 100)
    sp, cp_ = divmod(rest, 10)
    parts = [f"{gp} gp" if gp else "", f"{sp} sp" if sp else "", f"{cp_} cp" if cp_ else ""]
    return " ".join(p for p in parts if p) or "0 gp"
