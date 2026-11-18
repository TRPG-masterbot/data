"""전투 (W6). 위치 없는 1인 전투 [05-1a 보완 C]. Engine이 이 클래스를 상속한다.

주사위·명중·피해·HP·사망은 모두 이 코드가 정한다. AI 서술기는 결과 문장만 받는다.

규칙 근거 (SRD 5.2, 실제로 읽은 쪽):
  전투 순서·우선권·행동 p.13-14, 공격 굴림·치명타·피해 p.6-7·15-16, 저항·취약·면역 p.17,
  0 HP·죽음 구원 p.17-18, 상태 용어집(Grappled·Restrained·Frightened·Incapacitated·Invisible·Hide),
  집중 용어집, 몬스터 블록 p.256·283·290·319-320·324·340-341·347·349·359, 무기 p.91, 방어구 p.92,
  직업 p.36-38·47-48·61-62·77-78, 주문 p.121·132·138·139·146·157·159, 레벨업 고정 HP p.23.
위치 기반 규칙(엄폐, 사거리, 근접 중 원거리 불리, 기회 공격, 이동)은 쓰지 않는다 [05-7 4-2].

전투 진행 구조:
  start_combat(조우 id)  우선권을 굴려 순서를 정하고 s.combat에 전투 상태를 만든다.
  _run_until_hero()      몬스터 차례를 자동으로 진행하다가 주인공 차례가 되면 멈춘다.
  combat_choices()       주인공 차례의 선택지 (id는 모두 "cmb:"로 시작) → Engine.available_choices가 돌려준다.
  combat_choose(id)      행동 1개 처리. 행동·보너스 행동을 다 쓰면 자동으로 차례를 넘긴다.
  _end_combat(결과)       s.combat = None, 결과를 s.vars["combat:조우 id"]에 남기고 조우의 on_victory 등 효과를 실행.
데이터: 몬스터 수치 monsters.json · 조우 encounters.json · 무기/방어구/주문/직업 표 rules.json.
self.c / self.s / self.dice / _damage / _death_save 등은 Engine(engine.py)에 있는 것을 쓴다.
"""
import math
import re

from .dice import DICE_EXPR
from .events import ChoiceView, Event

OUTCOME_KO = {"victory": "승리", "retreat": "물러남", "avoid": "전투 회피", "peace": "적대 해제", "defeat": "패배"}
DTYPE_KO = {"slashing": "베기", "piercing": "관통", "bludgeoning": "타격", "fire": "화염", "cold": "냉기",
            "radiant": "광휘", "necrotic": "사령", "acid": "산성", "force": "역장", "poison": "독", "psychic": "정신"}
TURN_UNDEAD_ROUNDS = 10        # 1분 [SRD p.37 Turn Undead]
FIEND_UNDEAD = ("fiend", "undead")   # Holy Water [SRD p.97]


def _mode(adv, dis):
    """이점과 불리가 함께 있으면 상쇄 [SRD p.7]."""
    if adv and dis:
        return "normal"
    return "advantage" if adv else "disadvantage" if dis else "normal"


class CombatMixin:
    """전투 기능 묶음. 혼자서는 쓰지 않고 Engine이 상속해서 쓴다 (engine.py와 파일만 나눈 것)."""

    # ================================================================== 캐릭터 전투 수치
    def class_value(self, char, key, default=0):
        """rules.json class_levels에서 이 캐릭터의 직업·레벨 값을 읽는다 (예: 도적 sneak_dice)."""
        return self.c.rules["class_levels"].get(char.cls, {}).get(str(char.level), {}).get(key, default)

    def init_resources(self, char):
        """레벨 표의 사용 횟수로 채운다 (긴 휴식·레벨업)."""
        table = self.c.rules["class_levels"].get(char.cls, {}).get(str(char.level), {})
        char.resources = {k: (list(v) if isinstance(v, list) else v) for k, v in table.items() if k != "sneak_dice"}

    def armor_class(self, char):
        """[SRD p.92] 방어구 기본값 + 민첩(상한) + 방패 2, Defense +1 [SRD p.88], 점액 부식 −n [SRD p.290]."""
        arm = self.c.rules["armor"][char.armor]
        dex = char.mod("dex") if arm["dex_max"] is None else min(char.mod("dex"), arm["dex_max"])
        ac = arm["base"] + dex - (char.armor_penalty if char.armor != "none" else 0)
        if char.shield:
            ac += self.c.rules["shield_bonus"]
        if char.fighting_style == "defense" and char.armor != "none":
            ac += 1
        return ac

    def weapon_proficient(self, char, wid):
        """무기 숙련 여부. martial_finesse_light = 도적처럼 군용 무기 중 Finesse·Light 무기만 숙련."""
        w = self.c.rules["weapons"][wid]
        if w["category"] == "simple":
            return "simple" in char.weapon_profs
        if "martial" in char.weapon_profs:
            return True
        return "martial_finesse_light" in char.weapon_profs and ("finesse" in w["props"] or "light" in w["props"])

    def weapon_attack(self, char, wid):
        """(능력치, 명중 보너스). 근접=근력, 원거리=민첩, Finesse=둘 중 높은 것 [SRD p.15]."""
        w = self.c.rules["weapons"][wid]
        if "finesse" in w["props"]:
            ab = max(("str", "dex"), key=char.mod)
        elif w["kind"] == "ranged":
            ab = "dex"
        else:
            ab = "str"
        bonus = char.mod(ab) + (char.proficiency_bonus if self.weapon_proficient(char, wid) else 0)
        return ab, bonus - char.weapon_penalty.get(wid, 0)

    def spell_attack_bonus(self, char):
        return char.mod(char.spellcasting) + char.proficiency_bonus      # [SRD p.106]

    def spell_dc(self, char):
        return 8 + char.mod(char.spellcasting) + char.proficiency_bonus  # [SRD p.106]

    def _slot_level(self, char, level):
        """쓸 수 있는 가장 낮은 슬롯 레벨 (없으면 None). 상위 슬롯 사용은 자동 [보완]."""
        slots = char.resources.get("slots", [])
        for lv in range(level, len(slots) + 1):
            if slots[lv - 1] > 0:
                return lv
        return None

    # ================================================================== 레벨업 [SRD p.23]
    def _level_up(self, char, to_level):
        """to_level까지 한 단계씩 올린다 ({"level_up": n} 효과, 전투 시험장의 레벨 맞추기)."""
        events = []
        fixed = self.c.rules["fixed_hp_per_level"][char.cls]
        while char.level < to_level:
            char.level += 1
            gain = fixed + char.mod("con")   # 고정 HP 방식 [SRD p.23 Fixed Hit Points by Class]
            char.hp_max += gain
            char.hp += gain
            char.hd_left += 1
            self.init_resources(char)        # 새 레벨의 사용 횟수로 채움 [보완]
            events.append(Event("status", f"{char.name}가 {char.level}레벨이 되었다. 최대 HP +{gain} → {char.max_hp}"))
        return events

    # ================================================================== 전투 시작
    def start_combat(self, enc_id, on_end=None):
        """전투 시작. on_end = {"victory": [효과…], "retreat": [...]} — 장면 JSON의 on_victory 등을 Engine._apply가 넘겨준다.

        s.combat 구조: enc(조우 id), round, order(행동 순서 id 목록, 주인공은 "hero"), idx(지금 차례 위치),
        enemies(몬스터 id → hp·conditions·out 등), turn(주인공 남은 행동 수), xp, guided, dodging, on_end.
        """
        enc = self.c.encounters[enc_id]
        enemies, order = {}, []
        events = [Event("combat", f"⚔ 전투 시작: {enc['name']}", {"encounter": enc_id})]
        hero = self.hero
        surprised = bool(enc.get("surprise_unless_flag")) and enc["surprise_unless_flag"] not in self.s.flags
        adv = bool(enc.get("init_advantage_flag")) and enc["init_advantage_flag"] in self.s.flags
        d, rolls = self.dice.d20(_mode(adv, surprised))   # 기습당하면 우선권 불리 [SRD p.13]
        hero_init = d + hero.mod("dex")
        tag = " (기습당함: 불리)" if surprised and not adv else " (이점)" if adv and not surprised else ""
        events.append(Event("check", f"{hero.name}의 우선권: d20 {rolls if len(rolls) > 1 else d} {hero.mod('dex'):+d} = {hero_init}{tag}"))
        order.append(("hero", hero_init, 0))
        for gi, grp in enumerate(enc["monsters"]):
            md = self.c.monsters[grp["monster"]]
            gd, _ = self.dice.d20()                        # 같은 종류는 한 번만 굴린다 [SRD p.13]
            init = gd + md["init"]
            for i in range(grp["count"]):
                mid = f"{grp['monster']}_{i + 1}"
                name = md["name"] + (f" {i + 1}" if grp["count"] > 1 else "")
                enemies[mid] = {"id": mid, "kind": grp["monster"], "name": name, "hp": md["hp"], "hp_max": md["hp"],
                                "conditions": {}, "used": {}, "out": ""}
                order.append((mid, init, 1 + gi))
            events.append(Event("check", f"{md['name']}의 우선권: {init}"))
        order.sort(key=lambda x: (-x[1], x[2]))           # 동점이면 PC 먼저 [보완]
        self.s.combat = {"enc": enc_id, "round": 1, "order": [o[0] for o in order], "idx": 0, "enemies": enemies,
                         "turn": None, "xp": 0, "guided": None, "dodging": False, "on_end": on_end or {}}
        events.append(Event("combat", "순서: " + " → ".join(hero.name if o == "hero" else enemies[o]["name"]
                                                          for o in self.s.combat["order"])))
        if any("shriek" in self.c.monsters[m["kind"]]["traits"] for m in enemies.values()):
            self.s.vars["noise:shriekers"] = True          # 다가오면 비명 [SRD p.283, 원본: 아래쪽 전부가 안다]
            events.append(Event("say", "버섯이 귀를 찢는 비명을 지른다! 아래쪽 전부가 이 소리를 들었을 것이다."))
        return events + self._run_until_hero()

    def living_enemies(self):
        """아직 싸우는 적 (out이 ""인 것). out = "dead"(쓰러짐) / "fled"(달아남)."""
        return [m for m in self.s.combat["enemies"].values() if not m["out"]]

    def wound_text(self, m):
        """적 HP 숫자 대신 화면에 보여 줄 상태 말 (멀쩡함 / 다침 / 피투성이)."""
        if m["hp"] >= m["hp_max"]:
            return "멀쩡함"
        return "피투성이" if m["hp"] * 2 <= m["hp_max"] else "다침"     # Bloodied = 절반 이하 [SRD p.16]

    # ================================================================== 진행
    def _run_until_hero(self):
        """순서표를 따라 몬스터 차례를 자동 진행하고, 주인공이 행동할 수 있는 차례가 오면 멈춘다.
        전투가 끝나거나 게임 오버가 되면 바로 돌아간다. 5000번은 무한 반복을 막는 안전장치."""
        events = []
        for _ in range(5000):
            cb = self.s.combat
            if not cb or self.s.ended:
                return events
            who = cb["order"][cb["idx"]]
            if who == "hero":
                ev, ready = self._hero_turn_start()
                events += ev
                if ready:
                    return events
            else:
                events += self._monster_turn(who)
            if not self.s.combat or self.s.ended:
                return events
            events += self._check_end()
            if not self.s.combat:
                return events
            self._next_slot()
        raise RuntimeError("전투가 끝나지 않음")

    def _next_slot(self):
        cb = self.s.combat
        cb["idx"] += 1
        if cb["idx"] >= len(cb["order"]):
            cb["idx"] = 0
            cb["round"] += 1

    def _hero_turn_start(self):
        """주인공 차례 시작. (Event 목록, 선택지를 받을 준비가 됐는지)를 돌려준다.
        쓰러져 있으면 죽음 구원만 굴리고 False — 혼자 안정 상태가 되면 1인 패배로 게임 오버."""
        cb, hero = self.s.combat, self.hero
        cb["dodging"] = False                                # Dodge는 다음 턴 시작까지 [SRD 용어집]
        cb["turn"] = {"action": 1, "bonus": 1, "sneak": False, "surged": False}
        events = [Event("combat", f"— {cb['round']}라운드: {hero.name}의 차례 —", {"round": cb["round"]})]
        if not hero.down:
            return events, True
        # 쓰러짐: 자기 턴마다 죽음 구원 [SRD p.17-18]. 적은 쓰러진 캐릭터를 공격하지 않음 [보완 05-6 D2]
        if "dying" in hero.conditions:
            events += self._death_save(hero)
        if "dead" in hero.conditions:
            return events, False
        if not hero.down:
            events.append(Event("status", f"{hero.name}가 다시 일어선다!"))
            return events, True
        if "stable" in hero.conditions:                      # 혼자 쓰러진 채 안정 → 1인 패배 [보완 05-6 D4]
            events += self._end_combat("defeat")
            self.s.ended = "game_over"
            events.append(Event("end", f"전투에서 쓰러져 패배했다. 게임 오버. 체크포인트({self.checkpoint_label})에서 다시 시작할 수 있다."))
        return events, False

    def _end_hero_turn(self):
        cb = self.s.combat
        g = cb.get("guided")
        if g and g["round"] < cb["round"]:                 # Guiding Bolt: 다음 턴이 끝날 때까지 [SRD p.138]
            cb["guided"] = None
        self._next_slot()
        return self._run_until_hero()

    def _check_end(self):
        if self.s.combat and not self.living_enemies():
            return self._end_combat("victory")
        return []

    def _end_combat(self, outcome, text=None):
        """전투 종료. outcome: victory / retreat / avoid / peace / defeat (OUTCOME_KO).
        붙잡힘을 풀고, 결과를 s.vars["combat:조우 id"]에 기록하고, 패배가 아니면 시간 경과 + 조우의 on_결과 효과 실행."""
        cb = self.s.combat
        enc_id, rounds, hero = cb["enc"], cb["round"], self.hero
        self.s.combat = None
        hero.conditions.pop("grappled", None)
        if hero.conditions.get("restrained") == "grapple":
            hero.conditions.pop("restrained")
        events = [Event("say", text)] if text else []
        events.append(Event("combat", f"전투 종료: {OUTCOME_KO[outcome]} ({rounds}라운드)",
                            {"outcome": outcome, "encounter": enc_id, "rounds": rounds, "xp": cb["xp"]}))
        self.s.vars[f"combat:{enc_id}"] = outcome
        self.s.vars["xp"] = self.s.vars.get("xp", 0) + cb["xp"]
        if outcome != "defeat":
            events += self._pass_time(max(1, math.ceil(rounds * 6 / 60)))   # 1라운드 = 6초 [SRD p.13]
            events += self._apply(cb["on_end"].get(outcome, []), hero)
        return events

    # ================================================================== 선택지
    def combat_choices(self):
        """주인공 차례의 선택지. 남은 행동(action)·보너스 행동(bonus) 수에 따라 보이는 것이 달라진다.
        id 형식: cmb:attack:무기:적 / cmb:cast:주문:적|self / cmb:holy:적 / cmb:turn / cmb:escape / cmb:dodge
                 cmb:option:완화수단 / cmb:use:아이템 / cmb:wind / cmb:surge / cmb:end
        키: 행동은 숫자(목록 순번), 완화 수단 x숫자, 보너스 b숫자, Action Surge v, 차례 마침 z."""
        cb, hero = self.s.combat, self.hero
        t, out = cb["turn"], []
        foes = self.living_enemies()
        spells = self.c.rules["spells"]
        n = [0]

        def add(cid, label, key=None):
            if key is None:
                n[0] += 1
                key = str(n[0])
            out.append(ChoiceView(cid, label, [hero.id], hero.id, key=key))

        if t["action"] > 0:
            for wid in hero.weapons:
                w = self.c.rules["weapons"][wid]
                ab, bonus = self.weapon_attack(hero, wid)
                for m in foes:
                    add(f"cmb:attack:{wid}:{m['id']}", f"공격: {w['name']} → {m['name']} ({self.wound_text(m)}) [명중 {bonus:+d}, {w['damage']}{hero.mod(ab):+d}]")
            for sid in hero.cantrips + hero.spells:
                sp = spells[sid]
                if sp["time"] != "action" or (sp["level"] and self._slot_level(hero, sp["level"]) is None):
                    continue
                slot = f", 슬롯 {self._slot_level(hero, sp['level'])}" if sp["level"] else ""
                if sp["kind"] in ("attack", "save", "missile"):
                    for m in foes:
                        add(f"cmb:cast:{sid}:{m['id']}", f"주문: {sp['name']} → {m['name']} ({self.wound_text(m)}){slot}")
                elif sp["kind"] == "heal" and hero.hp < hero.max_hp:
                    add(f"cmb:cast:{sid}:self", f"주문: {sp['name']} (자신 치유{slot})")
                elif sp["kind"] == "ward" and hero.concentration != sid and self.s.has_item(sp["material"]):
                    add(f"cmb:cast:{sid}:self", f"주문: {sp['name']} (성수 1병 소모, 집중{slot})")
            if self.s.has_item("holy_water"):
                for m in foes:
                    add(f"cmb:holy:{m['id']}", f"성수 투척 → {m['name']} (남은 {self.s.inventory['holy_water']}병)")
            if ("turn_undead" in hero.features and hero.resources.get("channel_divinity", 0) > 0
                    and any(self.c.monsters[m["kind"]]["type"] == "undead" for m in foes)):
                add("cmb:turn", f"Turn Undead (Channel Divinity 남은 {hero.resources['channel_divinity']}회)")
            if "grappled" in hero.conditions:
                g = next(iter(hero.conditions["grappled"].values()))
                add("cmb:escape", f"붙잡힘에서 벗어난다 (근력(운동)/민첩(곡예) DC {g['dc']})")
            add("cmb:dodge", "회피 (다음 차례까지 공격받을 때 불리)")
            for i, opt in enumerate(self.c.encounters[cb["enc"]].get("options", []), 1):
                if opt["result"] == "retreat" and ("grappled" in hero.conditions or "restrained" in hero.conditions):
                    continue                                   # 속도 0이면 물러날 수 없다 [SRD 용어집 Grappled]
                if self.cond(opt.get("requires"), hero):
                    add(f"cmb:option:{opt['id']}", opt["label"], key=f"x{i}")
        if t["bonus"] > 0:
            for i, item_id in enumerate(self.c.use_order, 1):   # 물약은 보너스 행동 [SRD p.99]
                if self.s.has_item(item_id) and self._usable(hero, self.c.items[item_id]["use"]):
                    add(f"cmb:use:{item_id}", f"(보너스) {self.c.items[item_id]['use']['label']}", key=f"b{i}")
            if "second_wind" in hero.features and hero.resources.get("second_wind", 0) > 0 and hero.hp < hero.max_hp:
                add("cmb:wind", f"(보너스) Second Wind: 1d10+{hero.level} 회복 (남은 {hero.resources['second_wind']}회)", key="b8")
            for sid in hero.spells:
                sp = spells[sid]
                if sp["time"] == "bonus" and hero.hp < hero.max_hp and self._slot_level(hero, sp["level"]):
                    add(f"cmb:cast:{sid}:self", f"(보너스) 주문: {sp['name']} (자신 치유, 슬롯 {self._slot_level(hero, sp['level'])})", key="b9")
        if (t["action"] == 0 and not t["surged"] and "action_surge" in hero.features
                and hero.resources.get("action_surge", 0) > 0):
            add("cmb:surge", "Action Surge: 행동을 한 번 더 한다", key="v")
        add("cmb:end", "차례를 마친다", key="z")
        return out

    def combat_choose(self, cid):
        """전투 선택지 1개 처리 (Engine.choose가 "cmb:" id를 여기로 보낸다). 행동 수를 줄이고 결과를 적용한 뒤,
        적이 모두 쓰러졌는지 확인하고, 더 할 행동이 없으면 차례를 자동으로 넘긴다."""
        parts = cid.split(":")[1:]
        kind = parts[0]
        cb, hero = self.s.combat, self.hero
        t = cb["turn"]
        if kind == "end":
            return self._end_hero_turn()
        if kind == "option":
            opt = next(o for o in self.c.encounters[cb["enc"]]["options"] if o["id"] == parts[1])
            return self._end_combat(opt["result"], opt.get("text"))
        events = []
        if kind == "attack":
            t["action"] -= 1
            events += self._hero_weapon_attack(parts[1], cb["enemies"][parts[2]])
        elif kind == "cast":
            sp = self.c.rules["spells"][parts[1]]
            t["bonus" if sp["time"] == "bonus" else "action"] -= 1
            events += self._cast(parts[1], None if parts[2] == "self" else cb["enemies"][parts[2]])
        elif kind == "holy":
            t["action"] -= 1
            events += self._holy_water(cb["enemies"][parts[1]])
        elif kind == "turn":
            t["action"] -= 1
            events += self._turn_undead()
        elif kind == "escape":
            t["action"] -= 1
            events += self._escape_grapple()
        elif kind == "dodge":
            t["action"] -= 1
            cb["dodging"] = True
            events.append(Event("combat", f"{hero.name}가 몸을 낮추고 공격을 피할 준비를 한다."))
        elif kind == "use":
            t["bonus"] -= 1
            events += self._use_item(parts[1], hero)
        elif kind == "wind":                                   # [SRD p.47] 보너스 행동, 1d10 + 전사 레벨
            t["bonus"] -= 1
            hero.resources["second_wind"] -= 1
            r, _ = self.dice.roll("1d10")
            before, hero.hp = hero.hp, min(hero.max_hp, hero.hp + r + hero.level)
            events.append(Event("status", f"Second Wind: 1d10({r})+{hero.level} 회복. HP {before} → {hero.hp}/{hero.max_hp}"))
        elif kind == "surge":                                  # [SRD p.47] 행동 1회 추가 (Magic 행동 제외)
            hero.resources["action_surge"] -= 1
            t["action"] += 1
            t["surged"] = True
            events.append(Event("combat", f"{hero.name}가 한계를 넘어 한 번 더 움직인다! (Action Surge)"))
        if self.s.combat and not self.s.ended:
            events += self._check_end()
        if self.s.combat and not self.s.ended:
            if not [v for v in self.combat_choices() if v.id != "cmb:end"]:
                events += self._end_hero_turn()                # 더 할 것이 없으면 자동으로 차례 종료
        return events

    # ================================================================== PC의 공격
    def _roll_damage(self, expr, crit=False):
        """치명타: 피해 주사위를 두 번 굴림 (수정치는 한 번) [SRD p.15]."""
        m = DICE_EXPR.match(expr)
        count, sides = int(m.group(1) or 1), int(m.group(2))
        bonus = int(m.group(3).replace(" ", "")) if m.group(3) else 0
        rolls = [self.dice.die(sides) for _ in range(count * (2 if crit else 1))]
        return sum(rolls) + bonus, rolls

    def _hero_attack_flags(self, m):
        """주인공이 적 m을 공격할 때의 (이점, 불리). 둘 다 있으면 _mode가 상쇄한다."""
        hero, cb = self.hero, self.s.combat
        adv = bool(cb.get("guided") and cb["guided"]["mid"] == m["id"])
        adv = adv or "restrained" in m["conditions"]
        dis = "poisoned" in hero.conditions or "restrained" in hero.conditions      # [SRD 용어집]
        grapplers = hero.conditions.get("grappled", {})
        dis = dis or (bool(grapplers) and m["id"] not in grapplers)                  # Grappled [SRD 용어집]
        dis = dis or "hidden" in m["conditions"]                                     # 보이지 않는 대상 [SRD p.14]
        return adv, dis

    def _attack_roll(self, m, bonus, label):
        """PC → 몬스터 공격 굴림. (명중, 치명타, 모드, 이벤트)."""
        md = self.c.monsters[m["kind"]]
        adv, dis = self._hero_attack_flags(m)
        mode = _mode(adv, dis)
        if self.s.combat.get("guided") and self.s.combat["guided"]["mid"] == m["id"]:
            self.s.combat["guided"] = None                    # 다음 공격 1회에만 [SRD p.138]
        d, rolls = self.dice.d20(mode)
        total = d + bonus
        crit = d == 20
        hit = d != 1 and (crit or total >= md["ac"])          # 1은 빗나감, 20은 명중 [SRD p.15]
        tag = {"advantage": " (이점)", "disadvantage": " (불리)"}.get(mode, "")
        res = "치명타!" if crit else "명중" if hit else "빗나감"
        text = f"{self.hero.name}의 {label}: d20 {rolls if len(rolls) > 1 else d} {bonus:+d} = {total} vs AC {md['ac']} → {res}{tag}"
        return hit, crit, mode, Event("check", text, {"roll": d, "total": total, "ac": md["ac"], "hit": hit, "crit": crit})

    def _hero_weapon_attack(self, wid, m):
        """무기 공격: 공격 굴림 → 피해(+능력 수정치) → 조건이 맞으면 암습 → 몬스터에게 피해 → 점액이면 무기 부식."""
        hero, w = self.hero, self.c.rules["weapons"][wid]
        md = self.c.monsters[m["kind"]]
        ab, bonus = self.weapon_attack(hero, wid)
        hit, crit, mode, ev = self._attack_roll(m, bonus, f"{w['name']} 공격")
        events = [ev]
        if not hit:
            return events
        dmg, rolls = self._roll_damage(w["damage"], crit)
        dmg = max(0, dmg + hero.mod(ab))
        why = f"{w['damage']}({'+'.join(map(str, rolls))}){hero.mod(ab):+d}"
        t = self.s.combat["turn"]
        if ("sneak_attack" in hero.features and not t["sneak"] and mode == "advantage"
                and ("finesse" in w["props"] or w["kind"] == "ranged")):
            # Sneak Attack: 이점이 있고 Finesse·원거리 무기. 1인이라 '아군이 5ft 안' 조건은 쓰지 않음 [SRD p.61-62]
            nd = self.class_value(hero, "sneak_dice", 1)
            extra, er = self._roll_damage(f"{nd}d6", crit)
            dmg += extra
            t["sneak"] = True
            why += f" + 암습 {nd}d6({'+'.join(map(str, er))})"
        events += self._hurt_monster(m, dmg, w["type"], crit, why)
        if "corrosive_form" in md["traits"] and dmg > 0 and w["kind"] == "melee" and not w.get("thrown"):
            # 비마법 무기: 피해를 준 뒤 명중 −1 누적, −5면 파괴 [SRD p.290]
            hero.weapon_penalty[wid] = hero.weapon_penalty.get(wid, 0) + 1
            if hero.weapon_penalty[wid] >= 5:
                hero.weapons.remove(wid)
                events.append(Event("status", f"{w['name']}이(가) 점액에 녹아 부서졌다!"))
            else:
                events.append(Event("status", f"{w['name']}이(가) 부식되었다 (명중 −{hero.weapon_penalty[wid]})."))
        return events

    def _cast(self, sid, m):
        """주문 시전. rules.json spells의 kind로 처리 방식이 갈린다:
        attack(주문 공격 굴림) / save(대상 구원 실패 시 피해) / missile(자동 명중) / heal(자신 회복) / ward(보호 결계, 집중).
        1레벨 이상 주문은 쓸 수 있는 가장 낮은 슬롯을 쓰고, 높은 슬롯이면 upcast만큼 주사위가 늘어난다."""
        hero, sp = self.hero, self.c.rules["spells"][sid]
        events = []
        up = 0
        if sp["level"]:
            lv = self._slot_level(hero, sp["level"])
            hero.resources["slots"][lv - 1] -= 1
            up = lv - sp["level"]
            events.append(Event("status", f"{sp['name']} 시전 (주문 슬롯 {lv}레벨 사용, 남은 슬롯 {hero.resources['slots']})"))
        if sp["kind"] == "attack":
            hit, crit, _, ev = self._attack_roll(m, self.spell_attack_bonus(hero), f"{sp['name']} 주문 공격")
            events.append(ev)
            if hit:
                expr = sp["damage"] if not up else _add_dice(sp["damage"], sp["upcast"], up)
                dmg, rolls = self._roll_damage(expr, crit)
                events += self._hurt_monster(m, dmg, sp["type"], crit, f"{expr}({'+'.join(map(str, rolls))})")
                if sp.get("guide") and not m["out"]:
                    self.s.combat["guided"] = {"mid": m["id"], "round": self.s.combat["round"]}
                    events.append(Event("status", f"{m['name']}이(가) 빛에 휩싸였다. 다음 공격은 이점을 받는다."))
        elif sp["kind"] == "save":
            ok, ev = self._monster_save(m, sp["save"], self.spell_dc(hero))
            events.append(ev)
            if not ok:
                dmg, rolls = self._roll_damage(sp["damage"])
                events += self._hurt_monster(m, dmg, sp["type"], False, f"{sp['damage']}({'+'.join(map(str, rolls))})")
        elif sp["kind"] == "missile":                           # 자동 명중 [SRD p.146]
            darts = sp["darts"] + up
            total, all_rolls = 0, []
            for _ in range(darts):
                v, r = self._roll_damage(sp["damage"])
                total += v
                all_rolls += r
            events += self._hurt_monster(m, total, sp["type"], False, f"{darts}발 × {sp['damage']}({'+'.join(map(str, all_rolls))})")
        elif sp["kind"] == "heal":
            expr = sp["heal"] if not up else _add_dice(sp["heal"], sp["upcast"], up)
            v, r = self._roll_damage(expr)
            amount = v + hero.mod(hero.spellcasting)
            before, hero.hp = hero.hp, min(hero.max_hp, hero.hp + amount)
            events.append(Event("status", f"{sp['name']}: {expr}({'+'.join(map(str, r))}){hero.mod(hero.spellcasting):+d} 회복. HP {before} → {hero.hp}/{hero.max_hp}"))
        elif sp["kind"] == "ward":                              # [SRD p.157] 성수 소모, 집중 10분
            self.s.inventory[sp["material"]] -= 1
            self._end_concentration(hero)
            hero.concentration = sid
            hero.conditions[sid] = sp["minutes"]
            events.append(Event("status", f"{hero.name}가 보호의 결계를 두른다. 언데드 등은 {hero.name}를 공격할 때 불리 (집중)."))
        return events

    def _monster_save(self, m, ability, dc):
        """몬스터 구원 굴림 (수정치는 monsters.json saves). (성공 여부, Event)."""
        md = self.c.monsters[m["kind"]]
        d, _ = self.dice.d20()
        total = d + md["saves"][ability]
        ok = total >= dc
        return ok, Event("check", f"{m['name']}의 {ability.upper()} 구원: d20 {d} {md['saves'][ability]:+d} = {total} vs DC {dc} → {'성공' if ok else '실패'}")

    def _holy_water(self, m):
        """[SRD p.97] 공격 1회 대신. 민첩 구원 DC = 8 + 던진 사람 민첩 + 숙련, 실패 시 마귀·언데드에 2d8 광휘."""
        hero = self.hero
        self.s.inventory["holy_water"] -= 1
        dc = 8 + hero.mod("dex") + hero.proficiency_bonus
        events = [Event("item", f"{hero.name}가 성수를 {m['name']}에게 던진다. (남은 {self.s.inventory['holy_water']}병)")]
        ok, ev = self._monster_save(m, "dex", dc)
        events.append(ev)
        if not ok and self.c.monsters[m["kind"]]["type"] in FIEND_UNDEAD:
            dmg, rolls = self._roll_damage("2d8")
            events += self._hurt_monster(m, dmg, "radiant", False, f"2d8({'+'.join(map(str, rolls))})")
        elif not ok:
            events.append(Event("combat", "성수가 튀었지만 아무 효과가 없다."))
        return events

    def _turn_undead(self):
        """[SRD p.37] 30ft 안의 언데드: 지혜 구원 실패 → Frightened + Incapacitated 1분. 피해를 받으면 끝."""
        hero = self.hero
        hero.resources["channel_divinity"] -= 1
        events = [Event("combat", f"{hero.name}가 성표를 높이 든다! (Turn Undead)")]
        for m in self.living_enemies():
            md = self.c.monsters[m["kind"]]
            if md["type"] != "undead":
                continue
            ok, ev = self._monster_save(m, "wis", self.spell_dc(hero))
            events.append(ev)
            if ok:
                continue
            got = [c for c in ("frightened", "incapacitated") if c not in md["condition_immune"]]
            for c in got:
                m["conditions"][c] = TURN_UNDEAD_ROUNDS
            m["conditions"]["turned"] = TURN_UNDEAD_ROUNDS
            names = {"frightened": "겁에 질림", "incapacitated": "행동 불능"}
            events.append(Event("status", f"{m['name']}: {', '.join(names[c] for c in got) or '효과 없음 (면역)'}"))
        return events

    def _escape_grapple(self):
        """[SRD p.182] 행동으로 근력(운동) 또는 민첩(곡예) 판정 vs 탈출 DC."""
        hero = self.hero
        mid, g = next(iter(hero.conditions["grappled"].items()))
        ok, ev = self.roll_check(hero, {"options": [{"ability": "str", "skill": "athletics"},
                                                    {"ability": "dex", "skill": "acrobatics"}], "dc": g["dc"]})
        events = [ev]
        if ok:
            self._release(mid)
            events.append(Event("status", f"{hero.name}가 {self.s.combat['enemies'][mid]['name']}에게서 빠져나왔다."))
        return events

    def _release(self, mid):
        """몬스터 mid가 주인공을 붙잡고 있었다면 풀어 준다 (탈출 성공, 그 몬스터가 쓰러짐·달아남).
        hero.conditions["grappled"] = {몬스터 id: {"dc", "restrained"}} — 여러 몬스터가 동시에 붙잡을 수 있는 구조."""
        hero = self.hero
        grapplers = hero.conditions.get("grappled")
        if not grapplers or mid not in grapplers:
            return
        grapplers.pop(mid)
        if not grapplers:
            hero.conditions.pop("grappled")
        if hero.conditions.get("restrained") == "grapple" and not any(g.get("restrained") for g in grapplers.values()):
            hero.conditions.pop("restrained")

    def _hurt_monster(self, m, dmg, dtype, crit, why):
        """몬스터가 피해를 받는 유일한 통로. 면역 → 0, 저항 → 절반, 취약 → 2배를 적용하고,
        0 HP면 Undead Fortitude(좀비) 구원을 먼저 본 뒤 쓰러뜨린다. 문어는 피해를 받으면 먹물을 뿜고 달아난다."""
        md = self.c.monsters[m["kind"]]
        note = ""
        if dtype in md["immune"]:
            dmg, note = 0, " · 면역"
        elif dtype in md["resist"]:
            dmg, note = dmg // 2, " · 저항"
        elif dtype in md["vulnerable"]:
            dmg, note = dmg * 2, " · 취약"
        m["hp"] = max(0, m["hp"] - dmg)
        events = [Event("damage", f"{m['name']}에게 {DTYPE_KO.get(dtype, dtype)} 피해 {dmg} ({why}{note}) → {self.wound_text(m) if m['hp'] else '쓰러짐'}",
                        {"target": m["id"], "amount": dmg, "hp": m["hp"]})]
        if dmg > 0 and "turned" in m["conditions"]:            # Turn Undead는 피해를 받으면 끝 [SRD p.37]
            for c in ("turned", "frightened", "incapacitated"):
                m["conditions"].pop(c, None)
        if m["hp"] == 0:
            if "undead_fortitude" in md["traits"] and dtype != "radiant" and not crit and dmg > 0:
                ok, ev = self._monster_save(m, "con", 5 + dmg)   # DC 5 + 받은 피해 [SRD p.340]
                events.append(ev)
                if ok:
                    m["hp"] = 1
                    events.append(Event("status", f"{m['name']}이(가) 쓰러졌다가 다시 몸을 일으킨다! (Undead Fortitude)"))
                    return events
            m["out"] = "dead"
            self.s.combat["xp"] += md["xp"]
            self._release(m["id"])
            events.append(Event("status", f"{m['name']}이(가) 쓰러졌다."))
        elif (dmg > 0 and "ink_cloud" in md["traits"] and not m["used"].get("ink")
              and self.c.encounters[self.s.combat["enc"]].get("underwater")):
            # Ink Cloud (1/Day): 물속에서 피해를 받으면 먹물을 뿜고 헤엄쳐 감 [SRD p.349, 원본 scn-v8-octopus]
            m["used"]["ink"] = True
            m["out"] = "fled"
            self._release(m["id"])
            events.append(Event("status", f"{m['name']}이(가) 먹물 구름을 뿜고 물속 깊이 숨어 버렸다."))
        return events

    # ================================================================== 몬스터 차례
    def _monster_turn(self, mid):
        """몬스터 1마리의 차례. 남은 라운드가 있는 상태를 1 줄이고, 행동 불능이 아니면 첫 번째 공격을 multiattack번 한다.
        몬스터 AI는 단순하다: 대상은 항상 주인공, 쓰러진 주인공은 공격하지 않는다."""
        cb = self.s.combat
        m = cb["enemies"][mid]
        if m["out"]:
            return []
        md = self.c.monsters[m["kind"]]
        for c, left in list(m["conditions"].items()):
            if isinstance(left, int):
                m["conditions"][c] = left - 1
                if m["conditions"][c] <= 0:
                    del m["conditions"][c]
        if "incapacitated" in m["conditions"]:
            return [Event("combat", f"{m['name']}은(는) 겁에 질려 움직이지 못한다.")]
        if not md["attacks"]:
            return []
        hero = self.hero
        if hero.down:                                            # 쓰러진 캐릭터는 공격하지 않음 [보완 05-6 D2]
            return []
        events = []
        for _ in range(md["multiattack"]):                       # Multiattack [SRD 몬스터 블록]
            if hero.down or self.s.ended:
                break
            events += self._monster_attack(m, md["attacks"][0])
        if "shadow_stealth" in md["traits"] and self.c.encounters[cb["enc"]].get("dim_light"):
            # Shadow Stealth: 어둑한 빛·어둠에서 보너스 행동으로 Hide (민첩(은신) DC 15) [SRD p.320, 용어집 Hide]
            d, _ = self.dice.d20()
            if d + md.get("stealth", 0) >= 15:
                m["conditions"]["hidden"] = None
                events.append(Event("combat", f"{m['name']}이(가) 어둠 속으로 스며들어 보이지 않는다."))
        return events

    def _monster_attack(self, m, atk):
        """몬스터 공격 1회: 명중 굴림 vs 주인공 AC → 피해(Engine._damage) → 공격의 추가 효과
        (최대 HP 흡수, 근력 흡수, 붙잡기, 방어구 부식). 추가 효과는 monsters.json attacks의 키로 켜진다."""
        cb, hero = self.s.combat, self.hero
        md = self.c.monsters[m["kind"]]
        adv = "hidden" in m["conditions"] or "restrained" in hero.conditions
        dis = cb["dodging"]
        ward = "protection_from_evil_and_good"
        if ward in hero.conditions and md["type"] in self.c.rules["spells"][ward]["vs_types"]:
            dis = True                                           # [SRD p.157]
        m["conditions"].pop("hidden", None)                      # 공격하면 숨은 상태가 끝남 [SRD 용어집 Hide]
        mode = _mode(adv, dis)
        ac = self.armor_class(hero)
        d, rolls = self.dice.d20(mode)
        total = d + atk["to_hit"]
        crit = d == 20
        hit = d != 1 and (crit or total >= ac)
        tag = {"advantage": " (이점)", "disadvantage": " (불리)"}.get(mode, "")
        events = [Event("check", f"{m['name']}의 {atk['name']}: d20 {rolls if len(rolls) > 1 else d} {atk['to_hit']:+d} = {total} vs AC {ac} → "
                                 f"{'치명타!' if crit else '명중' if hit else '빗나감'}{tag}", {"hit": hit, "crit": crit})]
        if not hit:
            return events
        expr = atk["damage"]
        if atk.get("bloodied_damage") and m["hp"] * 2 <= m["hp_max"]:
            expr = atk["bloodied_damage"]                        # 쥐 떼: 피투성이면 1d4 [SRD p.359]
        dmg, r = self._roll_damage(expr, crit)
        why = f"{m['name']}의 {atk['name']} {expr}({'+'.join(map(str, r))}) {DTYPE_KO.get(atk['type'], atk['type'])}"
        if self.resists(hero, atk["type"]):
            dmg //= 2
            why += " · 저항"
        before_down = hero.down
        events += self._damage(hero, dmg, why, critical=crit)
        if "dead" in hero.conditions:
            return events
        if atk.get("max_hp_drain") and dmg > 0:                  # Life Drain [SRD p.324]
            hero.hp_max_loss += dmg
            hero.hp = min(hero.hp, hero.max_hp)
            events.append(Event("status", f"{hero.name}의 최대 HP가 {dmg} 줄었다 → {hero.max_hp}"))
        if atk.get("str_drain"):                                 # Draining Swipe [SRD p.320]
            n, _ = self.dice.roll(atk["str_drain"])
            hero.ability_loss["str"] = hero.ability_loss.get("str", 0) + n
            events.append(Event("status", f"{hero.name}의 근력이 {n} 줄었다 → {hero.score('str')}"))
            if hero.score("str") <= 0:
                return events + self._die(hero, "근력 0 — 원본: 그 자리에서 새 그림자가 된다")
        if hero.down or before_down:
            return events
        if atk.get("grapple"):                                   # [SRD 블록: 중형 이하 대상, 탈출 DC]
            g = hero.conditions.setdefault("grappled", {})
            g[m["id"]] = {"dc": atk["grapple"]["dc"], "restrained": bool(atk["grapple"].get("restrained"))}
            text = f"{hero.name}가 {m['name']}에게 붙잡혔다 (Grappled, 탈출 DC {atk['grapple']['dc']})"
            if atk["grapple"].get("restrained"):
                hero.conditions.setdefault("restrained", "grapple")
                text += " · Restrained"
            events.append(Event("status", text + "."))
        if atk.get("corrode_armor") and hero.armor != "none":    # 비마법 방어구 AC −1, 10이 되면 파괴 [SRD p.290]
            hero.armor_penalty += 1
            arm = self.c.rules["armor"][hero.armor]
            dex = hero.mod("dex") if arm["dex_max"] is None else min(hero.mod("dex"), arm["dex_max"])
            if arm["base"] + dex - hero.armor_penalty <= 10:
                events.append(Event("status", f"{arm['name']}이(가) 녹아 부서졌다!"))
                hero.armor, hero.armor_penalty = "none", 0
            else:
                events.append(Event("status", f"{arm['name']}이(가) 부식되었다 (AC −{hero.armor_penalty} → {self.armor_class(hero)})."))
        return events

    # ================================================================== 집중 [SRD 용어집 Concentration]
    def _end_concentration(self, char):
        """집중 중인 주문을 끝낸다 (주문 효과는 같은 이름의 condition으로 저장되어 있어 그것을 지운다)."""
        if char.concentration:
            char.conditions.pop(char.concentration, None)
            char.concentration = ""

    def _concentration_check(self, char, amount):
        """피해를 받았을 때 집중 유지 건강 구원 (DC = 피해의 절반, 10~30). Engine._damage가 부른다."""
        if not char.concentration or amount <= 0:
            return []
        name = self.c.rules["spells"][char.concentration]["name"]
        if char.down:
            self._end_concentration(char)
            return [Event("status", f"{name}의 집중이 끊겼다.")]
        dc = min(30, max(10, amount // 2))
        ok, ev = self.roll_save(char, "con", dc)
        if ok:
            return [ev]
        self._end_concentration(char)
        return [ev, Event("status", f"{name}의 집중이 끊겼다.")]


def _add_dice(base, extra, times):
    """'4d6' + '1d6' × 2 → '6d6' (같은 주사위만)."""
    b, e = re.match(r"(\d+)d(\d+)", base), re.match(r"(\d+)d(\d+)", extra)
    return f"{int(b.group(1)) + int(e.group(1)) * times}d{b.group(2)}" + base[b.end():]
