"""Gradio 화면과 게임 처리 함수를 연결합니다.

입력: 모델·검색 함수가 담긴 api와, 화면에서 받은 입력문·선택지 ID·접속 토큰입니다.
처리: PlaySessions에서 해당 사용자의 게임을 찾고 대화·행동 처리를 호출합니다.
render는 처리 후 저장된 대화, HP, 소지품, 가능한 행동을 읽어 화면 값으로 정리합니다.
출력: 실행 가능한 Gradio Blocks 화면과 각 구성요소를 갱신할 값입니다.

게임 진행과 응답 생성은 gameplay.py에 연결됩니다. 캐릭터 수치는 game.model,
금액 표기는 game.engine을 사용합니다. 화면을 다시 그릴 때는 저장된 추천을 읽습니다."""
import html
import json
import math
import re
from copy import deepcopy
from functools import wraps
from pathlib import Path
from queue import Empty, Queue
import tempfile
from threading import Event, Thread
import time
from types import GeneratorType

import gradio as gr

from game.engine import fmt_money
from game.model import ABILITY_KO
from gameplay import GameEngine, PlaySessions
from portrait_view import register_portraits, present_messages, conversation_text


COPY_CONVERSATION_JS = """async (text) => {
    if (!text) return '대화 전체 복사 · 내용 없음';
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
        } else {
            throw new Error('fallback');
        }
        return '대화 전체 복사 · 완료';
    } catch (_) {
        const field = document.createElement('textarea');
        field.value = text;
        field.style.cssText = 'position:fixed;left:-9999px;top:0';
        field.setAttribute('readonly', '');
        const focused = document.activeElement;
        document.body.appendChild(field);
        field.select();
        let copied = false;
        try { copied = document.execCommand('copy'); }
        finally { field.remove(); if (focused && focused.focus) focused.focus(); }
        return copied ? '대화 전체 복사 · 완료' : '대화 전체 복사 · 다시 시도';
    }
}"""


class _TypingView:
    """안전하게 전달된 본문을 화면 사본에서만 짧게 이어 표시합니다."""
    tick = 0.035
    finish_budget = 0.35

    def __init__(self):
        self.clear()

    def clear(self):
        self.view, self.context, self.target, self.position = None, None, "", 0
        self.last_emit = -float("inf")

    @property
    def pending(self):
        return self.view is not None and self.position < len(self.target)

    def offer(self, view):
        chat = view[0]
        if not chat or chat[-1].get("role") != "assistant":
            return False
        target, context = chat[-1]["content"], chat[:-1]
        if (context != self.context or not target.startswith(self.target[:self.position])):
            heading = re.match(r"^\*\*[^\n]+\*\*\n", target)
            self.position = heading.end() if heading else 0
            self.last_emit = -float("inf")
        self.view, self.context, self.target = list(view), context, target
        return True

    def advance(self, now, finish_by=None):
        if not self.pending or now - self.last_emit < self.tick:
            return None
        remaining = len(self.target) - self.position
        step = max(2, math.ceil(remaining / 12))
        if finish_by is not None:
            frames = max(1, int((finish_by - now) / self.tick))
            step = max(step, math.ceil(remaining / frames))
        self.position = min(len(self.target), self.position + step)
        self.last_emit = now
        view = list(self.view)
        view[0] = [*self.context, {**self.view[0][-1], "content": self.target[:self.position]}]
        return view

    def matches_final(self, view):
        return bool(self.view is not None and view[0] and view[0][-1].get("role") == "assistant"
                    and view[0][:-1] == self.context)


SKILL_ABILITIES = {
    "athletics": ("운동", "str"), "acrobatics": ("곡예", "dex"),
    "sleight_of_hand": ("손재주", "dex"), "stealth": ("은신", "dex"),
    "arcana": ("비전", "int"), "history": ("역사", "int"),
    "investigation": ("수사", "int"), "nature": ("자연", "int"), "religion": ("종교", "int"),
    "animal_handling": ("동물 다루기", "wis"), "insight": ("통찰", "wis"),
    "medicine": ("의학", "wis"), "perception": ("감지", "wis"), "survival": ("생존", "wis"),
    "deception": ("기만", "cha"), "intimidation": ("위협", "cha"),
    "performance": ("공연", "cha"), "persuasion": ("설득", "cha"),
}
RESOURCE_LABELS = {"second_wind": "재기의 바람", "action_surge": "행동 연쇄",
                   "arcane_recovery": "비전 회복", "channel_divinity": "신성 변환"}

GAME_LAYOUT_CSS = """
.gradio-container {width:100% !important;max-width:1440px !important;margin:0 auto!important;padding:16px!important;box-sizing:border-box;background:#15151c;color:#ece6db;}
.gradio-container *,.gradio-container *::before,.gradio-container *::after {box-sizing:border-box;}
.gradio-container .main {padding:0!important;gap:10px!important;}
.gradio-container {--body-background-fill:#15151c;--block-background-fill:#1b1b24;--block-border-color:#33303b;--border-color-primary:#33303b;--input-background-fill:#20202b;--button-primary-background-fill:#9d793b;--button-primary-text-color:#fff9eb;--body-text-color:#ece6db;--block-label-text-color:#c9bfae;--input-text-color:#ece6db;--body-text-color-subdued:#aba597;}
#game-heading {margin-bottom:6px;}
#game-heading h1 {font-size:27px!important;letter-spacing:.08em;color:#e2c48b;margin:0!important;}
#game-heading p {font-size:12px;color:#aaa494;margin-top:3px!important;}
#game-layout {display:grid!important;grid-template-columns:minmax(0,2.8fr) minmax(270px,1fr);gap:14px;align-items:start;}
#game-layout>.game-main-column,#game-layout>.game-side-column {min-width:0!important;width:100%!important;max-width:100%;}
#game-layout .block,#game-layout .form,.game-composer>*,.game-status-row>* {min-width:0!important;max-width:100%;}
#game-chat {border:1px solid #36313e!important;border-radius:14px!important;background:#191922!important;}
#game-chat div {background-color:transparent!important;}
.game-chat-tools {align-items:center;gap:8px!important;min-height:30px;}
.game-chat-tools .prose {font-size:12px;color:#bcb3a2;}
.game-chat-tools button {font-size:12px!important;min-height:30px!important;}
#game-chat .prose {overflow-wrap:anywhere;word-break:normal;}
#game-chat .message {width:100%!important;max-width:100%!important;border:0!important;background:transparent!important;box-shadow:none!important;}
#game-chat .message-row {padding:4px 8px!important;}
#game-chat pre,#game-chat code {white-space:pre-wrap;overflow-wrap:anywhere;}
#game-chat table {display:block;max-width:100%;overflow-x:auto;font-size:12px;}
.trpg-chat-row {display:flex;align-items:flex-start;gap:12px;margin:5px 0 10px;line-height:1.65;}
.trpg-avatar {width:42px!important;height:42px!important;min-width:42px!important;object-fit:cover;border-radius:50%!important;border:1px solid #66553d;display:flex!important;align-items:center;justify-content:center;background:#2d2c36;color:#d7c097;font-size:12px;font-weight:700;}
.trpg-chat-main {min-width:0;flex:1;}
.trpg-speaker {color:#e3c58d;font-size:12px;font-weight:700;margin-bottom:3px;letter-spacing:.03em;}
.trpg-message-body {color:#dedbd4;font-size:14px;line-height:1.75;overflow-wrap:anywhere;}
.trpg-message-body p {margin:0 0 7px!important;}
.trpg-message-body p:last-child {margin-bottom:0!important;}
.trpg-chat-row.user .trpg-speaker {color:#b6c9c3;}
.trpg-chat-row.user .trpg-message-body {color:#bfcac6;}
.trpg-chat-row.system .trpg-speaker {color:#aaa394;}
.trpg-chat-row.system .trpg-avatar {border-color:#3c3946;color:#9996a4;}
.game-side-column td,.game-side-column th {white-space:normal!important;overflow-wrap:anywhere;}
.game-side-column table,.game-side-column table * {background-color:#20202a!important;color:#d8d3c8!important;border-color:#393540!important;}
#game-inventory,#game-inventory * {background-color:#20202a!important;color:#dedbd4!important;border-color:#393540!important;}
.gradio-container button {color:#e1d7c6!important;}
.gradio-container button.secondary {background:#302e39!important;border-color:#4a424d!important;}
.gradio-container .label-wrap,.gradio-container summary {color:#c4b9a5!important;}
.game-side-column .tabitem {padding:9px!important;}
.game-side-column h3 {font-size:15px!important;margin:5px 0!important;}
.game-side-column p,.game-side-column td,.game-side-column th {font-size:12px!important;}
.game-composer {align-items:center;gap:7px;}
.game-composer button,.game-map-actions button {white-space:normal;line-height:1.3;}
.game-status-row {align-items:end;gap:8px;}
.game-composer textarea {font-size:14px!important;}
#game-recommendations {border:0!important;background:transparent!important;padding:0!important;}
#game-recommendations .gallery {gap:6px!important;}
#game-recommendations button {font-size:12px!important;text-align:left!important;}
#game-recommendations .label-wrap {font-size:11px!important;}
.trpg-stats {display:flex;gap:12px;align-items:center;justify-content:space-between;border:1px solid #4b4132;border-radius:12px;padding:12px 16px;background:linear-gradient(100deg,#24222a,#1b1b24);margin-bottom:2px;}
.trpg-place {font-weight:700;font-size:17px;color:#e2c48b;}
.trpg-hero {font-size:12px;color:#bcb6a9;margin-top:3px;}
.trpg-vitals {min-width:230px;font-size:12px;color:#d0cabb;text-align:right;}
.trpg-hp-track {height:5px;margin:6px 0;border-radius:4px;background:#36343e;overflow:hidden;}
.trpg-hp-fill {height:100%;background:#96b9a4;border-radius:4px;}
.trpg-resource {font-size:11px;color:#aaa394;}
.trpg-map-card {border:1px solid #39343f;border-radius:9px;padding:9px 10px;background:#20202a;flex:1 1 105px;min-width:0;color:#dedbd1;}
.trpg-map-card.current {border-color:#a58650;}
@media(max-width:980px) {#game-layout {grid-template-columns:minmax(0,1fr);} .game-side-column {margin-top:0!important;}}
@media(max-width:600px) {.gradio-container {padding:8px!important;}.game-composer .game-input {flex-basis:100%!important;}.trpg-stats {align-items:flex-start;gap:8px;padding:10px;}.trpg-vitals {min-width:145px;max-width:52%;}.trpg-place {font-size:15px;}.trpg-avatar {width:34px!important;height:34px!important;min-width:34px!important;}.trpg-chat-row {gap:9px;}.trpg-message-body {font-size:13px;}}
"""


def resource_summary(engine, content):
    """엔진이 보관하는 주문 슬롯·직업 자원을 화면용 한 줄로 만듭니다.

    현재 남은 횟수는 hero.resources에서, 최대 횟수는 Content의 직업·레벨 표에서 읽습니다.
    캐릭터의 종류에 따라 해당하는 항목만 표시하며 게임 수치를 직접 갱신하지 않습니다."""
    hero = engine.hero
    maximum = content.rules["class_levels"].get(hero.cls, {}).get(str(hero.level), {})
    rows = []
    for level, remaining in enumerate(hero.resources.get("slots", []), 1):
        limits = maximum.get("slots", [])
        limit = limits[level - 1] if level <= len(limits) else remaining
        rows.append(f"{level}레벨 주문 슬롯 {remaining}/{limit}")
    for key, label in RESOURCE_LABELS.items():
        if key in hero.resources and maximum.get(key, 0) > 0:
            rows.append(f"{label} {hero.resources[key]}/{maximum.get(key, hero.resources[key])}회")
    if hero.cls == "rogue":
        dice = maximum.get("sneak_dice", 0)
        if dice:
            used = bool(engine.s.combat and engine.s.combat.get("turn", {}).get("sneak"))
            rows.append(f"기습 공격 {dice}d6 · " + ("이번 차례 사용" if used else "조건 충족 시 차례당 1회"))
    return " · ".join(rows) or "별도 사용 횟수 자원 없음"


def character_sheet(engine, content):
    """현재 캐릭터의 능력치·숙련·장착 장비를 Markdown으로 정리합니다.

    engine과 content를 받아 수정치·구원·주문·자원 등 표시 값을 구합니다.
    탐사 가방과 장착 장비는 출처가 다르므로 구분해 보여 줍니다."""
    hero = engine.hero
    species = {"Human": "인간", "Elf": "엘프", "Dwarf": "드워프"}.get(hero.species, hero.species)
    background = {"Soldier": "군인", "Criminal": "범죄자", "Sage": "현자", "Acolyte": "시종"}.get(hero.background, hero.background)
    rows = [f"### {hero.name} · {hero.cls_ko} {hero.level}레벨",
            f"{species} · {background} · 숙련 보너스 **{hero.proficiency_bonus:+d}**",
            "", "능력 | 점수 | 수정치 | 구원", "---|---:|---:|---:"]
    for ability, label in ABILITY_KO.items():
        proficient = ability in hero.save_profs
        bonus = hero.mod(ability) + (hero.proficiency_bonus if proficient else 0)
        rows.append(f"{label} | {hero.score(ability)} | {hero.mod(ability):+d} | {bonus:+d}" + (" ✓" if proficient else ""))
    skills = []
    for skill in hero.skill_profs:
        if skill in SKILL_ABILITIES:
            label, ability = SKILL_ABILITIES[skill]
            skills.append(f"{label} {hero.mod(ability) + hero.proficiency_bonus:+d}")
        else:
            skills.append(skill)
    weapons = ", ".join(content.rules["weapons"][wid]["name"] for wid in hero.weapons) or "없음"
    armor = content.rules["armor"][hero.armor]["name"] + (" + 방패" if hero.shield else "")
    rows += ["", "✓ 구원 숙련", "", "**기술 숙련:** " + (", ".join(skills) or "없음"),
             "", f"**무기:** {weapons}", "", f"**방어구:** {armor}",
             "", "**소지금:** " + fmt_money(engine.s.gold_cp),
             "", f"**회복 주사위:** {hero.hd_left}/{hero.hd_max}개 · d{hero.hit_die}",
             "", "**자원:** " + resource_summary(engine, content),
             "", "**상태:** " + (", ".join(hero.conditions) or "정상")]
    if hero.tool_profs:
        names = {"thieves_tools": "도둑 도구"}
        rows += ["", "**도구 숙련:** " + ", ".join(names.get(key, key) for key in hero.tool_profs)]
    if hero.spellcasting:
        spells = content.rules["spells"]
        rows += ["", f"**주문 능력:** {ABILITY_KO[hero.spellcasting]} · 명중 {engine.spell_attack_bonus(hero):+d} · 구원 DC {engine.spell_dc(hero)}",
                 "", "**소마법:** " + (", ".join(spells[sid]["name"] for sid in hero.cantrips) or "없음"),
                 "", "**준비된 전투 주문:** " + (", ".join(spells[sid]["name"] for sid in hero.spells) or "없음")]
    if hero.concentration:
        rows += ["", "**집중 중:** " + content.rules["spells"].get(hero.concentration, {}).get("name", hero.concentration)]
    # 가방에는 탐사 중 꺼내 쓰는 물품을, 캐릭터 시트에는 별도 장착 장비를 표시합니다.
    equipped = engine.s.vars.get("setup", {}).get("equipped_items", {})
    if equipped:
        rows += ["", "**장착 장비:** " + " · ".join(
            f"{content.items[item]['name']} {quantity}개" for item, quantity in equipped.items() if quantity > 0)]
    return "\n".join(rows)


def map_markup(locations):
    """공개 지도 목록을 읽기 전용 HTML 카드로 표시합니다."""
    cards = []
    for location in locations:
        current = location.get("current") is True
        reachable = location.get("reachable") is True
        label = "현재 장소" if current else ("이동 가능" if reachable else "현재 이동 불가")
        people = ", ".join(html.escape(person["name"]) for person in location.get("npcs", []))
        cards.append(
            '<div class="trpg-map-card' + (' current' if current else '') + '">'
            f'<div style="font-size:10px;color:#b4a58c">{label}</div>'
            f'<div style="font-weight:700;font-size:12px;margin:3px 0">{html.escape(location["name"])}</div>'
            + (f'<div style="font-size:11px;color:#bcb6aa">{people}</div>' if people else '') + '</div>')
    if not cards:
        return '<p style="color:#bdd5c9">현재 공개된 지도 정보가 없습니다.</p>'
    extra = ('<p style="font-size:12px;color:#bdd5c9">장소 버튼은 8곳까지 표시합니다. '
             '나머지 공개 경로는 현재 가능한 행동에서 선택하세요.</p>' if len(locations) > 8 else "")
    return ('<div style="margin:2px 0"><div style="display:flex;flex-wrap:wrap;gap:6px">'
            + "".join(cards) + '</div>' + extra + '</div>')


def build_game_demo(api):
    """화면 구성요소와 이벤트 처리 함수를 만들고 Gradio 화면을 반환합니다.

    api는 gameplay.PlaySessions에 전달되어 실제 대화·검색·게임 처리에 쓰입니다.
    안쪽 함수들은 화면 입력과 세션 토큰을 받고, 공통 render 순서로 화면 값을 돌려줍니다."""
    register_portraits()
    sessions = PlaySessions(api)
    busy_sessions = set()
    polled_recommendations = {}
    content = sessions.content
    characters = [(f"{name} · {role}", cid) for cid, name, role in content.characters()]
    profiles = {c["dialogue"]["name"]: c for c in content.cards.values() if c.get("dialogue")}
    map_profile_names = list(profiles)[:5]
    room_choices = [(room["name"], rid) for rid, room in content.rooms.items()]
    encounters = [(data["name"], eid) for eid, data in content.encounters.items()]
    stream_enabled = callable(api.get("output_stream"))

    def preview_character(character_id):
        """새 게임을 만들기 전에 선택한 사전 제작 캐릭터의 실제 초기값을 보여 줍니다."""
        return character_sheet(GameEngine(content=content, char_id=character_id), content)

    def navigation_rows(game):
        # 비공개 장소를 content.rooms에서 별도로 확장하지 않습니다.
        rows = []
        for location in game.map_locations():
            if not isinstance(location, dict) or not isinstance(location.get("id"), str) or not isinstance(location.get("name"), str):
                continue
            people = [{"name": person["name"], "card_id": person.get("card_id"),
                       "can_meet": person.get("can_meet") is True}
                      for person in location.get("npcs", [])
                      if isinstance(person, dict) and isinstance(person.get("name"), str)]
            rows.append({"id": location["id"], "name": location["name"],
                         "current": location.get("current") is True,
                         "reachable": location.get("reachable") is True, "npcs": people})
        return rows

    def navigation_updates(game):
        locations = navigation_rows(game)
        room_updates = []
        for index in range(8):
            location = locations[index] if index < len(locations) else None
            room_updates.append(gr.update(
                value=(("📍 " if location["current"] else "이동 · ") + location["name"]) if location else "장소",
                visible=location is not None,
                interactive=bool(location and not location["current"] and location["reachable"])))
        meet_updates = []
        for index in range(5):
            name = map_profile_names[index] if index < len(map_profile_names) else None
            person = next((person for location in locations for person in location["npcs"] if person["name"] == name), None)
            meet_updates.append(gr.update(value=(name + " 만나기") if name else "인물 만남",
                                          visible=person is not None,
                                          interactive=bool(person and person["can_meet"])))
        return [map_markup(locations), *room_updates, *meet_updates]

    def render_snapshot(token):
        """세션 토큰으로 게임을 찾아 대화·수치·선택지의 최신 표시 값을 반환합니다.

        여기서는 저장된 상태와 추천을 읽어 HP 막대, 가방, 전투, 개발자 자료를 구성합니다.
        반환 목록의 순서는 아래 view 구성요소 목록과 맞아야 한 번에 화면이 갱신됩니다."""
        game = sessions.get(token)
        engine, state, hero = game.engine, game.engine.s, game.engine.hero
        ratio = max(0, min(100, int(100 * hero.hp / max(hero.max_hp, 1))))
        dev_tag = " · 개발자 테스트 기록 있음" if state.vars.get("developer_used") else ""
        turn_info = ""
        if state.combat:
            turn = state.combat.get("turn", {})
            turn_info = f" · 이번 차례: 행동 {turn.get('action', 0)} / 보너스 행동 {turn.get('bonus', 0)}"
            if "reaction" in turn:
                turn_info += f" / 반응 {turn['reaction']}"
        stats = f'''<div class="trpg-stats">
<div><div class="trpg-place">{html.escape(content.room_name(state.room))}</div>
<div class="trpg-hero">{html.escape(hero.name)} · {html.escape(hero.cls_ko)} {hero.level}레벨 · AC {engine.armor_class(hero)} · {fmt_money(state.gold_cp)}</div></div>
<div class="trpg-vitals"><div>HP {hero.hp}/{hero.max_hp} · 회복 주사위 {hero.hd_left}/{hero.hd_max} · {state.minutes}분 경과{dev_tag}</div>
<div class="trpg-hp-track"><div class="trpg-hp-fill" style="width:{ratio}%"></div></div>
<div class="trpg-resource">{html.escape(resource_summary(engine, content) + turn_info)}</div></div></div>'''
        inventory = [[content.items[item]["name"], quantity] for item, quantity in state.inventory.items() if quantity > 0]
        equipment = character_sheet(engine, content)
        if state.combat:
            cb = state.combat
            enemy_rows = [f"- {enemy['name']}: {engine.wound_text(enemy)}" for enemy in engine.living_enemies()]
            turn = cb.get("turn", {})
            budget = f"행동 **{turn.get('action', 0)}** · 보너스 행동 **{turn.get('bonus', 0)}**"
            # 반응 상태가 구현된 경우에만 실제 남은 값을 표시합니다.
            if "reaction" in turn:
                budget += f" · 반응 **{turn['reaction']}**"
            combat = f"### ⚔ {content.encounters[cb['enc']]['name']}\n{cb['round']}라운드 · {budget}\n\n" + "\n".join(enemy_rows)
            combat += "\n\n행동·보너스 행동은 자신의 차례 시작에 다시 준비됩니다. 보너스 행동은 해당 기능이 있을 때 사용합니다."
        else:
            combat = "현재 전투가 없습니다. 전투 중에는 이번 차례의 행동·보너스 행동 잔여 횟수가 표시됩니다."
        choices = [(v.label + (" · " + v.check_text if v.check_text else ""), v.id) for v in game.choices()]
        npc_options = [("장면 / 게임 마스터", "")] + game.nearby_npcs()
        pending = game.pending
        pending_text = (f"**🎲 대기: {pending['label']}**\n\n{pending.get('check_text','')}" if pending else "대기 중인 판정이 없습니다.")
        quest = game.quest_journal() or "아직 맡은 의뢰가 없습니다."
        # 생성은 게임의 대화 처리에서 끝났습니다. 화면 갱신은 저장된 추천만 읽습니다.
        recommendation_rows = [[item["label"]] for item in game.suggestions[:3]]
        logs = "\n".join(f"[{event['kind']}] {event['text']}" for event in game.audit[-35:])
        return [game.messages[-100:], stats, inventory, equipment, combat,
                gr.update(choices=choices, value=None), gr.update(choices=npc_options, value=game.npc),
                gr.update(value=pending_text, visible=bool(pending)), gr.update(visible=bool(pending), value="🎲 굴린다 / 실행"),
                game.last_status, json.dumps(game.debug, ensure_ascii=False, indent=2),
                json.dumps(game.rag_state(), ensure_ascii=False, indent=2),
                quest, logs, game.engine.developer,
                gr.update(visible=game.engine.developer), "",
                gr.update(samples=recommendation_rows, visible=bool(recommendation_rows)),
                *navigation_updates(game), gr.update(visible=bool(pending))]

    def present(view):
        """대화 사본에만 화자 초상화를 붙이고 실제 이력과 생성 입력은 그대로 둡니다."""
        view = list(view)
        view[0] = present_messages(view[0])
        return view

    def render(token):
        with sessions.lock:
            return present(render_snapshot(token))

    def copy_conversation(token):
        # 화면의 최근 100개 제한과 관계없이 이 게임에 실제 표시된 전체 이력을 읽습니다.
        if not token:
            return ""
        with sessions.lock:
            return conversation_text(sessions.get(token).messages)

    def launch_recommendations(game):
        """확정된 본문 뒤에 선택적 추천을 만들며, 다음 입력은 이 작업을 취소할 수 있습니다."""
        job = game.take_recommendation_job()
        if job is None:
            return

        def run_job():
            try:
                value = game.generate_recommendations(job)
            except Exception as error:
                value = {"error": repr(error), "suggestions": []}
            with sessions.lock:
                game.apply_recommendations(job, value)
        Thread(target=run_job, name="trpg-recommendations", daemon=True).start()

    def poll_recommendations(token):
        # 진행 중인 턴을 기다리거나 덮어쓰지 않고 추천 영역만 갱신합니다.
        if not token or token in busy_sessions or not sessions.lock.acquire(blocking=False):
            return gr.skip()
        try:
            game = sessions.get(token)
            rows = [[item["label"]] for item in game.suggestions[:3]]
            key = (game.revision, tuple(row[0] for row in rows))
            if polled_recommendations.get(token) == key:
                return gr.skip()
            polled_recommendations[token] = key
            return gr.update(samples=rows, visible=bool(rows))
        finally:
            sessions.lock.release()

    def record_error(game, operation, error):
        game.last_status = "실행 실패: " + str(error)
        game.debug = {"오류": repr(error), "단계": getattr(operation, "__name__", "callback")}
        if getattr(error, "raw_response", None):
            game.debug["실제 모델 원문"] = error.raw_response

    def perform_sync(token, operation, *args):
        """하나의 화면 동작을 실행한 뒤 같은 형식으로 화면을 다시 그립니다.

        sessions.lock 안에서 게임을 찾고 operation을 호출합니다. 예외가 나면 실행 상태와
        개발자 자료에 오류를 남겨 render가 함께 보여 주도록 합니다."""
        with sessions.lock:
            game = sessions.get(token)
            try:
                game.cancel_recommendations()
                operation(game, *args)
                launch_recommendations(game)
            except Exception as error:
                record_error(game, operation, error)
            return render(token)

    def perform_stream(token, operation, args, preview_text=None):
        """작업 스레드의 확정 상태 사본과 안전한 생성 조각만 화면에 전달합니다."""
        packets = Queue()
        stopped = Event()
        worker = None
        last_base = None
        initial_snapshot = None

        def publish(kind, value, base=None):
            if stopped.is_set():
                return False
            packets.put((kind, value, base))
            return True

        def run():
            with sessions.lock:
                try:
                    game = sessions.get(token)
                    draft = {"text": "", "speaker": "", "kind": ""}
                    base = deepcopy(initial_snapshot)
                    last_chunk = 0.0

                    def sink(event):
                        nonlocal base, last_chunk
                        if stopped.is_set():
                            return False
                        kind = event.get("event")
                        if kind not in {"start", "chunk", "end", "error", "status"}:
                            return True
                        if kind == "start":
                            draft.update(text="", speaker=event.get("speaker") or "", kind=event.get("kind") or "")
                            last_chunk = 0.0
                        elif kind == "error":
                            draft.update(text="", speaker="", kind="")
                        if kind in {"chunk", "end"}:
                            text = event.get("text")
                            if isinstance(text, str):
                                draft.update(text=text, speaker=event.get("speaker", draft["speaker"]) or "",
                                             kind=event.get("kind") or draft["kind"])
                            if kind == "chunk" and time.monotonic() - last_chunk < 0.08:
                                return True
                            last_chunk = time.monotonic()
                        if kind in {"start", "error"}:
                            # 생성 구간마다 실제 이력만 새로 읽고, 큰 상태/디버그 렌더는 완료 때 갱신합니다.
                            base = list(base)
                            base[0] = deepcopy(game.messages[-100:])
                        view = deepcopy(base)
                        if kind in {"chunk", "end"}:
                            view[9] = "내용 확인 중" if kind == "end" else "답변 작성 중"
                        elif kind == "status":
                            # 상태 이벤트도 공개된 짧은 진행 문구만 허용합니다.
                            view[9] = "내용 확인 중" if event.get("text") == "내용 확인 중" else "답변 작성 중"
                        elif kind == "error":
                            view[9] = "내용 확인 중"
                        else:
                            view[9] = "답변 작성 중"
                        if draft["text"]:
                            heading = ("**" + draft["speaker"] + "**\n"
                                       if draft["speaker"] and draft["kind"] != "scene_dialogue" else "")
                            displayed = draft["text"] if heading and draft["text"].startswith(heading) else heading + draft["text"]
                            last = base[0][-1] if base[0] else {}
                            if last.get("role") != "assistant" or last.get("content") != displayed:
                                view[0] = [*base[0], {"role": "assistant", "content": displayed, "speaker": draft["speaker"]}]
                        packet_kind = "reset" if kind in {"start", "error"} else "chunk"
                        return publish(packet_kind, view, base)

                    with api["output_stream"](sink):
                        operation(game, *args)
                    launch_recommendations(game)
                except Exception as error:
                    if not stopped.is_set():
                        try:
                            record_error(sessions.get(token), operation, error)
                        except Exception:
                            pass
                finally:
                    if not stopped.is_set():
                        try:
                            publish("done", deepcopy(render_snapshot(token)))
                        except Exception as error:
                            publish("error", str(error))

        try:
            with sessions.lock:
                game = sessions.get(token)
                game.cancel_recommendations()
                busy_sessions.add(token)
                last_base = deepcopy(render_snapshot(token))
                initial_snapshot = deepcopy(last_base)
                first = deepcopy(last_base)
                text = preview_text(game) if callable(preview_text) else preview_text
                if isinstance(text, str) and text.strip():
                    first[0] = [*first[0], {"role": "user", "content": text}]
                first[9] = "답변 작성 중"
                first[17] = gr.update(samples=[], visible=False)
            # 실제 messages에는 저장하지 않는 입력 미리보기입니다.
            yield first
            worker = Thread(target=run, name="trpg-output-stream", daemon=True)
            worker.start()
            deadline = time.monotonic() + float(api.get("stream_timeout_seconds", 180))
            queued_control = None
            typing = _TypingView()
            while True:
                if time.monotonic() >= deadline:
                    stopped.set()
                    timed_out = deepcopy(last_base)
                    timed_out[9] = "응답 대기를 중단했습니다."
                    yield timed_out
                    break
                try:
                    packet = queued_control if queued_control is not None else packets.get(
                        timeout=typing.tick if typing.pending else 0.2)
                    queued_control = None
                    if packet[0] == "chunk":
                        # 이미 쌓인 누적 초안은 최신 것만 보내고 제어 이벤트 순서는 보존합니다.
                        while True:
                            try:
                                following = packets.get_nowait()
                            except Empty:
                                break
                            if following[0] != "chunk":
                                if following[0] in {"reset", "error"}:
                                    packet = following
                                else:
                                    queued_control = following
                                break
                            packet = following
                    kind, value, base = packet
                except Empty:
                    frame = typing.advance(time.monotonic())
                    if frame is not None:
                        yield frame
                        continue
                    if not worker.is_alive():
                        failed = deepcopy(last_base)
                        failed[9] = "응답 처리가 중단되었습니다."
                        yield failed
                        break
                    continue
                if kind == "error":
                    failed = deepcopy(last_base)
                    failed[9] = "실행 실패: " + value
                    yield failed
                    break
                if base is not None:
                    last_base = base
                if kind == "done":
                    # 처리 완료 후의 긴 대기열은 짧게 따라잡고 실제 최종 상태로 확정합니다.
                    if typing.matches_final(value) and typing.offer(value):
                        finish_by = min(deadline, time.monotonic() + typing.finish_budget)
                        while typing.pending and time.monotonic() < finish_by:
                            frame = typing.advance(time.monotonic(), finish_by)
                            if frame is not None:
                                yield frame
                            if typing.pending:
                                time.sleep(typing.tick)
                    yield value
                    break
                if kind == "reset":
                    typing.clear()
                    yield value
                elif typing.offer(value):
                    frame = typing.advance(time.monotonic())
                    if frame is not None:
                        yield frame
                else:
                    typing.clear()
                    yield value
        finally:
            # close/timeout 이후의 sink는 False를 반환해 생성 측의 취소 처리를 요청합니다.
            stopped.set()
            busy_sessions.discard(token)
            if worker is not None:
                worker.join(timeout=2)

    def perform(token, operation, *args, preview_text=None):
        if stream_enabled:
            return perform_stream(token, operation, args, preview_text)
        return perform_sync(token, operation, *args)

    def stream_callback(callback):
        if not stream_enabled:
            return callback

        @wraps(callback)
        def streamed(*args, **kwargs):
            result = callback(*args, **kwargs)
            if isinstance(result, GeneratorType):
                for view in result:
                    yield present(view)
            else:
                yield present(result)
        return streamed

    def close_session(token):
        with sessions.lock:
            sessions.close(token)
            polled_recommendations.pop(token, None)
            busy_sessions.discard(token)

    def start(character, previous):
        """이전 세션을 정리하며 새 게임을 시작하고 새 토큰과 첫 화면 값을 반환합니다."""
        token = sessions.start(character, previous)
        return [token, *render(token)]

    def send(message, token):
        preview = message.strip() if isinstance(message, str) and 1 <= len(message.strip()) <= 1200 else None
        return perform(token, lambda game: game.send(message), preview_text=preview)

    def choose_suggestion(selected, token):
        # 클릭 당시 문구를 확인하여 늦게 도착한 이전 추천을 다른 행동으로 실행하지 않습니다.
        label = selected[0] if isinstance(selected, list) and selected else None
        def operation(game):
            index = next((index for index, item in enumerate(game.suggestions)
                          if item.get("label") == label), None)
            if index is None:
                raise ValueError("추천이 바뀌었습니다. 현재 표시된 항목을 선택해 주세요.")
            game.choose_suggestion(index)
        return perform(token, operation, preview_text=label)

    def choose(choice, token):
        """지금 가능한 선택지 ID를 확인한 뒤 선택 문장을 대화에 남기고 행동을 제안합니다.

        선택지의 label은 사용자가 읽는 문장이고, 실제 실행 연결에는 choice ID를 사용합니다."""
        def action(game):
            if not choice:
                raise ValueError("행동을 먼저 선택하세요.")
            view = next((v for v in game.choices() if v.id == choice), None)
            if view is None:
                raise ValueError("현재 실행할 수 없는 행동입니다.")
            game.messages.append({"role": "user", "content": "**" + view.label + "**"})
            game.propose(choice, direct=True)
        def preview(game):
            selected = next((item for item in game.choices() if item.id == choice), None)
            return "**" + selected.label + "**" if selected else None
        return perform(token, action, preview_text=preview)

    def select_npc(name, token):
        return perform(token, lambda game: game.talk_to(name))

    def map_travel(index, token):
        def operation(game):
            locations = navigation_rows(game)
            if not 0 <= index < min(8, len(locations)):
                raise ValueError("현재 지도에 없는 장소입니다.")
            location = locations[index]
            if location["current"] or not location["reachable"]:
                raise ValueError("현재 이동할 수 없는 장소입니다.")
            game.travel_to(location["id"])
        return perform(token, operation)

    def map_meet(index, token):
        def operation(game):
            if not 0 <= index < len(map_profile_names):
                raise ValueError("현재 만날 수 없는 인물입니다.")
            name = map_profile_names[index]
            people = [person for location in navigation_rows(game) for person in location["npcs"]]
            if not any(person["name"] == name and person["can_meet"] for person in people):
                raise ValueError("공개된 지도에서 현재 만날 수 없는 인물입니다.")
            game.visit_npc(name)
        return perform(token, operation)

    def map_callback(handler, index):
        def callback(token):
            return handler(index, token)
        return callback

    def roll(token):
        return perform(token, lambda game: game.resolve_pending())

    def cancel(token):
        def operation(game):
            game.pending = None
            game.refresh_suggestions()
            game.last_status = "대기 중인 행동을 취소했습니다."
        return perform(token, operation)

    def toggle_developer(enabled, token):
        return perform(token, lambda game: game.set_developer(enabled))

    def dev_meet(name, token):
        return perform(token, lambda game: game.developer_jump(profiles[name]["room"], name))

    def dev_move(room, token):
        return perform(token, lambda game: game.developer_jump(room))

    def dev_success(token):
        def operation(game):
            game._record(game.engine.arm_success())
            if game.pending:
                game.pending["revision"] = game.revision
            game.last_status = "다음 플레이어 판정 1회 성공 보장. 적의 굴림·일반 주사위 합계에는 적용하지 않습니다."
        return perform(token, operation)

    def dev_win(token):
        def operation(game):
            events = game.engine.force_victory()
            game.pending = None
            game._record(events)
            game.refresh_suggestions()
            game.last_status = "개발자 전투 승리를 적용했습니다."
        return perform(token, operation)

    def dev_heal(token):
        def operation(game):
            events = game.engine.full_heal()
            game.pending = None
            game._record(events)
            game.refresh_suggestions()
            game.last_status = "개발자 회복을 적용했습니다."
        return perform(token, operation)

    def dev_combat(encounter, token):
        return perform(token, lambda game: game.developer_encounter(encounter))

    def free_check(ability, token):
        return perform(token, lambda game: game.free_check(ability))

    def utility_dice(expression, token):
        return perform(token, lambda game: game.utility_dice(expression))

    def rule_mode(mode, token):
        def operation(game):
            game.engine.require_developer()
            if mode not in {"available", "verified"}:
                raise ValueError("지원하지 않는 검색 방식입니다.")
            game.rule_mode = mode
            game.last_status = "룰 질문 검색 방식: " + mode
        return perform(token, operation)

    def save(token):
        """세션의 저장 자료를 임시 JSON 파일로 만들고 다운로드 버튼에 연결합니다.

        게임 내용을 JSON으로 바꾸는 일은 game.save_data가 담당합니다."""
        with sessions.lock:
            game = sessions.get(token)
            path = Path(tempfile.gettempdir()) / ("trpg-save-" + token[:12] + ".json")
            path.write_text(json.dumps(game.save_data(), ensure_ascii=False, indent=2), encoding="utf-8")
            return gr.update(value=str(path), visible=True)

    def restore(file, token):
        """업로드한 저장 JSON을 읽어 현재 게임의 복원 절차에 전달합니다.

        파일 선택과 크기를 먼저 확인한 뒤 restore_data를 호출하고, 공통 perform 경로로
        복원 결과와 오류를 화면에 반영합니다."""
        def operation(game):
            if not file:
                raise ValueError("저장 파일을 먼저 선택하세요.")
            path = Path(file)
            if path.stat().st_size > 2_000_000:
                raise ValueError("저장 파일이 너무 큽니다.")
            game.restore_data(json.loads(path.read_text(encoding="utf-8-sig")))
        return perform(token, operation)

    with gr.Blocks(title="잠긴 종 · 플레이 테스트", analytics_enabled=False,
                   fill_width=True, fill_height=False) as demo:
        token = gr.State(value=None, time_to_live=3600, delete_callback=close_session)
        gr.Markdown("# 잠긴 종\n우물 아래에서 들려오는 소리, 사라진 사람들의 행방을 찾아서.", elem_id="game-heading")
        with gr.Row(elem_classes="game-composer"):
            character = gr.Dropdown(choices=characters, value=characters[0][1], label="플레이어 캐릭터",
                                    show_label=False, container=False, scale=3, min_width=160)
            start_button = gr.Button("게임 시작 / 새 게임", variant="primary", scale=1, min_width=160)
        stats = gr.HTML("<p>게임을 시작하면 HP·금액·현재 위치가 표시됩니다.</p>")
        map_buttons = []
        meet_buttons = []
        with gr.Row(elem_id="game-layout"):
            with gr.Column(scale=3, min_width=0, elem_classes="game-main-column"):
                npc = gr.Dropdown(choices=[("장면 / 게임 마스터", "")], value="", label="대화 상대",
                                  show_label=False, container=False)
                with gr.Row(elem_classes="game-chat-tools"):
                    gr.Markdown("이야기")
                    copy_button = gr.Button("대화 전체 복사", size="sm", scale=0, min_width=156,
                                            elem_id="game-copy-conversation")
                copy_buffer = gr.Textbox(visible=False)
                chat = gr.Chatbot(label="이야기", height="clamp(260px, calc(100vh - 490px), 500px)",
                                  min_width=0, elem_id="game-chat", sanitize_html=True, layout="panel",
                                  show_label=False, buttons=[])
                recommendations = gr.Dataset(components=["textbox"], type="values", samples=[],
                                             label="다음에 해볼 말이나 행동", elem_id="game-recommendations",
                                             layout="gallery", samples_per_page=3, visible=False)
                with gr.Row(elem_classes="game-composer"):
                    message = gr.Textbox(label="말 또는 행동", show_label=False,
                                         placeholder="말이나 **행동**을 입력하세요. /규칙 질문", lines=1, max_lines=6,
                                         scale=6, min_width=0, elem_classes="game-input",
                                         elem_id="game-message", info="Enter 전송 · Shift+Enter 줄바꿈")
                    action_mark = gr.Button("행동 **", size="sm", scale=1, min_width=72)
                    send_button = gr.Button("보내기", variant="primary", scale=1, min_width=80)
                with gr.Row():
                    choice = gr.Dropdown(choices=[], label="현재 가능한 행동", scale=4)
                    action_button = gr.Button("행동 실행", scale=1)
                pending = gr.Markdown("", visible=False)
                with gr.Row():
                    roll_button = gr.Button("🎲 굴린다 / 실행", variant="primary", visible=False)
                    cancel_button = gr.Button("취소", size="sm", visible=False)
            with gr.Column(scale=1, min_width=0, elem_classes="game-side-column"):
                with gr.Accordion("지도 · 인물", open=True):
                    map_view = gr.HTML('<p style="color:#bdd5c9">게임을 시작하면 공개된 장소와 인물이 표시됩니다.</p>')
                    for first in (0, 4):
                        with gr.Row(elem_classes="game-map-actions"):
                            for index in range(first, first + 4):
                                map_buttons.append(gr.Button("장소", visible=False, interactive=False, size="sm", min_width=0))
                    with gr.Row(elem_classes="game-map-actions"):
                        for index in range(5):
                            meet_buttons.append(gr.Button("인물 만남", visible=False, interactive=False, size="sm", min_width=0))
                with gr.Tabs():
                    with gr.Tab("가방"):
                        inventory = gr.Dataframe(headers=["휴대품", "수량"], datatype=["str", "number"], interactive=False, value=[],
                                                 elem_id="game-inventory")
                    with gr.Tab("캐릭터"):
                        equipment = gr.Markdown("장비가 표시됩니다.")
                        with gr.Accordion("시작 전 캐릭터 시트", open=False):
                            preview = gr.Markdown(preview_character(characters[0][1]))
                    with gr.Tab("전투"):
                        combat = gr.Markdown("현재 전투가 없습니다.")
                        gr.Markdown("SRD 5.2 기반: 우선권·명중·치명타·피해·HP·구원·직업 자원. 현재 전투는 위치·사거리·기회 공격을 생략한 테스트 방식입니다.")
                    with gr.Tab("의뢰·저장"):
                        quest = gr.Markdown("진행 중인 의뢰가 표시됩니다.")
                        save_button = gr.Button("저장 파일 만들기")
                        download = gr.DownloadButton("저장 파일 다운로드", visible=False)
                        save_file = gr.File(label="저장 파일 선택", file_types=[".json"], type="filepath")
                        load_button = gr.Button("불러오기")
        with gr.Row(elem_classes="game-status-row"):
            status = gr.Textbox(label="실행 상태", interactive=False, lines=1, max_lines=2, scale=8, min_width=0)
            refresh_button = gr.Button("새로고침", size="sm", scale=1, min_width=80)
        with gr.Accordion("개발자 도구", open=False):
            developer = gr.Checkbox(label="개발자 모드", value=False)
            with gr.Group(visible=False) as dev_group:
                gr.Markdown("### 개발자 테스트\n강제 이동·승리·성공은 테스트 기록에 표시됩니다. 일반 플레이 결과와 구분하세요.")
                with gr.Row():
                    dev_npc = gr.Dropdown(choices=list(profiles), value=next(iter(profiles)), label="누구를 바로 만날까요?")
                    dev_meet_button = gr.Button("바로 만나기")
                    dev_room = gr.Dropdown(choices=room_choices, value=room_choices[0][1], label="이동 장소")
                    dev_move_button = gr.Button("바로 이동")
                with gr.Row():
                    dev_success_button = gr.Button("다음 플레이어 판정 성공")
                    dev_win_button = gr.Button("전투 즉시 승리")
                    dev_heal_button = gr.Button("HP·자원 회복")
                with gr.Row():
                    dev_encounter = gr.Dropdown(choices=encounters, value="rats", label="시험 전투")
                    dev_combat_button = gr.Button("시험 전투 시작")
                    mode = gr.Radio(choices=[("빠른 룰 검색", "available"), ("근거 검증 포함", "verified")], value="available", label="룰 질문 검색")
                with gr.Accordion("자유 판정·주사위 도구", open=False):
                    with gr.Row():
                        ability = gr.Dropdown(choices=[(ko, en) for en, ko in ABILITY_KO.items()], value="wis", label="시험할 능력")
                        check_button = gr.Button("DC 15 판정 준비")
                        expression = gr.Textbox(value="1d20", label="일반 주사위")
                        dice_button = gr.Button("일반 주사위 굴리기")
                examples = gr.Dataset(components=["textbox"], type="values", samples=[
                    ["안녕하세요"], ["제가 도와드릴 일이 있나요?"], ["인형은 어떤 물건인가요?"],
                    ["**주변을 둘러본다.**"], ["그 얘기는 잠깐 쉬고 기분이 어떠세요?"],
                    ["/규칙 공격 굴림이 방어도와 같으면 명중인가요?"],
                ], label="개발자 대화 시험 — 누르면 전송")
                with gr.Accordion("근거·모델 입력·판정 기록", open=False):
                    debug = gr.Textbox(label="실제 모델 입력과 처리 결과", lines=12, interactive=False)
                    state_view = gr.Textbox(label="현재 게임 상태(JSON)", lines=10, interactive=False)
                    logs = gr.Textbox(label="최근 엔진·주사위 사건", lines=12, interactive=False)
        with gr.Accordion("플레이 범위 · 규칙 출처", open=False):
            gr.Markdown("현재 시나리오 플레이 범위: 마을·우물 갱도. 개발자 시험 전투는 다른 조우도 불러옵니다.\n\nIncludes modified SRD 5.2 material by Wizards of the Coast LLC under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).")

        # render의 반환값과 화면 구성요소를 같은 순서로 연결합니다.
        # 아래 이벤트들은 이 공통 목록을 갱신하므로 한 동작 뒤 여러 영역이 함께 바뀝니다.
        view = [chat, stats, inventory, equipment, combat, choice, npc, pending, roll_button,
                status, debug, state_view, quest, logs, developer, dev_group, message, recommendations,
                map_view, *map_buttons, *meet_buttons, cancel_button]
        recommendation_timer = gr.Timer(0.8)
        recommendation_timer.tick(poll_recommendations, [token], [recommendations], queue=False, api_name=False, show_progress="hidden")
        # 한 게임 처리 흐름의 중복 실행을 줄이도록 대화·선택·굴림 이벤트를 같은 대기열에 묶습니다.
        group = {"concurrency_id": "gameplay", "concurrency_limit": 1}
        if stream_enabled:
            group["stream_every"] = _TypingView.tick
        character.input(preview_character, [character], [preview], api_name="preview_character", queue=False)
        start_button.click(start, [character, token], [token, *view], api_name="start_game", **group)
        copy_button.click(copy_conversation, [token], [copy_buffer], api_name="copy_conversation",
                          show_progress="hidden", **group).then(
            fn=None, inputs=[copy_buffer], outputs=[copy_button], js=COPY_CONVERSATION_JS, queue=False)
        send_button.click(stream_callback(send), [message, token], view, api_name="send_message", **group)
        message.submit(stream_callback(send), [message, token], view, api_name="submit_message", **group)
        recommendations.click(stream_callback(choose_suggestion), [recommendations, token], view,
                              api_name="choose_suggestion", **group)
        action_button.click(stream_callback(choose), [choice, token], view, api_name="choose_action", **group)
        npc.input(stream_callback(select_npc), [npc, token], view, api_name="select_npc", **group)
        roll_button.click(stream_callback(roll), [token], view, api_name="roll_pending", **group)
        cancel_button.click(stream_callback(cancel), [token], view, api_name="cancel_pending", **group)
        developer.input(stream_callback(toggle_developer), [developer, token], view, api_name="toggle_developer", **group)
        dev_meet_button.click(stream_callback(dev_meet), [dev_npc, token], view, api_name="developer_meet", **group)
        dev_move_button.click(stream_callback(dev_move), [dev_room, token], view, api_name="developer_move", **group)
        dev_success_button.click(stream_callback(dev_success), [token], view, api_name="developer_success", **group)
        dev_win_button.click(stream_callback(dev_win), [token], view, api_name="developer_victory", **group)
        dev_heal_button.click(stream_callback(dev_heal), [token], view, api_name="developer_heal", **group)
        dev_combat_button.click(stream_callback(dev_combat), [dev_encounter, token], view, api_name="developer_combat", **group)
        check_button.click(stream_callback(free_check), [ability, token], view, api_name="free_check", **group)
        dice_button.click(stream_callback(utility_dice), [expression, token], view, api_name="utility_dice", **group)
        mode.input(stream_callback(rule_mode), [mode, token], view, api_name="rule_mode", **group)
        examples.click(stream_callback(lambda selected, session: send(selected[0], session)), [examples, token], view,
                       api_name="send_example", **group)
        save_button.click(save, [token], [download], api_name="save_game", **group)
        load_button.click(stream_callback(restore), [save_file, token], view, api_name="load_game", **group)
        refresh_button.click(render, [token], view, api_name="refresh_state", **group)
        for index, button in enumerate(map_buttons):
            button.click(stream_callback(map_callback(map_travel, index)), [token], view,
                         api_name=f"map_travel_{index}", **group)
        for index, button in enumerate(meet_buttons):
            button.click(stream_callback(map_callback(map_meet, index)), [token], view,
                         api_name=f"map_meet_{index}", **group)
        action_mark.click(fn=None, inputs=[message], outputs=[message],
                          js=api["ACTION_MARKER_JS"].replace("#player-message", "#game-message"), queue=False)
    # Gradio 6은 launch(css=...)에서 스타일을 받습니다.
    demo.trpg_layout_css = GAME_LAYOUT_CSS
    return demo
