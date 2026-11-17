"""Gradio 화면과 게임 처리 함수를 연결합니다.

입력: 모델·검색 함수가 담긴 api와, 화면에서 받은 입력문·선택지 ID·접속 토큰입니다.
처리: PlaySessions에서 해당 사용자의 게임을 찾고 대화·행동 처리를 호출합니다.
render는 처리 후 저장된 대화, HP, 소지품, 가능한 행동을 읽어 화면 값으로 정리합니다.
출력: 실행 가능한 Gradio Blocks 화면과 각 구성요소를 갱신할 값입니다.

게임 진행과 응답 생성은 gameplay.py에 연결됩니다. 캐릭터 수치는 game.model,
금액 표기는 game.engine을 사용합니다. 화면을 다시 그릴 때는 저장된 추천을 읽습니다."""
import html
import json
from pathlib import Path
import tempfile

import gradio as gr

from game.engine import fmt_money
from game.model import ABILITY_KO
from gameplay import GameEngine, PlaySessions


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


def build_game_demo(api):
    """화면 구성요소와 이벤트 처리 함수를 만들고 Gradio 화면을 반환합니다.

    api는 gameplay.PlaySessions에 전달되어 실제 대화·검색·게임 처리에 쓰입니다.
    안쪽 함수들은 화면 입력과 세션 토큰을 받고, 공통 render 순서로 화면 값을 돌려줍니다."""
    sessions = PlaySessions(api)
    content = sessions.content
    characters = [(f"{name} · {role}", cid) for cid, name, role in content.characters()]
    profiles = {c["dialogue"]["name"]: c for c in content.cards.values() if c.get("dialogue")}
    room_choices = [(room["name"], rid) for rid, room in content.rooms.items()]
    encounters = [(data["name"], eid) for eid, data in content.encounters.items()]

    def preview_character(character_id):
        """새 게임을 만들기 전에 선택한 사전 제작 캐릭터의 실제 초기값을 보여 줍니다."""
        return character_sheet(GameEngine(content=content, char_id=character_id), content)

    def render(token):
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
        stats = f'''<div style="border:1px solid #43695e;border-radius:12px;padding:16px;background:#14231f;color:#eef5f1">
<div style="font-size:22px;font-weight:700">{html.escape(content.room_name(state.room))}</div>
<div style="margin:7px 0">{html.escape(hero.name)} · {html.escape(hero.cls_ko)} {hero.level}레벨 · AC {engine.armor_class(hero)} · 💰 {fmt_money(state.gold_cp)}</div>
<div>♥ HP {hero.hp}/{hero.max_hp} · 회복 주사위 {hero.hd_left}/{hero.hd_max}개 (d{hero.hit_die}) · 경과 {state.minutes}분{dev_tag}</div>
<div style="height:8px;margin-top:8px;border-radius:6px;background:#31433c"><div style="height:100%;border-radius:6px;background:#a9d4b8;width:{ratio}%"></div></div>
<div style="margin-top:10px">{html.escape(resource_summary(engine, content) + turn_info)}</div></div>'''
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
                pending_text, gr.update(visible=bool(pending), value="🎲 굴린다 / 실행"),
                game.last_status, json.dumps(game.debug, ensure_ascii=False, indent=2),
                json.dumps(game.rag_state(), ensure_ascii=False, indent=2),
                quest, logs, game.engine.developer,
                gr.update(visible=game.engine.developer), "",
                gr.update(samples=recommendation_rows, visible=bool(recommendation_rows))]

    def perform(token, operation, *args):
        """하나의 화면 동작을 실행한 뒤 같은 형식으로 화면을 다시 그립니다.

        sessions.lock 안에서 게임을 찾고 operation을 호출합니다. 예외가 나면 실행 상태와
        개발자 자료에 오류를 남겨 render가 함께 보여 주도록 합니다."""
        with sessions.lock:
            game = sessions.get(token)
            try:
                operation(game, *args)
            except Exception as error:
                game.last_status = "실행 실패: " + str(error)
                game.debug = {"오류": repr(error), "단계": getattr(operation, "__name__", "callback")}
                if getattr(error, "raw_response", None):
                    game.debug["실제 모델 원문"] = error.raw_response
            return render(token)

    def start(character, previous):
        """이전 세션을 정리하며 새 게임을 시작하고 새 토큰과 첫 화면 값을 반환합니다."""
        token = sessions.start(character, previous)
        return [token, *render(token)]

    def send(message, token):
        return perform(token, lambda game: game.send(message))

    def choose_suggestion(index, token):
        # 추천 문장도 직접 입력과 같은 게임 처리 경로를 사용합니다.
        return perform(token, lambda game: game.choose_suggestion(index))

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
        return perform(token, action)

    def select_npc(name, token):
        return perform(token, lambda game: game.talk_to(name))

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

    with gr.Blocks(title="잠긴 종 · 플레이 테스트", analytics_enabled=False) as demo:
        token = gr.State(value=None, time_to_live=3600, delete_callback=sessions.close)
        gr.Markdown("# 잠긴 종\n캐릭터를 고르고 모험을 시작하세요. 말이나 행동을 직접 입력하거나 아래 제안을 선택할 수 있습니다.")
        with gr.Row():
            character = gr.Dropdown(choices=characters, value=characters[0][1], label="플레이어 캐릭터", info="사전 제작된 1레벨 캐릭터입니다. 능력치·숙련·장비·직업 자원이 정해져 있습니다.")
            start_button = gr.Button("게임 시작 / 새 게임", variant="primary")
        with gr.Accordion("시작 전 캐릭터 시트 확인", open=False):
            preview = gr.Markdown(preview_character(characters[0][1]))
        stats = gr.HTML("<p>게임을 시작하면 HP·금액·현재 위치가 표시됩니다.</p>")
        with gr.Row():
            with gr.Column(scale=3):
                npc = gr.Dropdown(choices=[("장면 / 게임 마스터", "")], value="", label="현재 장소의 대화 상대")
                chat = gr.Chatbot(label="이야기", height=600, sanitize_html=True)
                recommendations = gr.Dataset(components=["textbox"], type="index", samples=[],
                                             label="다음에 해볼 말이나 행동 · 누르면 전송",
                                             layout="gallery", samples_per_page=3, visible=False)
                message = gr.Textbox(label="말 또는 행동", placeholder="말을 건네거나 **행동**을 입력하세요. 규칙 질문: /규칙 질문", lines=1, max_lines=6,
                                     elem_id="game-message", info="Enter 전송 · Shift+Enter 줄바꿈")
                with gr.Row():
                    action_mark = gr.Button("행동 표시 **", size="sm")
                    send_button = gr.Button("보내기", variant="primary")
                with gr.Row():
                    choice = gr.Dropdown(choices=[], label="현재 가능한 행동", scale=4)
                    action_button = gr.Button("행동 실행", scale=1)
                pending = gr.Markdown("대기 중인 판정이 없습니다.")
                with gr.Row():
                    roll_button = gr.Button("🎲 굴린다 / 실행", variant="primary", visible=False)
                    cancel_button = gr.Button("대기 행동 취소", size="sm")
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("가방"):
                        inventory = gr.Dataframe(headers=["휴대품", "수량"], datatype=["str", "number"], interactive=False, value=[])
                    with gr.Tab("캐릭터"):
                        equipment = gr.Markdown("장비가 표시됩니다.")
                    with gr.Tab("전투"):
                        combat = gr.Markdown("현재 전투가 없습니다.")
                        gr.Markdown("SRD 5.2 기반: 우선권·명중·치명타·피해·HP·구원·직업 자원. 현재 전투는 위치·사거리·기회 공격을 생략한 테스트 방식입니다.")
                    with gr.Tab("의뢰·저장"):
                        quest = gr.Markdown("진행 중인 의뢰가 표시됩니다.")
                        save_button = gr.Button("저장 파일 만들기")
                        download = gr.DownloadButton("저장 파일 다운로드", visible=False)
                        save_file = gr.File(label="저장 파일 선택", file_types=[".json"], type="filepath")
                        load_button = gr.Button("불러오기")
        status = gr.Textbox(label="실행 상태", interactive=False)
        refresh_button = gr.Button("상태 새로고침", size="sm")
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
        gr.Markdown("현재 시나리오 플레이 범위: 마을·우물 갱도. 개발자 시험 전투는 다른 조우도 불러옵니다.\n\nIncludes modified SRD 5.2 material by Wizards of the Coast LLC under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).")

        # render의 반환값과 화면 구성요소를 같은 순서로 연결합니다.
        # 아래 이벤트들은 이 공통 목록을 갱신하므로 한 동작 뒤 여러 영역이 함께 바뀝니다.
        view = [chat, stats, inventory, equipment, combat, choice, npc, pending, roll_button,
                status, debug, state_view, quest, logs, developer, dev_group, message, recommendations]
        # 한 게임 처리 흐름의 중복 실행을 줄이도록 대화·선택·굴림 이벤트를 같은 대기열에 묶습니다.
        group = {"concurrency_id": "gameplay", "concurrency_limit": 1}
        character.input(preview_character, [character], [preview], api_name="preview_character", queue=False)
        start_button.click(start, [character, token], [token, *view], api_name="start_game", **group)
        send_button.click(send, [message, token], view, api_name="send_message", **group)
        message.submit(send, [message, token], view, api_name="submit_message", **group)
        recommendations.click(choose_suggestion, [recommendations, token], view,
                              api_name="choose_suggestion", **group)
        action_button.click(choose, [choice, token], view, api_name="choose_action", **group)
        npc.input(select_npc, [npc, token], view, api_name="select_npc", **group)
        roll_button.click(roll, [token], view, api_name="roll_pending", **group)
        cancel_button.click(cancel, [token], view, api_name="cancel_pending", **group)
        developer.input(toggle_developer, [developer, token], view, api_name="toggle_developer", **group)
        dev_meet_button.click(dev_meet, [dev_npc, token], view, api_name="developer_meet", **group)
        dev_move_button.click(dev_move, [dev_room, token], view, api_name="developer_move", **group)
        dev_success_button.click(dev_success, [token], view, api_name="developer_success", **group)
        dev_win_button.click(dev_win, [token], view, api_name="developer_victory", **group)
        dev_heal_button.click(dev_heal, [token], view, api_name="developer_heal", **group)
        dev_combat_button.click(dev_combat, [dev_encounter, token], view, api_name="developer_combat", **group)
        check_button.click(free_check, [ability, token], view, api_name="free_check", **group)
        dice_button.click(utility_dice, [expression, token], view, api_name="utility_dice", **group)
        mode.input(rule_mode, [mode, token], view, api_name="rule_mode", **group)
        examples.click(lambda selected, session: send(selected[0], session), [examples, token], view,
                       api_name="send_example", **group)
        save_button.click(save, [token], [download], api_name="save_game", **group)
        load_button.click(restore, [save_file, token], view, api_name="load_game", **group)
        refresh_button.click(render, [token], view, api_name="refresh_state", **group)
        action_mark.click(fn=None, inputs=[message], outputs=[message],
                          js=api["ACTION_MARKER_JS"].replace("#player-message", "#game-message"), queue=False)
    return demo
