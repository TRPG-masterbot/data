"""현재 게임 상태를 선택 카드·판정·전투 화면으로 표현합니다.

굴림·성공 여부·HP는 계산하지 않고 엔진의 정의와 확정 사건만 읽습니다.
인물·장소의 공개 여부는 gameplay.choices가 결정합니다.
"""
from html import escape
from pathlib import Path
from urllib.parse import quote


def discovery_data(game):
    """진행 연결부가 공개한 정보만 화면에 사용합니다."""
    reader = getattr(game, "discovery_view", None)
    if callable(reader):
        return reader()
    # 초기 준비·이전 저장 파일에서도 새 인물이나 경로를 임의로 노출하지 않습니다.
    locations = game.map_locations()
    people = [{**person, "here": location.get("current") is True,
               "room_id": location["id"], "room_name": location["name"]}
              for location in locations for person in location.get("npcs", [])]
    return {"current_room": {"id": game.engine.s.room,
                             "name": game.content.room_name(game.engine.s.room)},
            "people": people, "routes": locations, "actions": [], "objective": {}}


def discovery_samples(game, data=None):
    """클릭할 문구도 현재 공개 목록에서 만들고, 클릭 때 같은 목록을 다시 확인합니다."""
    data = discovery_data(game) if data is None else data
    routes = [["[이동] " + row["name"]] for row in data.get("routes", [])
              if not row.get("current") and row.get("reachable")]
    people = [[("[대화] " if row.get("here") and row.get("can_meet") else
                "[찾아가기] " if row.get("can_meet") else "[알게 된 인물] ")
               + row["name"] + (" · " + row.get("room_name", "") if row.get("room_name") else "")]
              for row in data.get("people", []) if row.get("can_meet")]
    return routes, people


def discovery_overview(game, data=None):
    """현재 장소·현장의 인물·현재 목표를 짧은 안내로 표시합니다."""
    data = discovery_data(game) if data is None else data
    here = [row["name"] for row in data.get("people", []) if row.get("here")]
    objective = data.get("objective") or {}
    goal = objective.get("summary") or objective.get("title") or "주변을 살피며 이야기를 이어가세요."
    if "requests" in data and not data["requests"]:
        goal = "눈앞의 인물과 이야기를 시작하세요."
    return ('<div class="discovery-overview"><span class="discovery-eyebrow">현재 주변</span>'
            '<b>' + escape(" · ".join(here) if here else "지금 대화할 인물이 없습니다") + '</b>'
            '<span class="discovery-goal">' + escape(goal) + '</span></div>')


def people_panel(game, data=None):
    """알게 된 인물의 공개 위치만 읽기 전용으로 표시합니다. 만남 실행은 별도 선택지입니다."""
    from portrait_view import _portrait_url
    rows = []
    data = discovery_data(game) if data is None else data
    for person in data.get("people", []):
        image = _portrait_url(person.get("name", ""), person.get("card_id"))
        portrait = '<img class="known-person-portrait" alt="" src="' + escape(image, quote=True) + '">' if image else ''
        where = "현재 장소에 있음" if person.get("here") else person.get("room_name") or "위치 미확인"
        state = "대화 가능" if person.get("here") and person.get("can_meet") else "알게 된 인물"
        rows.append('<article class="known-person">' + portrait + '<div><h3>'
                    + escape(str(person.get("name", "인물"))) + '</h3><p>' + escape(str(where))
                    + '</p><small>' + state + '</small></div></article>')
    return '<div class="known-people">' + (''.join(rows) or '<p class="empty-note">아직 만난 인물이 없습니다.</p>') + '</div>'


def discovery_actions(game, data=None):
    """실제 공개된 단서와 사건만 표시하며 인물 카드 전문을 다시 내보내지 않습니다."""
    data = discovery_data(game) if data is None else data
    clues = []
    for clue in data.get("clues", []):
        if not isinstance(clue, dict):
            continue
        title = str(clue.get("title") or "발견한 단서")
        text = clue.get("summary") or clue.get("text") or ""
        clues.append('<article class="clue-card"><h3>' + escape(title) + '</h3>'
                     + ('<p>' + escape(str(text)) + '</p>' if text else '') + '</article>')
    body = '<div class="clue-list">' + (''.join(clues) or '<p class="empty-note">아직 기록된 단서가 없습니다. 직접 듣거나 발견한 정보가 여기에 쌓입니다.</p>') + '</div>'
    # 행동 조건도 진행 연결부가 공개한 문장만 사용하며 내부 조건·카드 전문은 읽지 않습니다.
    rows = []
    for action in data.get("actions", []):
        requirements = action.get("requires")
        if isinstance(requirements, str):
            requirements = [requirements]
        elif not isinstance(requirements, (list, tuple)):
            # 진행 연결부의 bool은 조건 유무 표지입니다. 설명 목록처럼 순회하거나
            # True/False를 플레이어에게 출력하지 않고, 전달된 설명이 있을 때만 씁니다.
            requirements = []
        detail = " · ".join(str(value) for value in requirements if type(value) in (str, int, float))
        if isinstance(action.get("check"), str):
            detail = " · ".join(value for value in (detail, action["check"]) if value)
        rows.append('<div class="discovery-action"><b>' + escape(action.get("label", "행동"))
                    + '</b>' + ('<small>' + escape(detail) + '</small>' if detail else '') + '</div>')
    return body + '<details class="known-action-details"><summary>지금 가능한 행동과 조건</summary><div class="discovery-action-list">' + ("".join(rows) or '<p>진행 중인 선택을 먼저 마쳐 주세요.</p>') + '</div></details>'


def scene_background(game, data=None):
    """그림은 생성 문장이 아니라 엔진의 실제 장소와 현재 대화 상대에 연결합니다."""
    room = game.engine.s.room
    if room == "village_square":
        column, line, zone = 0, 0, "village"
    elif room in {"halvik_house", "halvik_study"}:
        column, line, zone = 1, 0, "house"
    elif room == "village_market":
        if data is None:
            reader = getattr(game, "discovery_view", None)
            shown_room = reader().get("current_room", {}).get("id", room) if callable(reader) else room
        else:
            shown_room = data.get("current_room", {}).get("id", room)
        in_inn = shown_room == "village_market:inn" or (shown_room == room and game.npc == "오단")
        column, line, zone = (0, 1, "inn") if in_inn else (2, 0, "forge")
    elif room == "well_side":
        opened = ("ws_open_well" in game.engine.s.done
                  or "well_entrance_known" in game.engine.s.flags)
        column, line, zone = (1, 1, "well") if opened else (0, 0, "village")
    elif room in {"shaft_upper", "ledge_60", "shaft_lower"}:
        column, line, zone = 2, 1, "shaft"
    else:
        column, line, zone = 0, 0, "unknown"
    folder = Path(__file__).resolve().parent / "assets/backgrounds"
    # 우물 안의 세 지점은 같은 좁은 우물 그림을 사용하며 이후 장소를 미리 보여주지 않습니다.
    background = folder / ("well-interior-v12.webp" if zone == "shaft" else "echo_town_scenes.webp")
    style = ""
    if background.is_file() and zone != "unknown":
        url = "./gradio_api/file=" + quote(background.as_posix(), safe="/:")
        placement = "background-size:cover;background-position:center" if zone == "shaft" else "background-position:" + str(column * 50) + "% " + str(line * 100) + "%"
        style = ' style="background-image:url(\'' + escape(url, quote=True) + '\');' + placement + '"'
    return '<div class="scene-art scene-' + zone + '"' + style + '></div><div class="scene-shade"></div>'


def action_entries(game):
    """현재 실행 가능한 행동에 표시용 분류와 판정 설명을 붙입니다."""
    rows = []
    if game.pending or game.engine.s.ended:
        return rows
    for choice in game.choices():
        if choice.id.startswith("npc:attack:"):
            continue  # 자유 입력의 공격 선언과 별도로, 기본 진행 카드는 탐사에 집중합니다.
        definition = game.content.choices.get(choice.id, {})
        tag = ("판정" if choice.check_text else "전투" if choice.id.startswith("cmb:") else
               "이동" if choice.id.startswith("move:") else
               "대화" if choice.id.startswith("talk:") or definition.get("interaction") == "talk" else "행동")
        if choice.id.startswith("meet:"):
            card = game.content.cards.get(choice.id[5:], {})
            tag = "대화" if card.get("room") == game.engine.s.room else "이동"
        title = choice.label.removeprefix("이동: ") if choice.id.startswith("move:") else choice.label
        rows.append({"id": choice.id, "label": f"[{tag}] {title}",
                     "detail": choice.check_text or ("선택 후 주사위를 굴립니다" if game._needs_roll(choice.id) else "")})
    # 추천된 실행 행동도 같은 카드로 보여 줍니다. 원본 선택지나 추천 목록은
    # 바꾸지 않고 표시 순서만 앞당겨, 다음 페이지에 숨은 추천이 생기지 않게 합니다.
    suggested = getattr(game, "suggestions", []) or []
    priority = {}
    for item in suggested[:3]:
        if isinstance(item, dict) and isinstance(item.get("action_id"), str):
            priority.setdefault(item["action_id"], len(priority))
    rows.sort(key=lambda row: priority.get(row["id"], len(priority)))
    return rows


def inventory_panel(game):
    """현재 휴대품만 작은 목록으로 표시합니다. 장착 장비는 캐릭터 탭에 둡니다."""
    rows = [(game.content.items[key]["name"], count)
            for key, count in game.engine.s.inventory.items() if count > 0]
    body = "".join(f'<div class="pack-row"><span>{escape(name)}</span><b>{count}</b></div>'
                   for name, count in rows)
    return '<div class="compact-pack">' + (body or '<p>가방이 비어 있습니다.</p>') + '</div>'


def dice_panel(game):
    """대기 판정·확정 굴림을 표시하고 개발자 지정 결과는 실제 주사위 값과 구분합니다."""
    pending = game.pending
    if pending:
        definition = game.content.choices.get(pending.get("choice"), {})
        check = definition.get("check")
        if pending.get("type") == "free_check":
            check = {"ability": pending["ability"], "dc": 15}
        details = escape(pending.get("check_text") or "선택한 행동을 실행합니다")
        target, formula = "행동 확인", "진행할 준비가 됐습니다"
        if check:
            actor = game.engine.s.member(pending.get("actor_id", game.engine.hero.id))
            selected = game.engine.selected_check(actor, check, pending.get("method_index"))
            bonus, name = game.engine.check_mod(actor, selected)
            target, formula = f"난이도 {selected['dc']}", f"d20 {bonus:+d} · {name}"
            details = "굴림 합계가 난이도 이상이면 성공합니다"
        elif pending.get("roll"):
            target, formula = "전투 판정", "선택한 공격·주문의 규칙을 적용합니다"
        return (f'<div class="dice-stage" aria-label="주사위 판정">'
                f'<div class="dice-glyph">{ "d20" if pending.get("roll") else "◆" }</div>'
                f'<div><div class="dice-eyebrow">{escape(target)}</div>'
                f'<h3>{escape(pending["label"])}</h3><b>{escape(formula)}</b><p>{details}</p></div></div>')
    records = (game.last_result or {}).get("public_event_records", [])
    candidates = [row for row in records if row.get("kind") == "check"
                 and isinstance(row.get("data"), dict)
                 and "roll" in row["data"] and ("success" in row["data"] or "hit" in row["data"])]
    roll = next((row for row in candidates if row["data"].get("side") != "enemy"),
                candidates[0] if candidates else None)
    if not roll:
        return ""
    data = roll["data"]
    success = data.get("success", data.get("hit")) is True
    result = "치명타" if data.get("crit") else "성공" if success else "실패"
    owner = "적의 굴림 · " if data.get("side") == "enemy" else ""
    override = (" · 개발자 결과 지정: " + ("성공" if success else "실패")
                if data.get("developer_override") else "")
    threshold = (f"난이도 {data['dc']}" if "dc" in data else f"방어도 {data['ac']}")
    return (f'<div class="dice-result {"success" if success else "failure"}">'
            f'<span class="dice-result-number">{escape(str(data["roll"]))}</span><div>'
            f'<b>{owner}{result}{override}</b><span>합계 {escape(str(data["total"]))} / {escape(threshold)}</span>'
            f'<small>{escape(roll["text"])}</small></div></div>')


def monster_portrait(enemy):
    """실제 몬스터 종류에 해당하는 작은 공용 그림 한 칸만 표시합니다."""
    # 768px WebP 한 장의 256px 칸 순서입니다. 이름이나 생성 문장으로 추측하지 않습니다.
    kinds = (
        "swarm_of_rats", "shrieker_fungus", "giant_crab",
        "gray_ooze", "zombie", "animated_armor",
        "shadow", "giant_octopus", "specter",
    )
    kind = enemy.get("kind")
    name = str(enemy.get("name") or "몬스터")
    atlas = Path(__file__).resolve().parent / "assets" / "monsters" / "encounter-monsters.webp"
    if kind not in kinds or not atlas.is_file():
        # 알 수 없는 종류나 누락된 파일을 다른 몬스터의 그림으로 대신하지 않습니다.
        return ('<span class="monster-portrait monster-portrait-fallback" role="img" '
                'aria-label="' + escape(name + " 이미지 없음", quote=True) + '">'
                '<span aria-hidden="true">이미지 없음</span></span>')
    index = kinds.index(kind)
    column, row = index % 3, index // 3
    url = "./gradio_api/file=" + quote(atlas.as_posix(), safe="/:")
    style = ("background-image:url('" + url + "');background-size:300% 300%;"
             + f"background-position:{column * 50}% {row * 50}%")
    return ('<span class="monster-portrait" role="img" aria-label="'
            + escape(name + " 모습", quote=True) + '" style="'
            + escape(style, quote=True) + '"></span>')


def encounter_panel(game):
    """전투의 차례 자원과 적의 실제 상태를 별도 전장 요약으로 만듭니다."""
    engine, combat = game.engine, game.engine.s.combat
    if not combat:
        return ""
    turn = combat.get("turn", {})
    # 같은 조우 안에서 쓰러진 대상도 남겨 명중 연출이 실제 대상에 연결되게 합니다.
    # 전투가 완전히 끝나면 위에서 빈 화면을 반환하며 이전 적을 새로 만들지 않습니다.
    roster = combat.get("enemies")
    current_enemies = roster.values() if isinstance(roster, dict) else engine.living_enemies()
    enemy_tokens = []
    for enemy in current_enemies:
        out = str(enemy.get("out") or "")
        wound = ({"dead": "쓰러짐", "fled": "달아남", "unconscious": "의식 없음"}.get(out, "전투에서 벗어남")
                 if out else engine.wound_text(enemy))
        token_class = "enemy-token is-out" if out else "enemy-token"
        enemy_tokens.append(
            '<div class="' + token_class + '" data-enemy-id="'
            + escape(str(enemy.get("id") or ""), quote=True) + '" data-enemy-kind="'
            + escape(str(enemy.get("kind") or ""), quote=True) + '" data-enemy-state="'
            + escape(out or "active", quote=True) + '">'
            + monster_portrait(enemy) + '<div class="enemy-caption"><b>'
            + escape(enemy["name"]) + '</b><small>' + escape(wound) + '</small></div></div>')
    enemies = "".join(enemy_tokens)
    return (f'<div class="encounter-board"><div class="encounter-heading">'
            f'<b>{escape(game.content.encounters[combat["enc"]]["name"])}</b>'
            f'<span>{combat["round"]} 라운드 · 당신의 차례</span></div>'
            f'<div class="enemy-line">{enemies}</div>'
            f'<div class="turn-budget">행동 <b>{turn.get("action", 0)}</b>'
            f' · 보너스 행동 <b>{turn.get("bonus", 0)}</b>'
            f' · HP <b>{engine.hero.hp}/{engine.hero.max_hp}</b></div></div>')


def journey_panel(game, data=None):
    """공개·수락 사건으로 확인된 의뢰만 진행 중과 완료 이력으로 나눕니다."""
    data = discovery_data(game) if data is None else data
    reader = getattr(game, "request_view", None)
    requests = reader() if callable(reader) else data.get("requests", [])
    requests = requests if isinstance(requests, list) else []
    objective = data.get("objective") or {}
    active, completed = [], []
    states = {"offered": "부탁을 들음", "accepted": "진행 중", "completed": "완료"}
    for request in requests:
        if not isinstance(request, dict) or request.get("status") not in states:
            continue
        status = request["status"]
        issuer = request.get("issuer") or {}
        title = request.get("title") or issuer.get("name", "마을 사람") + "의 부탁"
        purpose = request.get("purpose") or ""
        card = '<article class="quest-card ' + status + '"><div class="quest-card-header"><h3>' + escape(str(title)) + '</h3><span>' + states[status] + '</span></div>'
        if issuer.get("name"):
            card += '<small>' + escape(str(issuer["name"])) + '</small>'
        if purpose:
            card += '<p>' + escape(str(purpose)) + '</p>'
        card += '</article>'
        (completed if status == "completed" else active).append(card)
    if requests:
        title = objective.get("title") or "현재 목표"
        next_step = objective.get("next_step") or objective.get("summary") or "주변에서 다음 단서를 찾아보세요."
    else:
        title, next_step = "이야기를 시작하세요", "눈앞의 인물에게 부탁의 내용을 들어보세요."
    top = '<article class="objective-card"><span>현재 목표</span><h2>' + escape(str(title)) + '</h2><p><b>다음 단계</b> ' + escape(str(next_step)) + '</p></article>'
    history = '<details class="quest-history"><summary>완료한 의뢰 ' + str(len(completed)) + '개</summary>' + (''.join(completed) or '<p class="empty-note">아직 완료한 의뢰가 없습니다.</p>') + '</details>'
    return top + '<div class="quest-list">' + (''.join(active) or '<p class="empty-note">진행 중인 의뢰가 없습니다.</p>') + '</div>' + history
