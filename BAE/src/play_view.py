"""현재 게임 상태를 선택 카드·판정·전투 화면으로 표현합니다.

굴림·성공 여부·HP는 계산하지 않고 엔진의 정의와 확정 사건만 읽습니다.
인물·장소의 공개 여부는 gameplay.choices가 결정합니다.
"""
from html import escape


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
    return rows


def inventory_panel(game):
    """현재 휴대품만 작은 목록으로 표시합니다. 장착 장비는 캐릭터 탭에 둡니다."""
    rows = [(game.content.items[key]["name"], count)
            for key, count in game.engine.s.inventory.items() if count > 0]
    body = "".join(f'<div class="pack-row"><span>{escape(name)}</span><b>{count}</b></div>'
                   for name, count in rows)
    return '<div class="compact-pack">' + (body or '<p>가방이 비어 있습니다.</p>') + '</div>'


def dice_panel(game):
    """대기 중인 판정 또는 이번 행동에서 확정한 굴림을 표시합니다."""
    pending = game.pending
    if pending:
        definition = game.content.choices.get(pending.get("choice"), {})
        check = definition.get("check")
        if pending.get("type") == "free_check":
            check = {"ability": pending["ability"], "dc": 15}
        details = escape(pending.get("check_text") or "선택한 행동을 실행합니다")
        target, formula = "행동 확인", "진행할 준비가 됐습니다"
        if check:
            selected = game.engine.best_option(game.engine.hero, check)
            bonus, name = game.engine.check_mod(game.engine.hero, selected)
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
    override = ' · 개발자 성공 보장' if data.get("developer_override") else ''
    threshold = (f"난이도 {data['dc']}" if "dc" in data else f"방어도 {data['ac']}")
    return (f'<div class="dice-result {"success" if success else "failure"}">'
            f'<span class="dice-result-number">{escape(str(data["roll"]))}</span><div>'
            f'<b>{owner}{result}{override}</b><span>합계 {escape(str(data["total"]))} / {escape(threshold)}</span>'
            f'<small>{escape(roll["text"])}</small></div></div>')


def encounter_panel(game):
    """전투의 차례 자원과 적의 실제 상태를 별도 전장 요약으로 만듭니다."""
    engine, combat = game.engine, game.engine.s.combat
    if not combat:
        return ""
    turn = combat.get("turn", {})
    enemies = "".join(f'<div class="enemy-token"><span>⚔</span><div><b>{escape(enemy["name"])}</b>'
                      f'<small>{escape(engine.wound_text(enemy))}</small></div></div>'
                      for enemy in engine.living_enemies())
    return (f'<div class="encounter-board"><div class="encounter-heading">'
            f'<b>{escape(game.content.encounters[combat["enc"]]["name"])}</b>'
            f'<span>{combat["round"]} 라운드 · 당신의 차례</span></div>'
            f'<div class="enemy-line">{enemies}</div>'
            f'<div class="turn-budget">행동 <b>{turn.get("action", 0)}</b>'
            f' · 보너스 행동 <b>{turn.get("bonus", 0)}</b>'
            f' · HP <b>{engine.hero.hp}/{engine.hero.max_hp}</b></div></div>')


def journey_panel(game):
    """현재 위치와 지금까지 알게 된 의뢰만 표시합니다."""
    text = game.quest_journal() or "낡은 인형을 안은 여인의 이야기를 들어보세요."
    # Markdown은 별도 컴포넌트에서 렌더합니다. HTML에 사용자 대사를 넣지 않습니다.
    return text
