"""터미널에서 플레이한다 (기획서: 1~2주차는 터미널 화면으로 대체 가능). 장면 파일을 모두 이어서 진행.

  python -m game.cli                         # 시작할 때 캐릭터를 고름, LLM 없이 진행
  python -m game.cli --char harin --seed 7   # 캐릭터·주사위 시드 지정
  python -m game.cli --narrator ollama       # Ollama 서술 (모델은 --model, 고정하지 않음)

웹 화면(server/ + web/)과 같은 엔진을 쓰는 글자 화면이다. 규칙 처리는 하지 않고,
engine.available_choices()를 출력 → 키 입력 → engine.choose()를 반복할 뿐이다.
tools/play_script.py와 Colab 셀 [16]이 play()를 그대로 다시 쓴다.
"""
import argparse
import sys

from .content import Content
from .engine import Engine, fmt_money
from .narrator import OllamaNarrator, TemplateNarrator

HELP = ("입력 방법: 숫자 = 이 방의 행동, 영문자 a~e = 이동, p숫자 = 구입, u숫자 = 소모품 사용, t = 짧은 휴식, "
        "r = 새 소식 듣기, i/n = 영웅적 영감으로 다시 굴리기/그대로 두기, "
        "s = 상태, k = 알게 된 정보, h = 도움말, q = 종료\n"
        "  · 전투 중: 숫자 = 행동(공격·주문·회피 등), b숫자 = 보너스 행동, x숫자 = 물러나기 등, v = Action Surge, z = 차례 마침\n"
        "  · 같은 행동은 턴이 바뀌어도 같은 키를 씁니다 (끝난 행동은 목록에서 빠지고, 번호는 당겨지지 않습니다).\n"
        "  · 자유 문장 입력(예: '줄을 당겨본다')은 아직 지원하지 않습니다. 목록의 키를 입력하세요.")
LINE = "─" * 60


def print_events(events):
    """엔진 Event를 한 줄씩 출력 (flag 기록은 내부 정보라 숨김)."""
    for e in events:
        if e.kind == "flag":
            continue
        if e.kind == "scene":
            print(f"\n{e.text}")
            continue
        print(("  " if e.kind != "move" else "") + e.text)


def status(engine):
    """'s' 입력 — 위치·HP·상태·소지품 요약."""
    s = engine.s
    print(f"-- 장면 {s.scene} · {engine.c.room_name(s.room)} · 소지금 {fmt_money(s.gold_cp)} · 경과 {s.minutes // 60}시간"
          f" · 체크포인트: {engine.checkpoint_label}")
    for m in s.party:
        cond = ", ".join(k for k in m.conditions) or "-"
        insp = " · 영웅적 영감 있음" if m.heroic_inspiration else ""
        print(f"   {m.name} ({m.cls_ko} {m.level}레벨, {m.species}) HP {m.hp}/{m.max_hp}  AC {engine.armor_class(m)}"
              f"  Hit Point Dice {m.hd_left}/{m.hd_max}(d{m.hit_die})  상태: {cond}{insp}")
        if m.resources:
            print(f"   남은 사용 횟수: {m.resources}")
    if s.combat:
        foes = ", ".join(f"{x['name']}({engine.wound_text(x)})" for x in engine.living_enemies())
        print(f"   전투 중 ({s.combat['round']}라운드): {foes}")
    inv = ", ".join(f"{engine.c.items[i]['name']}×{n}" for i, n in s.inventory.items() if n) or "-"
    print(f"   소지품: {inv}")


def known(engine):
    """'k' 입력 — 지금까지 알게 된 카드 목록."""
    for cid in engine.player_known_cards():
        card = engine.c.cards[cid]
        print(f"  - [{card['label']}] {engine._view_text(cid)[0][:70]}…")


def pick_actor(engine, view, ask):
    # 보완 A: 판정이 없는 행동은 묻지 않고 추천 캐릭터가 한다. (1인에서는 항상 그 캐릭터)
    if len(view.actors) <= 1 or not view.check_text:
        return view.recommended or (view.actors[0] if view.actors else None)
    names = {m.id: m.name for m in engine.s.party}
    opts = " / ".join(f"{i + 1}.{names[a]}" for i, a in enumerate(view.actors))
    try:
        ans = ask(f"   누가 할까요? {opts} (Enter = 추천: {names[view.recommended]}) > ").strip().lstrip("﻿")
    except EOFError:
        ans = ""
    if ans.isdigit() and 1 <= int(ans) <= len(view.actors):
        return view.actors[int(ans) - 1]
    return view.recommended


def choose_character(content, ask):
    chars = content.characters()
    print("플레이할 캐릭터를 고르세요 (1인 1캐릭터):")
    for i, (_, name, cls_ko) in enumerate(chars, 1):
        print(f"  {i}. {name} ({cls_ko})")
    while True:
        try:
            ans = ask("> ").strip().lstrip("﻿")
        except EOFError:
            return None
        if ans.isdigit() and 1 <= int(ans) <= len(chars):
            return chars[int(ans) - 1][0]
        print(f"1~{len(chars)} 중 숫자를 입력하세요.")


def play(engine, narrator, ask=input):
    """게임 반복문. ask는 입력 함수라서, 테스트·play_script는 input 대신 정해진 답을 돌려주는 함수를 넣는다."""
    s = engine.s
    print(f"「잠긴 종」 장면 {s.scene} · {engine.c.scene_titles[s.scene]} — {engine.hero.name}({engine.hero.cls_ko})\n{HELP}\n")
    print_events(engine.opening)
    while True:
        if engine.s.ended == "game_over":
            print(f"\n{LINE}\n게임 오버. r = 체크포인트({engine.checkpoint_label})에서 다시 시작, q = 종료")
            try:
                ans = ask("> ").strip().lstrip("﻿").lower()
            except EOFError:
                break
            if ans == "r":
                print_events(engine.restart_from_checkpoint())
                continue
            break
        if engine.s.ended:
            break
        views = engine.available_choices()
        if not views:
            print("더 할 수 있는 행동이 없습니다.")
            break
        print(f"\n{LINE}")
        for v in views:
            extra = f"  [{v.check_text}]" if v.check_text else ""
            print(f" {v.key:>3}. {v.label}{extra}")
        try:
            ans = ask("> ").strip().lstrip("﻿").lower()
        except EOFError:  # 입력이 끝나면 조용히 종료
            print()
            break
        if ans == "q":
            break
        if ans == "s":
            status(engine)
            continue
        if ans == "k":
            known(engine)
            continue
        if ans in ("h", "?"):
            print(HELP)
            continue
        by_key = {v.key: v for v in views}
        if ans not in by_key:
            print(f"'{ans}'은(는) 지금 고를 수 있는 키가 아닙니다. 목록 왼쪽의 키를 입력하세요. (h = 도움말)")
            continue
        view = by_key[ans]
        actor = pick_actor(engine, view, ask)
        events = engine.choose(view.id, actor)
        print_events(events)
        text = narrator.narrate(engine, events, actor)
        if text:
            print(f"\n  ✦ {text}")
    if engine.s.ended and engine.s.ended != "game_over":
        print(f"\n=== 종료 ({engine.s.ended}) ===")
        status(engine)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--char", default=None, help="bran / harin / seo / maren (생략하면 시작할 때 선택)")
    p.add_argument("--narrator", choices=["template", "ollama"], default="template")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--combat", default=None,
                   help="테스트용: 이 조우로 바로 전투 시작 (rats, shriekers, crabs, ooze, zombies, armor, shadows, octopus, specter, ending2_hold)")
    p.add_argument("--potions", type=int, default=0, help="테스트용: 시작 물약 개수")
    a = p.parse_args(argv)
    narrator = OllamaNarrator(a.ollama_url, a.model) if a.narrator == "ollama" else TemplateNarrator()
    content = Content()
    if a.combat and a.combat not in content.encounters:
        print(f"없는 조우: {a.combat} (가능: {', '.join(content.encounters)})")
        return
    char_id = a.char or choose_character(content, input)
    if char_id is None:
        return
    engine = Engine(content=content, seed=a.seed, char_id=char_id)
    if a.potions:
        engine.s.inventory["potion_healing"] = a.potions
    if a.combat:  # 장면에 아직 연결되지 않은 전투를 직접 해 보기 위한 시험 모드 (W6)
        level = content.encounters[a.combat]["level"]
        engine.opening = (engine._level_up(engine.hero, level) +
                          [engine.opening[0]] + engine.start_combat(a.combat))
    play(engine, narrator)


if __name__ == "__main__":
    main()
