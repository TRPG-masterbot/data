"""선택한 캐릭터의 시작 장비·휴대품·금액을 준비합니다.

입력: 새 게임 엔진과 data/starting_equipment.json의 캐릭터별 장비 구성입니다.
처리: 직업·배경의 장비 묶음을 합친 뒤 휴대품과 장착 장비의 ID·수량을 확인합니다.
출력: GameState의 inventory·gold_cp·vars.setup을 채우고 최초 적용 여부를 반환합니다.

gameplay.py의 새 게임 준비 과정에서 호출합니다. setup에는 시작 구성과 미휴대 물품도
기록하며, 불러온 게임에 시작 장비가 다시 지급되지 않도록 적용 표식을 확인합니다."""
from collections import Counter
import json
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent / "data" / "starting_equipment.json"


def apply_starting_equipment(engine):
    """정의된 시작 장비를 새 게임의 휴대품·장착 기록·금액으로 반영합니다.

    입력은 캐릭터가 선택된 engine입니다. 직업·배경 장비 묶음을 합친 뒤 ID와 수량,
    휴대·장착 물품의 중복 배정을 확인하고 state에 기록합니다.
    처음 적용하면 True, 이미 적용 표식이 있으면 False를 반환합니다.
    setup에는 전체 시작 장비 중 현재 휴대하지 않는 물품도 따로 남깁니다."""
    state = engine.s
    setup = state.vars.get("setup") or {}
    if not isinstance(setup, dict):
        raise ValueError("시작 설정 기록이 올바르지 않습니다.")
    if setup.get("starting_equipment_applied"):
        return False
    if state.inventory or state.gold_cp:
        raise ValueError("이미 소지품이나 금액이 있는 상태에는 시작 장비를 다시 지급할 수 없습니다.")

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    character_id = engine.hero.id
    selected = data["characters"].get(character_id)
    if selected is None:
        raise ValueError("시작 장비가 정의되지 않은 캐릭터입니다: " + character_id)

    inventory = Counter()
    gold_cp = 0

    def add_items(values):
        for item_id, quantity in values.items():
            if item_id not in engine.c.items or type(quantity) is not int or quantity <= 0:
                raise ValueError("시작 소지품의 ID 또는 수량이 올바르지 않습니다: " + item_id)
        inventory.update(values)

    for package_id in selected["packages"]:
        package = data["packages"][package_id]
        if type(package["gold_cp"]) is not int or package["gold_cp"] < 0:
            raise ValueError("시작 금액이 올바르지 않습니다.")
        gold_cp += package["gold_cp"]
        add_items(package["items"])
        for pack_id in package.get("packs", []):
            add_items(data["packs"][pack_id]["items"])

    if type(gold_cp) is not int or gold_cp < 0:
        raise ValueError("시작 금액이 올바르지 않습니다.")
    for item_id, quantity in inventory.items():
        if item_id not in engine.c.items or type(quantity) is not int or quantity <= 0:
            raise ValueError("시작 소지품의 ID 또는 수량이 올바르지 않습니다: " + item_id)

    carried = selected.get("carry_selection", dict(inventory))
    equipped = selected.get("equipped_selection", {})
    for group in [carried, equipped]:
        if not isinstance(group, dict) or any(item not in inventory or type(quantity) is not int or quantity < 1
                                              or quantity > inventory[item] for item, quantity in group.items()):
            raise ValueError("휴대·장착 구성은 시작 장비의 실제 수량 안에서 선택해야 합니다.")
    used = Counter(carried) + Counter(equipped)
    if any(quantity > inventory[item] for item, quantity in used.items()):
        raise ValueError("같은 시작 물품을 휴대품과 장착 장비에 중복 배정했습니다.")
    state.inventory = dict(carried)
    state.gold_cp = gold_cp
    state.vars["setup"] = {
        "ruleset": data["ruleset"],
        "schema_version": data["schema_version"],
        "character_id": character_id,
        "starting_equipment_applied": True,
        "packages": list(selected["packages"]),
        "starting_gold_cp": gold_cp,
        "equipped_items": dict(equipped),
        "not_carried": {item: quantity - used[item] for item, quantity in inventory.items() if quantity > used[item]},
        "carry_note": selected.get("carry_note", ""),
    }
    return True
