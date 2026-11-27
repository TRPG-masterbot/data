"""저장 JSON을 현재 게임에 적용하기 전에 검사하는 출력 파트의 계약입니다.

validate_save_payload(content, data)는 값을 읽기만 하며 오류는 ValueError로 알립니다.
원본 엔진·카드·입력 자료를 수정하거나 효과를 시험 실행하지 않습니다. 구조·참조
일관성 검사이며, 저장된 자연어의 의미나 게임을 실제로 플레이했는지를 증명하지는 않습니다.
"""
from dataclasses import MISSING, fields
import json
import math
import random

from campaign_flow import CONVERSATION_MILESTONES, INTRODUCTION_SOURCES
from game.content import npc_name
from game.model import ABILITIES, Character, GameState


# 원본 village_market 안의 두 화면 공간입니다. 인물 카드의 장소를 바꾸지 않습니다.
PRESENTATION_ROOM_BY_NPC = {
    "scn-m-npc-smith": "village_market:forge",
    "scn-m-npc-innkeeper": "village_market:inn",
}
_OUTCOMES = {"victory", "retreat", "avoid", "peace", "defeat"}


def _bad(path, reason):
    raise ValueError(f"저장 자료 {path}: {reason}")


def _mapping(value, path):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _bad(path, "문자열 키를 가진 사전이 필요합니다.")
    return value


def _integer(value, path, low=0, high=None):
    # bool은 Python에서 int의 하위 형식이지만 금액·차례로 받아들이지 않습니다.
    if type(value) is not int or value < low or (high is not None and value > high):
        _bad(path, "허용 범위의 정수가 필요합니다.")
    return value


def _text(value, path, nonempty=False, limit=20000):
    if not isinstance(value, str) or len(value) > limit or (nonempty and not value.strip()):
        _bad(path, "길이와 내용이 올바른 문자열이 필요합니다.")
    return value


def _boolean(value, path):
    if type(value) is not bool:
        _bad(path, "참 또는 거짓 값이 필요합니다.")


def _list(value, path):
    if not isinstance(value, list):
        _bad(path, "목록이 필요합니다.")
    return value


def _strings(value, path, allowed=None):
    for index, item in enumerate(_list(value, path)):
        _text(item, f"{path}[{index}]", nonempty=True)
        if allowed is not None and item not in allowed:
            _bad(f"{path}[{index}]", "현재 자료에 없는 참조입니다.")
    return value


def _counts(value, path, allowed=None, high=9999, nullable=False):
    for key, count in _mapping(value, path).items():
        if allowed is not None and key not in allowed:
            _bad(path + "." + key, "현재 자료에 없는 항목입니다.")
        if count is None and nullable:
            continue
        _integer(count, path + "." + key, high=high)


def _messages(records, path, content):
    for index, message in enumerate(_list(records, path)):
        location = f"{path}[{index}]"
        _mapping(message, location)
        if message.get("role") not in {"assistant", "user"}:
            _bad(location + ".role", "사용자 또는 응답 역할이 필요합니다.")
        _text(message.get("content"), location + ".content")
        context = message.get("scene_context")
        if context is not None:
            # 새 저장은 발언 당시 위치를 보존합니다. 옛 저장에 이 필드가 없으면
            # 통과시키며, 현재 방이나 안내 목적지로 과거 위치를 추정하지 않습니다.
            _mapping(context, location + ".scene_context")
            room = context.get("room_id")
            if (type(context.get("scene_id")) is not int or room not in content.rooms
                    or context["scene_id"] != content.rooms[room]["scene"]):
                _bad(location + ".scene_context", "저장된 발언의 장면과 장소가 일치하지 않습니다.")
            places = {room, *(value for value in PRESENTATION_ROOM_BY_NPC.values()
                              if value.partition(":")[0] == room)}
            if context.get("place_id") not in places:
                _bad(location + ".scene_context.place_id", "발언 장소에 속한 공간이어야 합니다.")
            _text(context.get("room_name"), location + ".scene_context.room_name", nonempty=True, limit=300)
        kind, speaker = message.get("kind"), message.get("speaker_id")
        if kind == "speech":
            if message["role"] != "assistant" or not npc_name(content.cards.get(speaker, {})):
                _bad(location, "NPC 발언의 화자를 확인할 수 없습니다.")
        if kind == "narration" and speaker not in (None, ""):
            _bad(location, "장면 설명에 NPC 화자가 지정되어 있습니다.")
        if kind == "guidance":
            if message["role"] != "assistant":
                _bad(location, "안내 사건은 시스템 응답이어야 합니다.")
            _text(message.get("guidance_id"), location + ".guidance_id", nonempty=True, limit=200)
            target = message.get("target_npc_id")
            if (target not in INTRODUCTION_SOURCES
                    or message.get("source_npc_id") != INTRODUCTION_SOURCES[target]
                    or message.get("room_id") != _target_room(content, target)):
                _bad(location, "안내 사건의 출처·대상·목적지가 일치하지 않습니다.")


def _target_room(content, identifier):
    return PRESENTATION_ROOM_BY_NPC.get(identifier, content.cards.get(identifier, {}).get("room"))


def _method(check, value, path):
    if value is not None:
        count = len(check["options"]) if "options" in check else 1
        _integer(value, path, high=count - 1)


def _pending_reference(content, state, identifier):
    """정의·대상 참조를 확인합니다. 남은 횟수와 실행 가능은 후보 엔진이 재확인합니다."""
    if identifier in content.choices:
        return content.choices[identifier].get("room") == state["room"]
    prefix, _, target = identifier.partition(":")
    if prefix in {"move", "revisit"}:
        return target in content.rooms
    if prefix in {"buy", "use"}:
        return target in content.items
    if prefix in {"wake", "kit", "aid"}:
        return target == state["party"][0]["id"]
    if prefix == "rest":
        return target == "short"
    if prefix == "talk":
        return bool(npc_name(content.cards.get(target, {})))
    if identifier.startswith("npc:attack:"):
        return identifier[len("npc:attack:"):] in content.npc_combat
    combat = state.get("combat")
    if prefix != "cmb" or not combat:
        return False
    parts, enemies = target.split(":"), combat["enemies"]
    if parts[0] in {"end", "dodge", "wind", "surge", "turn", "escape"}:
        return len(parts) == 1
    if parts[0] in {"unarmed", "subdue", "holy"}:
        return len(parts) == 2 and parts[1] in enemies
    if parts[0] == "attack":
        return len(parts) == 3 and parts[1] in content.rules["weapons"] and parts[2] in enemies
    if parts[0] == "cast":
        return (len(parts) == 3 and parts[1] in content.rules["spells"]
                and parts[2] in {"self", *enemies})
    if parts[0] == "use":
        return len(parts) == 2 and parts[1] in content.items
    if parts[0] == "option":
        return len(parts) == 2 and any(option["id"] == parts[1]
            for option in content.encounters[combat["enc"]].get("options", []))
    return False


def _introductions(content, values, messages, path):
    for target, receipt in _mapping(values, path).items():
        location = path + "." + target
        _mapping(receipt, location)
        if (target not in INTRODUCTION_SOURCES or receipt.get("target_npc_id") != target
                or receipt.get("source_npc_id") != INTRODUCTION_SOURCES[target]
                or receipt.get("room_id") != _target_room(content, target)):
            _bad(location, "소개 대상·직전 화자·목적지가 일치하지 않습니다.")
        index = _integer(receipt.get("message_index"), location + ".message_index")
        if index >= len(messages):
            _bad(location + ".message_index", "실제로 저장된 메시지를 가리켜야 합니다.")
        message = messages[index]
        if message.get("kind") == "guidance":
            if (not receipt.get("guidance_id") or any(message.get(key) != receipt.get(key)
                    for key in ("guidance_id", "source_npc_id", "target_npc_id", "room_id"))):
                _bad(location, "표시된 안내 사건과 소개 기록이 다릅니다.")
        else:
            # v20 저장의 호환 범위: 실제 출처 NPC의 완료된 발언이 있어야 합니다.
            # 이름 포함 검사는 문장 의미를 증명하지 않으므로 새 기록은 guidance를 씁니다.
            if (receipt.get("guidance_id") or message.get("role") != "assistant"
                    or message.get("kind") != "speech"
                    or message.get("speaker_id") != receipt["source_npc_id"]
                    or npc_name(content.cards[target]) not in message.get("content", "")
                    or message.get("generation_status") == "error"):
                _bad(location, "실제 출처 NPC 발언을 확인할 수 없습니다.")


def _conditions(values, path, enemy_ids=None):
    for name, value in _mapping(values, path).items():
        location = path + "." + name
        if value is None:
            continue
        if name == "dying":
            _mapping(value, location)
            if set(value) != {"success", "fail"}:
                _bad(location, "죽음 구원 성공·실패 횟수가 필요합니다.")
            for key in value:
                _integer(value[key], location + "." + key, high=2)
        elif name == "grappled":
            if not _mapping(value, location):
                _bad(location, "붙잡은 적 기록이 비어 있습니다.")
            for mid, grapple in value.items():
                _mapping(grapple, location + "." + mid)
                if enemy_ids is None or mid not in enemy_ids:
                    _bad(location, "현재 전투에 없는 적이 붙잡고 있습니다.")
                _integer(grapple.get("dc"), location + ".dc", 1, 100)
                _boolean(grapple.get("restrained"), location + ".restrained")
        elif name == "restrained" and value == "grapple":
            if not values.get("grappled"):
                _bad(location, "붙잡힘의 출처 기록이 없습니다.")
        else:
            _integer(value, location)


def _character(content, character, path, enemy_ids):
    _mapping(character, path)
    definitions = {item["id"]: item for item in content._chars}
    identifier = character.get("id")
    if identifier not in definitions:
        _bad(path + ".id", "현재 선택할 수 없는 캐릭터입니다.")
    required = {item.name for item in fields(Character) if item.default is MISSING and item.default_factory is MISSING}
    allowed = {item.name for item in fields(Character)}
    if not required <= set(character) or set(character) - allowed:
        _bad(path, "필수 캐릭터 항목이 없거나 알 수 없는 항목이 있습니다.")
    for key in ("name", "cls_ko", "species", "background"):
        _text(character[key], path + "." + key, nonempty=True, limit=200)
    cls = character.get("cls")
    if cls != definitions[identifier]["cls"] or cls not in content.rules["class_levels"]:
        _bad(path + ".cls", "캐릭터와 직업이 일치하지 않습니다.")
    level = _integer(character.get("level"), path + ".level", 1, 20)
    levels = content.rules["class_levels"][cls]
    if str(level) not in levels:
        _bad(path + ".level", "현재 룰 자료에 없는 레벨입니다.")
    abilities = _mapping(character.get("abilities"), path + ".abilities")
    if set(abilities) != set(ABILITIES):
        _bad(path + ".abilities", "여섯 능력치가 모두 필요합니다.")
    for key, score in abilities.items():
        _integer(score, path + ".abilities." + key, 1, 30)
    maximum = _integer(character.get("hp_max"), path + ".hp_max", 1, 10000)
    loss = _integer(character.get("hp_max_loss", 0), path + ".hp_max_loss", high=10000)
    _integer(character.get("hp"), path + ".hp", high=max(0, maximum - loss))
    _integer(character.get("hd_left"), path + ".hd_left", high=level)
    if character.get("hit_die", 8) not in {6, 8, 10, 12}:
        _bad(path + ".hit_die", "허용되지 않는 회복 주사위입니다.")
    for key in ("darkvision", "proficiency_bonus", "armor_penalty"):
        _integer(character.get(key, 0), path + "." + key, high=1000)
    for key in ("shield", "heroic_inspiration"):
        _boolean(character.get(key, False), path + "." + key)
    for key in ("fighting_style", "spellcasting", "concentration"):
        _text(character.get(key, ""), path + "." + key, limit=200)
    if character.get("spellcasting", "") not in {"", *ABILITIES}:
        _bad(path + ".spellcasting", "없는 주문 능력치입니다.")
    if character.get("armor", "none") not in content.rules["armor"]:
        _bad(path + ".armor", "현재 룰에 없는 방어구입니다.")
    for key in ("save_profs", "skill_profs", "tool_profs", "traits", "tools", "weapon_profs", "features"):
        _strings(character.get(key, []), path + "." + key, set(ABILITIES) if key == "save_profs" else None)
    for key, catalog in (("weapons", "weapons"), ("cantrips", "spells"), ("spells", "spells")):
        _strings(character.get(key, []), path + "." + key, content.rules[catalog])
    concentration = character.get("concentration", "")
    if concentration and (concentration not in content.rules["spells"]
                          or concentration not in character.get("conditions", {})):
        _bad(path + ".concentration", "집중 주문과 상태가 일치하지 않습니다.")
    full = {key: value for key, value in levels[str(level)].items() if key != "sneak_dice"}
    resources = _mapping(character.get("resources", {}), path + ".resources")
    if set(resources) != set(full):
        _bad(path + ".resources", "직업·레벨의 자원 항목과 다릅니다.")
    for key, maximum in full.items():
        value = resources[key]
        if isinstance(maximum, list):
            if len(_list(value, path + ".resources." + key)) != len(maximum):
                _bad(path + ".resources." + key, "슬롯 단계 수가 다릅니다.")
            for index, limit in enumerate(maximum):
                _integer(value[index], f"{path}.resources.{key}[{index}]", high=limit)
        else:
            _integer(value, path + ".resources." + key, high=maximum)
    _counts(character.get("weapon_penalty", {}), path + ".weapon_penalty", content.rules["weapons"])
    _counts(character.get("ability_loss", {}), path + ".ability_loss", set(ABILITIES))
    _conditions(character.get("conditions", {}), path + ".conditions", enemy_ids)


def _combat(content, value, path):
    if value is None:
        return None
    _mapping(value, path)
    required = {"enc", "round", "order", "idx", "enemies", "turn", "xp", "guided", "dodging", "on_end"}
    if set(value) != required or value.get("enc") not in content.encounters:
        _bad(path, "필수 전투 항목 또는 조우 참조가 올바르지 않습니다.")
    encounter = content.encounters[value["enc"]]
    expected = {f"{group['monster']}_{index + 1}": group["monster"]
                for group in encounter["monsters"] for index in range(group["count"])}
    enemies = _mapping(value["enemies"], path + ".enemies")
    if set(enemies) != set(expected):
        _bad(path + ".enemies", "조우의 적 목록과 일치하지 않습니다.")
    for mid, enemy in enemies.items():
        location = path + ".enemies." + mid
        _mapping(enemy, location)
        if set(enemy) != {"id", "kind", "name", "hp", "hp_max", "conditions", "used", "out"}:
            _bad(location, "필수 적 상태 항목이 없습니다.")
        if enemy["id"] != mid or enemy["kind"] != expected[mid]:
            _bad(location, "적 식별자와 몬스터 자료가 다릅니다.")
        maximum = content.monsters[expected[mid]]["hp"]
        _integer(enemy["hp_max"], location + ".hp_max", 1)
        if enemy["hp_max"] != maximum:
            _bad(location + ".hp_max", "몬스터의 기본 최대 HP와 다릅니다.")
        _integer(enemy["hp"], location + ".hp", high=maximum)
        _text(enemy["name"], location + ".name", nonempty=True, limit=200)
        if enemy["out"] not in {"", "dead", "fled", "unconscious"}:
            _bad(location + ".out", "알 수 없는 전투 이탈 상태입니다.")
        if (enemy["hp"] == 0) != (enemy["out"] == "dead"):
            _bad(location, "적 HP와 사망 상태가 다릅니다.")
        _conditions(enemy["conditions"], location + ".conditions")
        for key, used in _mapping(enemy["used"], location + ".used").items():
            if type(used) not in {bool, int} or (type(used) is int and used < 0):
                _bad(location + ".used." + key, "기능 사용 기록이 올바르지 않습니다.")
    order = _strings(value["order"], path + ".order", {"hero", *enemies})
    if len(order) != len(set(order)) or set(order) != {"hero", *enemies}:
        _bad(path + ".order", "각 전투 참가자가 한 번씩 필요합니다.")
    index = _integer(value["idx"], path + ".idx", high=len(order) - 1)
    _integer(value["round"], path + ".round", 1)
    # 서버는 적 차례를 끝낸 뒤 사용자 입력을 받으므로 저장 가능한 차례는 hero입니다.
    if order[index] != "hero":
        _bad(path + ".idx", "사용자 입력을 기다리는 차례가 아닙니다.")
    turn = _mapping(value["turn"], path + ".turn")
    if set(turn) != {"action", "bonus", "sneak", "surged", "slot_spent"}:
        _bad(path + ".turn", "현재 차례의 필수 항목이 없습니다.")
    _integer(turn["action"], path + ".turn.action", high=2)
    _integer(turn["bonus"], path + ".turn.bonus", high=1)
    for key in ("sneak", "surged", "slot_spent"):
        _boolean(turn[key], path + ".turn." + key)
    if turn["action"] > 1 and not turn["surged"]:
        _bad(path + ".turn.action", "추가 행동을 얻은 기록이 없습니다.")
    _integer(value["xp"], path + ".xp")
    if value["xp"] != sum(content.monsters[enemy["kind"]].get("xp", 0)
                          for enemy in enemies.values() if enemy["out"] == "dead"):
        _bad(path + ".xp", "쓰러뜨린 적의 경험치 합계와 다릅니다.")
    _boolean(value["dodging"], path + ".dodging")
    guided = value["guided"]
    if guided is not None:
        _mapping(guided, path + ".guided")
        if set(guided) != {"mid", "round"} or guided["mid"] not in enemies:
            _bad(path + ".guided", "안내된 공격 대상이 현재 적이 아닙니다.")
        _integer(guided["round"], path + ".guided.round", 1, value["round"])
    on_end = _mapping(value["on_end"], path + ".on_end")
    if set(on_end) - _OUTCOMES:
        _bad(path + ".on_end", "알 수 없는 전투 종료 결과입니다.")
    # 원본 장면에서 이 조우에 연결한 후속 효과만 허용합니다. 빈 값은 개발자 시험 전투입니다.
    candidates = [{}]
    def collect(node):
        if isinstance(node, dict):
            if node.get("combat") == value["enc"]:
                candidates.append({key[3:]: child for key, child in node.items() if key.startswith("on_")})
            for child in node.values():
                collect(child)
        elif isinstance(node, list):
            for child in node:
                collect(child)
    collect(content.scenes)
    if on_end not in candidates:
        _bad(path + ".on_end", "원본 장면에 없는 전투 후속 효과입니다.")
    return set(enemies)


def _state(content, state, messages, path):
    _mapping(state, path)
    if set(state) - {item.name for item in fields(GameState)}:
        _bad(path, "알 수 없는 게임 상태 항목이 있습니다.")
    room = state.get("room")
    if room not in content.rooms:
        _bad(path + ".room", "현재 자료에 없는 장소입니다.")
    scene = _integer(state.get("scene"), path + ".scene", 1)
    if scene not in content.scene_titles or scene < content.rooms[room]["scene"]:
        _bad(path + ".scene", "장소와 진행 장면이 일치하지 않습니다.")
    _integer(state.get("minutes", 0), path + ".minutes")
    _integer(state.get("gold_cp", 0), path + ".gold_cp")
    ended = _text(state.get("ended", ""), path + ".ended", limit=200)
    if ended and ended not in {"game_over", *(f"scene{number}_complete" for number in content.scene_titles)}:
        _bad(path + ".ended", "알 수 없는 종료 상태입니다.")
    enemy_ids = _combat(content, state.get("combat"), path + ".combat")
    if enemy_ids and ended:
        _bad(path, "종료된 게임에 진행 중 전투가 남아 있습니다.")
    party = _list(state.get("party"), path + ".party")
    if len(party) != 1:
        _bad(path + ".party", "현재 게임은 플레이어 캐릭터 한 명을 사용합니다.")
    _character(content, party[0], path + ".party[0]", enemy_ids)
    _counts(state.get("inventory", {}), path + ".inventory", content.items)
    _counts(state.get("stock", {}), path + ".stock", content.items, nullable=True)
    for key in ("flags", "done", "once_done"):
        _strings(state.get(key, []), path + "." + key)
    visited = _strings(state.get("visited", []), path + ".visited", content.rooms)
    if any(content.rooms[place]["scene"] > scene for place in visited):
        _bad(path + ".visited", "진행 장면보다 뒤의 방문 기록이 있습니다.")
    for card_id, actors in _mapping(state.get("shown", {}), path + ".shown").items():
        if card_id not in content.cards:
            _bad(path + ".shown", "현재 자료에 없는 카드입니다.")
        _strings(actors, path + ".shown." + card_id, {party[0]["id"]})
    for index, record in enumerate(_list(state.get("failed", []), path + ".failed")):
        if (not isinstance(record, (list, tuple)) or len(record) != 2
                or record[0] not in content.choices or record[1] != party[0]["id"]):
            _bad(f"{path}.failed[{index}]", "실패 판정과 캐릭터 참조가 올바르지 않습니다.")
    values = _mapping(state.get("vars", {}), path + ".vars")
    if values.get("last_long_rest_end") is not None:
        _integer(values["last_long_rest_end"], path + ".vars.last_long_rest_end", high=state.get("minutes", 0))
    _strings(values.get("campaign_dialogues", []), path + ".vars.campaign_dialogues", CONVERSATION_MILESTONES)
    _introductions(content, values.get("campaign_introductions", {}), messages, path + ".vars.campaign_introductions")
    for trap_id, trap in _mapping(state.get("traps", {}), path + ".traps").items():
        if trap_id not in content.traps:
            _bad(path + ".traps", "현재 자료에 없는 함정입니다.")
        _mapping(trap, path + ".traps." + trap_id)
        _integer(trap.get("count", 0), path + ".traps." + trap_id + ".count")
        _boolean(trap.get("disarmed", False), path + ".traps." + trap_id + ".disarmed")
    _mapping(state.get("deferred", {}), path + ".deferred")
    for record in _list(state.get("log", []), path + ".log"):
        _mapping(record, path + ".log[]")
        _text(record.get("choice"), path + ".log[].choice", nonempty=True, limit=300)
        if record.get("actor") is not None and record["actor"] != party[0]["id"]:
            _bad(path + ".log[].actor", "파티에 없는 행동자입니다.")
        _strings(record.get("events"), path + ".log[].events")
    pending = state.get("pending")
    if pending is not None:
        _mapping(pending, path + ".pending")
        if enemy_ids or ended:
            _bad(path + ".pending", "전투·종료와 동시에 비전투 결정을 기다릴 수 없습니다.")
        if pending.get("type") == "inspire":
            choice = content.choices.get(pending.get("choice"), {})
            if (not choice.get("check") or pending.get("actor") != party[0]["id"]
                    or not party[0].get("heroic_inspiration") or choice.get("room") != room):
                _bad(path + ".pending", "사용할 영감·판정·행동자가 일치하지 않습니다.")
            _method(choice["check"], pending.get("method_index"), path + ".pending.method_index")
        elif pending.get("type") == "hit_dice":
            if set(pending) != {"type"} or party[0]["hp"] < 1 or party[0]["hd_left"] < 1:
                _bad(path + ".pending", "짧은 휴식 회복 주사위를 사용할 상태가 아닙니다.")
        else:
            _bad(path + ".pending.type", "알 수 없는 대기 상태입니다.")


def _validate_save_payload(content, data):
    """저장을 순수하게 검사합니다. 성공은 None, 실패는 항목 경로를 담은 ValueError입니다."""
    _mapping(data, "save")
    if data.get("schema") != "trpg_game_save_1":
        _bad("schema", "지원하지 않는 저장 형식입니다.")
    try:
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError, RecursionError, OverflowError) as error:
        raise ValueError("저장 자료는 순환·NaN이 없는 JSON 값이어야 합니다.") from error
    _integer(data.get("revision", 0), "revision")
    _text(data.get("npc", ""), "npc", limit=200)
    _boolean(data.get("scene_focus_explicit", False), "scene_focus_explicit")
    messages = data.get("messages", [])
    _messages(messages, "messages", content)
    histories = _mapping(data.get("histories", {}), "histories")
    for name, history in histories.items():
        _messages(history, "histories." + name, content)
    _state(content, data.get("state"), messages, "state")
    # RNG는 새 임시 난수기에만 적용합니다. 실제 세션의 다음 굴림을 바꾸지 않습니다.
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, list) else value
    try:
        rng = data["rng"]
        if (not isinstance(rng, (list, tuple)) or len(rng) != 3 or type(rng[0]) is not int
                or rng[0] != 3 or not isinstance(rng[1], (list, tuple)) or len(rng[1]) != 625
                or any(type(item) is not int for item in rng[1])
                or any(not 0 <= item <= 2**32 - 1 for item in rng[1][:-1])
                or not 0 <= rng[1][-1] <= 624
                or (rng[2] is not None and (type(rng[2]) not in {int, float} or not math.isfinite(rng[2])))):
            _bad("rng", "난수기 상태 형식이 올바르지 않습니다.")
        random.Random(0).setstate(tuples(rng))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("저장 자료 rng: 다음 주사위 상태를 복원할 수 없습니다.") from error
    pending = data.get("pending_choice")
    if pending is not None:
        _mapping(pending, "pending_choice")
        if pending.get("type") != "choice" or data["state"].get("pending"):
            _bad("pending_choice", "동시에 두 결정을 기다릴 수 없습니다.")
        _text(pending.get("choice"), "pending_choice.choice", nonempty=True, limit=300)
        _integer(pending.get("revision"), "pending_choice.revision")
        if pending["revision"] != data.get("revision", 0):
            _bad("pending_choice.revision", "현재 저장 상태와 다른 시점의 판정입니다.")
        _boolean(pending.get("roll"), "pending_choice.roll")
        if pending.get("actor_id") != data["state"]["party"][0]["id"]:
            _bad("pending_choice.actor_id", "현재 캐릭터와 다릅니다.")
        choice = content.choices.get(pending["choice"], {})
        if choice.get("check"):
            _method(choice["check"], pending.get("method_index"), "pending_choice.method_index")
        elif pending.get("method_index") is not None:
            _bad("pending_choice.method_index", "원본 판정 수단이 없는 행동입니다.")
        if not _pending_reference(content, data["state"], pending["choice"]):
            _bad("pending_choice.choice", "현재 자료에 없는 행동 또는 대상입니다.")
        # 현재 choices·수단 존재 여부는 원본 엔진을 읽는 후보 세션의 검사와 함께 확정합니다.
    checkpoint = data.get("engine_checkpoint")
    if checkpoint is not None:
        _mapping(checkpoint, "engine_checkpoint")
        if set(checkpoint) != {"state", "label"}:
            _bad("engine_checkpoint", "상태 사본과 지점 이름이 필요합니다.")
        _text(checkpoint["label"], "engine_checkpoint.label", nonempty=True, limit=300)
        _state(content, checkpoint["state"], messages, "engine_checkpoint.state")


def validate_save_payload(content, data):
    """검사 실패는 항상 ValueError입니다. 예상과 다른 중첩 참조 형식도 적용하지 않습니다."""
    try:
        return _validate_save_payload(content, data)
    except (TypeError, KeyError, IndexError, AttributeError, RecursionError, OverflowError) as error:
        raise ValueError("저장 자료의 중첩 항목·참조 형식이 올바르지 않습니다.") from error
