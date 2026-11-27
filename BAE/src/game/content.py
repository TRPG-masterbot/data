"""JSON으로 작성된 게임 자료를 읽어 엔진이 찾기 쉬운 형태로 묶습니다.

입력: game/data의 카드·장면·아이템·규칙·몬스터·캐릭터 파일입니다.
scene_paths를 지정하면 선택한 장면 파일들을 읽고, 기본값은 scene*.json 전체입니다.
처리: 방과 선택지를 ID별 사전으로 합치고 중복 ID·장면 번호·함정 참조를 검사합니다.
인물 전투 정의가 있으면 해당 카드에 연결된 몬스터·조우 자료도 준비합니다.
출력: Engine과 CombatMixin이 self.c로 읽는 Content 객체입니다.

가격·규칙·장면 정의는 이 객체에서 읽고, 게임 중 달라지는 재고·HP·진행 기록은
model.GameState에 따로 둡니다. new_party는 캐릭터 정의로 새 Character를 만듭니다."""
import json
from copy import deepcopy
from pathlib import Path

from .model import Character

DATA = Path(__file__).resolve().parent / "data"


def _load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _load_path(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def npc_name(card):
    """원본 NPC 카드의 공개 본문에서 ``이름 —`` 앞 표시 이름만 읽습니다.

    인물·장소·공개 이름 형식이 없는 카드는 빈 문자열을 반환합니다.
    원본 카드와 숨은 본문은 수정하지 않으며 별도 인물 프로필을 만들지 않습니다.
    """
    if not isinstance(card, dict) or card.get("kind") != "npc" or not card.get("room"):
        return ""
    public = card.get("player_text")
    if not isinstance(public, str):
        return ""
    name, separator, _ = public.partition(" —")
    return name.strip() if separator and "\n" not in name else ""


class Content:
    """게임 정의 자료를 ID별로 찾을 수 있게 모아 두는 객체입니다.

    여러 게임이 같은 정의 자료를 읽을 수 있도록 가격·장면·카드 원본과, 각 게임의
    변동 상태를 구분합니다. 변동 재고·HP·선택 완료 기록은 GameState가 맡습니다."""

    def __init__(self, scene_paths=None):
        """게임 JSON을 읽고 장면들의 방·선택지·공유 효과를 하나로 연결합니다.

        scene_paths가 주어지면 해당 파일들을, 아니면 data/scene*.json을 읽습니다.
        ID 중복·장면 불일치·없는 참조는 읽는 단계에서 오류로 알려 줍니다."""
        cards = _load("cards.json")
        self.cards = {c["id"]: c for c in cards["cards"]}
        self.room_names = cards["rooms"]
        paths = scene_paths or sorted(DATA.glob("scene*.json"))
        self.scenes = sorted((_load_path(p) for p in paths), key=lambda s: s["scene"])
        self.rooms, self.choices, self.card_views, self.buy_reactions, self.effect_sets = {}, {}, {}, {}, {}
        self._room_choices = {}
        self.scene_titles, self.scene_status = {}, {}
        for sc in self.scenes:
            n = sc["scene"]
            self.scene_titles[n] = sc.get("title", f"장면 {n}")
            self.scene_status[n] = sc.get("status", "playable")
            for rid, room in sc["rooms"].items():
                if rid in self.rooms:
                    raise ValueError(f"방 ID 중복: {rid}")
                if room["scene"] != n:
                    raise ValueError(f"방 {rid}의 장면 번호({room['scene']})가 파일({n})과 다름")
                self.rooms[rid] = room
            for ch in sc.get("choices", []):
                if ch["id"] in self.choices:
                    raise ValueError(f"선택지 ID 중복: {ch['id']}")
                self.choices[ch["id"]] = ch
                self._room_choices.setdefault(ch["room"], []).append(ch)
            self.card_views.update({k: v for k, v in sc.get("card_views", {}).items() if not k.startswith("_")})
            self.buy_reactions.update({k: v for k, v in sc.get("buy_reactions", {}).items() if not k.startswith("_")})
            # "…_effects"로 끝나는 최상위 목록은 선택지에서 {"use": 이름}으로 불러 쓰는 효과 묶음
            self.effect_sets.update({k: v for k, v in sc.items() if k.endswith("_effects") and isinstance(v, list)})
        first = self.scenes[0]
        self.start_scene = first["scene"]
        self.start_room = first["start_room"]
        items = _load("items.json")
        self.items = items["items"]
        self.shop_order = items["shop_order"]
        self.use_order = [i for i in self.shop_order if "use" in self.items[i]]  # 사용 선택지의 고정 키 순서 (W5)
        self.traps = _load("traps.json")["traps"]
        self.monsters = _load("monsters.json")["monsters"]        # W6
        self.encounters = _load("encounters.json")["encounters"]
        self.npc_combat = {}
        npc_path = DATA / "npc_encounters.json"
        if npc_path.is_file():
            definitions = _load("npc_encounters.json")
            for card_id in definitions["cards"]:
                card = self.cards.get(card_id, {})
                if card.get("kind") != "npc" or not npc_name(card):
                    raise ValueError("전투 대상의 인물 카드가 없습니다: " + card_id)
                key = "npc:" + card_id
                monster_key = "npc_" + card_id  # 전투 선택지의 ':' 구분자와 대상 ID가 섞이지 않게 합니다.
                template = deepcopy(definitions["template"])
                template["name"] = npc_name(card)
                self.npc_combat[card_id] = template
                self.monsters[monster_key] = template
                self.encounters[key] = {"name": template["name"] + "와의 충돌", "npc_card_id": card_id,
                    "card": card_id, "room": card["room"], "scene": card["scene_id"], "level": 1,
                    "monsters": [{"monster": monster_key, "count": 1}],
                    "options": [{"id": "withdraw", "label": "공격을 멈추고 물러난다", "result": "retreat",
                                 "text": "공격을 멈추고 거리를 벌렸다. 충돌한 사실과 부상은 남는다."}]}
        self.rules = _load("rules.json")
        self._check_trap_refs()
        self._chars = _load("characters.json")["characters"]

    def _check_trap_refs(self):
        """장면의 중첩 자료를 돌며 함정·전투 참조가 실제 정의에 있는지 검사합니다."""
        def walk(x):
            if isinstance(x, dict):
                for k in ("trap", "disarm_trap", "trap_active"):
                    if k in x and x[k] not in self.traps:
                        raise ValueError(f"없는 함정 ID: {x[k]}")
                if "combat" in x and isinstance(x["combat"], str) and x["combat"] not in self.encounters:
                    raise ValueError(f"없는 전투 ID: {x['combat']}")
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(self.scenes)

    def choices_in_room(self, room_id):
        """그 방의 선택지를 데이터 순서대로 (고정 키의 기준)."""
        return self._room_choices.get(room_id, [])

    def exits(self, room_id):
        """불러온 장면에 있는 방으로만 이동 가능 (아직 없는 장면의 방은 숨김)."""
        return [r for r in self.rooms[room_id]["exits"] if r in self.rooms]

    def characters(self):
        """선택 가능한 사전 제작 캐릭터 목록 [(id, 이름, 직업)]."""
        return [(c["id"], c["name"], c["cls_ko"]) for c in self._chars]

    def new_party(self, char_id):
        """사전 제작 캐릭터 ID에 해당하는 새 Character 하나를 목록으로 반환합니다.

        새 게임마다 별도 Character를 만들므로 원래 캐릭터 정의와 진행 중 HP·자원을 구분합니다."""
        for c in self._chars:
            if c["id"] == char_id:
                return [Character(**c)]
        raise ValueError(f"없는 캐릭터: {char_id} (가능: {[c['id'] for c in self._chars]})")

    def room_name(self, room_id):
        """방 이름. 장면 파일에 있으면 그 이름, 아직 장면 파일이 없는 방은 cards.json의 방 이름."""
        r = self.rooms.get(room_id)
        return r["name"] if r else self.room_names.get(room_id, room_id)
