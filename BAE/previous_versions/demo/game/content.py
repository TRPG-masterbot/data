"""게임 데이터 불러오기. 데이터 생성은 tools/build_cards.py (카드), 나머지는 game/data/*.json.

장면 파일: game/data/scene*.json 을 모두 읽어 하나로 합친다 (W3).
  - 파일마다 "scene"(번호), "title", "rooms", "choices"를 가진다.
  - 장면 파일을 추가하면 엔진 수정 없이 방·선택지가 연결된다.
  - 방 ID와 선택지 ID는 모든 장면에서 유일해야 한다 (불러올 때 검사).

불러온 결과(Content 속성)를 쓰는 곳: engine.py·combat.py는 self.c.xxx로, 서버는 Game.content로 읽는다.
JSON 파일과 속성의 대응:
  cards.json → cards, room_names · scene*.json → scenes, rooms, choices, card_views, buy_reactions, effect_sets
  items.json → items, shop_order, use_order · traps.json → traps · monsters.json → monsters
  encounters.json → encounters · rules.json → rules · characters.json → _chars (new_party로 Character 생성)
"""
import json
from pathlib import Path

from .model import Character

DATA = Path(__file__).resolve().parent / "data"


def _load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _load_path(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Content:
    """읽기 전용 게임 데이터. 게임 중에 바뀌는 값(상점 재고 등)은 GameState에 둔다."""

    def __init__(self, scene_paths=None):
        """scene_paths를 주면 그 장면 파일만 읽는다 (테스트에서 장면 추가를 흉내 낼 때). 기본은 data/scene*.json 전부."""
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
        self.rules = _load("rules.json")
        self._check_trap_refs()
        self._chars = _load("characters.json")["characters"]

    def _check_trap_refs(self):
        """장면 데이터가 부르는 함정 ID가 traps.json에 있는지 불러올 때 검사."""
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
        """1인 1캐릭터 (05-6 ①): 선택한 캐릭터 한 명만 만든다."""
        for c in self._chars:
            if c["id"] == char_id:
                return [Character(**c)]
        raise ValueError(f"없는 캐릭터: {char_id} (가능: {[c['id'] for c in self._chars]})")

    def room_name(self, room_id):
        """방 이름. 장면 파일에 있으면 그 이름, 아직 장면 파일이 없는 방은 cards.json의 방 이름."""
        r = self.rooms.get(room_id)
        return r["name"] if r else self.room_names.get(room_id, room_id)
