"""확정된 엔진 사건을 화면 효과용 자료로 바꿉니다.

사용 순서: begin(실행 전) → capture(사건 확정 후·생성 중·완료 시) → markup(화면).
capture를 여러 번 호출해도 이미 읽은 사건은 다시 효과로 만들지 않습니다.
markup은 자료를 소비하지 않습니다. 브라우저는 session·sequence로 중복을 거릅니다.
이 모듈은 주사위를 굴리거나 게임 수치를 바꾸거나 모델을 호출하지 않습니다.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from html import escape
import json
from secrets import token_hex
from threading import RLock
import weakref


RESET_OPERATIONS = {
    "start", "start_game", "new_game", "load", "load_game", "restore", "restore_data",
    "developer_restore", "developer_restore_checkpoint", "restore_checkpoint",
    "restore_safety_checkpoint", "recover_checkpoint", "safe_restore",
}


def _small_string(value, limit=120):
    return value[:limit] if isinstance(value, str) else ""


def _audit(game):
    value = getattr(game, "audit", [])
    return value if isinstance(value, list) else []


def _tail_signature(rows, position):
    if not position:
        return ""
    # 내용은 비교용 해시만 남깁니다. 엔진 설명·개인 대화는 화면 속성으로 보내지 않습니다.
    text = json.dumps(rows[position - 1], ensure_ascii=False, sort_keys=True, default=str)
    return sha256(text.encode("utf-8")).hexdigest()


def _enemies(game):
    engine = game.engine
    combat = engine.s.combat or {}
    result = {}
    for identifier, enemy in combat.get("enemies", {}).items():
        if not isinstance(enemy, dict):
            continue
        outcome = enemy.get("out")
        wound = {"dead": "쓰러짐", "unconscious": "의식을 잃음", "fled": "달아남"}.get(outcome)
        if not wound:
            wound = engine.wound_text(enemy)
        result[identifier] = {
            "id": _small_string(identifier), "kind": _small_string(enemy.get("kind")),
            "name": _small_string(enemy.get("name"), 80), "wound": _small_string(wound, 40),
        }
    return result


@dataclass
class _ViewState:
    session: str = field(default_factory=lambda: token_hex(10))
    sequence: int = 0
    epoch: int = 0
    effects: list = field(default_factory=list)
    enemies: list = field(default_factory=list)


@dataclass
class FeedbackCapture:
    """한 번의 화면 작업이 읽은 감사 기록 위치입니다. 저장 파일에는 넣지 않습니다."""
    game_id: int
    epoch: int
    audit_id: int
    position: int
    tail_signature: str
    operation: str
    enemies: dict
    action_id: str = ""
    event_id: str = ""
    player_critical: bool = False


class CombatFeedbackTracker:
    """접속별 효과 순서를 관리합니다. 화면 재표시와 실제 새 사건을 구분합니다."""

    def __init__(self, max_sessions=64):
        self.max_sessions = max(1, int(max_sessions))
        self._states = OrderedDict()
        self._lock = RLock()

    def _state(self, game):
        key = id(game)
        entry = self._states.get(key)
        if entry and entry[0]() is game:
            self._states.move_to_end(key)
            return entry[1]
        # 실제 PlaySession은 약한 참조를 사용합니다. 작은 확인용 객체도 지원합니다.
        try:
            owner = weakref.ref(game, lambda _ref: self._discard(key))
        except TypeError:
            owner = lambda: game
        state = _ViewState()
        self._states[key] = (owner, state)
        while len(self._states) > self.max_sessions:
            self._states.popitem(last=False)
        return state

    def _discard(self, key):
        with self._lock:
            self._states.pop(key, None)

    def close(self, game):
        self._discard(id(game))

    def _reset(self, state):
        # 되돌리기는 게임 revision을 낮출 수 있지만 화면 효과 번호는 낮추지 않습니다.
        state.epoch += 1
        state.sequence += 1
        state.effects = []
        state.enemies = []

    def reset(self, game):
        """저장 불러오기·기록 복구 후 이전 피해 효과를 재생하지 않도록 비웁니다."""
        with self._lock:
            self._reset(self._state(game))
            return self.packet(game)

    def begin(self, game, operation=""):
        with self._lock:
            state, rows = self._state(game), _audit(game)
            pending = getattr(game, "pending", None) or {}
            event = getattr(game, "last_result", None) or {}
            return FeedbackCapture(id(game), state.epoch, id(rows), len(rows),
                _tail_signature(rows, len(rows)), str(operation), _enemies(game),
                _small_string(pending.get("choice")), _small_string(event.get("event_id")))

    def _target(self, identifier, game, known):
        if identifier in known:
            return known[identifier]
        # 전투 시작과 종료가 한 작업에서 끝나도 실제 대상 ID로 종류를 찾습니다.
        # 전체 몬스터 목록을 화면에 보내지 않고, 사건에 나온 대상 하나만 읽습니다.
        monsters = getattr(game.engine.c, "monsters", {})
        for kind in sorted(monsters, key=len, reverse=True):
            if identifier == kind or identifier.startswith(kind + "_"):
                row = {"id": _small_string(identifier), "kind": _small_string(kind),
                       "name": _small_string(monsters[kind].get("name"), 80), "wound": ""}
                known[identifier] = row
                return row
        return None

    def capture(self, game, before, operation=None):
        """새 확정 사건만 읽습니다. 스트리밍 중 반복 호출해도 같은 피해는 한 번입니다."""
        with self._lock:
            if not isinstance(before, FeedbackCapture) or before.game_id != id(game):
                raise ValueError("이 게임에서 시작한 화면 작업인지 확인하세요.")
            state, rows = self._state(game), _audit(game)
            if before.epoch != state.epoch:
                return self.packet(game)
            name = str(operation if operation is not None else before.operation)
            if name in RESET_OPERATIONS:
                self._reset(state)
                return self.packet(game)
            replaced = (id(rows) != before.audit_id or len(rows) < before.position
                        or _tail_signature(rows, before.position) != before.tail_signature)
            if replaced:
                # 불러온 과거 기록 전체를 새 전투로 재생하지 않습니다.
                self._reset(state)
                before.epoch, before.audit_id, before.position = state.epoch, id(rows), len(rows)
                before.tail_signature, before.enemies = _tail_signature(rows, len(rows)), _enemies(game)
                return self.packet(game)

            fresh = rows[before.position:]
            before.position = len(rows)
            before.tail_signature = _tail_signature(rows, before.position)
            if not fresh:
                return self.packet(game)
            after_enemies = _enemies(game)
            known = {**before.enemies, **after_enemies}
            event = getattr(game, "last_result", None) or {}
            # 감사 기록은 확정됐지만 last_result가 아직 이전 턴일 수도 있습니다.
            # 실행 전 판정 대상이 있으면 우선하고, 새 결과의 행동 ID만 보충합니다.
            action_id = before.action_id
            if not action_id and event.get("event_id") != before.event_id:
                action_id = (event.get("engine_delta") or {}).get("executed_action_id")
            action_target = action_id.rsplit(":", 1)[-1] if isinstance(action_id, str) and action_id.startswith("cmb:") else ""
            effects, used_targets = [], OrderedDict()
            player_critical = before.player_critical
            hero_id = game.engine.hero.id
            for row in fresh:
                if not isinstance(row, dict):
                    continue
                data = row.get("data") or {}
                if not isinstance(data, dict):
                    continue
                kind = row.get("kind")
                if kind == "check" and data.get("side") == "player" and type(data.get("hit")) is bool:
                    player_critical = bool(data.get("crit"))
                    if data["hit"] is False:
                        target = self._target(action_target, game, known)
                        effect = {"kind": "player_miss"}
                        if target:
                            effect["target"] = target["id"]
                            used_targets[target["id"]] = target
                        effects.append(effect)
                if kind != "damage" or type(data.get("amount")) is not int:
                    continue
                amount = data["amount"]
                if data.get("char") == hero_id and amount > 0:
                    # 0HP에서 맞는 별도 사건에는 char 필드가 없습니다. 실제 HP 감소와 구분합니다.
                    # 순수 HP 차이는 회복·죽음 구원과 합쳐질 수 있으므로 판정 기준으로 쓰지 않습니다.
                    effects.append({"kind": "player_damage", "target": _small_string(hero_id), "amount": amount})
                elif isinstance(data.get("target"), str) and amount >= 0:
                    target = self._target(data["target"], game, known)
                    if not target:
                        continue
                    target = dict(target)
                    if data.get("hp") == 0 and target["id"] not in after_enemies:
                        target["wound"] = "쓰러짐"
                    used_targets[target["id"]] = target
                    effects.append({"kind": "player_hit" if amount else "player_no_damage",
                                    "target": target["id"], "amount": amount, "critical": player_critical})
            before.enemies = known
            # 판정과 실제 피해가 서로 다른 생성 프레임에서 도착해도 유지합니다.
            before.player_critical = player_critical
            if effects:
                state.sequence += 1
                state.effects = effects[-32:]
                state.enemies = list(used_targets.values())[-32:]
            return self.packet(game)

    finish = capture

    def packet(self, game):
        """이미 전송한 효과도 유지합니다. 브라우저가 sequence로 중복 표시를 막습니다."""
        with self._lock:
            state = self._state(game)
            return {"schema": 1, "session": state.session, "sequence": state.sequence,
                    "effects": deepcopy(state.effects), "enemies": deepcopy(state.enemies)}

    def markup(self, game):
        payload = json.dumps(self.packet(game), ensure_ascii=False, separators=(",", ":"))
        return ('<div class="trpg-combat-feedback" data-combat-feedback="'
                + escape(payload, quote=True) + '" aria-hidden="true"></div>')
