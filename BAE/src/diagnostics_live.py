"""진행 중 요청을 게임 잠금과 분리해 기록하고 마지막 확정 사본을 보관합니다.

원본 세션·게임 상태를 읽거나 잠그지 않습니다. 호출자가 capture로 넘긴 공개
사본만 보관하며, 모든 반환값은 내부 기록과 분리된 사본입니다.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import math
import re
from threading import RLock
import time
from uuid import uuid4


_CURRENT = ContextVar("trpg_live_request_diagnostics", default=None)
_PRIVATE_KEY = re.compile(
    r"(?i)(?:authorization|cookie|password|secret|credential|api[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|session[_-]?(?:id|token)|"
    r"^token$|(?:base[_-]?)?url$|endpoint|(?:^|_)path$|^host$)")
_LOCATION = re.compile(
    r"(?i)(?:\b(?:https?|wss?|ftp)://[^\s\"'<>]+|"
    r"[a-z]:[\\/][^\s\"'<>]+|\\\\[^\s\"'<>]+|"
    r"/(?:content|home|users|tmp|var)/[^\s\"'<>]+)")
_AUTH_TEXT = re.compile(r"(?i)(?:\bbearer\s+[^\s,;]+|"
                        r"\b(?:api[_-]?key|access[_-]?token|authorization|password|secret)\s*[:=]\s*[^\s,;]+)")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _public(value, session=None, depth=0):
    """호출자가 전달한 진단 필드에서 인증값·주소·경로만 제거합니다."""
    if depth > 24:
        return "[depth_limit]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        result = _AUTH_TEXT.sub("[private]", _LOCATION.sub("[location]", value))
        if isinstance(session, str) and session:
            result = result.replace(session, "[session]")
        return result
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, BaseException):
        return {"code": type(value).__name__, "message": _public(str(value), session, depth + 1)}
    if isinstance(value, dict):
        return {key: _public(item, session, depth + 1) for key, item in value.items()
                if isinstance(key, str) and not _PRIVATE_KEY.search(key)}
    if isinstance(value, (list, tuple)):
        return [_public(item, session, depth + 1) for item in value]
    return None


class RequestDiagnostics:
    """세션별 큐·현재 단계·완료 기록을 짧은 자체 잠금으로 관리합니다."""

    def __init__(self, max_requests=128, max_events=96):
        self.max_requests = max(1, int(max_requests))
        self.max_events = max(2, int(max_events))
        self._lock = RLock()
        self._sessions = {}
        self._requests = {}

    def _session(self, session):
        return self._sessions.setdefault(session, {"requests": [], "cache": None,
                                                   "dropped_count": 0})

    def _event(self, record, stage, details=None):
        event = {"stage": stage, "status": record["status"], "at": _now(),
                 "elapsed_seconds": round(record.get("_ended", time.monotonic()) - record["_started"], 6),
                 "details": details or {}}
        record["stage"] = stage
        record["updated_at"] = event["at"]
        record["_stage_started"] = time.monotonic()
        record["timeline"].append(event)
        if len(record["timeline"]) > self.max_events:
            # 접수 시각은 남겨 두고 오래된 중간 단계부터 줄입니다.
            del record["timeline"][1]
            record["dropped_events"] += 1

    def register(self, session, kind, input_text=None):
        if session is None:
            return None
        clean_kind = _public(str(kind), session)
        clean_input = _public(input_text, session) if isinstance(input_text, str) else None
        request_id = "rq_" + uuid4().hex
        with self._lock:
            state = self._session(session)
            while len(state["requests"]) >= self.max_requests:
                old = next((item for item in state["requests"] if item["_finished"]),
                           state["requests"][0])
                state["requests"].remove(old)
                self._requests.pop(old["request_id"], None)
                state["dropped_count"] += 1
            record = {"request_id": request_id, "kind": clean_kind,
                      "input_text": clean_input, "status": "queued", "stage": "queued",
                      "registered_at": _now(), "finished_at": None, "error": None,
                      "timeline": [], "dropped_events": 0,
                      "_started": time.monotonic(), "_finished": False}
            self._event(record, "queued")
            state["requests"].append(record)
            self._requests[request_id] = (session, record)
        return request_id

    def claim(self, session, kind):
        """동일 종류의 대기 요청 중 가장 먼저 등록된 요청을 가져옵니다."""
        if session is None:
            return None
        clean_kind = _public(str(kind), session)
        with self._lock:
            state = self._session(session)
            record = next((row for row in state["requests"]
                           if row["kind"] == clean_kind and row["status"] == "queued"), None)
            if record is None:
                request_id = self.register(session, kind)
                record = self._requests[request_id][1]
            record["status"] = "running"
            self._event(record, "claimed")
            return record["request_id"]

    @contextmanager
    def bind(self, request_id):
        token = _CURRENT.set((self, request_id))
        try:
            yield request_id
        finally:
            _CURRENT.reset(token)

    def stage(self, request_id, stage, **details):
        if request_id is None:
            return False
        with self._lock:
            entry = self._requests.get(request_id)
            if entry is None or entry[1]["_finished"]:
                return False
            session, record = entry
        # 중첩 메타데이터 복사·정리는 자체 잠금 밖에서 합니다.
        clean_stage = _public(str(stage), session)
        clean_details = _public(details, session)
        with self._lock:
            if self._requests.get(request_id) is not entry or record["_finished"]:
                return False
            self._event(record, clean_stage, clean_details)
        return True

    def finish(self, request_id, status="completed", error=None):
        if request_id is None:
            return False
        with self._lock:
            entry = self._requests.get(request_id)
            if entry is None or entry[1]["_finished"]:
                return False
            session, record = entry
        clean_error = _public(error, session)
        clean_status = _public(str(status), session)
        with self._lock:
            if self._requests.get(request_id) is not entry or record["_finished"]:
                return False
            record["status"] = clean_status
            record["error"] = clean_error
            record["finished_at"] = _now()
            record["_finished"] = True
            record["_ended"] = time.monotonic()
            self._event(record, clean_status, {"error": clean_error} if clean_error else {})
        return True

    def capture(self, session, messages, turn_records, session_meta, current_state):
        """호출 시점에 이미 확보한 공개 상태를 불변 사본으로 교체합니다."""
        if session is None:
            return False
        data = _public(deepcopy({"messages": messages, "turn_records": turn_records,
                                 "session_meta": session_meta, "current_state": current_state}), session)
        data["captured_at"] = _now()
        data["_captured_monotonic"] = time.monotonic()
        with self._lock:
            self._session(session)["cache"] = data
        return True

    def snapshot(self, session):
        """게임 잠금 없이 마지막 확정 사본과 현재 요청 단계의 사본을 줍니다."""
        with self._lock:
            state = self._sessions.get(session) if session is not None else None
            cached = state["cache"] if state else None
            requests = []
            for row in state["requests"] if state else []:
                # 이벤트·details는 생성 뒤 변경하지 않으므로 참조 목록만 짧게 복사합니다.
                item = {key: value for key, value in row.items() if not key.startswith("_")}
                item["timeline"] = list(row["timeline"])
                item["pending"] = not row["_finished"]
                item["elapsed_seconds"] = round((row.get("_ended", time.monotonic())
                                                  - row["_started"]), 6)
                item["stage_elapsed_seconds"] = round(max(0.0, row.get("_ended", time.monotonic())
                                                          - row["_stage_started"]), 6)
                requests.append(item)
            dropped = state["dropped_count"] if state else 0
        result = deepcopy(cached) if cached else {"messages": [], "turn_records": [],
                                                 "session_meta": {}, "current_state": {},
                                                 "captured_at": None}
        captured = result.pop("_captured_monotonic", None)
        result.update(cache_available=cached is not None,
                      cache_age_seconds=round(time.monotonic() - captured, 6) if captured is not None else None,
                      live_requests=deepcopy(requests), dropped_count=dropped,
                      history_count=len(requests), snapshot_at=_now())
        return result

    def close(self, session):
        """접속을 닫으면 해당 사용자의 사본·요청 기록만 비웁니다."""
        with self._lock:
            state = self._sessions.pop(session, None)
            if state:
                for request in state["requests"]:
                    self._requests.pop(request["request_id"], None)

    def progress(self, session):
        """화면의 짧은 진행 표시용이며 확정 대화·상태를 복사하지 않습니다."""
        with self._lock:
            state = self._sessions.get(session) if session is not None else None
            active = [row for row in state["requests"] if not row["_finished"]] if state else []
            if not active:
                return {"request_id": None, "kind": None, "status": "idle", "stage": "idle",
                        "elapsed_seconds": 0.0, "stage_elapsed_seconds": 0.0,
                        "registered_at": None, "updated_at": None, "pending_count": 0}
            row = active[-1]
            now = time.monotonic()
            return {"request_id": row["request_id"], "kind": row["kind"], "status": row["status"],
                    "stage": row["stage"], "elapsed_seconds": round(now - row["_started"], 6),
                    "stage_elapsed_seconds": round(now - row["_stage_started"], 6),
                    "registered_at": row["registered_at"], "updated_at": row["updated_at"],
                    "pending_count": len(active)}


def current_stage(stage, **details):
    """현재 요청에 묶인 계측에만 전달합니다. 요청 밖 호출은 안전한 무동작입니다."""
    current = _CURRENT.get()
    return current[0].stage(current[1], stage, **details) if current is not None else False


def current_request_id():
    current = _CURRENT.get()
    return current[1] if current is not None else None


def public_diagnostics(value):
    """직접 내보내는 실행 메타데이터에서도 인증값·주소·경로를 제거합니다."""
    return _public(value)
