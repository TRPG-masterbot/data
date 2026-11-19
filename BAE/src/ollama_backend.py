"""Ollama의 준비·본문 스트리밍·사용량을 연결하는 HTTP 전송 계층입니다.

추론 내용은 전달하지 않으며, 취소 시 열린 연결을 닫습니다.
게임 상태나 답변 검증은 이 모듈에서 변경하지 않습니다.
"""

import http.client
import json
import socket
import threading
import time
from queue import Empty, Queue
from urllib.parse import urlsplit

import requests


class OllamaBackend:
    """로컬 또는 지정한 Ollama 서버의 한 모델을 사용합니다."""

    def __init__(self, model, base_url="http://127.0.0.1:11434", *,
                 keep_alive="30m", num_ctx=6144, timeout=120):
        self.model = str(model).strip()
        self.base_url = str(base_url).rstrip("/")
        self.keep_alive, self.num_ctx = keep_alive, int(num_ctx)
        self.timeout = float(timeout)
        self.digest, self.details = None, {}
        self._url = urlsplit(self.base_url)
        if (not self.model or self._url.scheme not in {"http", "https"}
                or not self._url.hostname or self._url.username or self._url.password):
            raise ValueError("Ollama 모델 이름과 http 또는 https 연결 주소를 확인하세요.")

    def _json(self, path, payload=None, timeout=None):
        try:
            response = requests.request("GET" if payload is None else "POST",
                self.base_url + path, json=payload, timeout=timeout or self.timeout)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError("Ollama 연결을 확인하세요. 서버 주소: " + self.base_url) from error
        if not isinstance(data, dict):
            raise RuntimeError("Ollama 응답 형식이 올바르지 않습니다.")
        if data.get("error"):
            raise RuntimeError("Ollama 요청 실패: " + str(data["error"]))
        return data

    def pull(self):
        """명시적으로 요청한 경우에만 모델 파일을 내려받습니다."""
        return self._json("/api/pull", {"model": self.model, "stream": False}, timeout=3600)

    def prepare(self, pull=False):
        """설치된 태그와 모델 정보를 확인하고 실제 digest를 보관합니다."""
        def installed():
            name = self.model if ":" in self.model.rsplit("/", 1)[-1] else self.model + ":latest"
            return next((row for row in self._json("/api/tags").get("models", [])
                         if name in {row.get("name"), row.get("model")}), None)
        entry = installed()
        if entry is None and pull:
            self.pull()
            entry = installed()
        if entry is None:
            raise RuntimeError("Ollama 모델이 없습니다. 먼저 실행하세요: ollama pull " + self.model)
        info = self._json("/api/show", {"model": self.model})
        self.digest, self.details = entry.get("digest"), info.get("details", {})
        return self

    def generate(self, messages, *, max_new_tokens, sample, thinking=False,
                 temperature=.7, top_p=.8, top_k=20, on_text=None, check_cancel=None):
        """실제 도착한 본문을 누적 전달하고 완료 프레임의 토큰 수를 반환합니다."""
        started = time.perf_counter()
        check_cancel = check_cancel or (lambda: None)
        check_cancel()
        options = {"num_ctx": self.num_ctx, "num_predict": int(max_new_tokens),
                   "temperature": float(temperature) if sample else 0, "repeat_penalty": 1.0}
        if sample:
            options.update(top_p=float(top_p), top_k=int(top_k))
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "think": bool(thinking), "keep_alive": self.keep_alive, "options": options}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        connection_type = http.client.HTTPSConnection if self._url.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(self._url.hostname, self._url.port, timeout=self.timeout)
        path = self._url.path.rstrip("/") + "/api/chat"
        incoming, stopped = Queue(), threading.Event()
        active_socket = {}

        def close_connection():
            # close()만으로는 다른 스레드의 readline이 즉시 끝나지 않을 수 있습니다.
            sock = connection.sock or active_socket.get("socket")
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()

        def receive():
            try:
                if stopped.is_set():
                    return
                connection.request("POST", path, body=body,
                    headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"})
                active_socket["socket"] = connection.sock
                if stopped.is_set():
                    return
                response = connection.getresponse()
                if response.status != 200:
                    error_body = response.read(4096).decode("utf-8", errors="replace")
                    raise RuntimeError(f"Ollama 응답 오류 ({response.status}): {error_body}")
                while not stopped.is_set():
                    line = response.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    frame = json.loads(line)
                    if not isinstance(frame, dict):
                        raise RuntimeError("Ollama 스트림 응답 형식이 올바르지 않습니다.")
                    incoming.put(("frame", frame))
                    if frame.get("done") is True or frame.get("error"):
                        break
            except Exception as error:
                incoming.put(("error", error))
            finally:
                incoming.put(("end", None))
                close_connection()

        worker = threading.Thread(target=receive, daemon=True, name="trpg-ollama-stream")
        worker.start()
        text, final = "", None
        try:
            while final is None:
                check_cancel()
                if time.perf_counter() - started >= self.timeout:
                    raise TimeoutError("Ollama 응답 시간이 제한을 넘었습니다.")
                try:
                    kind, value = incoming.get(timeout=.1)
                except Empty:
                    continue
                if kind == "error":
                    if isinstance(value, RuntimeError):
                        raise value
                    raise RuntimeError("Ollama 응답을 받지 못했습니다. 서버 연결을 확인하세요.") from value
                if kind == "end":
                    raise RuntimeError("Ollama 응답이 완료되기 전에 연결이 종료되었습니다.")
                if value.get("error"):
                    raise RuntimeError("Ollama 생성 오류: " + str(value["error"]))
                fragment = (value.get("message") or {}).get("content", "")
                if not isinstance(fragment, str):
                    raise RuntimeError("Ollama 본문 형식이 올바르지 않습니다.")
                if fragment:
                    text += fragment
                    check_cancel()
                    if on_text is not None:
                        on_text(text)
                if value.get("done") is True:
                    final = value
            check_cancel()
            counts = [final.get(key) for key in ("prompt_eval_count", "eval_count")]
            if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
                raise RuntimeError("Ollama 완료 응답에 실제 토큰 사용량이 없습니다.")
            return {"text": text.strip(), "input_tokens": counts[0], "output_tokens": counts[1],
                    "generation_limit": int(max_new_tokens), "seconds": round(time.perf_counter() - started, 3),
                    "done_reason": final.get("done_reason"), "model": self.model,
                    "model_digest": self.digest}
        finally:
            stopped.set()
            close_connection()
            worker.join(timeout=2)
