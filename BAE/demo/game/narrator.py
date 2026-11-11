"""AI 마스터 서술 (Qwen). 판정·상태는 엔진이 이미 끝낸 뒤, 결과를 문장으로만 바꾼다.

정보 접근 원칙 (analysis/05_시스템설계/05-2a 3장):
  Qwen에게는 이번 행동으로 '플레이어에게 표시된' 카드 문장, 판정 결과, 짧은 상태 요약만 넘긴다.
  열람 가능하지만 표시되지 않은 정보(함정, 비밀 카드)는 넘기지 않는다.

백엔드:
  - TemplateNarrator: LLM 없이 진행 (서술 없음, 엔진 문장만 출력)
  - OllamaNarrator:   기존 Colab 노트북(TRPG_Rule.ipynb)과 같은 Ollama /api/chat 호출 (예: qwen2.5:7b)
  - CallableNarrator: Colab에서 transformers로 띄운 Qwen 등, messages -> str 함수를 직접 연결

세 클래스 모두 narrate(engine, events, actor_id) 하나만 가진다. 돌려준 문장은 화면에 덧붙여 보여 줄 뿐,
게임 상태에는 아무 영향이 없다. 쓰는 곳: game/cli.py play(), server/webapp.py Game.reply().
"""
import json
import re
import urllib.error
import urllib.request

SYSTEM_PROMPT = """너는 한국어 TRPG의 진행자(마스터)다. 아래 규칙을 반드시 지킨다.
1. 판정 결과(성공/실패), 피해 숫자, 획득한 물건은 [결과]에 적힌 그대로만 말한다. 바꾸거나 새로 만들지 않는다.
2. [이번에 드러난 정보]와 [결과]에 없는 사실(숨겨진 함정, 비밀, 다른 장소의 정보)을 지어내거나 암시하지 않는다.
3. 플레이어가 다음에 무엇을 해야 하는지 지시하지 않는다.
4. 2~4문장, 현재형 서술. 주사위 숫자나 DC는 다시 말하지 않는다.
5. 전투에서 명중·빗나감·피해·쓰러짐·승패는 [결과]에 적힌 그대로만 묘사한다. 결과에 없는 공격, 추가 피해, 적의 죽음이나 도주를 만들지 않는다."""

THINK_TAG = re.compile(r"<think>.*?</think>", re.S)


def build_messages(engine, events, actor_id=None):
    """엔진 이벤트 → Qwen 입력. 표시된 카드(card 이벤트)와 결과 문장만 담는다."""
    s = engine.s
    revealed = [e.text for e in events if e.kind == "card"]
    results = [e.text for e in events if e.kind in ("check", "say", "damage", "item", "gold", "status", "rest", "end", "combat")]
    actor = s.member(actor_id).name if actor_id else "일행"
    party = ", ".join(f"{m.name}({m.cls_ko}, HP {m.hp}/{m.max_hp})" for m in s.party)
    user = (f"[장소] {engine.c.room_name(s.room)}\n[파티] {party}\n[행동한 사람] {actor}\n"
            f"[이번에 드러난 정보]\n" + ("\n".join(revealed) or "(없음)") +
            "\n[결과]\n" + ("\n".join(results) or "(특별한 결과 없음)") +
            "\n\n위 내용만 바탕으로 장면을 묘사해 줘.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def clean(text):
    """모델이 붙이는 <think>…</think> 생각 과정을 지우고 앞뒤 공백을 정리한다."""
    return THINK_TAG.sub("", text or "").strip()


class TemplateNarrator:
    name = "template"

    def narrate(self, engine, events, actor_id=None):
        return None  # 엔진 문장을 그대로 보여 준다


class CallableNarrator:
    name = "callable"

    def __init__(self, fn):
        self.fn = fn

    def narrate(self, engine, events, actor_id=None):
        try:
            return clean(self.fn(build_messages(engine, events, actor_id)))
        except Exception as e:  # 서술 실패가 게임 진행을 막지 않게 한다
            return f"(서술 생성 실패: {e})"


class OllamaNarrator:
    name = "ollama"

    def __init__(self, base_url="http://localhost:11434", model="qwen2.5:7b", timeout=60, temperature=0.4):
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def narrate(self, engine, events, actor_id=None):
        body = json.dumps({
            "model": self.model,
            "messages": build_messages(engine, events, actor_id),
            "stream": False,
            "options": {"temperature": self.temperature},
        }).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return clean(data["message"]["content"])
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            return f"(서술 생성 실패: {e})"
