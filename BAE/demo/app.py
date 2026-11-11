"""잠긴 종 초반 플레이용 로컬 웹 앱."""

import argparse
from collections import OrderedDict
from contextlib import ExitStack
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
from secrets import token_urlsafe
import sys
from threading import Timer
from uuid import uuid4
import webbrowser

from game.content import Content
from game.engine import Engine


HERE = Path(__file__).resolve().parent
# 업로드용 사본의 외부 모델·룰북 경로는 이 저장소를 기준으로 잡는다.
RUNTIME = HERE / ".runtime"
RULEBOOK = RUNTIME / "Rules.pdf"
sys.path[:0] = [str(RUNTIME / name) for name in ("team_deps", "team_llama", "deps")]

import rag


ACTION_VERBS = (
    r"말(?:을)?걸|만나|대화|이야기|얘기|물어|묻",
    r"듣|엿듣",
    r"받|챙기|집어|줍",
    r"열|개방",
    r"살피|살펴|조사|확인|들여다",
    r"내려|하강",
    r"공격|때리|휘두르",
    r"구매|사겠|산다|사고싶",
)
TRAVEL = r"이동|가볼|가보|가자|갈래|가겠|간다|들어가|돌아가|향하"
APPROACH = r"다가|가까이|곁으로|말(?:을)?걸|대화|이야기|얘기|인사"
DETAIL_QUERY = re.compile(
    r"(?:(?:그\s*(?:얘기|이야기|부분)|그럼|그래서|조금(?:만)?|좀|더|길게|자세히|자세하게)\s*[, ]*)*"
    r"(?:해\s*줘|알려\s*(?:줘|주세요)?|들려\s*(?:줘|주세요)?|이야기\s*해\s*(?:줘|주세요)?|"
    r"말\s*해\s*(?:줘|주세요)?|설명\s*해\s*(?:줘|주세요)?|그게\s*무슨\s*뜻(?:이야|이에요)?|"
    r"그다음(?:은)?|그래서|계속|자세히|자세하게)(?:요)?[.?!…]*"
)
ACTION_WITHHELD = re.compile(
    r"(?:지(?:는)?\s*(?:않|말|못)|싶지는\s*않|척(?:만)?|하려다\s*말|"
    r"(?<![가-힣])안\s+(?:가|해|하|받|내려|열|들어|공격|살피|조사))"
)
META_RULE_QUESTION = re.compile(r"규칙상|룰북|(?<![A-Za-z])(?:d20|AC)(?![A-Za-z])", re.I)
RULE_COMMAND = re.compile(r"^/(?:룰|규칙)(?:\s+|$)")
RULE_QUESTION = re.compile(
    r"규칙상|룰북|판정|명중|붙잡힌 상태|짧은 휴식|긴 휴식|"
    r"(?<![가-힣])이점|불리점|내성 굴림|방어도|주사위|이동 속도|기회공격|\b(?:d20|AC)\b",
    re.IGNORECASE,
)


def npc_name(card):
    if card["kind"] != "npc":
        return None
    name = card.get("speaker")
    if name:
        return name
    # 인물 카드의 '이름 — 소개' 머리말을 사용합니다.
    found = re.match(r"^([가-힣]{2,12})\s*[—–-]\s+", card["player_text"])
    return found[1] if found else None


def npc_address(message, name):
    return re.match(
        rf"^{re.escape(name)}(?:\s*(?:님|씨|야)(?=[\s,.!?…]|$)|(?=[,.!?…]|$))[\s,.!?…]*",
        message,
    )


def match_action(message, choices):
    """현재 가능한 엔진 선택지에서만 대상을 찾는다. 동점이면 자동 실행하지 않는다."""
    compact = re.sub(r"\s+", "", message)
    if re.search(r"(?:하지말|안할|말까|않을)|^(?:왜|어떻게)|(?:해야해|뭐야|누구야|어디야)[?!]*$", compact):
        return None
    matches = {}
    for choice in choices:
        label = re.sub(r"\s+", "", choice.label)
        if compact.rstrip(".!") == label:
            return choice
        if label.startswith("이동:"):
            target = re.escape(label.removeprefix("이동:"))
            if re.search(target + rf"(?:으로|로|에)?(?:{TRAVEL})", compact):
                matches[choice.id] = (choice, 10)
            continue
        if not any(re.search(verb, compact) and re.search(verb, label) for verb in ACTION_VERBS):
            continue
        words = re.findall(r"[가-힣]{2,}", choice.label)
        anchors = {
            re.sub(r"(?:에게|한테|까지|에서|으로|을|를|와|과|이|가|의|에)$", "", word)
            for word in words if not any(re.match(verb, word) for verb in ACTION_VERBS)
        }
        score = sum(len(anchor) for anchor in anchors if len(anchor) >= 2 and anchor in compact)
        if score:
            matches[choice.id] = (choice, score)
    if not matches:
        return None
    highest = max(score for _, score in matches.values())
    best = [choice for choice, score in matches.values() if score == highest]
    return best[0] if len(best) == 1 else None


class Game:
    """한 플레이 세션의 화면 기록과 NPC 대화 상태를 관리한다. 판정과 게임 상태는 Engine이 관리한다."""
    def __init__(self, engine):
        self.engine = engine
        self.turn = 0
        self.log = []
        self.public_cards = {}
        self.clues = []
        self.indexed_cards = set()
        self.scenario_passages = {}
        self.scenario_db = None
        self.speaker = None
        self.suggested_line = ""
        self.conversations = {}
        self.dialogue_focus = {}
        self.record(engine.opening)

    def record(self, events):
        """엔진이 이 캐릭터에게 공개한 카드 조각만 화면과 세션 검색 자료에 넣는다."""
        known = self.engine.player_known_cards(self.engine.hero.id)
        for event in events:
            if event.kind == "flag":
                continue
            card_id = event.data.get("id")
            entry = {"kind": event.kind, "text": event.text}
            if event.kind == "card":
                if card_id not in known:
                    continue
                fragment = event.data.get("part", card_id)
                card = self.engine.c.cards[card_id]
                speaker = npc_name(card)
                if speaker and card["room"] == self.engine.s.room:
                    self.speaker = card_id
                    self.suggested_line = ""
                    entry["speaker"] = speaker
                if fragment not in self.public_cards and card["kind"] in ("clue", "rumor", "npc"):
                    self.clues.append({
                        "category": {"clue": "단서", "rumor": "소문", "npc": "인물"}[card["kind"]],
                        "text": re.sub(r"^\[[^\]]+\]\s*", "", event.text),
                        "room": self.engine.c.room_name(self.engine.s.room),
                        "turn": self.turn,
                    })
                self.public_cards[fragment] = (card_id, event.text)
            self.log.append(entry)

    def state(self):
        engine = self.engine
        hero = engine.hero
        known_cards = engine.player_known_cards(hero.id)
        return {
            "turn": self.turn,
            "scene": engine.s.scene,
            "room": engine.c.room_name(engine.s.room),
            "hero": {"name": hero.name, "role": hero.cls_ko, "hp": hero.hp, "max_hp": hero.max_hp},
            "speaker": npc_name(engine.c.cards[self.speaker]) if self.speaker else None,
            "suggested_line": self.suggested_line,
            "quest": [
                {"label": "실종자 가족 세라와 이야기하기", "done": "scn-q1-npc-sera" in known_cards},
                {"label": "리나의 인형 받기", "done": engine.s.inventory.get("rina_doll", 0) > 0},
                {"label": "우물 입구 조사하기", "done": "well_entrance_known" in engine.s.flags},
                {"label": "우물 아래 갱도에 도착하기", "done": engine.s.scene >= 2},
            ],
            "items": [
                {"name": engine.c.items[item].get("name", item), "quantity": count}
                for item, count in engine.s.inventory.items() if count > 0
            ],
            "clues": self.clues,
            "choices": [
                {"id": choice.id, "label": choice.label, "check": choice.check_text}
                for choice in engine.available_choices()
            ],
            "log": self.log,
        }

    def close(self):
        if self.scenario_db is not None:
            rag.drop_collection(self.scenario_db)

    def choose(self, choice_id, turn, message=None, kind="action"):
        """턴 번호와 선택지 ID를 검사한 뒤 실제 행동을 엔진에 한 번 전달한다."""
        if turn != self.turn:
            raise ValueError("장면이 바뀌었습니다. 현재 선택지를 다시 골라 주세요.")
        choice = next((choice for choice in self.engine.available_choices() if choice.id == choice_id), None)
        if choice is None:
            raise ValueError("지금 화면에 있는 행동을 골라 주세요.")

        self.log.append({"kind": kind, "text": message or choice.label})
        room = self.engine.s.room
        events = self.engine.choose(choice.id)
        self.turn += 1
        if room != self.engine.s.room:
            self.speaker = None
            self.suggested_line = ""
        self.record(events)

    def nearby_npcs(self, choices):
        content = self.engine.c
        known = self.engine.player_known_cards(self.engine.hero.id)
        introductions = {}
        for choice in choices:
            for effect in content.choices.get(choice.id, {}).get("effects", []):
                card_id = effect.get("show")
                name = npc_name(content.cards[card_id]) if card_id else None
                if name and name in choice.label:
                    introductions[card_id] = choice.id
        nearby = {}
        for card_id in dict.fromkeys([*known, *introductions]):
            card = content.cards[card_id]
            name = npc_name(card)
            if name and card["room"] == self.engine.s.room and self.engine.eligible(card_id, self.engine.hero):
                nearby[card_id] = (name, introductions.get(card_id))
        return nearby

    def npc_reply(self, card_id, name, message):
        """NPC별 대화를 관리한다. 후속 질문에는 직전 주제와 확인한 출처를 이어 쓴다."""
        recent = self.conversations.setdefault(card_id, [])
        # 부른 이름은 화면에 남기고 모델에는 실제 대사만 전달한다.
        address = npc_address(message, name)
        line = (message[address.end():] if address else message) or message
        focus = self.dialogue_focus.get(card_id)
        followup = bool(focus and DETAIL_QUERY.fullmatch(line))
        subject, previous_source = focus if followup else (line, None)
        query = subject
        try:
            evidence = self.scenario_evidence(query, card_id, name, previous_source)
            answer, self.suggested_line, source = rag.reply_as_npc(
                name, line, evidence, recent, followup=followup, topic=subject
            )
        except (RuntimeError, ValueError):
            answer = "지금은 답하기 어렵군요."
            self.suggested_line = ""
            source = None
        if source is not None or (not followup and not rag.is_social(name, line)):
            self.dialogue_focus[card_id] = (subject, source)
        recent.append((line, answer))
        del recent[:-3]
        self.log.append({"kind": "npc", "speaker": name, "text": answer})

    def act(self, message, turn):
        """행동 보류를 먼저 처리한다. 연결할 수 없는 시도는 상태 변경 없이 묘사한다."""
        if ACTION_WITHHELD.search(message):
            self.log.append({"kind": "action", "text": message})
            self.log.append({"kind": "say", "text": "일행은 잠시 행동을 보류한다."})
            return
        choices = self.engine.available_choices()
        nearby = self.nearby_npcs(choices)
        target = next((card for card, (name, _) in nearby.items() if name in message), None)
        if target and re.search(APPROACH, re.sub(r"\s+", "", message)) and not re.search(r"않|하지\s*말", message):
            name, introduction = nearby[target]
            if introduction is not None:
                self.choose(introduction, turn, message=message)
            else:
                self.speaker = target
                self.suggested_line = ""
                self.log.append({"kind": "action", "text": message})
                self.log.append({"kind": "say", "text": f"{name}의 곁에 선다."})
            return
        choice = match_action(message, choices)
        if choice is not None:
            self.choose(choice.id, turn, message=message)
            return
        self.log.append({"kind": "action", "text": message})
        if self.speaker in nearby and re.search(r"손을 흔들|고개를 끄덕|고개를 숙|인사", message):
            self.npc_reply(self.speaker, nearby[self.speaker][0], f"({message})")
            return
        room = self.engine.c.room_name(self.engine.s.room)
        try:
            scenes = [text for _, text in self.scenario_evidence(f"{room} {message}")]
            rules = rag.rule_evidence(message) if RULE_QUESTION.search(message) else []
            narration = rag.describe_attempt(message, scenes, room, rules)
        except (RuntimeError, ValueError):
            narration = "주변을 살피며 다음 행동을 정한다."
        self.log.append({"kind": "say", "text": narration})

    def chat(self, message, turn):
        """별표로 표시한 행동, 명시적 룰 질문, NPC 대사를 각각의 처리 흐름에 보낸다."""
        if turn != self.turn:
            raise ValueError("장면이 바뀌었습니다. 지금 장면에서 다시 입력해 주세요.")
        if message.startswith("**") or message.endswith("**"):
            action = message[2:-2].strip()
            if not (message.startswith("**") and message.endswith("**") and action):
                raise ValueError("별표 사이에 행동을 적어 주세요. 예: **세라에게 다가간다**")
            self.act(action, turn)
            return
        choices = self.engine.available_choices()
        nearby = self.nearby_npcs(choices)
        addressed = next((card for card, (name, _) in nearby.items() if npc_address(message, name)), None)
        active_npc = self.speaker in nearby
        if RULE_COMMAND.match(message) or (
            not addressed and (META_RULE_QUESTION.search(message) or (not active_npc and RULE_QUESTION.search(message)))
        ):
            question = RULE_COMMAND.sub("", message, count=1)
            self.log.append({"kind": "speech", "text": message})
            try:
                answer = rag.reply_to_rule(question, rag.rule_evidence(question))
            except (RuntimeError, ValueError):
                answer = "지금은 룰북에서 답을 확인하기 어렵습니다."
            self.log.append({"kind": "rule", "text": answer})
            return

        speaker = addressed or self.speaker
        if speaker not in nearby:
            if len(nearby) == 1:
                speaker = next(iter(nearby))
        if speaker not in nearby:
            self.log.append({"kind": "speech", "text": message})
            narration = (
                "곁에 있던 사람들이 고개를 돌린다. 누구에게 말을 건넬까?"
                if nearby else "목소리가 주변에 울린다. 대답은 돌아오지 않는다."
            )
            self.log.append({"kind": "say", "text": narration})
            return
        name, introduction = nearby[speaker]
        if introduction is not None:
            self.choose(introduction, turn, message=message, kind="speech")
        else:
            self.log.append({"kind": "speech", "text": message})
            self.speaker = speaker
        self.npc_reply(speaker, name, message)

    def search_public(self, question, card_ids=None, limit=4):
        """이번 플레이에서 본 카드만 색인하고 지정된 카드 범위 안에서 검색한다."""
        if not self.public_cards or card_ids == []:
            return []
        if self.scenario_db is None:
            self.scenario_db = rag.create_collection(f"drowned_bell_public_{uuid4().hex}")
        new_ids = [key for key in self.public_cards if key not in self.indexed_cards]
        if new_ids:
            passages = []
            for fragment in new_ids:
                card_id, text = self.public_cards[fragment]
                for number, passage in enumerate(rag.split_scenario(text)):
                    passages.append((f"{fragment}:{number}", card_id, passage))
            if passages:
                documents = [text for _, _, text in passages]
                rag.add_documents(
                    self.scenario_db, [key for key, _, _ in passages], documents,
                    [{"card_id": card_id, "passage_id": key} for key, card_id, _ in passages],
                )
                self.scenario_passages.update({key: (card_id, text) for key, card_id, text in passages})
            self.indexed_cards.update(new_ids)
        where = {"card_id": {"$in": card_ids}} if card_ids is not None else None
        return rag.query(self.scenario_db, question, limit=limit, where=where)

    def scenario_evidence(self, question, speaker_id=None, name=None, previous_source=None):
        """현재 방에서 공개된 자료를 NPC의 카드와 그 NPC에게 귀속된 카드로 좁힌다."""
        room = self.engine.s.room
        cards = self.engine.c.cards
        allowed = set()
        for card_id, _ in self.public_cards.values():
            card = cards[card_id]
            if card["room"] != room:
                continue
            if speaker_id is None:
                if card["kind"] == "desc":
                    allowed.add(card_id)
                continue
            if card_id == speaker_id or name in card.get("known_by_npcs", ()):
                allowed.add(card_id)
        hits = self.search_public(question, sorted(allowed), limit=4)
        evidence = {meta["passage_id"]: text for text, meta in hits}
        selected = {}
        previous = self.scenario_passages.get(previous_source)
        if previous and previous[0] in allowed:
            selected[previous_source] = previous[1]
        identity = f"{speaker_id}:0"
        if speaker_id in allowed and identity in self.scenario_passages:
            selected[identity] = self.scenario_passages[identity][1]
        selected.update(evidence)
        if speaker_id is None:
            return list(selected.items())[:4]

        # Auxiliary cards often name the subject in the first sentence and give
        # the actual clue later. Restore only the fragment the player was shown.
        expanded, seen = [], set()
        for key, passage in selected.items():
            fragment = key.rsplit(":", 1)[0]
            card_id, shown = self.public_cards[fragment]
            if card_id != speaker_id and len(shown) <= 510:
                if fragment in seen:
                    continue
                seen.add(fragment)
                passage = shown
            expanded.append((key, passage))
        return expanded[:4]

    def hint(self, kind, question):
        if kind == "rules":
            candidates = rag.rule_evidence(question)
        elif kind == "scenario":
            candidates = self.search_public(question)
        else:
            raise ValueError("규칙 또는 발견한 단서를 선택해 주세요.")

        if not candidates:
            return {"source": "", "text": "지금 확인할 수 있는 자료가 없습니다."}
        text, metadata = candidates[0]
        source = (
            f"{rag.RULEBOOK_LABEL} · {metadata['page']}쪽"
            if kind == "rules" else f"공개된 단서 · {metadata['card_id']}"
        )
        return {"source": source, "text": text}


class Handler(BaseHTTPRequestHandler):
    def game_session(self):
        try:
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            key = cookies["bell_session"].value if "bell_session" in cookies else None
        except CookieError:
            key = None
        games = self.server.games
        if key in games:
            games.move_to_end(key)
            return key, games[key]
        if len(games) >= 64:
            _, oldest = games.popitem(last=False)
            if oldest is not None:
                oldest.close()
        key = token_urlsafe(24)
        games[key] = None
        self.new_session = key
        return key, None

    def game_view(self, game):
        return {"characters": self.server.characters, "game": game.state() if game is not None else None}

    def reply(self, body, status=200, content_type="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        session = getattr(self, "new_session", None)
        if session is not None:
            secure = "; Secure" if self.server.shared else ""
            self.send_header("Set-Cookie", f"bell_session={session}; Path=/; HttpOnly; SameSite=Lax{secure}")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.reply((HERE / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
        elif self.path == "/api/state":
            _, game = self.game_session()
            self.reply(self.game_view(game))
        else:
            self.reply({"error": "이 주소의 화면을 찾을 수 없습니다."}, status=404)

    def do_POST(self):
        if self.headers.get_content_type() != "application/json":
            self.reply({"error": "게임 화면에서 다시 요청해 주세요."}, status=415)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 8192:
                raise ValueError("입력이 너무 길거나 비어 있습니다.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("게임 화면에서 다시 요청해 주세요.")
            key, game = self.game_session()
            if self.path == "/api/choose":
                if game is None:
                    raise ValueError("먼저 캐릭터를 골라 주세요.")
                game.choose(data.get("id"), data.get("turn"))
                self.reply(self.game_view(game))
            elif self.path == "/api/chat":
                if game is None:
                    raise ValueError("먼저 캐릭터를 골라 주세요.")
                message = data.get("message")
                if not isinstance(message, str) or not 0 < len(message.strip()) <= 200:
                    raise ValueError("할 말이나 행동을 1~200자로 입력해 주세요.")
                game.chat(message.strip(), data.get("turn"))
                self.reply(self.game_view(game))
            elif self.path == "/api/new":
                character = data.get("character")
                if character not in [option["id"] for option in self.server.characters]:
                    raise ValueError("목록에 있는 캐릭터를 골라 주세요.")
                new_game = Game(Engine(content=self.server.content, char_id=character))
                if game is not None:
                    game.close()
                self.server.games[key] = new_game
                self.reply(self.game_view(new_game))
            else:
                self.reply({"error": "지원하지 않는 요청입니다."}, status=404)
        except ValueError as error:
            self.reply({"error": str(error)}, status=400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", action="store_true", help="기본 브라우저에서 게임 열기")
    parser.add_argument("--share", action="store_true", help="다른 기기에서 열 수 있는 임시 링크 생성")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=Path, help="대화에 사용할 GGUF 모델 파일")
    parser.add_argument("--rulebook", type=Path, default=RULEBOOK, help="팀 베이직 룰 한국어판 Rules.pdf")
    parser.add_argument("--llama-server", type=Path, help="로컬 GPU 모델 실행 파일")
    parser.add_argument("--model-url", help="이미 실행 중인 로컬 모델의 주소")
    args = parser.parse_args()
    if args.llama_server and (not args.model or args.model_url):
        parser.error("--llama-server에는 --model이 필요하며 --model-url과 함께 쓸 수 없습니다.")
    print("게임 마스터를 준비하는 중입니다. 첫 화면까지 잠시 기다려 주세요.", flush=True)
    models = RUNTIME / "models"
    preferred = models / "qwen-answer-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    other_shard = preferred.with_name(preferred.name.replace("00001-of", "00002-of"))
    model = args.model or (preferred if preferred.is_file() and other_shard.is_file() else None)
    with ExitStack() as stack:
        model_url, model_key = args.model_url, None
        if args.llama_server:
            model_url, model_key = stack.enter_context(rag.local_model(args.llama_server, args.model))
        rag.initialize(models, args.rulebook, model, model_url=model_url, model_key=model_key)
        content = Content()
        server = stack.enter_context(HTTPServer(("127.0.0.1", args.port), Handler))
        server.content = content
        server.characters = []
        for char_id, name, role in content.characters():
            hero = content.new_party(char_id)[0]
            server.characters.append({"id": char_id, "name": name, "role": role, "hp": hero.max_hp})
        server.games = OrderedDict()
        server.shared = args.share
        address = f"http://127.0.0.1:{server.server_port}/"
        if args.share:
            sys.path.insert(0, str(RUNTIME / "gradio_deps"))
            from gradio.networking import setup_tunnel

            address = setup_tunnel("127.0.0.1", server.server_port, token_urlsafe(32), None, None)
        print(f"게임 주소 {address}\n이 창을 닫으면 게임이 종료됩니다.", flush=True)
        if args.open:
            Timer(0.25, webbrowser.open, args=(address,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
