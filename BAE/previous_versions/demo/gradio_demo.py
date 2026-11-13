"""Gradio UI for the same Engine, Game, and RAG used by the browser demo."""

import argparse
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from secrets import token_urlsafe
from threading import RLock

import gradio as gr

from app import Content, Engine, Game, RULEBOOK, RUNTIME, rag


class Sessions:
    """Keep game objects on the server; a browser session only keeps its token and turn."""

    def __init__(self, content):
        self.content = content
        self.games = OrderedDict()
        self.lock = RLock()
        self.characters = [(f"{name} · {role}", character) for character, name, role in content.characters()]

    def close(self, token):
        with self.lock:
            game = self.games.pop(token, None)
            if game is not None:
                game.close()

    def get(self, token):
        game = self.games.get(token)
        if game is None:
            raise gr.Error("캐릭터를 선택하고 게임을 시작해 주세요.")
        self.games.move_to_end(token)
        return game

    @staticmethod
    def view(game):
        state = game.state()
        hero = state["hero"]
        items = ", ".join(f"{item['name']} ×{item['quantity']}" for item in state["items"]) or "없음"
        status = (
            f"{state['room']} · 장면 {state['scene']} · {state['turn']}턴\n"
            f"{hero['name']} · {hero['role']} · HP {hero['hp']}/{hero['max_hp']}\n"
            f"소지품: {items}"
        )
        labels = {"rule": "룰북", "check": "판정", "action": "내 행동", "speech": "내 대사"}
        history = []
        for event in state["log"][-80:]:
            kind = event["kind"]
            label = event.get("speaker") or labels.get(kind)
            text = f"{label} · {event['text']}" if label else event["text"]
            history.append({"role": "user" if kind in ("action", "speech") else "assistant", "content": text})
        choices = [
            (f"{choice['label']} · {choice['check']}" if choice["check"] else choice["label"], choice["id"])
            for choice in state["choices"]
        ]
        return state["turn"], status, history, gr.update(choices=choices, value=None, interactive=bool(choices)), state["suggested_line"]

    def start(self, character, previous):
        if character not in {value for _, value in self.characters}:
            raise gr.Error("플레이할 캐릭터를 골라 주세요.")
        with self.lock:
            self.close(previous)
            if len(self.games) >= 64:
                self.close(next(iter(self.games)))
            token = token_urlsafe(24)
            game = Game(Engine(content=self.content, char_id=character))
            self.games[token] = game
            return token, *self.view(game), ""

    def choose(self, token, turn, choice):
        if not choice:
            raise gr.Error("행동을 선택해 주세요.")
        with self.lock:
            game = self.get(token)
            try:
                game.choose(choice, turn)
            except ValueError as error:
                raise gr.Error(str(error)) from error
            return self.view(game)

    def chat(self, token, turn, message):
        message = (message or "").strip()
        if not 0 < len(message) <= 200:
            raise gr.Error("할 말이나 행동을 1~200자로 입력해 주세요.")
        with self.lock:
            game = self.get(token)
            try:
                game.chat(message, turn)
            except ValueError as error:
                raise gr.Error(str(error)) from error
            return *self.view(game), ""


def make_demo(content):
    sessions = Sessions(content)
    with gr.Blocks(title="잠긴 종 · Gradio 프로토타입", analytics_enabled=False) as demo:
        token = gr.State(value=None, time_to_live=3600, delete_callback=sessions.close)
        turn = gr.State(value=0)
        gr.Markdown("# 잠긴 종\n캐릭터를 골라 에코 타운의 우물과 실종 사건을 조사하세요.")
        with gr.Row():
            character = gr.Dropdown(choices=sessions.characters, label="플레이어 캐릭터")
            start = gr.Button("게임 시작 / 새 게임", variant="primary")
        status = gr.Textbox(label="현재 상태", interactive=False, lines=3)
        history = gr.Chatbot(label="이야기", height=450, render_markdown=False, sanitize_html=True)
        with gr.Row():
            choices = gr.Dropdown(choices=[], label="가능한 행동", interactive=False)
            choose = gr.Button("행동 실행")
        with gr.Row():
            message = gr.Textbox(label="자유 입력", placeholder="대사 또는 **행동**을 입력하세요. 규칙 질문은 /규칙으로 시작합니다.")
            send = gr.Button("보내기", variant="primary")
        with gr.Row():
            suggestion = gr.Textbox(label="추천 대사", interactive=False)
            use_suggestion = gr.Button("추천 대사 가져오기")
        gr.Markdown("룰북 검색 자료는 팀의 D&D 베이직 룰 한국어판(2018)입니다.")
        gr.Markdown(
            "Includes material from the [System Reference Document 5.2](https://www.dndbeyond.com/srd) "
            "by Wizards of the Coast LLC under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)."
        )

        view = [turn, status, history, choices, suggestion]
        group = {"concurrency_id": "game", "concurrency_limit": 1}
        start.click(sessions.start, [character, token], [token, *view, message], api_name="start_game", **group)
        choose.click(sessions.choose, [token, turn, choices], view, api_name="choose_action", **group)
        send.click(sessions.chat, [token, turn, message], [*view, message], api_name="send_message", **group)
        message.submit(sessions.chat, [token, turn, message], [*view, message], api_name="submit_message", **group)
        use_suggestion.click(lambda text: text, suggestion, message, queue=False)
    return demo.queue(default_concurrency_limit=1, max_size=16)


def main():
    parser = argparse.ArgumentParser(description="잠긴 종 Gradio 프로토타입")
    runtime = parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument("--llama-server", type=Path, help="llama-server 실행 파일")
    runtime.add_argument("--model-url", help="별칭 qwen을 쓰는 로컬 모델 서버")
    parser.add_argument("--model", type=Path, help="GGUF 모델 파일")
    parser.add_argument("--rulebook", type=Path, default=RULEBOOK, help="팀 베이직 룰 한국어판 Rules.pdf")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-open", action="store_true", help="브라우저를 자동으로 열지 않음")
    args = parser.parse_args()
    if args.llama_server and not args.model:
        parser.error("--llama-server에는 --model을 함께 지정하세요.")
    if not args.rulebook.is_file():
        parser.error("팀의 한국어판 Rules.pdf를 --rulebook에 지정하세요.")

    with ExitStack() as stack:
        model_url, model_key = args.model_url, None
        if args.llama_server:
            model_url, model_key = stack.enter_context(rag.local_model(args.llama_server, args.model))
        rag.initialize(RUNTIME / "models", args.rulebook, model_url=model_url, model_key=model_key)
        demo = make_demo(Content())
        stack.callback(demo.close)
        demo.launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=not args.no_open)


if __name__ == "__main__":
    main()
