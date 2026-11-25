"""저장된 대화는 그대로 두고 채팅 화면용 초상화와 안전한 본문을 만듭니다."""
from html import escape
from pathlib import Path
from urllib.parse import quote

from response_guard import contains_han, _language_issue


PORTRAIT_DIR = Path(__file__).resolve().parent / "assets" / "portraits"
# 채팅·인물 목록용 256px WebP입니다. 원본 그림은 이전 배포본에 보관합니다.
PORTRAIT_FILES = {
    "세라": "sera.webp",
    "할빅": "halvik-v12.webp",
    "메라": "mera.webp",
    "그란": "gran-v12.webp",
    "오단": "odan-v12.webp",
}
PORTRAIT_CARD_FILES = {
    "scn-q1-npc-sera": "sera.webp",
    "scn-s1-npc-halvik": "halvik-v12.webp",
    "scn-s1-npc-mera": "mera.webp",
    "scn-m-npc-smith": "gran-v12.webp",
    "scn-m-npc-innkeeper": "odan-v12.webp",
}

# 입력 종류 표시는 저장된 발언을 고치지 않고 화면용 HTML에만 덧붙입니다.
# 고정 SVG만 사용하므로 사용자 입력에서 이미지 주소나 HTML을 실행하지 않습니다.
# 얇은 외곽선 대신 채운 도형을 사용합니다. 파일 다운로드가 없는 작은 SVG입니다.
ACTION_ICON_SVG = ('<svg class="trpg-input-icon" xmlns="http://www.w3.org/2000/svg" '
                   'width="16" height="16" viewBox="0 0 24 24" fill="currentColor" '
                   'aria-hidden="true" focusable="false">'
                   '<path d="M13.49 5.48c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2z'
                   'M9.89 19.38l1-4.4 2.1 2v6h2v-7.5l-2.1-2 .6-3c1.3 1.5 3.3 2.5 5.5 2.5v-2'
                   'c-1.9 0-3.5-1-4.3-2.4l-1-1.6c-.4-.6-1-1-1.7-1-.3 0-.5.1-.8.1l-5.2 2.2v4.7h2'
                   'v-3.4l1.8-.7-1.6 8.1-4.9-1-.4 2 7 1.4z"/></svg>')
SPEECH_ICON_SVG = ('<svg class="trpg-input-icon" xmlns="http://www.w3.org/2000/svg" '
                   'width="16" height="16" viewBox="0 0 24 24" fill="currentColor" '
                   'aria-hidden="true" focusable="false">'
                   '<path d="M6 17h3l2-4V7H5v6h3zm8 0h3l2-4V7h-6v6h3z"/></svg>')
SEND_ICON_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
                 'viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">'
                 '<path d="M2 21 23 12 2 3v7l15 2-15 2z"/></svg>')


class _PresentedText(str):
    """이미 변환한 Python 값만 식별하며 사용자 HTML 문자열은 신뢰하지 않습니다."""


def _image_as_text(tokens, index, *args, **kwargs):
    # 대화에서 제시한 외부 이미지 주소는 브라우저가 요청하지 않습니다.
    return escape(tokens[index].content, quote=True)


def _action_strong_open(tokens, index, *args, **kwargs):
    # 이 앱에서 **표시**는 행동입니다. __강조__와 코드 안의 별표는 구분합니다.
    if tokens[index].markup != "**":
        return "<strong>"
    return ('<span class="trpg-inline-action" role="group" aria-label="행동" title="행동">'
            + ACTION_ICON_SVG + "<strong>")


def _action_strong_close(tokens, index, *args, **kwargs):
    return "</strong></span>" if tokens[index].markup == "**" else "</strong>"


def _build_markdown(mark_actions=False):
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return None
    renderer = MarkdownIt("commonmark", {"html": False, "breaks": True})
    renderer.enable("table")
    renderer.renderer.rules["image"] = _image_as_text
    if mark_actions:
        renderer.renderer.rules["strong_open"] = _action_strong_open
        renderer.renderer.rules["strong_close"] = _action_strong_close
    return renderer


_MARKDOWN = _build_markdown()
_ACTION_MARKDOWN = _build_markdown(mark_actions=True)


def register_portraits():
    """초상화·장소 배경의 정확한 폴더만 Gradio 정적 경로로 등록합니다."""
    if not PORTRAIT_DIR.is_dir():
        return False
    import gradio as gr
    backgrounds = PORTRAIT_DIR.parent / "backgrounds"
    gr.set_static_paths([str(PORTRAIT_DIR.resolve())] + ([str(backgrounds.resolve())] if backgrounds.is_dir() else []))
    return True


def _portrait_url(speaker, speaker_id=None):
    filename = PORTRAIT_CARD_FILES.get(speaker_id) or PORTRAIT_FILES.get(speaker)
    if filename is None:
        return None
    directory = PORTRAIT_DIR.resolve()
    image = (directory / filename).resolve()
    # 고정 파일명만 허용하며 링크로 다른 디렉터리를 열어 두지 않습니다.
    if image.parent != directory or not image.is_file():
        return None
    return "./gradio_api/file=" + quote(image.as_posix(), safe="/:")


def _body_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block["text"] for block in content
                         if isinstance(block, dict) and block.get("type") == "text"
                         and isinstance(block.get("text"), str))
    return ""


def _speaker_and_body(message, body):
    if message["role"] == "user":
        kind = message.get("input_kind")
        label = "플레이어 · 말하기" if kind == "speech" else "플레이어 · 행동" if kind == "action" else "플레이어"
        return label, body
    if message.get("kind") == "narration":
        return "게임 마스터", body
    if "speaker" in message:
        # 명시적인 빈 값은 GM/시스템입니다. 본문의 인물 이름으로 바꾸지 않습니다.
        value = message["speaker"]
        speaker = value.strip()[:80] if isinstance(value, str) else ""
    else:
        # 예전 저장 파일의 첫 줄 표지만 읽으며, 본문에서 화자를 추론하지 않습니다.
        first, separator, _ = body.partition("\n")
        candidate = first[2:-2].strip() if first.startswith("**") and first.endswith("**") else ""
        speaker = candidate if separator and candidate in PORTRAIT_FILES else ""
    if speaker:
        first, separator, rest = body.partition("\n")
        if separator and first.strip() == "**" + speaker + "**":
            body = rest
    return speaker or "게임 마스터", body


def _action_marker_state(body):
    """실제 Markdown 행동 표시와 그 밖의 글이 있는지 화면 표시용으로 확인합니다.

    코드·이스케이프·밑줄 강조를 행동으로 오인하지 않습니다. Markdown 라이브러리를
    읽을 수 없는 환경은 완전한 **행동** 한 구간만 예전 저장 이력의 행동으로 봅니다.
    이 판별로 게임 행동을 실행하거나 저장된 input_kind를 바꾸지는 않습니다.
    """
    if _MARKDOWN is None:
        text = body.strip()
        full = (text.startswith("**") and text.endswith("**") and text.count("**") == 2
                and bool(text[2:-2].strip()) and not text.startswith("***"))
        return full, False
    has_action, has_other = False, False
    for block in _MARKDOWN.parse(body):
        depth = 0
        if block.type in {"fence", "code_block", "html_block"} and block.content.strip():
            has_other = True
        for token in block.children or []:
            if token.type == "strong_open" and token.markup == "**":
                has_action = True
                depth += 1
            elif token.type == "strong_close" and token.markup == "**":
                depth = max(0, depth - 1)
            elif depth == 0 and token.content.strip():
                has_other = True
    return has_action, has_other


def _player_input_kind(message, body):
    """명시된 입력 종류를 우선하고, 혼합 입력과 이전 저장 이력만 보완합니다."""
    kind = message.get("input_kind")
    if kind == "action":
        return "action"
    if kind == "mixed":
        return "mixed"
    has_action, has_other = _action_marker_state(body)
    if has_action and has_other:
        return "mixed"
    if kind == "speech":
        return "speech"
    return "action" if has_action else "speech"


def _input_badge(kind):
    label = {"action": "행동", "speech": "말하기", "mixed": "말·행동"}[kind]
    icon = ACTION_ICON_SVG if kind in {"action", "mixed"} else SPEECH_ICON_SVG
    return ('<span class="trpg-input-badge" title="' + label + '">'
            + icon + '<span>' + label + '</span></span>')


def _render_body(body, mark_actions=False):
    if _MARKDOWN is None:
        return "<p>" + escape(body, quote=True).replace("\n", "<br>\n") + "</p>"
    renderer = _ACTION_MARKDOWN if mark_actions else _MARKDOWN
    return renderer.render(body)


def _public_speaker_and_body(message):
    """예전 저장 이력도 같은 언어 정책으로 표시하며 사용자 원문은 유지합니다."""
    speaker, body = _speaker_and_body(message, _body_text(message.get("content", "")))
    language = _language_issue(body)
    if message["role"] == "assistant" and ((language and language["severity"] == "error") or contains_han(speaker)):
        return "게임 마스터", "한국어로 완성되지 않은 이전 응답은 표시하지 않았습니다."
    return speaker, body


def conversation_text(messages):
    """공개된 전체 대화를 복사용 문자열로 만들며 화면 HTML과 진단 자료는 넣지 않습니다."""
    rows = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            continue
        speaker, body = _public_speaker_and_body(message)
        if message["role"] == "user":
            rows.append("user: " + body)
        else:
            heading = "**" + speaker + "**\n" if speaker != "게임 마스터" else ""
            rows.append("assistant: " + heading + body)
    return "\n\n".join(rows)


def present_messages(messages):
    """원본 메시지를 변형하지 않고 Gradio의 role/content 목록을 반환합니다.

    초상화는 각 메시지에 저장된 화자를 사용합니다. 현재 선택한 NPC가 바뀌어도
    기존 대화의 얼굴은 바뀌지 않습니다. HTML은 서버에서 만든 바깥 틀만 허용합니다.
    """
    rendered = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            continue
        role = message["role"]
        content = message.get("content", "")
        if isinstance(content, _PresentedText):
            rendered.append({"role": role, "content": content})
            continue
        speaker, body = _public_speaker_and_body(message)
        input_kind = _player_input_kind(message, body) if role == "user" else None
        # 이름과 입력 종류를 서로 다른 요소로 두어 색·아이콘·읽기 순서가 분명해집니다.
        heading = ("플레이어 " + _input_badge(input_kind)) if role == "user" else escape(speaker, quote=True)
        image = _portrait_url(speaker, message.get("speaker_id")) if role == "assistant" and speaker != "게임 마스터" else None
        if image:
            avatar = ('<img class="trpg-avatar" src="' + escape(image, quote=True)
                      + '" alt="' + escape(speaker, quote=True) + '" loading="lazy">')
        else:
            mark = "나" if role == "user" else "◇"
            avatar = '<span class="trpg-avatar trpg-avatar-fallback" aria-hidden="true">' + mark + "</span>"
        role_class = ("user user-" + input_kind) if role == "user" else "system" if speaker == "게임 마스터" else "npc"
        html = ('<div class="trpg-chat-row ' + role_class + '">' + avatar
                + '<div class="trpg-chat-main"><div class="trpg-speaker">'
                + heading + '</div><div class="trpg-message-body">'
                + _render_body(body, mark_actions=input_kind == "mixed") + "</div></div></div>")
        rendered.append({"role": role, "content": _PresentedText(html)})
    return rendered
