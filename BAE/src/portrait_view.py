"""저장된 대화는 그대로 두고 채팅 화면용 초상화와 안전한 본문을 만듭니다."""
from html import escape
from pathlib import Path
from urllib.parse import quote

from response_guard import contains_han


PORTRAIT_DIR = Path(__file__).resolve().parent / "assets" / "portraits"
PORTRAIT_FILES = {
    "세라": "sera.png",
    "할빅": "halvik.png",
    "메라": "mera.png",
    "그란": "gran.png",
    "오단": "odan.png",
}


class _PresentedText(str):
    """이미 변환한 Python 값만 식별하며 사용자 HTML 문자열은 신뢰하지 않습니다."""


def _image_as_text(tokens, index, *args, **kwargs):
    # 대화에서 제시한 외부 이미지 주소는 브라우저가 요청하지 않습니다.
    return escape(tokens[index].content, quote=True)


def _build_markdown():
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return None
    renderer = MarkdownIt("commonmark", {"html": False, "breaks": True})
    renderer.enable("table")
    renderer.renderer.rules["image"] = _image_as_text
    return renderer


_MARKDOWN = _build_markdown()


def register_portraits():
    """초상화 파일이 있는 정확한 폴더 하나만 Gradio 정적 경로로 등록합니다."""
    if not PORTRAIT_DIR.is_dir():
        return False
    import gradio as gr
    gr.set_static_paths([str(PORTRAIT_DIR.resolve())])
    return True


def _portrait_url(speaker):
    filename = PORTRAIT_FILES.get(speaker)
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
        return "플레이어", body
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


def _render_body(body):
    if _MARKDOWN is None:
        return "<p>" + escape(body, quote=True).replace("\n", "<br>\n") + "</p>"
    return _MARKDOWN.render(body)


def _public_speaker_and_body(message):
    """예전 저장 이력도 같은 언어 정책으로 표시하며 사용자 원문은 유지합니다."""
    speaker, body = _speaker_and_body(message, _body_text(message.get("content", "")))
    if message["role"] == "assistant" and (contains_han(body) or contains_han(speaker)):
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
        image = _portrait_url(speaker) if role == "assistant" else None
        if image:
            avatar = ('<img class="trpg-avatar" src="' + escape(image, quote=True)
                      + '" alt="' + escape(speaker, quote=True) + '" loading="lazy">')
        else:
            mark = "나" if role == "user" else "◇"
            avatar = '<span class="trpg-avatar trpg-avatar-fallback" aria-hidden="true">' + mark + "</span>"
        role_class = "user" if role == "user" else "system" if speaker == "게임 마스터" else "npc"
        html = ('<div class="trpg-chat-row ' + role_class + '">' + avatar
                + '<div class="trpg-chat-main"><div class="trpg-speaker">'
                + escape(speaker, quote=True) + '</div><div class="trpg-message-body">'
                + _render_body(body) + "</div></div></div>")
        rendered.append({"role": role, "content": _PresentedText(html)})
    return rendered
