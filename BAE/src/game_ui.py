"""Gradio 화면과 게임 처리 함수를 연결합니다.

입력: 모델·검색 함수가 담긴 api와, 화면에서 받은 입력문·선택지 ID·접속 토큰입니다.
처리: PlaySessions에서 해당 사용자의 게임을 찾고 대화·행동 처리를 호출합니다.
render는 처리 후 저장된 대화, HP, 소지품, 가능한 행동을 읽어 화면 값으로 정리합니다.
출력: 실행 가능한 Gradio Blocks 화면과 각 구성요소를 갱신할 값입니다.

게임 진행과 응답 생성은 gameplay.py에 연결됩니다. 캐릭터 수치는 game.model,
금액 표기는 game.engine을 사용합니다. 화면을 다시 그릴 때는 저장된 추천을 읽습니다."""
import html
import json
import math
import re
from copy import deepcopy
from functools import wraps
from pathlib import Path
from queue import Empty, Queue
import tempfile
from threading import Event, Thread
import time
import traceback
from types import GeneratorType, SimpleNamespace
from urllib.parse import quote as quote_icon

import gradio as gr

from game.content import npc_name
from game.engine import fmt_money
from game.model import ABILITY_KO
from gameplay import GameEngine, PlaySessions
from portrait_view import register_portraits, present_messages, ACTION_ICON_SVG, SEND_ICON_SVG
from play_view import (action_entries, inventory_panel, dice_panel, encounter_panel, journey_panel,
                       discovery_data, discovery_samples, discovery_overview, discovery_actions, scene_background, people_panel)
from diagnostic_export import build_copy_report, build_plain_conversation
from diagnostics_live import RequestDiagnostics, current_request_id, current_stage
from action_requirements import equipment_requirements_panel
from combat_feedback import CombatFeedbackTracker


COPY_CONVERSATION_JS = """async (text) => {
    const isDiagnostics = typeof text === 'string' && text.startsWith('TRPG 전체');
    const label = isDiagnostics ? '전체 진단 복사' : '대화 복사';
    if (!text) return label + ' · 내용 없음';
    if (isDiagnostics) {
        const client = window.__trpgControls?.diagnostics();
        if (client) text += '\\n\\n[브라우저 입력·대기 진단]\\n```json\\n' + JSON.stringify(client, null, 2) + '\\n```';
    }
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
        } else {
            throw new Error('fallback');
        }
        return label + ' · 완료';
    } catch (_) {
        const field = document.createElement('textarea');
        field.value = text;
        field.style.cssText = 'position:fixed;left:-9999px;top:0';
        field.setAttribute('readonly', '');
        const focused = document.activeElement;
        document.body.appendChild(field);
        field.select();
        let copied = false;
        try { copied = document.execCommand('copy'); }
        finally { field.remove(); if (focused && focused.focus) focused.focus(); }
        return label + (copied ? ' · 완료' : ' · 다시 시도');
    }
}"""


GAME_HELP = """### 이야기 이어가기

아래 입력칸에 상대에게 할 말이나 행동을 적으세요. **보내기** 또는 Enter로 전송하고 Shift+Enter로 줄을 바꿉니다.

달리는 사람 모양의 **행동 버튼**을 누르면 커서 자리에 빈 `****`가 생깁니다. 가운데에 행동을 적은 뒤 화살표 버튼으로 보내세요. 기존에 쓴 말은 지워지지 않습니다. 직접 `**우물 안을 살핀다**`처럼 입력해도 됩니다.

**추천과 선택** · `[대화]`는 말하기 추천, `[행동]`·`[이동]`은 행동 추천입니다. 누르면 바로 전달됩니다. 장면 선택지는 지금 실행할 수 있는 행동입니다.

**주사위 판정** · 행동을 고른 뒤 가능한 판정 방법과 수정치·난이도를 확인하세요. 방법을 선택하고 주사위를 한 번 굴리면 그 결과로 이야기가 이어집니다. 실패해도 같은 결과를 다시 굴리지 않고 다른 선택을 찾아 진행합니다.

**새 여정과 저장** · 위에서 캐릭터를 고른 뒤 새 여정을 시작합니다. 기록 탭에서 저장하거나 이어 할 수 있습니다.

**지도·인물·정보** · 발견한 길, 알게 된 사람, 실제로 들은 단서를 각 탭에서 확인합니다. 도입 글이 나타나는 중에는 이야기를 누르면 바로 끝까지 읽을 수 있습니다.

**개발자 모드** · 모드를 켜도 일반 이야기의 선택지와 추천은 발견·진행 순서를 따릅니다. 보라색 전용 도구에서만 이동·만남·판정·전투 조건을 건너뛰어 시험합니다. 사망 후에는 기록 목록을 새로고침하고 이전 기록을 선택해 복구합니다. 사용한 조작은 기록에 남습니다.
"""


GAME_CONTROLS_JS = """async () => {
    window.__trpgControls?.dispose();
    // Gradio가 DOM을 늦게 붙이는 경우에도 초기화를 조용히 포기하지 않습니다.
    const locate = () => ({field: document.querySelector('#game-message textarea'),
        action: document.querySelector('#game-mark-action'), send: document.querySelector('#game-send-message'),
        clock: document.querySelector('#game-turn-clock .trpg-turn-clock')});
    let nodes = locate();
    for (let attempt = 0; attempt < 100 && Object.values(nodes).some(node => !node); attempt++) {
        await new Promise(resolve => setTimeout(resolve, 50));
        nodes = locate();
    }
    const {field, action, send, clock} = nodes;
    const app = document.querySelector('.gradio-container');
    const chat = document.querySelector('#game-chat');
    if (!field || !action || !send || !clock) {
        window.__trpgControlInitFailure = {at: new Date().toISOString(),
            missing: Object.entries(nodes).filter(([, node]) => !node).map(([key]) => key)};
        return [];
    }
    window.__trpgControlInitFailure = null;
    let composing = false;
    let deferredAction = false;
    let commitFrame = 0;
    let restoreFrame = 0;
    let serial = 0;
    let active = null;
    let interval = null;
    let revealFrame = 0;
    let revealText = [];
    let revealRows = [];
    let hasStarted = false;
    const requests = new Map();
    const seconds = clock.querySelector('.trpg-turn-seconds');
    const stageLabel = clock.querySelector('.trpg-turn-stage');
    const clientRequests = [];
    const clientErrors = [];
    const observeError = event => {
        const message = event.message || event.reason?.message || String(event.reason || '브라우저 오류');
        clientErrors.push({at: new Date().toISOString(), type: event.type, message: String(message).slice(0, 1500)});
        if (clientErrors.length > 20) clientErrors.shift();
    };
    window.addEventListener('error', observeError);
    window.addEventListener('unhandledrejection', observeError);
    action.setAttribute('aria-label', '행동 표시 추가');
    action.setAttribute('title', '커서 위치에 행동 표시 추가');
    send.setAttribute('aria-label', '메시지 보내기');
    send.setAttribute('title', '보내기 · Enter');

    const completeOpening = () => {
        if (revealFrame) cancelAnimationFrame(revealFrame);
        revealFrame = 0;
        revealText.forEach(row => { row.node.nodeValue = row.text; });
        revealRows.forEach(row => { row.style.visibility = ''; });
        revealText = []; revealRows = [];
        if (app) app.dataset.revealing = 'false';
    };
    const setStage = stage => {
        completeOpening();
        if (app) app.dataset.trpgStage = stage;
        // Gradio의 스타일 범위와 관계없이 한 화면만 표시하고 나머지는 접근성 트리에서도 숨깁니다.
        for (const [identifier, screenStage] of [['game-landing', 'landing'],
                ['game-character-screen', 'characters'], ['game-play-screen', 'play']]) {
            const screen = document.querySelector('#' + identifier);
            if (!screen) continue;
            const visible = stage === screenStage;
            screen.style.setProperty('display', visible ? 'flex' : 'none', 'important');
            screen.hidden = !visible;
            screen.inert = !visible;
            screen.setAttribute('aria-hidden', String(!visible));
        }
        if (stage === 'play') window.dispatchEvent(new Event('resize'));
    };
    const selectCharacter = identifier => {
        document.querySelectorAll?.('.character-card').forEach(card => {
            const selected = card.id === 'character-card-' + identifier;
            card.setAttribute('aria-pressed', String(selected));
            card.dataset.selected = String(selected);
        });
        return identifier;
    };
    const revealOpening = () => {
        completeOpening();
        if (!chat || !app) return;
        const bodies = [...chat.querySelectorAll('.trpg-chat-row:not(.user) .trpg-message-body')]
            .filter(body => body.textContent.length > 30);
        for (const body of bodies) {
            const row = body.closest('.trpg-chat-row');
            if (row) revealRows.push(row);
            const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT);
            let node;
            while ((node = walker.nextNode())) {
                if (node.nodeValue) revealText.push({node, text: node.nodeValue, row});
            }
        }
        const total = revealText.reduce((sum, row) => sum + row.text.length, 0);
        if (!total) return;
        revealRows.forEach(row => { row.style.visibility = 'hidden'; });
        revealText.forEach(row => { row.node.nodeValue = ''; });
        app.dataset.revealing = 'true';
        const started = performance.now();
        const scroller = chat.querySelector('.bubble-wrap');
        if (scroller) scroller.scrollTop = 0;
        const draw = () => {
            const count = Math.min(total, Math.floor((performance.now() - started) * .085));
            let remaining = count;
            let current = null;
            for (const row of revealText) {
                row.node.nodeValue = row.text.slice(0, Math.max(0, remaining));
                if (remaining > 0 && row.row) { row.row.style.visibility = ''; current = row.row; }
                remaining -= row.text.length;
            }
            if (scroller && current) {
                const bottom = current.getBoundingClientRect().bottom - scroller.getBoundingClientRect().bottom;
                if (bottom > 0) scroller.scrollTop += bottom;
            }
            if (count >= total) completeOpening();
            else revealFrame = requestAnimationFrame(draw);
        };
        revealFrame = requestAnimationFrame(draw);
    };
    const startFinished = notice => {
        finish('start_game');
        // gr.State는 브라우저에 토큰을 노출하지 않습니다. 서버가 렌더까지 완료한 공개 신호만 확인합니다.
        if (notice !== '여정을 시작했습니다.') return;
        hasStarted = true;
        setStage('play');
        // Gradio가 마지막 응답 HTML을 그린 다음, 저장값을 건드리지 않고 표시만 펼칩니다.
        revealFrame = requestAnimationFrame(() => { revealFrame = requestAnimationFrame(revealOpening); });
    };

    const stopClock = () => {
        if (interval !== null) clearInterval(interval);
        interval = null;
        active = null;
        clock.dataset.running = 'false';
    };
    const cancelAll = () => {
        clientRequests.filter(row => row.status === 'pending').forEach(row => {
            row.status = 'cancelled'; row.elapsed_seconds = (performance.now() - row.started) / 1000;
        });
        requests.clear(); stopClock();
    };
    const paint = () => {
        if (active) seconds.textContent = ((performance.now() - active.started) / 1000).toFixed(1) + '초';
    };
    const begin = kind => {
        completeOpening();
        const now = performance.now();
        if (active && active.kind === kind && now - active.started < 150) return;
        // 취소는 이전 시계를 끝내고, 취소 처리 자체의 짧은 경과 시간부터 표시합니다.
        if (kind === 'cancel_pending') cancelAll();
        if (interval !== null) clearInterval(interval);
        const operation = {id: ++serial, kind, input: field.value, started: now, started_at: new Date().toISOString(), status: 'pending'};
        clientRequests.push(operation);
        if (clientRequests.length > 100) clientRequests.shift();
        if (stageLabel) stageLabel.textContent = '입력 전달 중';
        const queue = requests.get(kind) || [];
        queue.push(operation.id);
        requests.set(kind, queue);
        active = operation;
        clock.dataset.running = 'true';
        paint();
        interval = setInterval(paint, 100);
    };
    const finish = kind => {
        const queue = requests.get(kind);
        if (!queue || !queue.length) return;
        const completed = queue.shift();
        if (!queue.length) requests.delete(kind);
        // 먼저 시작한 요청의 종료가 새 요청의 시계를 끄지 않게 합니다.
        const row = clientRequests.find(item => item.id === completed);
        if (row) { row.status = 'completed'; row.elapsed_seconds = (performance.now() - row.started) / 1000; }
        if (active && active.id === completed) { paint(); stopClock(); }
    };

    const updateStatus = data => {
        if (!data || !active || !data.request_id || data.status === 'idle') return;
        if (stageLabel) stageLabel.textContent = data.label || '처리 중';
        active.server = data;
    };
    const diagnosticsSnapshot = () => ({schema: 'trpg_client_trace.v1', captured_at: new Date().toISOString(),
        online: navigator.onLine, visibility: document.visibilityState,
        errors: clientErrors.slice(), control_initialization: window.__trpgControlInitFailure,
        requests: clientRequests.map(row => ({id: row.id, kind: row.kind, input: row.input,
            started_at: row.started_at, status: row.status,
            elapsed_seconds: row.elapsed_seconds ?? (performance.now() - row.started) / 1000, server: row.server || null}))});
    const nativeSend = () => begin('send_message');
    const nativeEnter = event => {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && !composing && event.keyCode !== 229) begin('submit_message');
    };
    send.addEventListener('click', nativeSend, true);
    field.addEventListener('keydown', nativeEnter, true);

    const insertAction = () => {
        completeOpening();
        if (composing) {
            // 한글 한 글자가 완성되기 전에는 문자열을 건드리지 않습니다.
            deferredAction = true;
            field.focus({preventScroll: true});
            return;
        }
        const before = field.value;
        const at = Number.isInteger(field.selectionStart) ? field.selectionStart : before.length;
        // 선택한 글자도 지우지 않고 선택 시작 위치에 빈 행동 칸만 삽입합니다.
        field.setRangeText('****', at, at, 'end');
        const value = field.value;
        field.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertText', data: '****'}));
        field.focus({preventScroll: true});
        field.setSelectionRange(at + 2, at + 2);
        if (restoreFrame) cancelAnimationFrame(restoreFrame);
        restoreFrame = requestAnimationFrame(() => {
            restoreFrame = 0;
            if (field.value === value) field.setSelectionRange(at + 2, at + 2);
        });
    };
    const keepCaret = event => { event.preventDefault(); };
    const compositionStart = () => { composing = true; };
    const compositionEnd = () => {
        composing = false;
        if (deferredAction) {
            deferredAction = false;
            commitFrame = requestAnimationFrame(() => {
                commitFrame = 0;
                if (composing) deferredAction = true;
                else insertAction();
            });
        }
    };
    const preserveIME = event => {
        // Enter의 기본 IME 확정 동작은 유지하고, 확정 중에 전송되는 것만 막습니다.
        if (event.key === 'Enter' && (composing || event.isComposing || event.keyCode === 229)) {
            event.stopImmediatePropagation();
        }
    };
    action.addEventListener('pointerdown', keepCaret);
    field.addEventListener('compositionstart', compositionStart);
    field.addEventListener('compositionend', compositionEnd);
    field.addEventListener('keydown', preserveIME, true);
    const dispose = () => {
        completeOpening();
        cancelAll();
        deferredAction = false;
        if (restoreFrame) cancelAnimationFrame(restoreFrame);
        if (commitFrame) cancelAnimationFrame(commitFrame);
        action.removeEventListener('pointerdown', keepCaret);
        send.removeEventListener('click', nativeSend, true);
        field.removeEventListener('keydown', nativeEnter, true);
        field.removeEventListener('compositionstart', compositionStart);
        field.removeEventListener('compositionend', compositionEnd);
        field.removeEventListener('keydown', preserveIME, true);
        chat?.removeEventListener('click', completeOpening);
        window.removeEventListener('pagehide', dispose);
        window.removeEventListener('error', observeError);
        window.removeEventListener('unhandledrejection', observeError);
    };
    window.addEventListener('pagehide', dispose);
    chat?.addEventListener('click', completeOpening);
    setStage(app?.dataset.trpgStage || 'landing');
    selectCharacter(document.querySelector('#game-character-select input')?.value || 'bran');
    window.__trpgControls = {insertAction, begin, finish, updateStatus, diagnostics: diagnosticsSnapshot, cancelAll, dispose, setStage, selectCharacter,
        startFinished, completeOpening, returnFromSelection: () => setStage(hasStarted ? 'play' : 'landing')};
    return [];
}"""


def turn_start_js(operation, *, roll=False):
    """기존 요청 전에 화면의 전체 처리 시계만 시작합니다. 반환 인수는 바꾸지 않습니다."""
    dice = "document.querySelector('#game-dice .dice-stage')?.classList.add('is-rolling');" if roll else ""
    return "(...args) => {window.__trpgControls?.begin(" + json.dumps(operation) + ");" + dice + "return args;}"


def turn_finish_js(operation):
    """Gradio의 성공·실패 공통 종료 뒤 시계를 멈춥니다."""
    return "() => {window.__trpgControls?.finish(" + json.dumps(operation) + ");return [];}"


class _TypingView:
    """안전하게 전달된 본문을 화면 사본에서만 짧게 이어 표시합니다."""
    tick = 0.035
    finish_budget = 0.35

    def __init__(self):
        self.clear()

    def clear(self):
        self.view, self.context, self.target, self.position = None, None, "", 0
        self.last_emit = -float("inf")

    @property
    def pending(self):
        return self.view is not None and self.position < len(self.target)

    def offer(self, view):
        chat = view[0]
        if not chat or chat[-1].get("role") != "assistant":
            return False
        target, context = chat[-1]["content"], chat[:-1]
        if (context != self.context or not target.startswith(self.target[:self.position])):
            heading = re.match(r"^\*\*[^\n]+\*\*\n", target)
            self.position = heading.end() if heading else 0
            self.last_emit = -float("inf")
        self.view, self.context, self.target = list(view), context, target
        return True

    def advance(self, now, finish_by=None):
        if not self.pending or now - self.last_emit < self.tick:
            return None
        remaining = len(self.target) - self.position
        step = max(2, math.ceil(remaining / 12))
        if finish_by is not None:
            frames = max(1, int((finish_by - now) / self.tick))
            step = max(step, math.ceil(remaining / frames))
        self.position = min(len(self.target), self.position + step)
        self.last_emit = now
        view = list(self.view)
        view[0] = [*self.context, {**self.view[0][-1], "content": self.target[:self.position]}]
        return view

    def matches_final(self, view):
        return bool(self.view is not None and view[0] and view[0][-1].get("role") == "assistant"
                    and view[0][:-1] == self.context)


SKILL_ABILITIES = {
    "athletics": ("운동", "str"), "acrobatics": ("곡예", "dex"),
    "sleight_of_hand": ("손재주", "dex"), "stealth": ("은신", "dex"),
    "arcana": ("비전", "int"), "history": ("역사", "int"),
    "investigation": ("수사", "int"), "nature": ("자연", "int"), "religion": ("종교", "int"),
    "animal_handling": ("동물 다루기", "wis"), "insight": ("통찰", "wis"),
    "medicine": ("의학", "wis"), "perception": ("감지", "wis"), "survival": ("생존", "wis"),
    "deception": ("기만", "cha"), "intimidation": ("위협", "cha"),
    "performance": ("공연", "cha"), "persuasion": ("설득", "cha"),
}
RESOURCE_LABELS = {"second_wind": "재기의 바람", "action_surge": "행동 연쇄",
                   "arcane_recovery": "비전 회복", "channel_divinity": "신성 변환"}

GAME_VIEWPORT_JS = """() => {
    // 화면의 남은 높이만 계산합니다. 게임 상태나 모델에는 접근하지 않습니다.
    const previous = window.__trpgViewportLayout;
    if (previous && previous.dispose) previous.dispose();
    const shell = document.querySelector('#game-story-shell');
    if (!shell) return [];
    let frame = 0;
    let lastHeight = '';
    const update = () => {
        frame = 0;
        if (!shell.isConnected || !shell.getClientRects().length) return;
        const viewport = window.visualViewport ? window.visualViewport.height : window.innerHeight;
        const top = Math.max(0, shell.getBoundingClientRect().top);
        const height = Math.max(180, Math.floor(viewport - top - 14)) + 'px';
        // 같은 값을 다시 쓰지 않아 ResizeObserver가 반복 실행되지 않습니다.
        if (height !== lastHeight) {
            shell.style.setProperty('--game-story-height', height);
            lastHeight = height;
        }
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
    const resize = new ResizeObserver(schedule);
    ['#game-story-shell', '.game-toolbar', '#game-statusbar', '#game-known-summary',
     '#game-main-tabs > .tab-nav'].forEach(selector => {
        const element = document.querySelector(selector);
        if (element) resize.observe(element);
    });
    const visibility = new MutationObserver(schedule);
    if (shell.parentElement) visibility.observe(shell.parentElement, {
        attributes: true, attributeFilter: ['style', 'class', 'hidden']
    });
    const tabs = document.querySelector('#game-main-tabs > .tab-nav');
    if (tabs) tabs.addEventListener('click', schedule);
    window.addEventListener('resize', schedule);
    window.visualViewport?.addEventListener('resize', schedule);
    window.__trpgViewportLayout = { dispose: () => {
        resize.disconnect(); visibility.disconnect();
        if (frame) cancelAnimationFrame(frame);
        if (tabs) tabs.removeEventListener('click', schedule);
        window.removeEventListener('resize', schedule);
        window.visualViewport?.removeEventListener('resize', schedule);
    }};
    schedule();
    return [];
}"""


GAME_LAYOUT_CSS = """
/* 공통 화면: 사이드바 없는 단일 열이며 모든 탭이 같은 폭을 사용합니다. */
.gradio-container {width:100%!important;max-width:1080px!important;margin:0 auto!important;padding:10px 20px 16px!important;background:#121617!important;color:#e5e2d8;--body-background-fill:#121617;--block-background-fill:#182124;--block-border-color:#334044;--border-color-primary:#334044;--input-background-fill:#1c272a;--button-primary-background-fill:#a98a51;--button-primary-text-color:#fff9eb;--body-text-color:#e5e2d8;--block-label-text-color:#b6b8ac;--input-text-color:#e5e2d8;--body-text-color-subdued:#9da9a3;}
.gradio-container *,.gradio-container *::before,.gradio-container *::after {box-sizing:border-box;}
.gradio-container .main {padding:0!important;gap:7px!important;}
.gradio-container button {color:#dddcca!important;}
.gradio-container button.secondary {background:#263539!important;border-color:#445458!important;}
.gradio-container button.primary {background:#a98a51!important;border-color:#b39b68!important;color:#fff5da!important;}
.gradio-container .label-wrap,.gradio-container summary {color:#bcc5b7!important;}
.game-toolbar {align-items:center;gap:8px!important;flex-wrap:wrap!important;position:sticky;top:0;z-index:20;background:#121617f2;padding:4px 0;backdrop-filter:blur(8px);}
#game-heading {min-width:190px!important;flex:3;}
#game-heading h1 {font-size:21px!important;color:#d7c29a;letter-spacing:.06em;margin:0!important;}
#game-heading p {font-size:11px;color:#91a49b;margin:2px 0!important;}
.game-toolbar>.form {flex:0 1 260px!important;min-width:180px!important;max-width:290px!important;background:transparent!important;border:0!important;box-shadow:none!important;padding:0!important;}
#game-character-select {width:100%!important;min-width:180px!important;background:transparent!important;border:0!important;box-shadow:none!important;padding:0!important;}
#game-character-select .wrap {background:#1c2b29!important;border-color:#496054!important;}
#game-character-select label {background:transparent!important;}
#game-layout,#game-main-tabs {display:flex!important;flex-direction:column!important;width:100%!important;max-width:100%!important;min-width:0!important;gap:0!important;}
#game-layout .block,#game-layout .form {min-width:0!important;max-width:100%;}
#game-main-tabs>.tab-nav {background:#171c1e!important;border-bottom:1px solid #45514c!important;gap:0!important;overflow-x:auto!important;flex-wrap:nowrap!important;}
#game-main-tabs>.tab-nav button {flex:1;min-width:82px;padding:9px 12px!important;font-size:13px!important;color:#a8b5ae!important;background:transparent!important;border-radius:0!important;white-space:nowrap;}
#game-main-tabs>.tab-nav button.selected {color:#e5d0a3!important;border-bottom:2px solid #bb9c61!important;}
#game-main-tabs>.tabitem {padding:0!important;border:0!important;}
#game-statusbar {padding:0!important;border:0!important;background:transparent!important;}
.trpg-stats {display:flex;gap:10px;align-items:center;justify-content:space-between;border:1px solid #454d44;border-radius:10px;padding:8px 12px;background:#1b2425;}
.trpg-place {font-weight:700;font-size:15px;color:#dbc59b;}
.trpg-hero,.trpg-vitals {font-size:11px;color:#bbc7bd;}.trpg-hero {margin-top:2px;}.trpg-vitals {min-width:190px;text-align:right;}
.trpg-hp-track {height:4px;margin:5px 0;border-radius:4px;background:#354440;overflow:hidden;}
.trpg-hp-fill {height:100%;background:#8daf94;border-radius:4px;}.trpg-resource {font-size:10px;color:#9cae9f;}
.discovery-overview {display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:6px 3px;color:#ccd5ca;font-size:12px;}
.discovery-eyebrow {color:#91aaa0;font-size:10px;}.discovery-overview b {font-size:12px;color:#d5c7a7;}.discovery-goal {flex:1;text-align:right;color:#b8c8bd;min-width:210px;}
/* 입력칸은 이야기의 아래에 고정되고, 높이가 부족하면 본문·선택지만 스크롤합니다. */
#game-story-shell {position:relative!important;isolation:isolate;display:flex!important;flex-direction:column!important;height:var(--game-story-height,calc(100dvh - 275px))!important;min-height:0!important;max-height:920px;overflow:hidden;border:1px solid #3b4948!important;border-radius:0 0 14px 14px!important;padding:10px!important;background:#1a2425!important;gap:6px!important;}
#game-story-shell>* {position:relative;z-index:1;flex-shrink:0;}
#game-story-content {display:flex!important;flex-direction:column!important;flex:1 1 0%!important;min-height:0!important;gap:6px!important;overflow-y:auto!important;overflow-x:hidden!important;scrollbar-gutter:stable;}
#game-story-content>* {flex-shrink:0;}
#game-scene-background {position:absolute!important;inset:0!important;z-index:0!important;min-height:0!important;padding:0!important;border:0!important;overflow:hidden!important;pointer-events:none!important;}
#game-scene-background .scene-art {position:absolute;inset:-3px;background-size:300% 200%;background-repeat:no-repeat;background-color:#29322f;filter:blur(1.2px);opacity:.66;transition:background-position .4s ease,opacity .4s ease;}
#game-scene-background .scene-shade {position:absolute;inset:0;background:linear-gradient(180deg,rgba(12,19,20,.55),rgba(12,19,20,.72) 65%,rgba(12,19,20,.91));}
#game-scene-background .scene-village {background-color:#3d493e;}#game-scene-background .scene-house,#game-scene-background .scene-inn {background-color:#4c3d2c;}#game-scene-background .scene-forge {background-color:#473c30;}#game-scene-background .scene-well {background-color:#35413c;}#game-scene-background .scene-shaft {background-color:#202f32;}
#game-chat {flex:1 1 180px!important;height:180px!important;min-height:140px!important;max-height:none!important;overflow:hidden!important;background:transparent!important;border:0!important;border-radius:0!important;box-shadow:none!important;}
#game-chat div {background-color:transparent!important;}
#game-chat .message {width:100%!important;max-width:100%!important;border:0!important;background:transparent!important;box-shadow:none!important;overflow:visible!important;max-height:none!important;}
#game-chat .message-row {padding:4px 0!important;}
#game-chat .bubble-wrap {overflow-y:auto!important;}
#game-chat pre,#game-chat code {white-space:pre-wrap;overflow-wrap:anywhere;}
#game-chat table {display:block;max-width:100%;overflow-x:auto;font-size:12px;}
.game-chat-tools {align-items:center;gap:6px!important;min-height:24px;}.game-chat-tools .prose {font-size:11px;color:#9da9a3;}
.game-chat-tools button {font-size:11px!important;min-height:26px!important;padding:3px 10px!important;}
.trpg-chat-row {display:flex;align-items:flex-start;gap:10px;margin:2px 0 12px;line-height:1.6;}
.trpg-avatar {width:38px!important;height:38px!important;min-width:38px!important;opacity:1!important;object-fit:cover;border-radius:50%!important;border:1px solid #9d8661;display:flex!important;align-items:center;justify-content:center;background:#283437;color:#d7c097;font-size:11px;font-weight:700;filter:none!important;}
img.trpg-avatar {filter:brightness(1.3) contrast(1.05)!important;}
.trpg-chat-main {min-width:0;flex:1;overflow:visible!important;max-height:none!important;}
.trpg-speaker {display:flex;align-items:center;flex-wrap:wrap;gap:6px;min-height:24px;line-height:1.5;overflow:visible!important;color:#e0cba2;font-size:11px;font-weight:700;margin-bottom:2px;}
.trpg-message-body {color:#f0f0e8;font-size:15px;line-height:1.8;overflow-wrap:anywhere;white-space:normal;max-height:none!important;overflow:visible!important;}
.trpg-message-body p {margin:0 0 6px!important;}.trpg-message-body p:last-child {margin-bottom:0!important;}
#game-chat .trpg-chat-row.npc .trpg-message-body {display:inline-block;max-width:100%;background:rgba(51,66,66,.75)!important;padding:8px 12px;border-radius:3px 12px 12px 12px;border:1px solid rgba(147,167,157,.17);}
.trpg-chat-row.user .trpg-speaker {color:#a6c3b4;}.trpg-chat-row.user .trpg-message-body {color:#c8ded3;}
.trpg-chat-row.system .trpg-avatar {display:none!important;}.trpg-chat-row.system .trpg-speaker {font-size:10px;letter-spacing:.05em;color:#a9b6ac;}.trpg-chat-row.system .trpg-message-body {padding:0 0 2px;background:transparent!important;}
#game-chat .trpg-message-body pre {font-size:14px;white-space:pre-wrap;}
/* Gradio가 입력칸을 감싸는 .form까지 같은 격자 안에서 너비를 받습니다. */
#game-unified-composer {display:grid!important;grid-template-columns:minmax(0,1fr) 34px 34px!important;flex-wrap:nowrap!important;padding:6px!important;margin-top:0!important;border:1px solid #4a5c59;border-radius:9px;background:rgba(33,45,46,.95)!important;gap:6px!important;align-items:center;}
#game-unified-composer>.form,#game-unified-composer>#game-message {grid-column:1!important;min-width:0!important;width:100%!important;max-width:100%!important;flex:unset!important;padding:0!important;background:transparent!important;border:0!important;box-shadow:none!important;}
#game-unified-composer #game-message {width:100%!important;min-width:0!important;max-width:100%!important;}
#game-unified-composer textarea {width:100%!important;background:transparent!important;border:0!important;box-shadow:none!important;font-size:14px!important;line-height:1.5!important;min-height:40px!important;padding:8px!important;white-space:pre-wrap;word-break:normal;}
#game-unified-composer button {width:100%!important;min-width:0!important;min-height:38px!important;padding:7px!important;border-radius:6px!important;white-space:nowrap!important;word-break:keep-all!important;}
#game-mark-action {grid-column:2!important;}#game-send-message {grid-column:3!important;}
#game-unified-composer .game-icon-button {display:flex!important;align-items:center;justify-content:center;width:34px!important;min-height:34px!important;height:34px!important;padding:5px!important;font-size:0!important;border:0!important;background:transparent!important;color:#f1f3ef!important;}
#game-unified-composer #game-send-message {color:#a5d7f8!important;}
#game-unified-composer .game-icon-button:hover {background:#3b5055!important;}
#game-unified-composer .game-icon-button:focus-visible {outline:2px solid #b9d9ec!important;outline-offset:2px;}
.game-icon-button::before {content:"";width:24px;height:24px;flex:0 0 24px;display:block;background:currentColor;mask-repeat:no-repeat;mask-position:center;mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;-webkit-mask-size:contain;}
.trpg-turn-clock {display:flex;align-items:center;justify-content:flex-end;gap:5px;font-size:11px;line-height:1.4;color:#bdcec6;white-space:nowrap;}
.trpg-turn-clock[data-running="false"] {visibility:hidden;}.trpg-turn-seconds {font-variant-numeric:tabular-nums;color:#d9c99e;}
#game-turn-clock {border:0!important;background:transparent!important;padding:0!important;min-width:0!important;flex:0 1 auto!important;}
.trpg-input-badge {display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto;gap:4px;min-height:22px;padding:2px 6px;margin-left:0;border-radius:4px;font-size:11px;line-height:16px;font-weight:600;color:#e2edf0;background:#293e47;vertical-align:middle;overflow:visible!important;}
.trpg-input-icon {display:block!important;width:16px!important;height:16px!important;min-width:16px;max-width:none!important;flex:0 0 16px!important;vertical-align:middle;overflow:visible!important;}
.user-action .trpg-input-badge,.user-mixed .trpg-input-badge {color:#e0c387;background:#413c2c;}
.trpg-inline-action {display:inline-flex;align-items:center;gap:4px;padding:1px 5px;border:1px solid #89744766;border-radius:5px;background:#4b42254d;color:#e2ce9d;}
.game-composer {align-items:center;gap:6px!important;}.game-composer>* {min-width:0!important;max-width:100%;}.game-composer button {line-height:1.3;min-height:36px!important;}
.game-input-help {font-size:10px!important;line-height:1.4!important;color:#a5b4ab!important;margin:0 2px!important;}
/* 현재 가능한 선택지만 표시하며, 많은 선택지는 이 영역 안에서 넘겨 봅니다. */
#game-actions,#game-recommendations,#game-action-recommendations {border:0!important;background:transparent!important;padding:0!important;gap:4px!important;overflow-y:auto!important;}
#game-actions {max-height:116px;}#game-recommendations {max-height:90px;}
#game-actions>.gallery,#game-recommendations>.gallery {display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr))!important;gap:5px!important;}
#game-actions button,#game-recommendations button {white-space:normal!important;width:100%!important;min-width:0!important;min-height:32px!important;text-align:left!important;padding:5px 8px!important;background:rgba(39,53,53,.86)!important;border:1px solid rgba(133,153,143,.25)!important;line-height:1.5!important;border-radius:6px!important;font-size:12px!important;}
#game-actions button:hover,#game-recommendations button:hover {border-color:#b59b63!important;background:#2a403a!important;}
#game-actions .label-wrap,#game-recommendations .label-wrap,#game-actions>.label,#game-recommendations>.label {font-size:10px!important;}
#game-actions .gallery-item .gallery {display:block!important;white-space:pre-line!important;font-size:12px!important;line-height:1.5!important;max-width:none!important;}
#game-recommendations .gallery-item .gallery {font-size:11px!important;}
/* 페이지 이동은 게임 행동과 다릅니다. Gradio의 번호·이전·다음 버튼은
   원래 순서와 클릭 연결을 유지하고, 행동 카드의 전체 폭 스타일만 덮어씁니다. */
#game-actions :is(.paginate,.pagination),#game-recommendations :is(.paginate,.pagination) {display:flex!important;flex-direction:row!important;align-items:center!important;justify-content:center!important;flex-wrap:wrap!important;gap:5px!important;width:fit-content!important;max-width:100%!important;margin:6px auto 2px!important;padding:3px 5px!important;font-size:0!important;line-height:1.4!important;color:#b6c8cf!important;border-top:1px solid #41545c!important;}
/* 설치된 Gradio Dataset의 .paginate는 여러 페이지일 때만 생깁니다.
   기본 Pages: 글자 대신 용도를 밝히고, 생략 기호와 실제 버튼은 계속 표시합니다. */
#game-actions :is(.paginate,.pagination)::before,#game-recommendations :is(.paginate,.pagination)::before {content:"선택지 페이지";font-size:11px;line-height:1.4;white-space:nowrap;margin-right:3px;}
#game-actions :is(.paginate,.pagination)>div,#game-recommendations :is(.paginate,.pagination)>div {font-size:11px!important;}
#game-actions :is(.paginate,.pagination) button,#game-recommendations :is(.paginate,.pagination) button {display:inline-flex!important;align-items:center!important;justify-content:center!important;flex:0 0 auto!important;width:auto!important;min-width:28px!important;max-width:none!important;min-height:27px!important;height:27px!important;padding:3px 8px!important;white-space:nowrap!important;text-align:center!important;font-size:12px!important;line-height:1!important;border:1px solid #526a76!important;border-radius:5px!important;background:#192a32!important;color:#d0e1e7!important;box-sizing:border-box!important;}
#game-actions :is(.paginate,.pagination) button:is(.current-page,.active,.selected,.current,[aria-current="page"],[aria-selected="true"]),#game-recommendations :is(.paginate,.pagination) button:is(.current-page,.active,.selected,.current,[aria-current="page"],[aria-selected="true"]) {background:#385667!important;border-color:#a1c2d0!important;color:#f1f7fa!important;font-weight:700!important;}
#game-actions :is(.paginate,.pagination) button:hover,#game-recommendations :is(.paginate,.pagination) button:hover {background:#294653!important;border-color:#b0cad5!important;}
#game-actions :is(.paginate,.pagination) button:focus-visible,#game-recommendations :is(.paginate,.pagination) button:focus-visible {outline:2px solid #badce9!important;outline-offset:2px!important;}
#game-actions :is(.paginate,.pagination) button:disabled,#game-recommendations :is(.paginate,.pagination) button:disabled {opacity:.45!important;cursor:default!important;}
#game-quest,#game-character {padding:12px!important;max-height:calc(100dvh - 300px);overflow:auto;}
.compact-pack {max-height:calc(100dvh - 350px);overflow:auto;padding:10px;}.pack-row {display:flex;justify-content:space-between;gap:10px;padding:8px 3px;border-bottom:1px solid #30423e;font-size:12px;}.pack-row b {color:#c9b27f;}
#game-known-places,#game-known-people {border:1px solid #364644!important;background:#1b2628!important;border-radius:9px!important;}
#game-known-places button,#game-known-people button {font-size:12px!important;line-height:1.6!important;padding:10px!important;}
.discovery-action-list {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin:7px 0;}
.discovery-action {display:flex;flex-direction:column;gap:3px;padding:10px;background:#1e2a2b;border:1px solid #354645;border-radius:7px;font-size:12px;}.discovery-action small {font-size:11px;color:#aebeb1;}
.known-cards {display:flex;flex-wrap:wrap;gap:7px;margin:6px 0 14px;}.known-card {min-width:150px;flex:1 1 160px;padding:10px 12px;border:1px solid #394b47;border-radius:8px;background:#1d2b2c;}
.known-card b,.known-card small {display:block;}.known-card b {font-size:13px;color:#decba6;}.known-card small {font-size:11px;color:#9bb3a7;margin-top:4px;}
/* 판정·전투 및 개발자 화면의 기능은 그대로 유지합니다. */
.dice-stage {display:flex;align-items:center;gap:14px;border:1px solid #a88e59;border-radius:10px;background:radial-gradient(ellipse at left,#394137,#192524);padding:10px 14px;}
.dice-glyph {display:flex;align-items:center;justify-content:center;width:58px;height:64px;flex-shrink:0;clip-path:polygon(50% 0,95% 25%,95% 75%,50% 100%,5% 75%,5% 25%);background:linear-gradient(135deg,#c0a56d,#75613c);color:#17211b;font-size:23px;font-weight:800;}
.dice-stage h3 {font-size:14px!important;color:#e3d4ad;margin:3px 0 5px!important;}.dice-stage b {font-size:13px;color:#d6dbbd;}.dice-stage p {font-size:11px;color:#a7b7a4;margin:4px 0 0!important;}.dice-eyebrow {font-size:11px;color:#d6b773;}
.dice-result {display:flex;align-items:center;gap:12px;padding:9px 14px;border:1px solid #42664d;border-radius:10px;background:#1c3027;}.dice-result.failure {border-color:#794b41;background:#342521;}.dice-result-number {font-size:27px;font-weight:800;color:#e1cf9c;min-width:36px;text-align:center;}.dice-result div {display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px;}.dice-result b {font-size:13px;color:#c1ddba;}.dice-result.failure b {color:#e1b5a0;}.dice-result span {font-size:12px;}.dice-result small {display:block;flex-basis:100%;font-size:10px;color:#aebbae;}
.dice-stage.is-rolling .dice-glyph {animation:dice-tumble .3s infinite alternate;}@keyframes dice-tumble {to {transform:rotate(14deg) scale(.88);filter:brightness(1.3);}}
.encounter-board {padding:10px 13px;border:1px solid #79654c;border-radius:10px;background:linear-gradient(100deg,#302b25,#1e2928);}.encounter-heading {display:flex;justify-content:space-between;gap:10px;font-size:12px;color:#dcc08c;}.encounter-heading span {font-size:11px;color:#aeb9aa;}.enemy-line {display:flex;gap:7px;flex-wrap:wrap;margin:9px 0;}.enemy-token {display:flex;align-items:center;gap:8px;border:1px solid #625a47;padding:5px 10px;border-radius:8px;background:#28302b;}.enemy-token>span {color:#ba9871;}.enemy-token b {display:block;font-size:12px;color:#d7d2ba;}.enemy-token small {display:block;font-size:10px;color:#adbaa8;}.turn-budget {font-size:11px;color:#acbea9;}.turn-budget b {color:#e0d2a5;}
#game-check-method {border-color:#796b47!important;}
#game-developer {margin-top:8px;border:1px solid #3c514b;border-radius:12px;background:#152320;}
.game-dev-row {align-items:stretch!important;gap:10px!important;flex-wrap:wrap!important;}
.game-dev-row>.game-dev-card {flex:1 1 360px!important;min-width:min(360px,100%)!important;padding:12px!important;border:1px solid #3a4c49!important;border-radius:10px!important;background:#1b2628!important;overflow:visible!important;}
.game-dev-card .game-dev-card {min-width:0!important;max-width:100%!important;padding:0!important;border:0!important;border-radius:0!important;background:transparent!important;flex:unset!important;overflow:visible!important;box-shadow:none!important;}
.game-dev-card .styler,.game-dev-card .form {background:transparent!important;}.game-dev-card h3 {color:#d9c28d;font-size:16px!important;margin:0 0 12px!important;}
.game-dev-card button,.game-map-actions button {white-space:nowrap!important;word-break:keep-all!important;min-height:36px!important;font-size:13px!important;}
.game-dev-help {font-size:12px;color:#a5bab0;}.trpg-map-card {border:1px solid #3b4c47;border-radius:8px;padding:8px;background:#22302e;flex:1 1 100px;min-width:0;}.trpg-map-card.current {border-color:#b39963;}
#game-start-notice {border:1px solid #8e784b;border-radius:9px;padding:9px 12px;background:#29352a;color:#ead7a9;font-size:13px;}
#game-help-overlay {position:fixed!important;inset:0!important;z-index:1000!important;background:rgba(4,10,10,.82)!important;padding:20px!important;overflow-y:auto!important;backdrop-filter:blur(5px);}
.game-help-card {max-width:640px!important;width:100%!important;margin:4vh auto!important;border:1px solid #8d7a4f!important;border-radius:16px!important;padding:24px!important;background:#1a2926!important;box-shadow:0 20px 80px #0008;}
.game-help-card .prose {font-size:14px!important;line-height:1.75!important;}.game-help-card button {white-space:nowrap;}footer {display:none!important;}
@media(max-width:700px) {.gradio-container {padding:8px!important;}#game-heading {min-width:150px!important;}#game-heading h1 {font-size:20px!important;}.game-toolbar {gap:5px!important;}.game-toolbar>.form {min-width:150px!important;}#game-character-select {min-width:150px!important;}.trpg-stats {align-items:flex-start;gap:6px;}.trpg-vitals {min-width:0;text-align:right;font-size:10px;}.trpg-hero {font-size:10px;}.discovery-overview {gap:5px;}.discovery-goal {text-align:left;min-width:100%;}#game-story-shell {padding:7px!important;}#game-actions>.gallery,#game-recommendations>.gallery,.discovery-action-list {grid-template-columns:1fr!important;}.trpg-message-body {font-size:14px;}}
@media(max-width:560px) {#game-main-tabs>.tab-nav button {min-width:70px;padding:8px!important;font-size:12px!important;}.dice-stage {padding:9px;gap:10px;}.game-chat-tools {flex-wrap:wrap!important;}.trpg-turn-clock {font-size:10px;}}
/* 시작·인물 선택은 화면 전환이며 아직 게임 행동을 실행하지 않습니다. */
.gradio-container:not([data-trpg-stage="play"]) #game-play-screen {display:none!important;}
.gradio-container:not([data-trpg-stage="characters"]) #game-character-screen {display:none!important;}
.gradio-container[data-trpg-stage="play"] #game-landing,.gradio-container[data-trpg-stage="characters"] #game-landing {display:none!important;}
#game-landing {position:relative!important;isolation:isolate;min-height:calc(100dvh - 56px);display:flex!important;align-items:center!important;justify-content:center!important;gap:24px!important;overflow:hidden;border:1px solid #45545a;border-radius:16px;padding:60px 24px!important;background:#162225!important;}
#game-landing>* {position:relative;z-index:1;}#game-landing-background {position:absolute!important;inset:0!important;z-index:0!important;padding:0!important;}
#game-landing-background .scene-art {position:absolute;inset:-4px;background-size:300% 200%;background-repeat:no-repeat;filter:blur(1.2px);opacity:.6;}
#game-landing-background .scene-shade {position:absolute;inset:0;background:linear-gradient(180deg,#111b2180,#111b21bb 55%,#111b21ed);}
#game-landing-title {text-align:center!important;max-width:650px!important;}#game-landing-title h1 {font-size:clamp(58px,9vw,110px)!important;line-height:1.25!important;letter-spacing:.15em!important;color:#f1f2ec!important;margin:12px 0 24px!important;}
#game-landing-title p {font-size:15px!important;line-height:1.8!important;color:#d2dde0!important;}
#game-landing-start {min-width:250px!important;max-width:320px!important;min-height:56px!important;font-size:18px!important;letter-spacing:.05em;}
.landing-footnote {font-size:12px!important;color:#b7c9cf!important;text-align:center;}
#game-character-screen {min-height:calc(100dvh - 56px);padding:36px 12px!important;gap:24px!important;justify-content:center!important;}
#game-character-screen h2 {font-size:28px!important;color:#edf2f0!important;}.character-card-grid {display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px!important;}
.character-card {width:100%!important;min-width:0!important;min-height:240px!important;padding:30px 14px!important;white-space:pre-line!important;text-align:left!important;line-height:2.15!important;font-size:15px!important;background:linear-gradient(145deg,#263842,#1b262f)!important;border:1px solid #4f6572!important;border-radius:12px!important;color:#e8efed!important;box-shadow:0 8px 30px #0002;}
.character-card[data-selected="true"] {border:2px solid #a8d7ee!important;background:linear-gradient(145deg,#304a57,#1b303e)!important;box-shadow:0 0 0 2px #8ac0db25;}
.character-card:hover {border-color:#b2d2e0!important;transform:translateY(-2px);}.character-card:focus-visible {outline:2px solid #d1eaff!important;outline-offset:3px;}
#game-character-select {display:none!important;}.character-screen-actions {align-items:center!important;justify-content:center!important;}#game-character-start {max-width:420px!important;min-height:48px!important;font-size:16px!important;}
#game-play-screen {width:100%!important;min-width:0!important;gap:7px!important;}
#game-play-screen .prose {font-size:14px!important;line-height:1.7!important;color:#dce6e5;}.trpg-speaker {font-size:13px!important;color:#cfdfdf!important;}.trpg-message-body {font-size:15px!important;line-height:1.8!important;}
.trpg-place {color:#dcebf1!important;}.discovery-eyebrow {font-size:12px!important;color:#b0c7cb!important;}.discovery-overview,.discovery-overview b {font-size:13px!important;}.discovery-goal {color:#c6d8d8!important;}
.trpg-opening-hint {display:none;font-size:12px;color:#c0d6df;}.gradio-container[data-revealing="true"] .trpg-opening-hint {display:inline;}
.gradio-container[data-revealing="true"] #game-chat {cursor:pointer;}
.objective-card,.quest-card,.clue-card {padding:16px 18px;margin:10px 0;border:1px solid #435b63;border-radius:10px;background:#1c2a31;color:#e1eceb;}
.objective-card {border-color:#7296a6;background:linear-gradient(120deg,#263d49,#1b2a33);}.objective-card>span {font-size:12px;color:#b6d0df;}.objective-card h2 {font-size:21px;margin:6px 0 12px;color:#ecf3f3;}.objective-card p,.quest-card p,.clue-card p {font-size:14px;line-height:1.75;white-space:pre-line;}.objective-card b {margin-right:5px;color:#b9d5e4;}
.quest-card-header {display:flex;justify-content:space-between;align-items:center;gap:10px;}.quest-card h3,.clue-card h3 {font-size:16px!important;margin:0 0 6px!important;color:#e4eeee;}.quest-card-header>span {font-size:12px;padding:3px 7px;border-radius:5px;background:#304e56;color:#d0e5eb;white-space:nowrap;}.quest-card small {font-size:12px;color:#b4c9cb;}.quest-card.completed {opacity:.85;}.quest-card.completed .quest-card-header>span {background:#294b3b;color:#c0ddcb;}
.quest-history,.known-action-details {padding:12px 3px;border-top:1px solid #3e5259;margin-top:16px;}.quest-history summary,.known-action-details summary {cursor:pointer;font-size:14px;color:#bed0d6;}.empty-note {font-size:14px;line-height:1.7;color:#adbec4;padding:12px 4px;}
.known-people {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;padding:12px 0;}.known-person {display:flex;align-items:center;gap:13px;padding:14px;border:1px solid #41585f;background:#1c2b31;border-radius:10px;}.known-person-portrait {height:60px;width:60px;object-fit:cover;border-radius:10px;filter:none!important;}.known-person h3 {font-size:16px!important;margin:0!important;color:#e8eeec;}.known-person p {font-size:13px;margin:5px 0!important;color:#bcd1d8;}.known-person small {font-size:12px;color:#a2b9bd;}
#game-developer {border:1px solid #83709e!important;background:#211e2b!important;color:#ded5ec!important;}#game-developer>label,#game-developer summary {color:#d9c9ef!important;}.game-dev-help {color:#d0bfe4!important;font-size:14px!important;}.game-dev-row>.game-dev-card {border-color:#655676!important;background:#272232!important;}.game-dev-card h3 {color:#e1d2f0!important;}.game-dev-card button {border-color:#756082!important;background:#3a304b!important;color:#ede4f4!important;}
@media(max-width:740px) {.character-card-grid {grid-template-columns:repeat(2,minmax(0,1fr));}.character-card {min-height:190px!important;padding:20px 16px!important;}.known-people {grid-template-columns:1fr;}}
@media(max-width:420px) {#game-character-screen {padding:22px 3px!important;}.character-card {font-size:13px!important;padding:18px 10px!important;min-height:180px!important;}#game-landing-title p {font-size:14px!important;}.trpg-message-body {font-size:14px!important;}}
"""

# 대화 배지와 버튼이 같은 작은 SVG를 사용하고 색상만 화면에서 지정합니다.
for _selector, _svg in (("#game-mark-action", ACTION_ICON_SVG), ("#game-send-message", SEND_ICON_SVG)):
    _icon_url = "data:image/svg+xml," + quote_icon(_svg, safe="")
    GAME_LAYOUT_CSS += (_selector + '::before {mask-image:url("' + _icon_url
                        + '");-webkit-mask-image:url("' + _icon_url + '");}\n')




_feedback_assets = Path(__file__).resolve().parent / "assets"
GAME_LAYOUT_CSS += "\n" + (_feedback_assets / "combat-feedback.css").read_text(encoding="utf-8")
_feedback_script = (_feedback_assets / "combat-feedback.js").read_text(encoding="utf-8")
GAME_CONTROLS_JS = GAME_CONTROLS_JS.replace(
    "    const dispose = () => {",
    "    const combatFeedback = (" + _feedback_script + ")();\n"
    "    const dispose = () => {\n        combatFeedback.dispose();",
)


def resource_summary(engine, content):
    """엔진이 보관하는 주문 슬롯·직업 자원을 화면용 한 줄로 만듭니다.

    현재 남은 횟수는 hero.resources에서, 최대 횟수는 Content의 직업·레벨 표에서 읽습니다.
    캐릭터의 종류에 따라 해당하는 항목만 표시하며 게임 수치를 직접 갱신하지 않습니다."""
    hero = engine.hero
    maximum = content.rules["class_levels"].get(hero.cls, {}).get(str(hero.level), {})
    rows = []
    for level, remaining in enumerate(hero.resources.get("slots", []), 1):
        limits = maximum.get("slots", [])
        limit = limits[level - 1] if level <= len(limits) else remaining
        rows.append(f"{level}레벨 주문 슬롯 {remaining}/{limit}")
    for key, label in RESOURCE_LABELS.items():
        if key in hero.resources and maximum.get(key, 0) > 0:
            rows.append(f"{label} {hero.resources[key]}/{maximum.get(key, hero.resources[key])}회")
    if hero.cls == "rogue":
        dice = maximum.get("sneak_dice", 0)
        if dice:
            used = bool(engine.s.combat and engine.s.combat.get("turn", {}).get("sneak"))
            rows.append(f"기습 공격 {dice}d6 · " + ("이번 차례 사용" if used else "조건 충족 시 차례당 1회"))
    return " · ".join(rows) or "별도 사용 횟수 자원 없음"


def character_sheet(engine, content):
    """현재 캐릭터의 능력치·숙련·장착 장비를 Markdown으로 정리합니다.

    engine과 content를 받아 수정치·구원·주문·자원 등 표시 값을 구합니다.
    탐사 가방과 장착 장비는 출처가 다르므로 구분해 보여 줍니다."""
    hero = engine.hero
    species = {"Human": "인간", "Elf": "엘프", "Dwarf": "드워프"}.get(hero.species, hero.species)
    background = {"Soldier": "군인", "Criminal": "범죄자", "Sage": "현자", "Acolyte": "시종"}.get(hero.background, hero.background)
    rows = [f"### {hero.name} · {hero.cls_ko} {hero.level}레벨",
            f"{species} · {background} · 숙련 보너스 **{hero.proficiency_bonus:+d}**",
            "", "능력 | 점수 | 수정치 | 구원", "---|---:|---:|---:"]
    for ability, label in ABILITY_KO.items():
        proficient = ability in hero.save_profs
        bonus = hero.mod(ability) + (hero.proficiency_bonus if proficient else 0)
        rows.append(f"{label} | {hero.score(ability)} | {hero.mod(ability):+d} | {bonus:+d}" + (" ✓" if proficient else ""))
    skills = []
    for skill in hero.skill_profs:
        if skill in SKILL_ABILITIES:
            label, ability = SKILL_ABILITIES[skill]
            skills.append(f"{label} {hero.mod(ability) + hero.proficiency_bonus:+d}")
        else:
            skills.append(skill)
    weapons = ", ".join(content.rules["weapons"][wid]["name"] for wid in hero.weapons) or "없음"
    armor = content.rules["armor"][hero.armor]["name"] + (" + 방패" if hero.shield else "")
    rows += ["", "✓ 구원 숙련", "", "**기술 숙련:** " + (", ".join(skills) or "없음"),
             "", f"**무기:** {weapons}", "", f"**방어구:** {armor}",
             "", "**소지금:** " + fmt_money(engine.s.gold_cp),
             "", f"**회복 주사위:** {hero.hd_left}/{hero.hd_max}개 · d{hero.hit_die}",
             "", "**자원:** " + resource_summary(engine, content),
             "", "**상태:** " + (", ".join(hero.conditions) or "정상")]
    if hero.tool_profs:
        names = {"thieves_tools": "도둑 도구"}
        rows += ["", "**도구 숙련:** " + ", ".join(names.get(key, key) for key in hero.tool_profs)]
    if hero.spellcasting:
        spells = content.rules["spells"]
        rows += ["", f"**주문 능력:** {ABILITY_KO[hero.spellcasting]} · 명중 {engine.spell_attack_bonus(hero):+d} · 구원 DC {engine.spell_dc(hero)}",
                 "", "**소마법:** " + (", ".join(spells[sid]["name"] for sid in hero.cantrips) or "없음"),
                 "", "**준비된 전투 주문:** " + (", ".join(spells[sid]["name"] for sid in hero.spells) or "없음")]
    if hero.concentration:
        rows += ["", "**집중 중:** " + content.rules["spells"].get(hero.concentration, {}).get("name", hero.concentration)]
    # 가방에는 탐사 중 꺼내 쓰는 물품을, 캐릭터 시트에는 별도 장착 장비를 표시합니다.
    equipped = engine.s.vars.get("setup", {}).get("equipped_items", {})
    if equipped:
        rows += ["", "**장착 장비:** " + " · ".join(
            f"{content.items[item]['name']} {quantity}개" for item, quantity in equipped.items() if quantity > 0)]
    return "\n".join(rows)


def character_card_label(content, character_id):
    """선택 카드의 이름·직업·능력치는 원본 캐릭터 정의에서만 읽습니다."""
    engine = GameEngine(content=content, char_id=character_id)
    hero = engine.hero
    species = {"Human": "인간", "Elf": "엘프", "Dwarf": "드워프"}.get(hero.species, hero.species)
    primary = max(ABILITY_KO, key=lambda ability: hero.score(ability))
    return (f"{hero.name}\n{hero.cls_ko} · {species}\n"
            f"HP {hero.max_hp} · AC {engine.armor_class(hero)}\n"
            f"{ABILITY_KO[primary]} {hero.score(primary)} · {hero.mod(primary):+d}")


def map_markup(locations):
    """공개 지도 목록을 읽기 전용 HTML 카드로 표시합니다."""
    cards = []
    for location in locations:
        current = location.get("current") is True
        reachable = location.get("reachable") is True
        label = "현재 장소" if current else ("이동 가능" if reachable else "현재 이동 불가")
        people = ", ".join(html.escape(person["name"]) for person in location.get("npcs", []))
        cards.append(
            '<div class="trpg-map-card' + (' current' if current else '') + '">'
            f'<div style="font-size:10px;color:#b4a58c">{label}</div>'
            f'<div style="font-weight:700;font-size:12px;margin:3px 0">{html.escape(location["name"])}</div>'
            + (f'<div style="font-size:11px;color:#bcb6aa">{people}</div>' if people else '') + '</div>')
    if not cards:
        return '<p style="color:#bdd5c9">현재 공개된 지도 정보가 없습니다.</p>'
    extra = ('<p style="font-size:12px;color:#bdd5c9">장소 버튼은 8곳까지 표시합니다. '
             '나머지 공개 경로는 현재 가능한 행동에서 선택하세요.</p>' if len(locations) > 8 else "")
    return ('<div style="margin:2px 0"><div style="display:flex;flex-wrap:wrap;gap:6px">'
            + "".join(cards) + '</div>' + extra + '</div>')


def build_game_demo(api):
    """화면 구성요소와 이벤트 처리 함수를 만들고 Gradio 화면을 반환합니다.

    api는 gameplay.PlaySessions에 전달되어 실제 대화·검색·게임 처리에 쓰입니다.
    안쪽 함수들은 화면 입력과 세션 토큰을 받고, 공통 render 순서로 화면 값을 돌려줍니다."""
    register_portraits()
    sessions = PlaySessions(api)
    # 게임 결과를 읽는 화면 전용 기록입니다. 저장 파일·엔진에는 넣지 않습니다.
    feedback = CombatFeedbackTracker()
    diagnostics = RequestDiagnostics()
    busy_sessions = set()
    active_streams = {}
    starting_sessions, started_requests = set(), {}
    polled_recommendations = {}
    content = sessions.content
    characters = [(f"{name} · {role}", cid) for cid, name, role in content.characters()]
    profiles = {npc_name(card): card for card in content.cards.values() if npc_name(card)}
    map_profile_names = list(profiles)[:5]
    room_choices = [(room["name"], rid) for rid, room in content.rooms.items()]
    encounters = [(data["name"], eid) for eid, data in content.encounters.items()]
    stream_enabled = callable(api.get("output_stream"))

    def recommendation_entries(game, primary_actions=None):
        """행동 문구는 현재 엔진 선택지를, 대화 문구는 실제 생성 결과를 사용합니다."""
        if game.pending or game.engine.s.combat or game.engine.s.ended:
            return []
        available = {choice.id: choice for choice in game.choices()}
        primary = action_entries(game) if primary_actions is None else primary_actions
        primary_ids = {row["id"] for row in primary}
        shown_actions = set()
        entries = []
        for index, item in enumerate(game.suggestions[:3]):
            action_id = item.get("action_id")
            if action_id:
                current = available.get(action_id)
                # 같은 실행은 위쪽 행동 카드에 한 번만 둡니다. 목록을 삭제하거나
                # 재번호하지 않아 실제 클릭은 원래 suggestion index로 연결됩니다.
                if current is None or action_id in primary_ids or action_id in shown_actions:
                    continue
                shown_actions.add(action_id)
                check = game.content.choices.get(action_id, {}).get("check")
                tag = "판정" if check else "이동" if action_id.startswith(("move:", "travel:", "meet:")) else "행동"
                entries.append({"index": index, "kind": "action", "label": f"[{tag}] {current.label}",
                                "text": current.label, "action_id": action_id})
            elif item.get("kind", "dialogue") == "dialogue" and isinstance(item.get("text"), str):
                entries.append({"index": index, "kind": "dialogue", "label": "[대화] " + item["text"],
                                "text": item["text"], "action_id": None})
        return entries

    def recommendation_samples(game, primary_actions=None):
        # 대화·행동을 한 추천 영역에 두고 분류 표지로 구분합니다.
        return [[item["label"]] for item in recommendation_entries(game, primary_actions)], []

    def failure_display(game):
        retry = getattr(game, "retryable_dialogue_turn_id", lambda: None)()
        failed = game.last_status.startswith(("실행 실패", "이야기 생성에 실패", "응답 대기", "응답 처리"))
        if not failed and retry is None:
            return "", None
        text = "**응답을 완료하지 못했습니다.**"
        if retry is not None:
            text += "\n\n질문은 대화에 남아 있습니다. ‘답변 다시 받기’는 같은 질문의 응답만 다시 생성합니다."
        else:
            text += "\n\n현재 게임 상태와 입력 내용을 확인해 주세요. 실제 처리 기록은 ‘대화·진단 전체 복사’로 확인할 수 있습니다."
        return text, retry

    def preview_character(character_id):
        """새 게임을 만들기 전에 선택한 사전 제작 캐릭터의 실제 초기값을 보여 줍니다."""
        return character_sheet(GameEngine(content=content, char_id=character_id), content)

    def navigation_rows(game, locations=None):
        # 비공개 장소를 content.rooms에서 별도로 확장하지 않습니다.
        rows = []
        for location in game.map_locations() if locations is None else locations:
            if not isinstance(location, dict) or not isinstance(location.get("id"), str) or not isinstance(location.get("name"), str):
                continue
            people = [{"name": person["name"], "card_id": person.get("card_id"),
                       "can_meet": person.get("can_meet") is True}
                      for person in location.get("npcs", [])
                      if isinstance(person, dict) and isinstance(person.get("name"), str)]
            rows.append({"id": location["id"], "name": location["name"],
                         "current": location.get("current") is True,
                         "reachable": location.get("reachable") is True, "npcs": people})
        return rows

    def navigation_updates(game, locations=None):
        locations = navigation_rows(game, locations)
        room_updates = []
        for index in range(8):
            location = locations[index] if index < len(locations) else None
            room_updates.append(gr.update(
                value=(("📍 " if location["current"] else "이동 · ") + location["name"]) if location else "장소",
                visible=bool(game.engine.developer and location is not None),
                interactive=bool(location and not location["current"] and location["reachable"])))
        meet_updates = []
        for index in range(5):
            name = map_profile_names[index] if index < len(map_profile_names) else None
            person = next((person for location in locations for person in location["npcs"] if person["name"] == name), None)
            meet_updates.append(gr.update(value=(name + " 만나기") if name else "인물 만남",
                                          visible=bool(game.engine.developer and person is not None),
                                          interactive=bool(person and person["can_meet"])))
        return [map_markup(locations), *room_updates, *meet_updates]

    def status_markup(game, public_room=None):
        """현재 수치와 새 확정 사건 표시를 만듭니다. AI·검색을 호출하지 않습니다."""
        engine, state, hero = game.engine, game.engine.s, game.engine.hero
        ratio = max(0, min(100, int(100 * hero.hp / max(hero.max_hp, 1))))
        dev_tag = " · 개발자 테스트 기록 있음" if state.vars.get("developer_used") else ""
        turn_info = ""
        if state.combat:
            turn = state.combat.get("turn", {})
            turn_info = f" · 이번 차례: 행동 {turn.get('action', 0)} / 보너스 행동 {turn.get('bonus', 0)}"
            if "reaction" in turn:
                turn_info += f" / 반응 {turn['reaction']}"
        if public_room is None:
            data = discovery_data(game)
            public_room = data.get("current_room", {}).get("name") or content.room_name(state.room)
        stats = f'''<div class="trpg-stats">
<div><div class="trpg-place">{html.escape(public_room)}</div>
<div class="trpg-hero">{html.escape(hero.name)} · {html.escape(hero.cls_ko)} {hero.level}레벨 · AC {engine.armor_class(hero)} · {fmt_money(state.gold_cp)}</div></div>
<div class="trpg-vitals"><div>HP {hero.hp}/{hero.max_hp} · 회복 주사위 {hero.hd_left}/{hero.hd_max} · {state.minutes}분 경과{dev_tag}</div>
<div class="trpg-hp-track"><div class="trpg-hp-fill" style="width:{ratio}%"></div></div>
<div class="trpg-resource">{html.escape(resource_summary(engine, content) + turn_info)}</div></div></div>'''
        return stats + feedback.markup(game)

    def render_snapshot(token):
        """세션 토큰으로 게임을 찾아 대화·수치·선택지의 최신 표시 값을 반환합니다.

        여기서는 저장된 상태와 추천을 읽어 HP 막대, 가방, 전투, 개발자 자료를 구성합니다.
        반환 목록의 순서는 아래 view 구성요소 목록과 맞아야 한 번에 화면이 갱신됩니다."""
        game = sessions.get(token)
        engine, state, hero = game.engine, game.engine.s, game.engine.hero
        # 같은 잠금 안의 한 화면 갱신에서만 공개 정보를 함께 사용합니다.
        # 다음 갱신과 클릭 검증은 다시 읽으므로 진행 조건이 바뀌어도 묵은 값이 남지 않습니다.
        discovery = discovery_data(game)
        place_samples, people_samples = discovery_samples(game, discovery)
        public_room = discovery.get("current_room", {}).get("name") or content.room_name(state.room)
        stats = status_markup(game, public_room)
        inventory = inventory_panel(game)
        equipment = character_sheet(engine, content)
        if state.combat:
            cb = state.combat
            enemy_rows = [f"- {enemy['name']}: {engine.wound_text(enemy)}" for enemy in engine.living_enemies()]
            turn = cb.get("turn", {})
            budget = f"행동 **{turn.get('action', 0)}** · 보너스 행동 **{turn.get('bonus', 0)}**"
            # 반응 상태가 구현된 경우에만 실제 남은 값을 표시합니다.
            if "reaction" in turn:
                budget += f" · 반응 **{turn['reaction']}**"
            combat = f"### ⚔ {content.encounters[cb['enc']]['name']}\n{cb['round']}라운드 · {budget}\n\n" + "\n".join(enemy_rows)
            combat += "\n\n행동·보너스 행동은 자신의 차례 시작에 다시 준비됩니다. 보너스 행동은 해당 기능이 있을 때 사용합니다."
        else:
            combat = "현재 전투가 없습니다. 전투 중에는 이번 차례의 행동·보너스 행동 잔여 횟수가 표시됩니다."
        choices = [(v.label + (" · " + v.check_text if v.check_text else ""), v.id) for v in game.choices()]
        npc_options = [("장면 / 게임 마스터", "")] + game.nearby_npcs()
        pending = game.pending
        roll_markup = dice_panel(game) + equipment_requirements_panel(game)
        scene_actions = action_entries(game)
        can_force_roll = bool(engine.developer and game.can_force_pending_roll())
        failure_text, retry_turn = failure_display(game)
        quest = journey_panel(game, discovery)
        # 생성은 게임의 대화 처리에서 끝났습니다. 화면 갱신은 저장된 추천만 읽습니다.
        recommendation_rows, action_recommendation_rows = recommendation_samples(game, scene_actions)
        methods = game.pending_check_options()
        method_choices = [(row["label"] + f" · {row['formula']} · DC {row['dc']}"
                           + (" · 추천" if row["recommended"] else ""), row["index"]) for row in methods]
        selected_method = next((row["index"] for row in methods if row["selected"]), None)
        logs = "\n".join(f"[{event['kind']}] {event['text']}" for event in game.audit[-35:])
        current_state = game.rag_state()
        capture_diagnostics(token, game, current_state)
        return [game.messages, stats, inventory, equipment, combat,
                gr.update(choices=choices, value=None), gr.update(choices=npc_options, value=profiles.get(game.npc, {}).get("id", "")),
                gr.update(value=roll_markup, visible=bool(roll_markup)), gr.update(visible=bool(pending), value="🎲 주사위 굴리기" if pending and pending.get("roll") else "실행하기"),
                game.last_status, json.dumps(game.debug, ensure_ascii=False, indent=2),
                json.dumps(current_state, ensure_ascii=False, indent=2),
                quest, logs, game.engine.developer,
                gr.update(visible=game.engine.developer), "",
                gr.update(samples=recommendation_rows, visible=bool(recommendation_rows)),
                *navigation_updates(game, discovery.get("routes", [])), gr.update(visible=bool(pending)),
                gr.update(samples=[[row["label"] + ("\n" + row["detail"] if row["detail"] else "")] for row in scene_actions],
                          visible=bool(scene_actions)),
                gr.update(value=encounter_panel(game), visible=bool(state.combat)),
                gr.update(samples=action_recommendation_rows, visible=bool(action_recommendation_rows)),
                gr.update(choices=method_choices, value=selected_method, visible=bool(methods)), "",
                gr.update(value=game.last_status,
                          visible=game.last_status.startswith(("실행 실패", "이야기 생성에 실패", "응답 대기"))),
                gr.update(value=hero.id, interactive=True),
                {"choice_id": pending.get("choice"), "revision": pending.get("revision"),
                 "type": pending.get("type"), "ability": pending.get("ability"),
                 "method_index": pending.get("method_index")} if pending else {},
                gr.update(visible=can_force_roll, interactive=can_force_roll),
                gr.update(visible=can_force_roll, interactive=can_force_roll),
                gr.update(value=failure_text, visible=bool(failure_text)),
                gr.update(visible=retry_turn is not None, interactive=retry_turn is not None), retry_turn,
                scene_background(game, discovery), discovery_overview(game, discovery),
                gr.update(samples=place_samples, visible=bool(place_samples)),
                gr.update(samples=people_samples, visible=bool(people_samples)),
                discovery_actions(game, discovery), people_panel(game, discovery)]

    def present(view):
        """대화 사본에만 화자 초상화를 붙이고 실제 이력과 생성 입력은 그대로 둡니다."""
        view = list(view)
        view[0] = present_messages(view[0])
        return view

    def render(token):
        with sessions.lock:
            return present(render_snapshot(token))

    def capture_diagnostics(token, game, current_state=None):
        """게임을 그리는 시점의 확정 사본만 저장합니다. 복사는 게임 잠금을 잡지 않습니다."""
        meta = deepcopy(game.diagnostics_meta)
        reader = api.get("diagnostics_metadata")
        if callable(reader):
            try:
                meta.update(reader())
            except Exception as error:
                meta["runtime_observation_error"] = str(error)
        request_id = current_request_id()
        latest = game.turn_records[-1] if game.turn_records else {}
        if request_id and latest.get("request_id") == request_id and latest.get("status") == "error":
            diagnostics.finish(request_id, status="error", error=(latest.get("output") or {}).get("error"))
        diagnostics.capture(token, game.messages, game.turn_records, meta,
                            game.rag_state() if current_state is None else current_state)

    def copy_conversation(token):
        snapshot = diagnostics.snapshot(token)
        return build_copy_report(snapshot["messages"], snapshot["turn_records"],
                                 snapshot["session_meta"], snapshot["current_state"],
                                 live_diagnostics=snapshot)

    def copy_plain_conversation(token):
        return build_plain_conversation(diagnostics.snapshot(token)["messages"])

    def poll_diagnostics(token):
        progress = diagnostics.progress(token)
        labels = {"queued": "처리 순서 대기 중", "lock_wait": "게임 상태 대기 중",
                  "routing": "입력 확인 중", "retrieval": "근거 검색 중", "handoff": "생성 자료 정리 중",
                  "model_wait": "모델 사용 순서 대기 중", "model_loading": "모델 준비 중",
                  "model_ready": "모델 연결 완료", "generating": "답변 작성 중", "validation": "내용 확인 중",
                  "recommendations": "선택지 생성 중", "render": "화면 갱신 중", "idle": "대기 중"}
        progress["label"] = labels.get(progress.get("stage"), "처리 중")
        detail = (progress["label"] + " · " + str(round(progress.get("elapsed_seconds", 0), 1)) + "초"
                  + "\n이 정보는 자동으로 전체 진단 복사에 포함됩니다.")
        return progress, detail

    def prepare_operation(kind):
        def prepare(*values):
            token = values[-1] if values else None
            entry = values[0] if len(values) > 1 else None
            if isinstance(entry, (list, tuple)) and entry:
                entry = entry[0]
            diagnostics.register(token, kind, entry if isinstance(entry, str) else None)
        return prepare

    def launch_recommendations(game, token):
        """확정된 본문 뒤에 선택적 추천을 만들며, 다음 입력은 이 작업을 취소할 수 있습니다."""
        job = game.take_recommendation_job()
        if job is None:
            return
        parent_request = current_request_id()
        request_id = diagnostics.register(token, "recommendations")
        diagnostics.claim(token, "recommendations")

        def run_job():
            with diagnostics.bind(request_id):
                try:
                    current_stage("recommendations", parent_request_id=parent_request)
                    try:
                        value = game.generate_recommendations(job)
                    except Exception as error:
                        value = {"error": repr(error), "suggestions": []}
                    with sessions.lock:
                        applied = game.apply_recommendations(job, value)
                        # 늦게 완성된 추천도 다음 사용자 입력 없이 곧바로 전체 복사에 담습니다.
                        if diagnostics.stage(request_id, "recommendations", result=value, applied=bool(applied)):
                            capture_diagnostics(token, game)
                    status = "error" if value.get("error") else "cancelled" if value.get("reason") == "cancelled" else "completed"
                    diagnostics.finish(request_id, status=status, error=value.get("error"))
                except Exception as error:
                    current_stage("error", error={"type": type(error).__name__, "message": str(error),
                                                  "traceback": traceback.format_exc()})
                    diagnostics.finish(request_id, status="error", error=str(error))
                finally:
                    diagnostics.finish(request_id)
        Thread(target=run_job, name="trpg-recommendations", daemon=True).start()

    def poll_recommendations(token):
        # 진행 중인 턴을 기다리거나 덮어쓰지 않습니다. 늦게 도착한 행동 추천은
        # 주 행동 목록의 앞쪽으로 함께 올려, 추천에서 숨겼는데 못 찾는 일을 막습니다.
        if not token or token in busy_sessions or not sessions.lock.acquire(blocking=False):
            return gr.skip(), gr.skip(), gr.skip()
        try:
            game = sessions.get(token)
            primary = action_entries(game)
            rows, action_rows = recommendation_samples(game, primary)
            action_samples = [[row["label"] + ("\n" + row["detail"] if row["detail"] else "")] for row in primary]
            key = (game.revision, tuple(row[0] for row in rows), tuple(row[0] for row in action_rows),
                   tuple(row[0] for row in action_samples))
            if polled_recommendations.get(token) == key:
                return gr.skip(), gr.skip(), gr.skip()
            polled_recommendations[token] = key
            return (gr.update(samples=rows, visible=bool(rows)),
                    gr.update(samples=action_rows, visible=bool(action_rows)),
                    gr.update(samples=action_samples, visible=bool(action_samples)))
        finally:
            sessions.lock.release()

    def record_error(game, operation, error):
        current_stage("error", error={"type": type(error).__name__, "message": str(error),
                                      "traceback": traceback.format_exc()})
        diagnostics.finish(current_request_id(), status="error", error=str(error))
        game.last_status = "실행 실패: " + str(error)
        game.debug = {"오류": repr(error), "단계": getattr(operation, "__name__", "callback")}
        if getattr(error, "raw_response", None):
            game.debug["실제 모델 원문"] = error.raw_response

    def perform_sync(token, operation, *args):
        """하나의 화면 동작을 실행한 뒤 같은 형식으로 화면을 다시 그립니다.

        sessions.lock 안에서 게임을 찾고 operation을 호출합니다. 예외가 나면 실행 상태와
        개발자 자료에 오류를 남겨 render가 함께 보여 주도록 합니다."""
        with sessions.lock:
            game = sessions.get(token)
            before = feedback.begin(game, getattr(operation, "__name__", ""))
            try:
                game.cancel_recommendations()
                operation(game, *args)
                launch_recommendations(game, token)
            except Exception as error:
                record_error(game, operation, error)
            finally:
                feedback.capture(game, before)
            return render(token)

    def perform_stream(token, operation, args, preview_text=None):
        """작업 스레드의 확정 상태 사본과 안전한 생성 조각만 화면에 전달합니다."""
        request_id = current_request_id()
        current_stage("lock_wait")
        packets = Queue()
        stopped = Event()
        worker = None
        last_base = None
        initial_snapshot = None

        def publish(kind, value, base=None):
            if stopped.is_set():
                return False
            packets.put((kind, value, base))
            return True

        def run():
            with diagnostics.bind(request_id):
                with sessions.lock:
                    if stopped.is_set():
                        return
                    before = None
                    try:
                        game = sessions.get(token)
                        before = feedback.begin(game, getattr(operation, "__name__", ""))
                        draft = {"text": "", "speaker": "", "kind": "", "blocks": []}
                        base = deepcopy(initial_snapshot)
                        last_chunk = 0.0

                        def sink(event):
                            nonlocal base, last_chunk
                            if stopped.is_set():
                                return False
                            kind = event.get("event")
                            if kind not in {"start", "chunk", "end", "error", "status"}:
                                return True
                            if kind == "start":
                                draft.update(text="", speaker=event.get("speaker") or "", kind=event.get("kind") or "", blocks=[])
                                last_chunk = 0.0
                            elif kind == "error":
                                draft.update(text="", speaker="", kind="", blocks=[])
                            if kind in {"chunk", "end"}:
                                if isinstance(event.get("blocks"), list):
                                    draft["blocks"] = deepcopy(event["blocks"])
                                text = event.get("text")
                                if isinstance(text, str):
                                    draft.update(text=text, speaker=event.get("speaker", draft["speaker"]) or "",
                                                 kind=event.get("kind") or draft["kind"])
                                if kind == "chunk" and time.monotonic() - last_chunk < 0.08:
                                    return True
                                last_chunk = time.monotonic()
                            if kind in {"start", "error"}:
                                # 생성 구간마다 실제 이력만 새로 읽고, 큰 상태/디버그 렌더는 완료 때 갱신합니다.
                                base = list(base)
                                base[0] = deepcopy(game.messages)
                            # 엔진 결과는 생성보다 먼저 확정됩니다. 그때 받은 사건만 즉시 표시합니다.
                            feedback.capture(game, before)
                            view = deepcopy(base)
                            view[1] = status_markup(game)
                            view[34] = gr.update(value=encounter_panel(game), visible=bool(game.engine.s.combat))
                            # 상태 이벤트도 공개된 두 진행 문구로만 표시합니다.
                            checking = kind in {"end", "error"} or (
                                kind == "status" and event.get("text") == "내용 확인 중")
                            view[9] = "내용 확인 중" if checking else "답변 작성 중"
                            if draft["blocks"]:
                                rows = [{"role": "assistant", "content": block["text"],
                                         "speaker": block.get("speaker") or "", "speaker_id": block.get("speaker_id"),
                                         "kind": block["kind"]}
                                        for block in draft["blocks"]
                                        if isinstance(block, dict) and block.get("kind") in {"narration", "speech"}
                                        and isinstance(block.get("text"), str) and block["text"].strip()]
                                view[0] = [*base[0], *rows]
                            elif draft["text"]:
                                heading = ("**" + draft["speaker"] + "**\n"
                                           if draft["speaker"] and draft["kind"] != "scene_dialogue" else "")
                                displayed = draft["text"] if heading and draft["text"].startswith(heading) else heading + draft["text"]
                                last = base[0][-1] if base[0] else {}
                                if last.get("role") != "assistant" or last.get("content") != displayed:
                                    view[0] = [*base[0], {"role": "assistant", "content": displayed, "speaker": draft["speaker"]}]
                            packet_kind = "reset" if kind in {"start", "error"} else "chunk"
                            return publish(packet_kind, view, base)

                        # 완성 블록이 오지 않아도 생성 측이 화면 종료를 즉시 확인합니다.
                        with api["output_stream"](sink, cancel_event=stopped):
                            operation(game, *args)
                        launch_recommendations(game, token)
                    except Exception as error:
                        if not stopped.is_set():
                            try:
                                record_error(sessions.get(token), operation, error)
                            except Exception:
                                pass
                    finally:
                        if not stopped.is_set():
                            try:
                                if before is not None:
                                    feedback.capture(game, before)
                                publish("done", deepcopy(render_snapshot(token)))
                            except Exception as error:
                                publish("error", str(error))

        try:
            with sessions.lock:
                game = sessions.get(token)
                game.cancel_recommendations()
                busy_sessions.add(token)
                active_streams[token] = stopped
                last_base = deepcopy(render_snapshot(token))
                # 모든 생성 프레임과 대기 종료 화면에서 이전 턴의 추천을 숨깁니다.
                last_base[17] = gr.update(samples=[], visible=False)
                last_base[33] = gr.update(samples=[], visible=False)
                last_base[35] = gr.update(samples=[], visible=False)
                last_base[43] = gr.update(value="", visible=False)
                last_base[44] = gr.update(visible=False, interactive=False)
                last_base[45] = None
                initial_snapshot = deepcopy(last_base)
                first = deepcopy(last_base)
                text = preview_text(game) if callable(preview_text) else preview_text
                preview = text if isinstance(text, dict) else {"content": text}
                if isinstance(preview.get("content"), str) and preview["content"].strip():
                    first[0] = [*first[0], {"role": "user", **preview}]
                first[9] = "답변 작성 중"
            # 실제 messages에는 저장하지 않는 입력 미리보기입니다.
            yield first
            worker = Thread(target=run, name="trpg-output-stream", daemon=True)
            worker.start()
            deadline = time.monotonic() + float(api.get("stream_timeout_seconds", 180))
            queued_control = None
            typing = _TypingView()
            while True:
                # 개발자 복구·접속 종료가 취소한 이전 화면은 새 상태를 덮어쓰지 않습니다.
                if stopped.is_set():
                    break
                if time.monotonic() >= deadline:
                    diagnostics.finish(request_id, status="timeout", error="응답 전체 대기 기한 초과")
                    stopped.set()
                    timed_out = deepcopy(last_base)
                    timed_out[9] = "응답 대기를 중단했습니다."
                    timed_out[38] = gr.update(value=timed_out[9], visible=True)
                    timed_out[43] = gr.update(value="**응답 대기를 중단했습니다.** 입력한 질문과 실제 처리 기록을 확인해 주세요.", visible=True)
                    if len(timed_out[0]) < len(first[0]):
                        timed_out[0] = deepcopy(first[0])
                    yield timed_out
                    break
                try:
                    packet = queued_control if queued_control is not None else packets.get(
                        timeout=typing.tick if typing.pending else 0.2)
                    queued_control = None
                    if packet[0] == "chunk":
                        # 이미 쌓인 누적 초안은 최신 것만 보내고 제어 이벤트 순서는 보존합니다.
                        while True:
                            try:
                                following = packets.get_nowait()
                            except Empty:
                                break
                            if following[0] != "chunk":
                                if following[0] in {"reset", "error"}:
                                    packet = following
                                else:
                                    queued_control = following
                                break
                            packet = following
                    kind, value, base = packet
                except Empty:
                    frame = typing.advance(time.monotonic())
                    if frame is not None:
                        yield frame
                        continue
                    if not worker.is_alive():
                        failed = deepcopy(last_base)
                        failed[9] = "응답 처리가 중단되었습니다."
                        failed[38] = gr.update(value=failed[9], visible=True)
                        failed[43] = gr.update(value="**응답 처리가 중단되었습니다.** 입력 내용과 실제 처리 기록을 확인해 주세요.", visible=True)
                        yield failed
                        break
                    continue
                if kind == "error":
                    failed = deepcopy(last_base)
                    failed[9] = "실행 실패: " + value
                    failed[38] = gr.update(value=failed[9], visible=True)
                    failed[43] = gr.update(value="**응답을 완료하지 못했습니다.** 입력 내용과 실제 처리 기록을 확인해 주세요.", visible=True)
                    yield failed
                    break
                if base is not None:
                    last_base = base
                if kind == "done":
                    # 처리 완료 후의 긴 대기열은 짧게 따라잡고 실제 최종 상태로 확정합니다.
                    if typing.matches_final(value) and typing.offer(value):
                        finish_by = min(deadline, time.monotonic() + typing.finish_budget)
                        while typing.pending and time.monotonic() < finish_by:
                            frame = typing.advance(time.monotonic(), finish_by)
                            if frame is not None:
                                yield frame
                            if typing.pending:
                                time.sleep(typing.tick)
                    yield value
                    break
                if kind == "reset":
                    typing.clear()
                    yield value
                elif typing.offer(value):
                    frame = typing.advance(time.monotonic())
                    if frame is not None:
                        yield frame
                else:
                    typing.clear()
                    yield value
        finally:
            # close/timeout 이후의 sink는 False를 반환해 생성 측의 취소 처리를 요청합니다.
            stopped.set()
            if active_streams.get(token) is stopped:
                active_streams.pop(token, None)
                busy_sessions.discard(token)
            if worker is not None:
                worker.join(timeout=2)

    def perform(token, operation, *args, preview_text=None):
        if stream_enabled:
            return perform_stream(token, operation, args, preview_text)
        return perform_sync(token, operation, *args)

    def stream_callback(callback, operation_name=None):
        kind = operation_name or callback.__name__

        @wraps(callback)
        def streamed(*args, **kwargs):
            token = args[-1] if args else None
            request_id = diagnostics.claim(token, kind)
            iterator = None
            try:
                with diagnostics.bind(request_id):
                    current_stage("lock_wait")
                    result = callback(*args, **kwargs)
                if isinstance(result, GeneratorType):
                    iterator = result
                    while True:
                        with diagnostics.bind(request_id):
                            try:
                                frame = next(iterator)
                            except StopIteration:
                                break
                        yield present(frame)
                else:
                    yield present(result)
            except GeneratorExit:
                diagnostics.finish(request_id, status="cancelled", error="화면 요청 연결 종료")
                raise
            except Exception as error:
                diagnostics.finish(request_id, status="error", error=str(error))
                raise
            finally:
                if iterator is not None:
                    with diagnostics.bind(request_id):
                        iterator.close()
                diagnostics.finish(request_id)
        return streamed

    def close_session(token):
        diagnostics.close(token)
        if token in active_streams:
            active_streams[token].set()
        with sessions.lock:
            if token in sessions.games:
                feedback.close(sessions.games[token])
            sessions.close(token)
            polled_recommendations.pop(token, None)
            busy_sessions.discard(token)

    def start(character, previous, request: gr.Request):
        """준비 안내를 먼저 보낸 뒤 새 게임을 만들며 중복 시작을 막습니다."""
        key = getattr(request, "session_hash", None) or previous or "initial"
        if key in starting_sessions:
            yield [gr.skip(), *[gr.skip() for _ in view], gr.skip()]
            return
        starting_sessions.add(key)
        first = [gr.skip() for _ in view]
        first[38] = gr.update(value="여정을 준비하고 있습니다. 첫 대화를 불러오는 중입니다.", visible=True)
        first[39] = gr.update(value=character, interactive=False)
        yield [gr.skip(), *first, gr.update(value="여정 준비 중…", interactive=False)]
        try:
            cached = started_requests.get(key, {})
            if (cached.get("previous") == previous and cached.get("character") == character
                    and cached.get("token") in sessions.games):
                next_token = cached["token"]
            else:
                next_token = sessions.start(character, previous)
                started_requests[key] = {"previous": previous, "character": character, "token": next_token}
            ready = render(next_token)
            ready[38] = gr.update(value="여정을 시작했습니다.", visible=False)
            final = [next_token, *ready, gr.update(value="이 인물로 시작", interactive=True)]
        except Exception as error:
            failed = [gr.skip() for _ in view]
            failed[38] = gr.update(value="여정을 시작하지 못했습니다. " + str(error), visible=True)
            failed[39] = gr.update(value=character, interactive=True)
            final = [gr.skip(), *failed, gr.update(value="이 인물로 시작", interactive=True)]
        finally:
            starting_sessions.discard(key)
        yield final

    def send(message, token):
        preview = message.strip() if isinstance(message, str) and 1 <= len(message.strip()) <= 1200 else None
        explicit_action = bool(preview and preview.startswith("**") and preview.endswith("**"))
        return perform(token, lambda game: game.send(message),
                       preview_text={"content": preview, "input_kind": "action" if explicit_action else None})

    def send_action(message, token):
        preview = message.strip() if isinstance(message, str) and 1 <= len(message.strip()) <= 1200 else None
        return perform(token, lambda game: game.send_action(message),
                       preview_text={"content": preview, "input_kind": "action"})

    def mark_action(message):
        """같은 입력칸의 행동 표시만 바꿉니다. 실제 실행은 보내기를 누를 때 합니다."""
        text = message.strip() if isinstance(message, str) else ""
        return text[2:-2].strip() if text.startswith("**") and text.endswith("**") else "**" + text + "**"

    def choose_discovered_place(selected, token):
        label = selected[0] if isinstance(selected, list) and selected else None
        def operation(game):
            data = discovery_data(game)
            labels, _ = discovery_samples(game)
            available = [row for row in data.get("routes", []) if not row.get("current") and row.get("reachable")]
            match = next((row for row, sample in zip(available, labels) if sample == [label]), None)
            if not match or match.get("current") or not match.get("reachable"):
                raise ValueError("지금 이동할 수 있는 경로를 선택해 주세요.")
            game.travel_to(match["id"])
        return perform(token, operation)

    def choose_discovered_person(selected, token):
        label = selected[0] if isinstance(selected, list) and selected else None
        def operation(game):
            data = discovery_data(game)
            _, labels = discovery_samples(game)
            available = [row for row in data.get("people", []) if row.get("can_meet")]
            match = next((row for row, sample in zip(available, labels) if sample == [label]), None)
            if not match or not match.get("can_meet"):
                raise ValueError("지금 만날 수 있는 인물을 선택해 주세요.")
            if match.get("here"):
                game.talk_to(match["name"])
            else:
                game.visit_npc(match["name"])
        return perform(token, operation)

    def choose_suggestion(selected, token):
        # 클릭 당시 문구를 확인하여 늦게 도착한 이전 추천을 다른 행동으로 실행하지 않습니다.
        label = selected[0] if isinstance(selected, list) and selected else None
        def preview(game):
            selected = next((item for item in recommendation_entries(game) if item["label"] == label), None)
            return selected.get("text") if selected else None
        def operation(game):
            entry = next((item for item in recommendation_entries(game) if item["label"] == label), None)
            if entry is None:
                raise ValueError("추천이 바뀌었습니다. 현재 표시된 항목을 선택해 주세요.")
            game.choose_suggestion(entry["index"])
        return perform(token, operation, preview_text=preview)

    def retry_dialogue(expected_turn_id, token):
        return perform(token, lambda game: game.retry_failed_dialogue(expected_turn_id))

    def choose(choice, token):
        """지금 가능한 선택지 ID를 확인한 뒤 선택 문장을 대화에 남기고 행동을 제안합니다.

        선택지의 label은 사용자가 읽는 문장이고, 실제 실행 연결에는 choice ID를 사용합니다."""
        def action(game):
            if not choice:
                raise ValueError("행동을 먼저 선택하세요.")
            view = next((v for v in game.choices() if v.id == choice), None)
            if view is None:
                raise ValueError("현재 실행할 수 없는 행동입니다.")
            game.last_input_kind = "action"
            game.messages.append({"role": "user", "content": "**" + view.label + "**",
                                  "input_kind": "action", "action_id": choice,
                                  "speaker_id": game.engine.hero.id, "speaker": game.engine.hero.name})
            game.propose(choice, direct=True)
        def preview(game):
            selected = next((item for item in game.choices() if item.id == choice), None)
            return {"content": "**" + selected.label + "**", "input_kind": "action",
                    "action_id": choice, "speaker_id": game.engine.hero.id,
                    "speaker": game.engine.hero.name} if selected else None
        return perform(token, action, preview_text=preview)

    def select_npc(name, token):
        return perform(token, lambda game: game.talk_to(name))

    def map_travel(index, token):
        def operation(game):
            game.engine.require_developer()
            locations = navigation_rows(game)
            if not 0 <= index < min(8, len(locations)):
                raise ValueError("현재 지도에 없는 장소입니다.")
            location = locations[index]
            if location["current"] or not location["reachable"]:
                raise ValueError("현재 이동할 수 없는 장소입니다.")
            game.travel_to(location["id"])
        return perform(token, operation)

    def map_meet(index, token):
        def operation(game):
            game.engine.require_developer()
            if not 0 <= index < len(map_profile_names):
                raise ValueError("현재 만날 수 없는 인물입니다.")
            name = map_profile_names[index]
            people = [person for location in navigation_rows(game) for person in location["npcs"]]
            if not any(person["name"] == name and person["can_meet"] for person in people):
                raise ValueError("공개된 지도에서 현재 만날 수 없는 인물입니다.")
            game.visit_npc(name)
        return perform(token, operation)

    def map_callback(handler, index):
        def callback(token):
            return handler(index, token)
        return callback

    def choose_scene_action(selected, token):
        def operation(game):
            matches = [row for row in action_entries(game)
                       if isinstance(selected, list) and selected == [row["label"] + ("\n" + row["detail"] if row["detail"] else "")]]
            if len(matches) != 1:
                raise ValueError("현재 장면의 선택지를 다시 골라 주세요.")
            row = matches[0]
            current = next(choice for choice in game.choices() if choice.id == row["id"])
            game._consume_suggestions()
            game.last_input_kind = "action"
            game.messages.append({"role": "user", "content": "**" + current.label + "**",
                                  "input_kind": "action", "action_id": row["id"],
                                  "speaker_id": game.engine.hero.id, "speaker": game.engine.hero.name})
            game.propose(row["id"], direct=True)
        return perform(token, operation)

    def roll(token):
        return perform(token, lambda game: game.resolve_pending())

    def force_pending_result(context, token, success):
        """개발자 권한과 화면에 표시한 판정의 일치 여부를 서버에서 다시 확인합니다."""
        def operation(game):
            if not game.engine.developer:
                raise ValueError("개발자 모드에서만 판정 결과를 시험할 수 있습니다.")
            pending = game.pending
            fields = {"choice_id": "choice", "revision": "revision", "type": "type",
                      "ability": "ability", "method_index": "method_index"}
            if (not isinstance(context, dict) or not pending
                    or not all(key in context and context[key] == pending.get(field) for key, field in fields.items())
                    or pending.get("revision") != game.revision or not game.can_force_pending_roll()):
                raise ValueError("판정 내용이 바뀌었습니다. 현재 행동과 판정 방법을 다시 확인하세요.")
            return game.resolve_pending_developer(success)
        return perform(token, operation)

    def dev_roll_success(context, token):
        return force_pending_result(context, token, True)

    def dev_roll_failure(context, token):
        return force_pending_result(context, token, False)

    def select_check_method(index, context, token):
        context = context if isinstance(context, dict) else {}
        return perform_sync(token, lambda game: game.select_check_method(
            index, choice_id=context.get("choice_id"), revision=context.get("revision")))

    def cancel(token):
        def operation(game):
            game.pending = None
            game.refresh_suggestions()
            game.last_status = "대기 중인 행동을 취소했습니다."
        return perform(token, operation)

    def toggle_developer(enabled, token):
        return perform(token, lambda game: game.set_developer(enabled))

    def dev_meet(name, token):
        return perform(token, lambda game: game.developer_jump(profiles[name]["room"], name))

    def dev_move(room, token):
        return perform(token, lambda game: game.developer_jump(room))

    def dev_success(token):
        def operation(game):
            game._record(game.engine.arm_success())
            if game.pending:
                game.pending["revision"] = game.revision
            game.last_status = "다음 플레이어 판정 1회 성공 보장. 적의 굴림·일반 주사위 합계에는 적용하지 않습니다."
        return perform(token, operation)

    def developer_result(token, method, status):
        def operation(game):
            events = getattr(game.engine, method)()
            game.pending = None
            game._record(events)
            game.refresh_suggestions()
            game.last_status = status
        return perform(token, operation)

    def dev_win(token):
        return developer_result(token, "force_victory", "개발자 전투 승리를 적용했습니다.")

    def dev_heal(token):
        return developer_result(token, "full_heal", "개발자 회복을 적용했습니다.")

    def dev_combat(encounter, token):
        return perform(token, lambda game: game.developer_encounter(encounter))

    def free_check(ability, token):
        return perform(token, lambda game: game.free_check(ability))

    def utility_dice(expression, token):
        return perform(token, lambda game: game.utility_dice(expression))

    def rule_mode(mode, token):
        def operation(game):
            game.engine.require_developer()
            if mode not in {"available", "verified"}:
                raise ValueError("지원하지 않는 검색 방식입니다.")
            game.rule_mode = mode
            game.last_status = "룰 질문 검색 방식: " + mode
        return perform(token, operation)

    def save(token):
        """세션의 저장 자료를 임시 JSON 파일로 만들고 다운로드 버튼에 연결합니다.

        게임 내용을 JSON으로 바꾸는 일은 game.save_data가 담당합니다."""
        with sessions.lock:
            game = sessions.get(token)
            path = Path(tempfile.gettempdir()) / ("trpg-save-" + token[:12] + ".json")
            path.write_text(json.dumps(game.save_data(), ensure_ascii=False, indent=2), encoding="utf-8")
            return gr.update(value=str(path), visible=True)

    def restore(file, token):
        """업로드한 저장 JSON을 읽어 현재 게임의 복원 절차에 전달합니다.

        파일 선택과 크기를 먼저 확인한 뒤 restore_data를 호출하고, 공통 perform 경로로
        복원 결과와 오류를 화면에 반영합니다."""
        def operation(game):
            if not file:
                raise ValueError("저장 파일을 먼저 선택하세요.")
            path = Path(file)
            if path.stat().st_size > 10_000_000:
                raise ValueError("저장 파일이 너무 큽니다.")
            game.restore_data(json.loads(path.read_text(encoding="utf-8-sig")))
            feedback.reset(game)  # 저장된 과거 피해를 새 피해로 재생하지 않습니다.
        return perform(token, operation)

    def developer_checkpoints(token):
        """복구 가능한 개발자 기록 목록만 읽고 아직 게임 상태를 바꾸지 않습니다."""
        with sessions.lock:
            game = sessions.get(token)
            game.engine.require_developer()
            rows = game.developer_checkpoints()
            choices = [(row["label"], row["id"]) for row in rows]
            return gr.update(choices=choices, value=choices[0][1] if choices else None)

    def developer_restore(identifier, token):
        # 세션 잠금을 기다리기 전에 이전 생성의 취소 신호부터 보냅니다.
        if token in active_streams:
            active_streams[token].set()
        def operation(game):
            game.developer_restore_checkpoint(identifier)
            feedback.reset(game)
        return perform(token, operation)

    with gr.Blocks(title="잠긴 종 · 에코 타운", analytics_enabled=False,
                   fill_width=True, fill_height=False) as demo:
        token = gr.State(value=None, time_to_live=3600, delete_callback=close_session)
        method_context = gr.State(value={})
        with gr.Column(elem_id="game-landing", min_width=0):
            gr.HTML(scene_background(SimpleNamespace(engine=GameEngine(content=content, char_id=characters[0][1]), npc="")),
                    elem_id="game-landing-background")
            gr.Markdown("에코 타운\n# 잠긴 종\n우물 아래에서 울려오는 소리를 따라, 사라진 사람들의 행방을 찾아서.",
                        elem_id="game-landing-title")
            landing_start = gr.Button("새 여정 시작", variant="primary", elem_id="game-landing-start")
            gr.Markdown("대화와 선택, 주사위로 이어 가는 모험", elem_classes="landing-footnote")
        character_buttons = []
        with gr.Column(elem_id="game-character-screen", min_width=0):
            gr.Markdown("## 이번 여정을 함께할 인물\n능력과 장비가 준비된 네 인물 중 하나를 고르세요.")
            with gr.Row(elem_classes="character-card-grid"):
                for label, identifier in characters:
                    button = gr.Button(character_card_label(content, identifier), elem_id="character-card-" + identifier,
                                       elem_classes="character-card", min_width=0)
                    character_buttons.append((identifier, button))
            character = gr.Dropdown(choices=characters, value=characters[0][1], label="선택한 인물",
                                   show_label=True, container=True, min_width=0, elem_id="game-character-select")
            with gr.Row(elem_classes="character-screen-actions"):
                selection_back = gr.Button("돌아가기", scale=0, min_width=100)
                start_button = gr.Button("이 인물로 시작", variant="primary", elem_id="game-character-start")
        journey_notice = gr.Markdown("", visible=False, elem_id="game-start-notice")
        with gr.Column(elem_id="game-play-screen", min_width=0):
            with gr.Row(elem_classes="game-toolbar"):
                gr.Markdown("# 잠긴 종\n에코 타운 · 사라진 사람들과 우물 아래의 소문", elem_id="game-heading")
                new_journey_button = gr.Button("새 여정", scale=0, min_width=88, size="sm")
                help_button = gr.Button("조작법 ?", scale=0, min_width=88, size="sm")
            stats = gr.HTML("<div class='trpg-stats'>캐릭터를 고르고 여정을 시작하세요.</div>", elem_id="game-statusbar")
            with gr.Group(visible=False, elem_id="game-help-overlay") as help_overlay:
                with gr.Column(elem_classes="game-help-card"):
                    with gr.Row():
                        gr.Markdown("## 조작법")
                        close_help = gr.Button("닫기 ×", scale=0, min_width=80, size="sm")
                    gr.Markdown(GAME_HELP)
            map_buttons, meet_buttons = [], []
            discovery_overview_view = gr.HTML("<div class='discovery-overview'><span>주변의 이야기를 들으며 장소와 인물을 발견하세요.</span></div>", elem_id="game-known-summary")
            with gr.Column(elem_id="game-layout"):
                with gr.Tabs(elem_id="game-main-tabs"):
                    with gr.Tab("스토리"):
                        with gr.Column(elem_id="game-story-shell", min_width=0):
                            scene_background_view = gr.HTML("<div class='scene-art scene-village'></div><div class='scene-shade'></div>", elem_id="game-scene-background")
                            # 입력칸 위의 이야기·선택지는 공간이 부족할 때만 함께 스크롤합니다.
                            with gr.Column(elem_id="game-story-content", min_width=0):
                                battle_view = gr.HTML("", visible=False, elem_id="game-battle")
                                with gr.Row(elem_classes="game-chat-tools"):
                                    gr.Markdown("이야기")
                                    gr.HTML('<div class="trpg-turn-clock" data-running="false" role="timer" aria-live="off" '
                                            'aria-label="전체 응답 처리 경과 시간" title="입력 분류·검색·응답 표시를 포함한 전체 요청 처리 시간">'
                                            '<span class="trpg-turn-stage">입력 전달 중</span><span class="trpg-turn-seconds">0.0초</span></div>'
                                            '<span class="trpg-opening-hint">이야기를 누르면 바로 읽기</span>',
                                            elem_id="game-turn-clock", scale=0, min_width=0)
                                    plain_copy_button = gr.Button("대화 복사", size="sm", scale=0, min_width=80, elem_id="game-copy-plain")
                                    copy_button = gr.Button("전체 진단 복사", size="sm", scale=0, min_width=135,
                                                            elem_id="game-copy-conversation")
                                copy_buffer = gr.Textbox(visible=False)
                                plain_copy_buffer = gr.Textbox(visible=False)
                                live_progress_buffer = gr.JSON(visible=False)
                                with gr.Accordion("대화·처리 진단", open=False):
                                    live_diagnostics_view = gr.Textbox(label="현재 처리 단계", interactive=False, lines=2, max_lines=3)
                                chat = gr.Chatbot(label="이야기", height=180,
                                                  min_width=0, elem_id="game-chat", sanitize_html=True, layout="panel",
                                                  show_label=False, buttons=[], autoscroll=True)
                                response_failure = gr.Markdown("", visible=False, elem_id="game-response-failure")
                                retry_button = gr.Button("답변 다시 받기", visible=False, size="sm", scale=0)
                                retry_turn = gr.State(value=None)
                                action_cards = gr.Dataset(components=["textbox"], type="values", samples=[],
                                                          label="지금 할 수 있는 일", elem_id="game-actions",
                                                          layout="gallery", samples_per_page=4, visible=False)
                                pending = gr.HTML("", visible=False, elem_id="game-dice")
                                check_method = gr.Dropdown(choices=[], value=None, label="판정 방법 · 수정치와 난이도",
                                                           visible=False, elem_id="game-check-method")
                                with gr.Row(elem_classes="game-composer"):
                                    roll_button = gr.Button("🎲 주사위 굴리기", variant="primary", visible=False)
                                    dev_roll_success_button = gr.Button("개발자 검사 · 성공", size="sm", visible=False)
                                    dev_roll_failure_button = gr.Button("개발자 검사 · 실패", size="sm", visible=False)
                                    cancel_button = gr.Button("다른 선택", size="sm", visible=False, scale=0, min_width=90)
                                recommendations = gr.Dataset(components=["textbox"], type="values", samples=[],
                                                             label="이어서 할 말과 행동", elem_id="game-recommendations",
                                                             layout="gallery", samples_per_page=3, visible=False)
                                action_recommendations = gr.Dataset(components=["textbox"], type="values", samples=[],
                                                                   label="행동 추천", elem_id="game-action-recommendations",
                                                                   layout="gallery", samples_per_page=3, visible=False)
                            with gr.Row(elem_classes="game-composer", elem_id="game-unified-composer"):
                                message = gr.Textbox(label="말 또는 행동", placeholder="말을 건네거나 **행동**을 입력하세요",
                                                     lines=1, max_lines=3, show_label=False, container=False,
                                                     scale=1, min_width=0, elem_id="game-message", elem_classes="game-input")
                                action_send_button = gr.Button("행동 표시 추가", scale=0, min_width=0, size="sm",
                                                               elem_id="game-mark-action", elem_classes="game-icon-button")
                                send_button = gr.Button("메시지 보내기", variant="primary", scale=0, min_width=0,
                                                        elem_id="game-send-message", elem_classes="game-icon-button")
                            # 이전 API의 입력·출력 순서를 유지하는 비표시 연결입니다.
                            action_message = gr.Textbox(visible=False)
                            action_submit_button = gr.Button("행동 전송", visible=False)
                            mark_action_compat = gr.Button("행동 표시 API", visible=False)
                            gr.Markdown("Enter 전송 · Shift+Enter 줄바꿈 · 행동 버튼으로 표시한 뒤 보내세요.",
                                        elem_classes="game-input-help")
                    with gr.Tab("퀘스트"):
                        quest = gr.HTML('<p class="empty-note">아직 들은 부탁이 없습니다.</p>', elem_id="game-quest")
                    with gr.Tab("현재 상태"):
                        with gr.Tabs():
                            with gr.Tab("가방"):
                                inventory = gr.HTML("", elem_id="game-inventory")
                            with gr.Tab("캐릭터"):
                                equipment = gr.Markdown("장비가 표시됩니다.", elem_id="game-character")
                                with gr.Accordion("선택한 캐릭터 미리 보기", open=False):
                                    preview = gr.Markdown(preview_character(characters[0][1]))
                    with gr.Tab("지도"):
                        gr.Markdown("### 발견한 길\n알게 된 장소만 표시합니다. 지금 갈 수 있는 길을 골라 이동하세요.")
                        map_view = gr.HTML("")
                        discovered_places = gr.Dataset(components=["textbox"], type="values", samples=[],
                                                        label="이동할 수 있는 곳", layout="gallery", samples_per_page=8,
                                                        visible=False, elem_id="game-known-places")
                    with gr.Tab("인물"):
                        people_view = gr.HTML('<p class="empty-note">아직 만난 인물이 없습니다.</p>', elem_id="game-known-people-list")
                        discovered_people = gr.Dataset(components=["textbox"], type="values", samples=[],
                                                        label="지금 만날 수 있는 인물", layout="gallery", samples_per_page=6,
                                                        visible=False, elem_id="game-known-people")
                    with gr.Tab("알아낸 정보"):
                        gr.Markdown("### 알아낸 정보\n직접 듣거나 발견한 단서가 기록됩니다.")
                        discovery_actions_view = gr.HTML("<p>여정을 시작하면 현재 가능한 행동이 표시됩니다.</p>", elem_id="game-known-actions")
                    with gr.Tab("기록"):
                        combat = gr.Markdown("전투 기록이 표시됩니다.")
                        with gr.Accordion("저장 · 불러오기", open=False):
                            save_button = gr.Button("저장 파일 만들기", size="sm")
                            download = gr.DownloadButton("저장 파일 다운로드", visible=False)
                            save_file = gr.File(label="저장 파일", file_types=[".json"], type="filepath")
                            load_button = gr.Button("불러오기", size="sm")
            with gr.Accordion("개발자 도구", open=False, elem_id="game-developer"):
                developer = gr.Checkbox(label="개발자 모드 켜기", value=False)
                with gr.Column(visible=False, elem_id="game-developer-cards") as dev_group:
                    gr.Markdown("**개발자 전용** · 일반 이야기의 선택지·추천은 발견·진행 순서를 유지합니다. 이 보라색 전용 도구에서만 조건을 건너뛰어 시험하며, 사용 기록은 여정에 남습니다.",
                                elem_classes="game-dev-help")
                    with gr.Row(elem_classes="game-dev-row"):
                        with gr.Group(elem_classes="game-dev-card"):
                            gr.Markdown("### 이동 · 인물")
                            with gr.Row():
                                dev_npc = gr.Dropdown(choices=list(profiles), value=next(iter(profiles), None),
                                                     label="만날 인물", min_width=160)
                                dev_meet_button = gr.Button("바로 만나기", min_width=110, scale=0)
                            with gr.Row():
                                dev_room = gr.Dropdown(choices=room_choices, value=room_choices[0][1],
                                                      label="이동 장소", min_width=160)
                                dev_move_button = gr.Button("바로 이동", min_width=110, scale=0)
                            with gr.Accordion("전체 지도 · 인물", open=False):
                                for first in (0, 4):
                                    with gr.Row(elem_classes="game-map-actions"):
                                        for index in range(first, first + 4):
                                            map_buttons.append(gr.Button("장소", visible=False, interactive=False,
                                                                         size="sm", min_width=110))
                                with gr.Row(elem_classes="game-map-actions"):
                                    for index in range(5):
                                        meet_buttons.append(gr.Button("인물 만남", visible=False, interactive=False,
                                                                      size="sm", min_width=100))
                            npc = gr.Dropdown(choices=[("장면 / 게임 마스터", "")], value="", label="현재 장소의 대화 상대")
                            choice = gr.Dropdown(choices=[], label="현재 가능한 전체 행동")
                            action_button = gr.Button("선택한 행동 실행", min_width=145)
                        with gr.Group(elem_classes="game-dev-card"):
                            gr.Markdown("### 판정 · 주사위")
                            dev_success_button = gr.Button("다음 판정 1회 성공", min_width=170)
                            with gr.Row():
                                ability = gr.Dropdown(choices=[(ko, en) for en, ko in ABILITY_KO.items()],
                                                     value="wis", label="시험할 능력", min_width=140)
                                check_button = gr.Button("DC 15 판정 준비", min_width=145, scale=0)
                            with gr.Row():
                                expression = gr.Textbox(value="1d20", label="일반 주사위", min_width=140)
                                dice_button = gr.Button("주사위 굴리기", min_width=145, scale=0)
                    with gr.Row(elem_classes="game-dev-row"):
                        with gr.Group(elem_classes="game-dev-card"):
                            gr.Markdown("### 전투 · 회복")
                            with gr.Row():
                                dev_win_button = gr.Button("전투 즉시 승리", min_width=150)
                                dev_heal_button = gr.Button("HP·자원 회복", min_width=150)
                            dev_encounter = gr.Dropdown(choices=encounters, value="rats", label="시험할 전투")
                            dev_combat_button = gr.Button("시험 전투 시작", min_width=150)
                            with gr.Accordion("이전 기록 복구 · 개발자 전용", open=False):
                                gr.Markdown("사망하거나 선택을 되돌려 시험할 때만 사용합니다. 일반 대화로 부활하지 않습니다.")
                                checkpoint_choice = gr.Dropdown(choices=[], value=None, label="되돌릴 기록", min_width=0)
                                with gr.Row():
                                    checkpoint_refresh = gr.Button("기록 새로고침", size="sm", min_width=120)
                                    checkpoint_restore = gr.Button("선택한 기록 복구", size="sm", min_width=140)
                        with gr.Group(elem_classes="game-dev-card"):
                            gr.Markdown("### 기록 · 검색")
                            mode = gr.Radio(choices=[("빠른 룰 검색", "available"), ("근거 검증 포함", "verified")],
                                            value="available", label="룰 질문 검색")
                            with gr.Row():
                                status = gr.Textbox(label="실행 상태", interactive=False, lines=2, max_lines=4,
                                                    scale=4, min_width=160)
                                refresh_button = gr.Button("새로고침", size="sm", scale=0, min_width=100)
                            with gr.Accordion("근거·모델 입력·판정 기록", open=False):
                                debug = gr.Textbox(label="실제 모델 입력과 처리 결과", lines=10, interactive=False)
                                state_view = gr.Textbox(label="현재 게임 상태(JSON)", lines=8, interactive=False)
                                logs = gr.Textbox(label="최근 엔진·주사위 사건", lines=10, interactive=False)
                            examples = gr.Dataset(components=["textbox"], type="values", samples=[
                                ["안녕하세요"], ["제가 도와드릴 일이 있나요?"], ["인형은 어떤 물건인가요?"],
                                ["**주변을 둘러본다.**"], ["/규칙 공격 굴림이 방어도와 같으면 명중인가요?"],
                            ], label="대화 시험")
            with gr.Accordion("규칙 출처", open=False, elem_id="game-rules-source"):
                gr.Markdown("Includes modified SRD 5.2 material by Wizards of the Coast LLC under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).")

        # 기존 0~50 출력 순서는 유지하고, 읽기 전용 인물 위치 요약만 마지막에 붙입니다.
        view = [chat, stats, inventory, equipment, combat, choice, npc, pending, roll_button,
                status, debug, state_view, quest, logs, developer, dev_group, message, recommendations,
                map_view, *map_buttons, *meet_buttons, cancel_button, action_cards, battle_view,
                action_recommendations, check_method, action_message, journey_notice, character, method_context,
                dev_roll_success_button, dev_roll_failure_button, response_failure, retry_button, retry_turn,
                scene_background_view, discovery_overview_view, discovered_places, discovered_people, discovery_actions_view,
                people_view]
        recommendation_timer = gr.Timer(0.8)
        recommendation_timer.tick(poll_recommendations, [token], [recommendations, action_recommendations, action_cards],
                                  queue=False, api_name=False, show_progress="hidden")
        # 짧은 진행 조회는 큰 대화·턴 기록을 복사하지 않으며 게임 잠금을 기다리지 않습니다.
        diagnostics_timer = gr.Timer(0.6)
        diagnostics_timer.tick(poll_diagnostics, [token], [live_progress_buffer, live_diagnostics_view],
                               queue=False, api_name=False, show_progress="hidden").then(
            fn=None, inputs=[live_progress_buffer], js="data => {window.__trpgControls?.updateStatus(data);return [];}",
            queue=False, api_name=False)
        group = {"concurrency_id": "gameplay", "concurrency_limit": 1}
        if stream_enabled:
            group["stream_every"] = _TypingView.tick
        landing_start.click(fn=None, js="() => {window.__trpgControls?.setStage('characters');return [];}", queue=False, api_name=False)
        new_journey_button.click(fn=None, js="() => {window.__trpgControls?.setStage('characters');return [];}", queue=False, api_name=False)
        selection_back.click(fn=None, js="() => {window.__trpgControls?.returnFromSelection();return [];}", queue=False, api_name=False)
        for identifier, button in character_buttons:
            button.click(fn=None, outputs=[character],
                         js="() => window.__trpgControls?.selectCharacter(" + json.dumps(identifier) + ")",
                         queue=False, api_name=False)
        character.input(preview_character, [character], [preview], api_name="preview_character", queue=False)
        # 이전 성공 신호를 먼저 지워, 서버에 도달하지 못한 요청도 성공으로 보이지 않게 합니다.
        start_button.click(fn=None, outputs=[journey_notice], queue=False, api_name=False,
                           js="() => {window.__trpgControls?.begin('start_game');return '여정을 준비하고 있습니다.';}").then(
            start, [character, token], [token, *view, start_button], api_name="start_game", show_progress="hidden", **group).then(
            fn=None, inputs=[journey_notice],
            js="notice => {window.__trpgControls?.startFinished(notice);return [];}", queue=False, api_name=False)
        help_button.click(lambda: gr.update(visible=True), outputs=[help_overlay], queue=False, api_name=False)
        close_help.click(lambda: gr.update(visible=False), outputs=[help_overlay], queue=False, api_name=False)
        plain_copy_button.click(copy_plain_conversation, [token], [plain_copy_buffer], api_name="copy_plain_conversation",
                                queue=False, show_progress="hidden").then(
            fn=None, inputs=[plain_copy_buffer], outputs=[plain_copy_button], js=COPY_CONVERSATION_JS, queue=False)
        copy_button.click(copy_conversation, [token], [copy_buffer], api_name="copy_conversation",
                          queue=False, show_progress="hidden").then(
            fn=None, inputs=[copy_buffer], outputs=[copy_button], js=COPY_CONVERSATION_JS, queue=False)
        # 실제 버튼은 커서 자리에 삽입만 합니다. 기존 API는 비표시 호환 연결로 남깁니다.
        action_send_button.click(fn=None, js="() => {window.__trpgControls?.insertAction();return [];}",
                                 queue=False, api_name=False)
        mark_action_compat.click(mark_action, [message], [message], queue=False, api_name="mark_action")
        gameplay_events = [
            (send_button.click, send, [message, token], "send_message"),
            (discovered_places.click, choose_discovered_place, [discovered_places, token], "travel_discovered"),
            (discovered_people.click, choose_discovered_person, [discovered_people, token], "meet_discovered"),
            (message.submit, send, [message, token], "submit_message"),
            (action_submit_button.click, send_action, [message, token], "send_action"),
            (action_message.submit, send_action, [action_message, token], "submit_action"),
            (recommendations.click, choose_suggestion, [recommendations, token], "choose_suggestion"),
            (action_recommendations.click, choose_suggestion, [action_recommendations, token], "choose_action_suggestion"),
            (retry_button.click, retry_dialogue, [retry_turn, token], "retry_dialogue"),
            (action_button.click, choose, [choice, token], "choose_action"),
            (action_cards.click, choose_scene_action, [action_cards, token], "choose_scene_action"),
            (npc.input, select_npc, [npc, token], "select_npc"),
            (roll_button.click, roll, [token], "roll_pending"),
            (dev_roll_success_button.click, dev_roll_success, [method_context, token], "developer_roll_success"),
            (dev_roll_failure_button.click, dev_roll_failure, [method_context, token], "developer_roll_failure"),
            (cancel_button.click, cancel, [token], "cancel_pending"),
            (check_method.input, select_check_method, [check_method, method_context, token], "select_check_method"),
            (developer.input, toggle_developer, [developer, token], "toggle_developer"),
            (dev_meet_button.click, dev_meet, [dev_npc, token], "developer_meet"),
            (dev_move_button.click, dev_move, [dev_room, token], "developer_move"),
            (dev_success_button.click, dev_success, [token], "developer_success"),
            (dev_win_button.click, dev_win, [token], "developer_victory"),
            (dev_heal_button.click, dev_heal, [token], "developer_heal"),
            (dev_combat_button.click, dev_combat, [dev_encounter, token], "developer_combat"),
            (checkpoint_restore.click, developer_restore, [checkpoint_choice, token], "developer_restore_checkpoint"),
            (check_button.click, free_check, [ability, token], "free_check"),
            (dice_button.click, utility_dice, [expression, token], "utility_dice"),
            (mode.input, rule_mode, [mode, token], "rule_mode"),
            (examples.click, lambda selected, session: send(selected[0], session), [examples, token], "send_example"),
        ]
        for event, callback, inputs, api_name in gameplay_events:
            event_options = dict(group)
            event_options["show_progress"] = "hidden"
            if api_name == "developer_restore_checkpoint":
                event_options.update(concurrency_id="gameplay-recovery", concurrency_limit=1)
            # 접수와 시계는 즉시 처리하고, 게임 상태 변경만 기존 순서 대기를 따릅니다.
            event(prepare_operation(api_name), inputs, [], queue=False, api_name=False, show_progress="hidden",
                  js=turn_start_js(api_name, roll=api_name in {"roll_pending", "developer_roll_success", "developer_roll_failure"})).then(
                stream_callback(callback, api_name), inputs, view, api_name=api_name, **event_options).then(
                fn=None, js=turn_finish_js(api_name), queue=False, api_name=False)
        save_button.click(save, [token], [download], api_name="save_game", **group)
        load_button.click(stream_callback(restore), [save_file, token], view, api_name="load_game",
                          js=turn_start_js("load_game"), show_progress="hidden", **group).then(
            fn=None, js=turn_finish_js("load_game"), queue=False, api_name=False)
        refresh_button.click(render, [token], view, api_name="refresh_state", **group)
        checkpoint_refresh.click(developer_checkpoints, [token], [checkpoint_choice],
                                 queue=False, api_name="developer_checkpoints", show_progress="hidden")
        for index, button in enumerate(map_buttons):
            button.click(stream_callback(map_callback(map_travel, index)), [token], view,
                         api_name=f"map_travel_{index}", js=turn_start_js(f"map_travel_{index}"),
                         show_progress="hidden", **group).then(fn=None, js=turn_finish_js(f"map_travel_{index}"),
                         queue=False, api_name=False)
        for index, button in enumerate(meet_buttons):
            button.click(stream_callback(map_callback(map_meet, index)), [token], view,
                         api_name=f"map_meet_{index}", js=turn_start_js(f"map_meet_{index}"),
                         show_progress="hidden", **group).then(fn=None, js=turn_finish_js(f"map_meet_{index}"),
                         queue=False, api_name=False)
        # 창 크기와 상단 정보 높이에 맞춰 이야기 공간만 조절합니다.
        demo.load(fn=None, js=GAME_VIEWPORT_JS, queue=False, api_name=False)
        demo.load(fn=None, js=GAME_CONTROLS_JS, queue=False, api_name=False)
    # Gradio 6은 launch(css=...)에서 스타일을 받습니다.
    demo.trpg_layout_css = GAME_LAYOUT_CSS
    return demo
