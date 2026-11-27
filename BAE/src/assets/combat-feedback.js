() => {
    // 화면 연출만 담당합니다. 주사위·HP·게임 상태는 이 코드에서 바꾸지 않습니다.
    window.__trpgCombatFeedback?.dispose();
    const shell = document.querySelector('#game-story-shell');
    const status = document.querySelector('#game-statusbar');
    if (!shell || !status) return {dispose() {}};
    const layer = document.createElement('div');
    layer.className = 'combat-fx-layer';
    layer.setAttribute('aria-hidden', 'true');
    shell.append(layer);
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const timers = new Set();
    const portraits = new Map();
    const history = [];
    let session = null;
    let sequence = -1;
    let disposed = false;
    let refreshFrame = 0;

    const later = (callback, delay) => {
        const id = setTimeout(() => {timers.delete(id); if (!disposed) callback();}, delay);
        timers.add(id);
        return id;
    };
    const rememberPortraits = () => {
        // 마지막 일격으로 전투가 끝나도 방금 보던 적의 그림을 잠깐 남깁니다.
        // 아직 만나지 않은 적은 읽지 않고, 실제 전장에 표시된 그림만 보관합니다.
        for (const node of shell.querySelectorAll('.enemy-token[data-enemy-id]')) {
            if (node.dataset.enemyId) {
                const portrait = node.cloneNode(true);
                portrait.classList.remove('combat-enemy-hit', 'combat-enemy-critical');
                for (const effect of portrait.querySelectorAll('.combat-fx-number,.combat-hit-slash')) effect.remove();
                portraits.set(node.dataset.enemyId, portrait);
            }
        }
        while (portraits.size > 24) portraits.delete(portraits.keys().next().value);
    };
    const clear = () => {
        for (const id of timers) clearTimeout(id);
        timers.clear();
        shell.classList.remove('combat-player-hurt', 'combat-player-critical-hurt');
        for (const node of [...layer.children]) node.remove();
        for (const node of shell.querySelectorAll('.combat-enemy-hit,.combat-enemy-critical')) {
            node.classList.remove('combat-enemy-hit', 'combat-enemy-critical');
        }
        // 상대 카드 안쪽에 붙인 숫자·타격선도 지웁니다. 복구 중 타이머가 취소돼도 남지 않습니다.
        for (const node of shell.querySelectorAll('.combat-fx-number,.combat-hit-slash')) node.remove();
    };
    const pulse = (node, className, duration) => {
        node.classList.remove(className);
        void node.offsetWidth;
        node.classList.add(className);
        later(() => node.classList.remove(className), duration);
    };
    const badge = (text, className, parent = layer) => {
        const node = document.createElement('span');
        node.className = 'combat-fx-number ' + className;
        node.textContent = text;
        parent.append(node);
        later(() => node.remove(), reducedMotion.matches ? 650 : 950);
    };
    const targetNode = (identifier, enemies) => {
        const node = [...shell.querySelectorAll('.enemy-token[data-enemy-id]')]
            .find(item => item.dataset.enemyId === identifier && item.getClientRects().length);
        if (node) return node;
        // 전투가 끝난 마지막 타격도 보입니다. 캐시가 없으면 공개된 이름만 표시합니다.
        const old = portraits.get(identifier);
        const enemy = enemies.find(item => item.id === identifier);
        if (!old && !enemy) return null;
        const ghost = old ? old.cloneNode(true) : document.createElement('div');
        ghost.classList.add('combat-last-target');
        ghost.classList.remove('is-out', 'combat-enemy-hit', 'combat-enemy-critical');
        if (!old) {
            ghost.classList.add('enemy-token');
            ghost.textContent = enemy.name;
        }
        ghost.removeAttribute('data-enemy-id');
        layer.append(ghost);
        later(() => ghost.remove(), 1000);
        return ghost;
    };
    const show = (effect, enemies) => {
        const amount = Number(effect.amount);
        if (effect.kind === 'player_damage' && Number.isFinite(amount) && amount > 0) {
            pulse(shell, 'combat-player-hurt', 480);
            badge('−' + amount + ' HP', 'combat-fx-incoming');
        } else if (effect.kind === 'player_hit' && Number.isFinite(amount) && amount > 0) {
            const target = targetNode(effect.target, enemies);
            if (target) {
                pulse(target, effect.critical ? 'combat-enemy-critical' : 'combat-enemy-hit', 480);
                badge((effect.critical ? '치명타 ' : '') + '−' + amount, 'combat-fx-outgoing', target);
                const slash = document.createElement('span');
                slash.className = 'combat-hit-slash';
                target.append(slash);
                later(() => slash.remove(), 420);
            }
        } else if (effect.kind === 'player_miss') {
            badge('빗나감', 'combat-fx-neutral');
        } else if (effect.kind === 'player_no_damage') {
            badge('피해 없음', 'combat-fx-neutral');
        }
    };
    const consume = (packet, baseline = false) => {
        if (!packet || packet.schema !== 1 || typeof packet.session !== 'string'
                || !Number.isSafeInteger(packet.sequence) || packet.sequence < 0) return;
        if (session !== packet.session) {
            clear();
            portraits.clear();
            session = packet.session;
            sequence = packet.sequence;
            rememberPortraits();
            // 새 게임·불러오기·복구 때는 지난 피해를 다시 연출하지 않습니다.
            return;
        }
        if (baseline || packet.sequence <= sequence) return;
        sequence = packet.sequence;
        const effects = Array.isArray(packet.effects) ? packet.effects.slice(0, 24) : [];
        const enemies = Array.isArray(packet.enemies) ? packet.enemies : [];
        rememberPortraits();
        if (!effects.length) {clear(); portraits.clear(); return;}
        history.push({sequence, kinds:effects.map(row => row.kind)});
        if (history.length > 20) history.shift();
        // 여러 적의 공격은 최대 1.2초 안에 표시하고 입력을 잠그지 않습니다.
        effects.forEach((effect, index) => later(() => show(effect, enemies), Math.min(index * 145, 1200)));
    };
    const read = (baseline = false) => {
        const node = status.querySelector('[data-combat-feedback]');
        if (!node) return;
        try {consume(JSON.parse(node.dataset.combatFeedback), baseline);} catch (_) { /* 잘못된 표시 자료는 무시합니다. */ }
    };
    const observer = new MutationObserver(() => {
        // 자기 이펙트 DOM은 관찰하지 않습니다. Gradio 갱신만 한 프레임에 묶습니다.
        if (refreshFrame) return;
        refreshFrame = requestAnimationFrame(() => {
            refreshFrame = 0;
            rememberPortraits();
            read();
        });
    });
    read(true);
    rememberPortraits();
    observer.observe(status, {subtree:true,childList:true,attributes:true});
    const battle = document.querySelector('#game-battle');
    if (battle) observer.observe(battle, {subtree:true,childList:true});
    const dispose = () => {
        if (disposed) return;
        disposed = true;
        observer.disconnect();
        if (refreshFrame) cancelAnimationFrame(refreshFrame);
        clear();
        layer.remove();
        portraits.clear();
        if (window.__trpgCombatFeedback?.dispose === dispose) delete window.__trpgCombatFeedback;
    };
    const api = {dispose, clear, diagnostics:() => ({session,sequence,history:history.slice()})};
    window.__trpgCombatFeedback = api;
    return api;
}
