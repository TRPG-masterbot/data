# 출력 실행 보조 함수: 노트북과 같은 전역 공간에서 원본 본문을 실행합니다.

def _bae_connect_and_open(namespace):
    import ast
    import hashlib
    import inspect
    import importlib
    import json
    from pathlib import Path
    import sys
    import textwrap
    import types

    changed = [name for name, original in namespace["BAE_TEAM_BINDINGS"].items()
               if namespace.get(name) is not original]
    if changed:
        raise RuntimeError("공통·RAG·입력 이름과 출력 코드가 충돌합니다: " + ", ".join(changed))
    required = ("search_rag", "build_llm_handoff", "can_search")
    if any(not callable(namespace.get(name)) for name in required):
        raise RuntimeError("위쪽 팀 입력·RAG 함수 준비를 먼저 완료하세요.")
    runtime = Path(namespace["BAE_RUNTIME_PATH"])
    game_cards = json.loads((runtime / "game/data/cards.json").read_text(encoding="utf-8"))
    if namespace["card_by_id"] != {card["id"]: card for card in game_cards["cards"]}:
        raise RuntimeError("팀 시나리오 카드와 게임 실행 자료의 버전이 다릅니다.")

    # 화면 보조 모듈도 방금 셀에서 정의한 검사 함수를 그대로 사용합니다.
    guard = types.ModuleType("response_guard")
    for name in ['_HAN',
 '_PROTOCOL',
 '_contains_response_json',
 'allowed_speaker_map',
 'contains_han',
 '_language_issue',
 'first_han_index',
 'inspect_response',
 'json',
 'parse_response_blocks',
 're',
 'response_blocks_schema',
 'response_requirements',
 'unescape',
 'unique_json_keys',
 'validate_response_block']:
        setattr(guard, name, namespace[name])
    sys.modules["response_guard"] = guard
    namespace["bind_team_backend"](namespace["search_rag"], namespace["build_llm_handoff"],
        scenario_cards=namespace["card_by_id"], can_search=namespace["can_search"], settings=namespace)

    # Colab에서도 실행 중인 함수·설정·보조 파일의 버전을 확인합니다.
    # 셀 소스를 찾지 못해도 실제 Python 코드로 식별하며, 경로와 실행 셀 번호는 제외합니다.
    import marshal

    def code_shape(code):
        constants = tuple(code_shape(item) if isinstance(item, types.CodeType) else item
                          for item in code.co_consts)
        return code.replace(co_filename="", co_firstlineno=0, co_consts=constants)

    def definition_shape(value):
        if inspect.isfunction(value):
            return {"code": hashlib.sha256(marshal.dumps(code_shape(value.__code__))).hexdigest()}
        if inspect.isclass(value):
            methods = {}
            for name, member in value.__dict__.items():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if inspect.isfunction(member):
                    methods[name] = definition_shape(member)
            return {"class": value.__qualname__, "bases": [base.__qualname__ for base in value.__bases__],
                    "methods": methods}
        raise TypeError("실행 코드 버전을 확인할 수 없습니다.")

    code = {name: definition_shape(namespace[name]) for name in ['OutputFormatError', 'OutputStreamCancelled', '_ResponseStream', '_attach_output_error', '_blocks_text', '_contains_response_json', '_emit_output_stream', '_engine_data', '_generate_npc_conversation', '_generate_public', '_generation_slot', '_inspect_npc_conversation', '_json_text', '_npc_conversation_schema', '_npc_view', '_plain_npc_conversation', '_prepared_confirmed_interaction', '_prepared_opening', '_prompt_characters', '_raise_if_cancelled', '_recent_messages', '_result', '_speaker_context', '_utterance_task', '_version_metadata', '_with_speaker_whitelist', 'allowed_speaker_map', 'bind_team_backend', 'build_context', 'build_messages', 'build_npc_conversation_messages', 'canonical_npc_name', 'close_memory_session', 'contains_han', 'create_memory_session', 'diagnostics_metadata', 'first_han_index', 'generate_action_result', 'generate_answer', 'generate_followup_question', 'generate_from_handoff', 'generation_for_display', 'get_handoff', 'inspect_response', 'load_dialogue_history', 'load_output_model', 'normalize_history', 'output_speaker', 'output_stream', 'parse_response_blocks', 'read_handoff', 'response_blocks_schema', 'response_requirements', 'run_generation', 'save_dialogue_history', 'unique_json_keys', 'validate_engine_result', 'validate_response_block']}
    for name in ("_bae_prepare_direct_helpers", "_bae_capture_team_bindings", "_bae_connect_and_open"):
        if callable(namespace.get(name)):
            code[name] = definition_shape(namespace[name])
    settings = {name: namespace[name] for name in ("MODEL_NAME", "OLLAMA_BASE_URL", "OLLAMA_NUM_CTX",
        "MAX_NEW_TOKENS", "MAX_INPUT_CHARACTERS", "HISTORY_MESSAGES", "DO_SAMPLE", "TEMPERATURE",
        "TOP_P", "TOP_K", "OUTPUT_SYSTEM", "NPC_CONVERSATION_SYSTEM", "NPC_CONVERSATION_FIRST_PASS",
        "NPC_CONVERSATION_REFERENCE_PASS", "OPENING_LINES", "OUTPUT_POLICY")}
    helper_hashes = {str(path.relative_to(runtime)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(runtime.rglob("*")) if path.is_file() and path.suffix in {".py", ".json", ".png", ".webp"}}
    fingerprint = hashlib.sha256(json.dumps({"code": code, "settings": settings, "helpers": helper_hashes,
        "python": list(sys.version_info[:2])}, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    namespace["_SOURCE_SHA256"] = fingerprint
    api = {name: namespace[name] for name in ['ContextVar',
 'DO_SAMPLE',
 'HISTORY_MESSAGES',
 'MAX_INPUT_CHARACTERS',
 'MAX_NEW_TOKENS',
 'MODEL_NAME',
 'OLLAMA_BASE_URL',
 'OLLAMA_KEEP_ALIVE',
 'OLLAMA_NUM_CTX',
 'OUTPUT_SYSTEM',
 'OUTPUT_VERSION',
 'OutputFormatError',
 'OutputStreamCancelled',
 'SUGGESTION_COUNT',
 'TEMPERATURE',
 'TOP_K',
 'TOP_P',
 '_ResponseStream',
 '_SOURCE_SHA256',
 '_blocks_text',
 '_dialogue_lock',
 '_dialogue_sessions',
 '_emit_output_stream',
 '_engine_data',
 '_generate_public',
 '_generation_slot',
 '_json_text',
 '_output_stream_sink',
 '_prompt_characters',
 '_raise_if_cancelled',
 '_recent_messages',
 '_result',
 '_speaker_context',
 '_utterance_task',
 '_version_metadata',
 '_with_speaker_whitelist',
 'bind_team_backend',
 'build_context',
 'build_messages',
 'canonical_npc_name',
 'close_memory_session',
 'contextmanager',
 'create_memory_session',
 'deepcopy',
 'diagnostics_metadata',
 'generate_action_result',
 'generate_answer',
 'generate_followup_question',
 'generate_from_handoff',
 'generation_for_display',
 'generation_lock',
 'get_handoff',
 'hashlib',
 'input_namespace',
 'json',
 'load_dialogue_history',
 'load_output_model',
 'normalize_history',
 'output_model',
 'output_speaker',
 'output_stream',
 're',
 'read_handoff',
 'run_generation',
 'save_dialogue_history',
 'secrets',
 'threading',
 'time',
 'validate_engine_result'] if name in namespace}
    api["_bae_loaded_runtime_fingerprint"] = fingerprint
    namespace["BAE_OUTPUT_API"] = api
    if namespace.get("_BAE_TEST_MODE", False):
        return None
    backend = namespace["load_output_model"]()
    # Colab의 요청 GPU와 실제 배정 GPU는 다를 수 있습니다. 종류를 바꾸지 않고
    # 실제 장치와 설치된 모델 정보를 읽어 실행 화면과 복사 진단에 표시합니다.
    import subprocess
    info = {"model": backend.model, "model_digest": backend.digest,
            "quantization": backend.details.get("quantization_level"), "gpu": None}
    try:
        detected = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5)
        if detected.returncode == 0:
            info["gpu"] = detected.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass  # GPU 조회 도구가 없는 환경에서도 기존 연결은 사용할 수 있습니다.
    namespace["_OUTPUT_RUNTIME_INFO"].update(info)
    print("출력 연결 준비 | 모델:", info["model"], "| 양자화:", info["quantization"],
          "| 실제 GPU:", info["gpu"] or "확인할 수 없음", flush=True)
    previous = namespace.get("bae_demo")
    if previous is not None:
        previous.close()
    # 같은 런타임에서 셀을 다시 실행해도 교체한 화면 보조 코드가 반영되게 합니다.
    # 이 출력 폴더에서 불러온 모듈·생성된 캐시만 정리하고 팀 모듈은 유지합니다.
    for name in ("game_ui", "gameplay", "play_view", "diagnostic_export"):
        loaded = sys.modules.get(name)
        file = getattr(loaded, "__file__", None)
        if file and Path(file).resolve().is_relative_to(runtime.resolve()):
            cache = getattr(loaded, "__cached__", None)
            if cache:
                cached = Path(cache).resolve()
                if cached.is_relative_to(runtime.resolve()) and cached.suffix == ".pyc":
                    cached.unlink(missing_ok=True)
            del sys.modules[name]
    importlib.invalidate_caches()
    import game_ui
    demo = game_ui.build_game_demo(api)
    options = {"share": "google.colab" in sys.modules, "inline": True, "height": 1100,
        "width": "100%", "show_error": True, "css": getattr(demo, "trpg_layout_css", None)}
    options.update(namespace.get("_BAE_LAUNCH_OPTIONS", {}))
    demo.queue(default_concurrency_limit=1).launch(**options)
    return demo
