# 잠긴 종 · RAG TRPG

팀의 입력 모듈과 룰·시나리오 RAG 자료를 받아 Qwen3.5-9B가 NPC 대사와 장면 설명을 생성하는 프로토타입입니다.

## 실행

1. `BAE` 폴더 전체를 팀 저장소의 `main`에 업로드합니다.
2. [Colab 열기](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/Output_Code.ipynb)에서 GPU 런타임을 선택하고 위부터 실행합니다.
3. Ollama·모델·검색기 준비가 끝나면 마지막 셀에서 게임 화면을 엽니다.

사용 모델은 `qwen3.5:9b`이며 Q4_K_M 양자화를 사용합니다. 첫 다운로드는 약 6.6GB입니다. 공유 주소는 Colab 런타임이 실행되는 동안 사용할 수 있습니다.

로컬 PC에서는 Ollama를 설치·실행하고 `ollama pull qwen3.5:9b`를 먼저 실행합니다.

## 처리 흐름

입력·RAG 자료·게임 상태(JSON) → 대화 문맥과 프롬프트 구성 → Qwen 생성 → 응답 점검 → Gradio 표시

이동·돈·아이템·주사위·전투 결과는 게임 엔진이 결정하고, 모델은 결과에 맞는 대사와 서술을 생성합니다.

## 주요 파일

| 파일·폴더 | 역할 |
|---|---|
| `Output_Code.ipynb` | 설치부터 게임 화면 실행까지 진행하는 메인 노트북 |
| `src/output_code.py` | 입력·RAG·대화 기억 연결, 프롬프트 구성과 답변 생성 |
| `src/ollama_backend.py` | Ollama 연결·스트리밍·생성 중단 처리 |
| `src/gameplay.py` | 대화·행동·이동·추천을 게임 진행에 연결 |
| `src/game_ui.py` | Gradio 화면과 사용자 입력 연결 |
| `src/play_view.py`·`src/portrait_view.py` | 선택지·게임 상태·초상화 표시 |
| `src/dialogue_projection.py`·`src/state_projection.py` | NPC의 지식과 실제 게임 상태 정리 |
| `src/response_guard.py` | 말투 지침과 응답 형식·일부 관계·상태 충돌 점검 |
| `src/game/` | 게임 엔진과 시나리오·규칙 자료 |
| `src/context_cards.json`·`src/transaction_bindings.json` | 인물 지식·말투와 의뢰·거래 연결 자료 |
| `src/assets/portraits/` | 캐릭터 초상화 |
| `src/requirements.txt` | 실행 패키지 목록 |
| `notebooks/`·`previous_versions/` | 기억 RAG 실습과 초기 실험 자료 |

## 팀 연결

대화는 `generate_from_handoff()`, 실행된 행동의 표현은 `generate_action_result()`로 요청하고 반환값의 `text`를 표시합니다.

현재 NPC·공개 근거·실제 게임 상태를 함께 전달합니다. 노트북과 `src/`는 같은 버전으로 교체해야 합니다.

## 구현 범위

- 장면 1~2의 8개 방에서 대화·준비·우물 탐사·판정·전투·귀환·저장을 구현했습니다.
- 일반 모드는 발견한 인물과 경로만 표시하며, 전체 바로가기와 강제 판정은 개발자 모드에서 사용합니다.
- 이후 가족 구출과 최종 결말은 미구현입니다.
- 자유 대화의 반복·사실 오류·말투는 개선 중입니다.

[기억 RAG 실습](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/notebooks/Memory_RAG.ipynb)
