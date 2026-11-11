# 잠긴 종 · BAE

팀의 1인 TRPG 프로토타입입니다. 게임 엔진이 이동, 주사위 판정, 상태와 단서 공개를 처리하고, RAG가 찾아온 근거를 Qwen에 전달해 답변과 대사를 생성합니다.

| 파일 | 내용 |
| --- | --- |
| [Rule_RAG.ipynb](Rule_RAG.ipynb) | 한국어 룰북 RAG 수업 실습. GPU 확인, Qwen 실행, PDF 추출, 임베딩·Chroma 검색, BM25·RRF 비교, 근거를 사용한 한국어 답변 |
| [Scenario_RAG.ipynb](Scenario_RAG.ipynb) | 공개 카드 검색. 현재 방과 NPC에 따라 근거가 달라지는 예제 및 공개 전후 테스트 |
| [demo](demo) | 게임 마스터 코드와 캐릭터 선택·대사 진행이 있는 웹 플레이 화면 |

## 노트북 실행

- **룰북** · [Colab에서 열기](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/Rule_RAG.ipynb). GPU 런타임을 선택하고 위에서 아래로 실행합니다. Qwen3.5-4B와 임베딩 모델을 내려받습니다. 한국어 [Rules.pdf](https://github.com/TRPG-masterbot/data/blob/93130b452d4351f0b051f1478a7e582e4156c549/Rules.pdf)는 노트북에서 자동으로 준비할 수 있습니다.
- **시나리오** · [Colab에서 열기](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/Scenario_RAG.ipynb). 기본 CPU 런타임에서 실행합니다. 포함된 예제로 검색 범위와 출처를 확인할 수 있습니다.

## 로컬에서 플레이

Python 3.12를 기준으로 안내합니다. 아래 파일을 `demo`에 준비합니다.

1. [llama.cpp 릴리스](https://github.com/ggml-org/llama.cpp/releases)에서 PC에 맞는 실행 파일을 받아 `.runtime/llama/`에 풉니다. Windows 예시는 Vulkan 빌드를 사용합니다.
2. [Qwen2.5-7B GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/main)에서 `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`와 `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf`를 받아 `.runtime/models/`에 함께 둡니다.
3. 팀의 [한국어판 Rules.pdf](https://github.com/TRPG-masterbot/data/blob/93130b452d4351f0b051f1478a7e582e4156c549/Rules.pdf)를 받아 `.runtime/Rules.pdf`에 둡니다.

Windows PowerShell에서 `demo` 폴더로 이동해 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py --open --llama-server ".runtime/llama/llama-server.exe" --model ".runtime/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf" --rulebook ".runtime/Rules.pdf"
```

브라우저가 열리면 **‘누구로 시작할까요?’**에서 브란(전사), 하린(도적), 세오(마법사), 마렌(성직자) 중 한 명을 고릅니다. **다음 문장 보기**를 누르면 대사와 묘사가 한 줄씩 나옵니다. 채팅 옆 **`**`** 버튼을 누르면 행동 표시가 삽입되고 그 사이에 바로 입력할 수 있습니다. 룰 질문은 `/규칙`을 붙입니다.

웹 플레이 주소는 [127.0.0.1:8765](http://127.0.0.1:8765/)입니다. 각자 PC에서 실행하는 로컬 주소입니다. Gradio 화면을 열려면 실행 명령의 `app.py --open` 부분을 `gradio_demo.py`로 바꾼 뒤 [127.0.0.1:7860](http://127.0.0.1:7860/)에서 플레이합니다.

## 코드와 판본

- `demo/app.py`는 입력을 분류하고, `demo/rag.py`는 근거를 검색해 모델에 전달합니다. 로컬 게임의 룰북 검색은 글자 조각을 비교하는 방식입니다. Colab 룰북 노트북은 의미 검색과 키워드 검색을 비교합니다.
- `demo/game/`은 [Kim의 팀 게임 엔진과 데이터](../Kim/TheDrownedBell/game)를 사용합니다. 룰 질문은 한국어 베이직 룰 2018년판으로 답하며, 엔진의 일부 판정에는 SRD 5.2가 반영돼 판본별 결과가 다를 수 있습니다.

This work includes material from the System Reference Document 5.2 by Wizards of the Coast LLC under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).
