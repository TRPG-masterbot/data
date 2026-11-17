# 잠긴 종 · RAG TRPG 프로토타입

엔진이 게임 상태를 확정하고 RAG 근거로 Qwen이 대사와 장면을 표현하는 플레이 실험입니다.
제 담당은 출력 파트입니다. 입력과 RAG 근거를 받아 NPC 대사·GM 서술을 생성하고 화면에 연결하는 역할입니다.

파일·폴더 (`BAE/` 기준) | 역할
---|---
`Output_Code.ipynb` | 준비부터 화면 실행까지 순서대로 진행하는 메인 노트북.
`src/output_code.py` | 출력 모델 준비, RAG 연결, 대화 생성과 결과 표시.
`src/gameplay.py` | 사용자 대화·행동을 엔진과 출력 처리에 연결.
`src/game_ui.py` | 플레이 화면과 버튼·입력 콜백.
`src/game/` | 상태·판정·전투를 처리하는 엔진과 게임 데이터.
`src/*_projection.py`·`src/*.json` | 상태·공개 지식 정리와 대화·거래 보조 자료.
`notebooks/Memory_RAG.ipynb` | CPU로 실행하는 별도 기억 RAG 실습.
`previous_versions/` | 초기 데모·룰 RAG·시나리오 RAG 원본 3개 기록.
`.gitignore` | 캐시·임시 파일 제외. 설치 목록은 `src/requirements.txt`.

## 흐름과 기술

사용자 입력 → 입력 해석·엔진 조건 확인 → 필요한 RAG 검색 → Qwen 표현 → 화면 표시.
첫 만남의 대사를 제외한 행동의 이야기는 카드 원문 대신 확정 사건을 바탕으로 생성합니다.
Qwen/Qwen3.5-9B · NF4 4비트 · PyTorch·Transformers·Accelerate·bitsandbytes · Python·JSON·Gradio. LangGraph는 룰 RAG의 선택적인 근거 검증 경로에서 사용합니다.
룰 검색: BGE-M3·Chroma 의미 검색 + BM25 단어 검색 + 가중 RRF 순위 결합.
시나리오 검색: E5-small·Chroma와 장면·공개 조건. 기억 검색도 E5 임베딩을 사용합니다.

## 연결 자료

- [입력 모듈](https://github.com/TRPG-masterbot/data/blob/fd77d6c9608397c3a4cb03ee78df087b8ba81c44/Jang/Input_Code_0.1_ver.ipynb): 검색 연결과 전달 데이터.
- [룰 모듈](https://github.com/TRPG-masterbot/data/blob/fd77d6c9608397c3a4cb03ee78df087b8ba81c44/Min/rule_rag.py) · [규칙 데이터](https://github.com/TRPG-masterbot/data/blob/fd77d6c9608397c3a4cb03ee78df087b8ba81c44/Min/rules_new.jsonl): SRD 5.2 규칙.
- [시나리오 모듈](https://github.com/TRPG-masterbot/data/blob/fd77d6c9608397c3a4cb03ee78df087b8ba81c44/Kim/sce_rag.py) · [원본 카드](https://github.com/TRPG-masterbot/data/blob/fd77d6c9608397c3a4cb03ee78df087b8ba81c44/Kim/TheDrownedBell/game/data/cards.json).

플레이 카드는 `src/game/data/cards.json`입니다. 외부 자료는 커밋 `fd77d6c960`·SHA-256을 확인해 준비합니다.

## 실행

**BAE 폴더째** `TRPG-masterbot/data`의 `main` 최상위에 올립니다. 기존 BAE는 이 새 폴더 구성으로 교체합니다.
`Output_Code.ipynb`를 새 런타임에서 위부터 실행합니다. L4 GPU를 권장하며 실제 배정은 가용량에 따릅니다. 모델·색인 준비 후 마지막 셀에서 화면·임시 공유 주소가 열립니다.
Enter 전송 / Shift+Enter 줄바꿈. 기억 실습은 위 표의 CPU용 별도 노트북입니다.

## 범위와 한계

마을·우물 갱도의 탐사·전투·귀환과 JSON 저장을 시험합니다. 자유 대화의 관점 혼동·근거 없는 내용·반복과 전체 룰 구현은 개선 중입니다.
규칙 출처와 수정 고지는 [`src/game/__init__.py`](src/game/__init__.py)에 있습니다.

## 실행 링크

새 탭은 Ctrl/Cmd+클릭 또는 휠 클릭으로 엽니다.

- [프로토타입 실행 — Colab](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/Output_Code.ipynb)
- [기억 RAG 실습 — Colab](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/notebooks/Memory_RAG.ipynb)
