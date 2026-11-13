# TRPG 출력 프로토타입

카드 기반 NPC 선택, 룰·시나리오 검색과 대화 기억으로 답변을 만드는 출력 실험입니다.
선택한 이름은 화자 역할에만 사용하며, 전달된 근거와 불확실성을 간결한 공통 대화 원칙으로 답하도록 지시합니다.
[`previous_versions/`](previous_versions/README.md)에는 최초 데모·룰 RAG·시나리오 RAG와 실험 기록 설명을 보관합니다. 현재 버전은 BAE 바로 아래에서 업데이트합니다.

## 처리 흐름

NPC 선택·질문 → 검색·기억 연결 → 상태·최근 대화로 프롬프트 구성 → Qwen 답변 한 번 생성 → Gradio 표시
최근 12메시지와 오래된 기억 최대 2개를 사용합니다. 기억은 접속자·NPC별 임시 DB이며 초기화·런타임 종료 시 사라집니다.
룰·시나리오 결과를 유지하고 출처 본문은 한 번씩 전달합니다. 플레이어 발언만 기억에 저장하며 시나리오의 행동 묘사나 AI 답변은 엔진의 확정 결과와 구분합니다.

## 연결 자료

공개 GitHub 커밋 `5b137eb7c2`의 코드를 SHA-256으로 확인하고 런타임 캐시에 준비합니다.

- [`Jang/Input_Code_0.1_ver.ipynb`](https://github.com/TRPG-masterbot/data/blob/5b137eb7c209e1950da79490f8588bd6d8ccde9f/Jang/Input_Code_0.1_ver.ipynb) — 검색어 확장·schema 1.0 전달 JSON
- [`Min/rule_rag.py`](https://github.com/TRPG-masterbot/data/blob/5b137eb7c209e1950da79490f8588bd6d8ccde9f/Min/rule_rag.py) — 룰 검색 함수 연결
- [`Min/rules_new.jsonl`](https://github.com/TRPG-masterbot/data/blob/5b137eb7c209e1950da79490f8588bd6d8ccde9f/Min/rules_new.jsonl) — SRD 5.2 영문 규칙
- [`Kim/sce_rag.py`](https://github.com/TRPG-masterbot/data/blob/5b137eb7c209e1950da79490f8588bd6d8ccde9f/Kim/sce_rag.py) — 현재 상태의 공개 조건을 적용한 시나리오 검색
- [`Kim/TheDrownedBell/game/data/cards.json`](https://github.com/TRPG-masterbot/data/blob/5b137eb7c209e1950da79490f8588bd6d8ccde9f/Kim/TheDrownedBell/game/data/cards.json) — 카드 본문·공개 조건

## 기술 스택

| 기술 | 역할 |
|---|---|
| Qwen3.5-9B · NF4 4비트 | GPU 답변 생성·선택한 일반 근거 재검토 |
| PyTorch·Transformers·Accelerate·bitsandbytes | 모델 로딩·토큰화·생성 |
| Sentence Transformers · BGE-M3 / E5-small(CPU) | 룰 / 시나리오·기억 임베딩 |
| Chroma · BM25 · RRF | 의미 검색·단어 검색·순위 결합 |
| Python · JSON · Gradio | NPC 선택·개발 확인용 질문 버튼·행동 입력·근거 확인 |

## 실행

압축을 풀고 **BAE 폴더째** 저장소의 `main` 브랜치 최상위에 올립니다. 이후 현재 파일을 같은 경로에서 교체합니다.
`Output_Code.ipynb`를 위부터 실행합니다. Colab은 L4 GPU를 선호하도록 저장했으며 마지막 셀에서 대화 화면·임시 공유 주소를 엽니다.
L4 배정은 가용량·계정에 따라 달라지며 유료 이용이 필요할 수 있습니다. 첫 다운로드·색인은 몇 분이 걸릴 수 있고 GPU 여유를 활용한 뒤 CPU로 검색합니다.
`Memory_RAG.ipynb`는 GPU 없이 CPU에서 실행하는 별도 기억 실습입니다.
`GROUNDING_REVIEW=False`가 기본값입니다. 같은 9B 모델의 일반 근거 재검토는 비교 실험에서 켤 수 있으며 추가 시간과 결과를 확인합니다. 오류 제거를 보장하지 않습니다.
기본 룰 검색은 입력 확장어를 사용해 추가 번역 LLM 호출을 생략합니다. 원본 전체 번역·검증·답변 경로와 성능 점수를 동일하게 보지 않습니다.

## 실행 링크

- [프로토타입 실행 — Colab](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/Output_Code.ipynb)
- [기억 RAG 실험 — Colab](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/Memory_RAG.ipynb)
