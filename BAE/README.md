# TRPG 출력 프로토타입

룰·시나리오와 이전 대화를 참고해 NPC 답변을 만드는 프로토타입입니다.
입력 모듈의 검색 결과를 받아 프롬프트·답변 생성·출처 표시를 처리합니다.

## 동작 방식

사용자 입력 → 룰·시나리오 검색 → 기억 검색 → 현재 상태·최근 대화로 프롬프트 구성 → Qwen 답변 → Gradio 표시

최근 12메시지와 오래된 기억 최대 2개를 사용합니다. 기억은 접속자·NPC별로 구분하고, AI 생성 답변은 장기 저장하지 않습니다. 초기화하거나 런타임을 종료하면 기억이 사라집니다.

## 가져오는 자료

공개 GitHub의 확인한 버전을 처음 받아 캐시로 사용합니다.

- [입력 모듈: `Jang/Input_Code_0.1_ver.ipynb`](https://github.com/TRPG-masterbot/data/blob/4ffec99072dcbe29150aadad892b777a8bbe24cd/Jang/Input_Code_0.1_ver.ipynb) — 검색어 확장·룰/시나리오 검색·JSON 전달
- [룰 데이터: `Min/rules_new.jsonl`](https://github.com/TRPG-masterbot/data/blob/4ffec99072dcbe29150aadad892b777a8bbe24cd/Min/rules_new.jsonl) — 분할된 SRD 5.2 영문 규칙
- [시나리오: `Kim/TheDrownedBell/game/data/cards.json`](https://github.com/TRPG-masterbot/data/blob/4ffec99072dcbe29150aadad892b777a8bbe24cd/Kim/TheDrownedBell/game/data/cards.json) — 카드 본문·공개 조건

## 기술 스택

| 기술 | 역할 |
|---|---|
| Qwen3.5-9B · NF4 4비트 | GPU 답변 생성·메모리 부담 완화 |
| PyTorch·Transformers·Accelerate·bitsandbytes | 모델 로딩·토큰화·실행 |
| Sentence Transformers · E5-small(CPU) | 질문·문장을 검색용 벡터로 변환 |
| FAISS · Chroma | 룰 검색 / 시나리오·기억 저장 및 검색 |
| BM25 · RRF | 단어 일치와 의미 검색의 순위 결합 |
| Python · JSON | 모듈 연결과 상태·검색 결과 전달 |
| Gradio | 대화·진행 시간·근거·프롬프트 확인 |

## 실행

6개 파일을 `TRPG-masterbot/data`의 `main` 브랜치 **`BAE` 폴더 안**에 올립니다.
`Output_Code.ipynb`를 Colab GPU에서 위부터 실행하면 마지막 셀에 대화 화면과 임시 공유 주소가 나옵니다. `Memory_RAG.ipynb`는 CPU 기억 검색 실험용입니다.

현재는 대화 실험용이며, 이동·전투 등 게임 상태 변경은 연결 전입니다.

## 실행 링크

업로드 후 사용할 수 있습니다. 새 탭은 **Ctrl/Cmd+클릭 또는 휠 클릭**으로 엽니다.

- [프로토타입 실행 — Colab](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/Output_Code.ipynb)
- [기억 RAG 실험 — Colab](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/Memory_RAG.ipynb)
- [코드 미리보기 — GitHub](https://github.com/TRPG-masterbot/data/blob/main/BAE/Output_Code.ipynb)
