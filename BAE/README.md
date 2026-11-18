# 잠긴 종 · RAG TRPG 프로토타입

말과 행동을 입력하면 현재 게임 상태와 검색 근거를 바탕으로 NPC 대사와 장면이 이어집니다. 지도에서 장소·인물을 선택하거나 직접 문장을 입력해 탐사·대화·전투를 시험할 수 있습니다.

## 실행

1. **BAE 폴더 전체**를 `TRPG-masterbot/data`의 `main` 최상위에 올립니다. 코드와 초상화 파일을 함께 교체해야 노트북의 버전 검사를 통과합니다.
2. `Output_Code.ipynb`를 새 Colab GPU 런타임에서 위부터 실행합니다. L4를 권장하며 실제 배정은 가용량에 따릅니다.
3. 검색기와 모델 준비가 끝나면 마지막 셀에서 플레이 화면과 임시 공유 주소를 엽니다. 저장소 파일을 받지 못하는 경우 같은 버전의 BAE ZIP을 선택할 수 있습니다.

첫 모델 다운로드·색인 준비와 준비 후 대화 응답에는 각각 별도의 시간이 걸립니다. 임시 공유 주소는 Colab 런타임이 실행되는 동안 사용할 수 있습니다.

## 화면 사용

- **대화:** Enter로 전송하고 Shift+Enter로 줄을 바꿉니다. NPC 대사 옆의 작은 초상화로 화자를 구분합니다.
- **복사:** `대화 전체 복사`로 플레이어·NPC·GM의 공개 대화를 텍스트로 복사합니다.
- **이동·만남:** 지도에서 공개된 장소나 만날 수 있는 인물을 선택합니다. 자유 입력도 같은 행동 확인 경로에 연결됩니다.
- **추천:** 다음에 할 말이나 행동을 고를 수 있습니다. 직접 입력한 문장도 함께 사용할 수 있습니다.
- **상태:** 가방·캐릭터·전투·의뢰에서 실제 엔진 값을 확인합니다. 주문 슬롯·직업별 사용 횟수·전투 차례 자원은 선택한 캐릭터와 규칙에 따라 표시합니다.
- **저장:** 게임 상태를 JSON으로 저장하고 불러옵니다. 개발자 모드의 이동·판정·전투 도구는 검사에 사용합니다.

## 파일 안내

파일·폴더 | 내용
---|---
`Output_Code.ipynb` | 설치·파일 준비·코드 설명·모델 준비·화면 실행을 순서대로 진행하는 메인 노트북.
`src/output_code.py` | 입력·RAG 연결, 대화 기억, 프롬프트 구성, Qwen 생성과 결과 처리.
`src/gameplay.py` | 대화·행동·공개 경로 이동과 추천을 게임 세션에 연결.
`src/game_ui.py` | 대화·초상화·지도·인물·상태와 입력 이벤트를 표시하는 Gradio 화면.
`src/portrait_view.py` | 메시지에 기록된 화자를 초상화와 연결하고 대화 본문을 안전한 화면 형식으로 변환.
`src/dialogue_projection.py`·`src/state_projection.py` | NPC가 알 수 있는 근거, 대화 역할, 엔진의 현재 상태를 출력 자료로 정리.
`src/response_guard.py` | 청자 호칭·가족 지칭을 구분한 말투 지침과 응답 형식·관계·상태 충돌 점검을 연결.
`src/context_cards.json`·`src/transaction_bindings.json` | 인물별 지식·말투와 실제 의뢰·거래 경로의 보조 자료.
`src/assets/portraits/` | 대사와 인물 표시에 사용하는 로컬 초상화 이미지.
`src/game/` | 판정·전투·이동·돈·인벤토리 상태와 게임 자료.
`src/requirements.txt` | 실행 패키지 범위.
`notebooks/Memory_RAG.ipynb` | CPU로 실행하는 별도 기억 RAG 실습.
`previous_versions/` | 초기 데모와 룰·시나리오 RAG 원본 실험 기록.

## 처리 흐름

입력 해석 → 엔진 조건 확인 → 필요한 검색과 기억 보충 → 화자·근거·최근 대화 구성 → Qwen 응답 → 화면 갱신.

Qwen3.5-9B를 NF4 4비트로 실행합니다. 모델 실행은 PyTorch·Transformers, 화면은 Gradio, 전달 데이터는 Python·JSON을 사용합니다. 룰 검색은 BGE-M3·Chroma·BM25·RRF, 시나리오 검색은 E5·Chroma와 장면·공개 조건을 사용합니다.

새 게임은 시나리오에 준비된 광장 배경·공개 소문과 세라의 부탁으로 시작합니다. 첫 만남에는 시나리오의 시작 대사를 사용합니다. 이후 대화는 질문과 최근 문맥을 바탕으로 생성하며, 관련 공개 근거와 NPC의 인지 범위를 함께 참고합니다. 과거 생성 발언은 시나리오의 확정 사실로 취급하지 않습니다. 이동·판정·돈·아이템의 실제 변경은 엔진이 결정하고 생성 모델은 그 결과를 표현합니다.

생성 답변의 한자 혼입을 점검하고 필요한 경우 한국어로 교정합니다. 교정 후에도 출력 기준을 통과하지 못한 본문은 공개 답변으로 사용하지 않습니다.

말투 지침은 대화 상대에 대한 존대와 가족·제삼자 지칭을 구분합니다. 다른 사람을 부르는 표현은 현재 화자와 그 사람의 관계를 기준으로 구성합니다.

## 연결 자료

- [입력 모듈](https://github.com/TRPG-masterbot/data/blob/fd77d6c9608397c3a4cb03ee78df087b8ba81c44/Jang/Input_Code_0.1_ver.ipynb)
- [룰 모듈](https://github.com/TRPG-masterbot/data/blob/fd77d6c9608397c3a4cb03ee78df087b8ba81c44/Min/rule_rag.py) · [규칙 자료](https://github.com/TRPG-masterbot/data/blob/fd77d6c9608397c3a4cb03ee78df087b8ba81c44/Min/rules_new.jsonl)
- [시나리오 모듈](https://github.com/TRPG-masterbot/data/blob/fd77d6c9608397c3a4cb03ee78df087b8ba81c44/Kim/sce_rag.py) · [원본 카드](https://github.com/TRPG-masterbot/data/blob/fd77d6c9608397c3a4cb03ee78df087b8ba81c44/Kim/TheDrownedBell/game/data/cards.json)

외부 파일은 지정한 커밋과 SHA-256을 확인해 준비합니다. 플레이에서 사용하는 카드는 `src/game/data/cards.json`입니다. 규칙 출처와 수정 고지는 `src/game/__init__.py`에 있습니다.

## 검토 범위

마을·우물 갱도의 탐사·전투·귀환과 저장을 시험하는 프로토타입입니다. 자유 대화의 관점·말투·사실 정확성·반복, 긴 대화의 문맥 유지와 응답 시간은 계속 확인해야 합니다. 형식·상태 검사 통과가 모든 대화 내용의 정확성을 뜻하지는 않습니다.

## 실행 링크

- [프로토타입 실행 — Colab](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/Output_Code.ipynb)
- [기억 RAG 실습 — Colab](https://colab.research.google.com/github/TRPG-masterbot/data/blob/main/BAE/notebooks/Memory_RAG.ipynb)
