# 「잠긴 종」 MVP — 장면 1

## 실행
```
python -m game.cli                      # 시작할 때 캐릭터 1명 선택 (1인 1캐릭터), LLM 없이 플레이
python -m game.cli --char harin --seed 7   # 캐릭터(bran/harin/seo/maren)·주사위 시드 지정
python -m game.cli --narrator ollama --model qwen2.5:7b   # Ollama 모델이 결과를 묘사 (모델은 고정하지 않음)
python -m unittest discover -s tests -t .                 # 테스트
python tools/play_script.py             # 정해진 경로로 장면 1 끝까지 자동 실행
python tools/build_cards.py             # 카드 데이터 다시 만들기 (원천: 상위 폴더 생성 스크립트 + 시나리오 PDF, pypdf 필요)
```
- Python 3.10 이상. 게임 실행에는 외부 라이브러리가 필요 없습니다(표준 라이브러리만 사용).

## 구조
| 파일 | 역할 |
|---|---|
| `engine.py` | 선택지, 판정, 효과(상태·flag·카드), 함정, 휴식, 0 HP 처리 — 결과는 모두 여기서 결정 |
| `narrator.py` | Qwen 서술. 표시된 카드·판정 결과만 받음. `OllamaNarrator`(Ollama), `CallableNarrator`(Colab transformers 함수 연결), `TemplateNarrator`(LLM 없음) |
| `cli.py` | 터미널 화면 |
| `data/cards.json` | 카드 166개 (본문 원문, 플레이어용 문장, GM 전용 문장 분리) |
| `data/scenario.jsonl` | 기존 Chroma 코드와 같은 메타데이터 형식(`scene_id`, `reveal_after`, `vis_*`, `req_flag`, `room`) |
| `data/scene1.json` | 장면 1의 방·선택지·판정·효과. 선택지마다 `source`(원본/보완) 표시 |
| `data/characters.json` | 사전 제작 4명 (SRD 근거 쪽 번호는 파일 머리말) |

## Colab에서 transformers로 띄운 Qwen 연결
```python
from game.engine import Engine
from game.narrator import CallableNarrator
from game import cli

def qwen_chat(messages):           # messages: [{"role": ..., "content": ...}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=256)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

cli.play(Engine(char_id="harin"), CallableNarrator(qwen_chat))
```

설계 근거: `analysis/05_시스템설계/05-2a_구현전_보완결정.md`
