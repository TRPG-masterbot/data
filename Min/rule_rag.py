"""rule_rag.py — 룰북 RAG (RuleRAG_basic_v1.5)를 다른 노트북에서 import해서 쓰는 파일

사용법 (Ollama 서버와 모델 bge-m3, qwen2.5:14b는 부르는 노트북에서 먼저 준비):
    import rule_rag
    rule_rag.load()                               # 룰북 받기 + 임베딩 + 검색 준비 (한 번만)
    e = rule_rag.evidence("넘어졌을 때 일어나려면 이동력이 얼마나 들어?")
    e["status"]    # "ok" / "refuse"
    e["text"]      # 영어 근거 본문 (ok일 때) → GM 프롬프트에 넣어서 한국어 설명
    e["terms"]     # 근거에 나온 용어 "영어=한국어" (GM 표기 통일용)

- import 때는 용어표·그래프 정의만 만들어요(1초 미만). 파일 받기·임베딩·검색 준비는 load()에서만 해요.
- 룰북 청크는 GitHub(REPO)에서 받아 현재 폴더에 저장하고, 임베딩도 현재 폴더에 저장해요.
- 동시 처리를 쓰려면 Ollama 서버를 켜기 전에 OLLAMA_NUM_PARALLEL=4 를 설정해야 해요.
  (설정 안 해도 결과는 같고, 속도만 느려져요)
- 노트북 RuleRAG_basic_v1.5의 Ex02~Ex16을 그대로 옮겼어요 (캐시, 질문해 보기, 실험 기록은 뺌).
- 실험1~3은 파일 맨 아래에 주석으로 있어요 (노트북 셀에 붙여서 주석을 풀고 실행).
- 노트북을 고치면 이 파일도 같이 고쳐야 해요.
"""

# ── 설정값 (바꿀 값은 여기만)

# ── 모델 ──
OLLAMA_URL = "http://localhost:11434"     # Ollama 서버 주소
EMBED_MODEL = "bge-m3"                    # 임베딩 모델 (글 → 숫자 벡터)
GEN_MODEL = "qwen2.5:14b"                 # 번역·선택·검증 모델
TRANSLATE_MODEL = GEN_MODEL               # 번역도 같은 14B 사용

# ── 경로 ──
REPO = "https://raw.githubusercontent.com/TRPG-masterbot/data/main/Min"   # 깃허브 데이터 폴더
OUT_PATH = "rules_new.jsonl"              # 받은 룰북 청크 파일 (현재 폴더)
DEV_PATH = "golden_dev2.json"             # 받은 개발 질문 세트 (현재 폴더, 실험용)
EMB_PATH = "rules_new_emb.npy"            # 임베딩 저장 파일 (현재 폴더)
COLLECTION = "rule_rag"                   # Chroma 컬렉션 이름 (다른 노트북의 컬렉션과 겹치지 않게)

# ── 검색 ──
TOP_K = 5                                 # 최종으로 가져올 청크 수
POOL_K = 10                               # 검색 하나당 후보 수 (합치기 전)
SEARCH_WEIGHTS = [1, 1, 1, 0.5, 0.5]      # 검색 비중: 임베딩 3개, 키워드 2개
RRF_C = 60                                # RRF 점수 = 비중 / (RRF_C + 순위)
NUM_CTX = 8192                            # 모든 LLM 호출 공통 num_ctx (같아야 모델을 다시 안 올림)

# ── 근거 고르기 ──
GEN_SEED = 42                             # 같은 질문 → 같은 결과
SELECT_CHARS = 300                        # 근거 고를 때 후보마다 보여 줄 글자 수
VERIFY_CHARS = 800                        # 근거 검증 때 보여 줄 글자 수

# ── 동시 처리 ──
NUM_PARALLEL = 4                          # 동시에 보낼 요청 수 (Ollama의 OLLAMA_NUM_PARALLEL과 같게)


# ── import
import os, re, json, math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict
import numpy as np, requests, chromadb
from tqdm.auto import tqdm
from langgraph.graph import StateGraph, START, END


# ── load()가 채우는 값 (import 직후에는 비어 있음)
chunks = None        # 룰북 청크 목록
collection = None    # Chroma 컬렉션
bm25 = None          # 키워드 검색기
parts = None         # 나뉜 조각 모음: (제목, 쪽) → {조각 번호: 청크}

pool = ThreadPoolExecutor(max_workers=NUM_PARALLEL)   # 요청을 동시에 보내는 일꾼들


# ── 준비 (load)

def download(name):                 # 깃허브에서 파일 하나 받아 현재 폴더에 저장
    res = requests.get(f"{REPO}/{name}", timeout=60)
    res.raise_for_status()                         # 파일 이름이나 주소가 틀리면 여기서 멈춤
    with open(name, "wb") as f:
        f.write(res.content)
    print("받음:", name, len(res.content), "bytes")


def embed(texts, model=EMBED_MODEL):   # 글 여러 개 → 벡터 여러 개
    res = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": texts, "options": {"num_ctx": 8192}},
        timeout=300,
    )
    res.raise_for_status()                       # 서버 오류면 여기서 멈춤
    return res.json()["embeddings"]


def load():                         # 검색 준비 (한 번만 부르면 돼요): 받기 → 청크 → 임베딩 → Chroma → BM25 → 조각 모음
    global chunks, collection, bm25, parts

    # 1. 깃허브에서 받기 (매번 최신으로)
    download("rules_new.jsonl")

    # 2. 청크 읽기 (규칙 청크만)
    chunks = []
    for line in open(OUT_PATH, encoding="utf-8"):
        c = json.loads(line)
        if c["meta"]["type"] == "rule":
            chunks.append(c)
    print("검색 대상 청크:", len(chunks))

    # 3. 임베딩: 저장된 게 있고 청크 수도 같으면 불러오기
    #    (청크 내용이 바뀌었는데 수가 같으면 옛 임베딩을 씀 → 그럴 땐 EMB_PATH 파일을 지우고 다시 load)
    saved = np.load(EMB_PATH) if os.path.exists(EMB_PATH) else None
    if saved is not None and saved.shape[0] == len(chunks):
        emb = saved
        print("저장된 임베딩 사용:", emb.shape)
    else:
        vecs = []
        for i in tqdm(range(0, len(chunks), 32)):
            batch = []
            for c in chunks[i:i + 32]:
                batch.append(c["text"])
            vecs += embed(batch)
        emb = np.array(vecs, dtype="float32")
        np.save(EMB_PATH, emb)
        print("새로 계산:", emb.shape)

    # 4. Chroma 컬렉션 (같은 런타임에서 다시 불러도 옛 벡터가 안 남게 지우고 새로)
    client = chromadb.Client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass                                       # 처음이면 지울 게 없어서 에러 → 무시
    collection = client.create_collection(
        name=COLLECTION,
        configuration={"hnsw": {"space": "cosine", "ef_search": 200}},
    )
    ids = []
    for i in range(len(chunks)):
        ids.append(str(i))                         # Chroma id = 청크 순서 번호 (문자열)
    collection.add(ids=ids, embeddings=emb.tolist())
    print("Chroma 벡터 수:", collection.count())

    # 5. BM25 (제목을 두 번 넣어서 제목 단어에 가중치)
    docs = []
    for c in chunks:
        title = c["meta"]["title"]
        docs.append(tokenize(title + " " + title + " " + c["text"]))
    bm25 = BM25(docs)

    # 6. 나뉜 조각끼리 모아 두기 (expand용)
    parts = {}
    for c in chunks:
        meta = c["meta"]
        if "part" not in meta:
            continue
        key = (meta["title"], meta["page"])
        if key not in parts:
            parts[key] = {}
        parts[key][meta["part"]] = c
    print("준비 완료")


# ======================================================================
# Ex02. 제목 비교

def norm(title):                    # 제목 다듬기 (꼬리표 제거 + 소문자)
    title = re.sub(r"\s*\[[^\]]*\]$", "", title)    # 예: "Prone [Condition]" → "Prone"
    title = re.sub(r"\s*\([^)]*\)$", "", title)     # 예: "Hide (Action)"     → "Hide"
    return title.strip().lower()


def hit(title, want):               # 정답인지 (정확한 비교)
    return norm(title) == norm(want)


def stem(title):                    # 단수/복수 차이 없애기 (단어 끝 s 제거)
    return re.sub(r"s\b", "", norm(title))          # 예: "attacks" → "attack"


def hit2(title, want):              # 정답인지 (느슨한 비교: 단수/복수 차이 무시)
    return stem(title) == stem(want)


# ======================================================================
# Ex03. 채점

def wants(item):                    # 정답을 항상 리스트로 ("Prone" → ["Prone"], 정답 없음 → [])
    want = item["want"]
    if want is None:                               # 거절 문항
        return []
    if isinstance(want, str):
        return [want]
    return want


def rank(found, item):              # 정답 등수 찾기 (없으면 None)
    titles = [c["meta"]["title"] for c in found]
    for i, title in enumerate(titles):
        if any(hit2(title, w) for w in wants(item)):
            return i + 1, titles                    # 등수(1부터), 검색된 제목 목록
    return None, titles


def counts(ranks):                  # 순위 세기 → (1등, 3등 안, 5등 안)
    top1 = sum(1 for r in ranks if r == 1)
    top3 = sum(1 for r in ranks if r is not None and r <= 3)
    top5 = sum(1 for r in ranks if r is not None)   # 검색이 5개만 가져오므로 "찾기만 하면" 5등 안
    return (top1, top3, top5)


def evaluate(search, items, misses=False):   # 질문 세트 전체 채점 (search = 검색 함수)
    ranks = []
    for g in tqdm(items, leave=False):
        found = search(g["q"])                      # 검색에는 질문(q)만
        r, titles = rank(found, g)                  # 정답(want)은 채점에만
        ranks.append(r)
        if misses and r != 1:                       # 1등 못 한 질문 출력
            print(f"[{r}] {g['q']}\n     정답: {g['want']} | 검색: {titles[:3]}")
    return counts(ranks)


# ======================================================================
# Ex05. 용어표
# 한국어 = 룰북 영어 용어. 번역②의 용어 목록과 GM 답변의 표기 통일에 같이 씀
# 기준: 룰북에서 이름이 정해진 "규칙 용어"만 넣음 (번역이 제각각이 되기 쉬운 것)
#       주문·속도·피해·무기처럼 누가 번역해도 같은 일반 단어는 넣지 않음
TERM_V2 = [
    # ── 판정·전투 ──
    ("이점", "Advantage"), ("불리점", "Disadvantage"), ("능력 판정", "Ability Check"),
    ("공격 굴림", "Attack Roll"), ("내성 굴림", "Saving Throw"), ("기회 공격", "Opportunity Attack"),
    ("반응", "Reaction"), ("추가 행동", "Bonus Action"), ("집중", "Concentration"),
    ("이니셔티브", "Initiative"), ("저항", "Resistance"), ("취약", "Vulnerability"), ("면역", "Immunity"),
    ("엄폐", "Cover"), ("치명타", "Critical Hit"), ("숙련", "Proficiency"), ("전문화", "Expertise"),
    ("긴 휴식", "Long Rest"), ("짧은 휴식", "Short Rest"), ("D20 판정", "D20 Test"),
    ("난이도 등급", "Difficulty Class"), ("방어도", "Armor Class"), ("피해 굴림", "Damage Roll"),
    ("죽음 내성 굴림", "Death Saving Throw"), ("체력", "Hit Points"), ("체력 주사위", "Hit Point Dice"),
    ("임시 체력", "Temporary Hit Points"), ("피투성이", "Bloodied"), ("안정", "Stable"),
    ("기습", "Surprise"), ("붙잡기", "Grappling"), ("맨손 타격", "Unarmed Strike"),
    ("주문 공격", "Spell Attack"), ("무기 공격", "Weapon Attack"), ("임시변통 무기", "Improvised Weapons"),
    ("도달 거리", "Reach"), ("영웅적 영감", "Heroic Inspiration"), ("내림", "Round Down"),
    ("동시 효과", "Simultaneous Effects"),
    # ── 능력치 ──
    ("근력", "Strength"), ("민첩", "Dexterity"), ("건강", "Constitution"),
    ("지능", "Intelligence"), ("지혜", "Wisdom"), ("카리스마", "Charisma"),
    # ── 상태이상 15 ──
    ("실명", "Blinded"), ("매혹", "Charmed"), ("청각 상실", "Deafened"), ("탈진", "Exhaustion"),
    ("겁에 질림", "Frightened"), ("붙잡힘", "Grappled"), ("행동 불능", "Incapacitated"), ("투명", "Invisible"),
    ("마비", "Paralyzed"), ("석화", "Petrified"), ("중독", "Poisoned"), ("넘어짐", "Prone"),
    ("포박", "Restrained"), ("기절", "Stunned"), ("무의식", "Unconscious"),
    # ── 행동 12 ──
    ("공격 행동", "Attack action"), ("질주 행동", "Dash action"), ("이탈 행동", "Disengage action"),
    ("회피 행동", "Dodge action"), ("도움 행동", "Help action"), ("은신 행동", "Hide action"),
    ("영향 행동", "Influence action"), ("마법 행동", "Magic action"), ("대기 행동", "Ready action"),
    ("수색 행동", "Search action"), ("연구 행동", "Study action"), ("사용 행동", "Utilize action"),
    # ── 이동·환경 ──
    ("험지", "Difficult Terrain"), ("기어가기", "Crawling"), ("오르기", "Climbing"), ("수영", "Swimming"),
    ("높이뛰기", "High Jump"), ("멀리뛰기", "Long Jump"), ("부양", "Hover"), ("낙하", "Falling"),
    ("등반 속도", "Climb Speed"), ("비행 속도", "Fly Speed"), ("수영 속도", "Swim Speed"),
    ("굴착 이동 속도", "Burrow Speed"), ("점유 공간", "Occupied Space"), ("빈 공간", "Unoccupied Space"),
    ("운반 용량", "Carrying Capacity"), ("탈수", "Dehydration"), ("영양실조", "Malnutrition"),
    ("질식", "Suffocation"), ("불붙음", "Burning"), ("물체 부수기", "Breaking Objects"),
    # ── 시야·감각 ──
    ("밝은 빛", "Bright Light"), ("희미한 빛", "Dim Light"), ("어둠", "Darkness"),
    ("약간 가려짐", "Lightly Obscured"), ("심하게 가려짐", "Heavily Obscured"),
    ("암시야", "Darkvision"), ("맹시", "Blindsight"), ("진동감지", "Tremorsense"), ("진시", "Truesight"),
    ("텔레파시", "Telepathy"), ("수동 감지", "Passive Perception"),
    # ── 사회·태도 ──
    ("성향", "Alignment"), ("태도", "Attitude"), ("우호적", "Friendly"), ("무관심", "Indifferent"), ("적대적", "Hostile"),
    # ── 피해 종류 13 ──
    ("산성", "Acid"), ("타격", "Bludgeoning"), ("냉기", "Cold"), ("화염", "Fire"), ("역장", "Force"),
    ("번개", "Lightning"), ("사령", "Necrotic"), ("관통", "Piercing"), ("독", "Poison"), ("정신", "Psychic"),
    ("광휘", "Radiant"), ("참격", "Slashing"), ("천둥", "Thunder"),
    ("피해 유형", "Damage Types"), ("피해 임계값", "Damage Threshold"),
    # ── 주문 ──
    ("캔트립", "Cantrip"), ("의식", "Ritual"), ("주문 시전 매개체", "Spellcasting Focus"),
    ("시전 시간", "Casting Time"), ("사거리", "Range"), ("구성요소", "Components"), ("지속 시간", "Duration"),
    ("음성", "Verbal"), ("동작", "Somatic"), ("물질", "Material"),
    ("효과 범위", "Area of Effect"), ("원뿔", "Cone"), ("정육면체", "Cube"), ("원기둥", "Cylinder"),
    ("발산", "Emanation"), ("직선", "Line"), ("구", "Sphere"), ("마법 효과", "Magical Effect"),
    ("순간이동", "Teleportation"), ("변신", "Shape-Shifting"), ("저주", "Curses"),
    ("방호", "Abjuration"), ("소환", "Conjuration"), ("예지", "Divination"), ("환혹", "Enchantment"),
    ("방출", "Evocation"), ("환영", "Illusion"), ("강령", "Necromancy"), ("변환", "Transmutation"),
    # ── 클래스 12 ──
    ("바바리안", "Barbarian"), ("바드", "Bard"), ("클레릭", "Cleric"), ("드루이드", "Druid"),
    ("파이터", "Fighter"), ("몽크", "Monk"), ("팔라딘", "Paladin"), ("레인저", "Ranger"),
    ("로그", "Rogue"), ("소서러", "Sorcerer"), ("워록", "Warlock"), ("위저드", "Wizard"),
    # ── 클래스 기능 ──
    ("분노", "Rage"), ("암습", "Sneak Attack"), ("야생 변신", "Wild Shape"), ("추가 공격", "Extra Attack"),
    ("바드의 영감", "Bardic Inspiration"), ("신성 변환", "Channel Divinity"), ("안수", "Lay On Hands"),
    ("메타매직", "Metamagic"), ("섬뜩한 기원", "Eldritch Invocations"), ("계약 마법", "Pact Magic"),
    ("행동 쇄도", "Action Surge"), ("재기의 바람", "Second Wind"), ("무기 마스터리", "Weapon Mastery"),
    # ── 무기 마스터리 속성 ──
    ("쪼개기", "Cleave"), ("스치기", "Graze"), ("연속 공격", "Nick"), ("밀치기", "Push"),
    ("약화", "Sap"), ("둔화", "Slow"), ("넘어뜨리기", "Topple"), ("괴롭히기", "Vex"),
    # ── 재주 ──
    ("경계", "Alert"), ("마법 입문", "Magic Initiate"), ("야만적 공격자", "Savage Attacker"),
    ("다재다능", "Skilled"), ("그래플러", "Grappler"), ("능력치 향상", "Ability Score Improvement"),
    ("궁술", "Archery"), ("대형 무기 전투", "Great Weapon Fighting"), ("쌍수 전투", "Two-Weapon Fighting"),
    # ── 기타 규칙 ──
    ("조율", "Attunement"), ("방어구 숙련", "Armor Training"), ("도전 지수", "Challenge Rating"),
    ("경험치", "Experience Points"), ("생물 유형", "Creature Type"), ("스탯 블록", "Stat Block"),
]

# 같은 한국어가 두 번 들어가면 뒤의 것이 앞을 덮어써서 → 미리 확인
seen = set()
for ko, en in TERM_V2:
    assert ko not in seen, f"한국어 중복: {ko}"
    seen.add(ko)

TERM_MAP = {}
for ko, en in TERM_V2:
    TERM_MAP[ko] = en

# 번역②에 보여 줄 용어: 용어표에서 뽑기 (예전 27개와 같은 순서)
CONDITIONS = ["실명", "매혹", "청각 상실", "탈진", "겁에 질림", "붙잡힘", "행동 불능", "투명",
              "마비", "석화", "중독", "넘어짐", "포박", "기절", "무의식"]
ACTIONS = ["공격 행동", "질주 행동", "이탈 행동", "회피 행동", "도움 행동", "은신 행동",
           "영향 행동", "마법 행동", "대기 행동", "수색 행동", "연구 행동", "사용 행동"]

RULE_TERMS = []
for ko in CONDITIONS + ACTIONS:
    RULE_TERMS.append(TERM_MAP[ko])

def glossary(text):                 # 글에 나오는 용어만 골라 "영어=한국어" 목록 만들기 (GM 프롬프트용)
    out = []
    for ko, en in TERM_MAP.items():
        if " " in en:
            # 두 단어 이상 (예: Attack Roll): 복수형 s 허용, 대소문자 무시
            pattern = r"\b" + re.escape(en) + r"s?\b"
            found = re.search(pattern, text, re.I)
        else:
            # 한 단어 (예: Prone): 대소문자까지 정확히 (일반 단어와 구분)
            pattern = r"\b" + re.escape(en) + r"\b"
            found = re.search(pattern, text)
        if found:
            out.append(f"{en}={ko}")
    if not out:
        return "(없음)"
    return ", ".join(out)


# ======================================================================
# Ex06. 번역 호출

def ask(prompt, model=None, temperature=0.0):      # 번역용 LLM 호출 (항상 같은 결과)
    options = {
        "temperature": temperature,        # 0 = 가장 확실한 답만
        "seed": GEN_SEED,
        "num_ctx": NUM_CTX,
    }
    res = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model or TRANSLATE_MODEL,       # model을 안 주면 기본 번역 모델
              "messages": [{"role": "user", "content": prompt}],
              "stream": False,
              "think": False,                          # 생각 모드 끄기 (qwen3 이상용, 2.5에는 영향 없음)
              "options": options},
        timeout=300,
    )
    res.raise_for_status()
    return res.json()["message"]["content"]


# ======================================================================
# Ex07. 번역 프롬프트

def clean(text):                    # 번역 결과 다듬기 (첫 줄만, "Query:"·따옴표·영어 아닌 글자 제거)
    text = text.strip().split("\n")[0]                       # 첫 줄만
    text = re.sub(r"^\s*query\s*:\s*", "", text, flags=re.I) # 앞의 "Query:" 지우기 (대소문자 무시)
    text = re.sub(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+", " ", text)   # 한자·일본어·한글 → 공백 (예: "Poison中毒效果" → "Poison")
    text = re.sub(r"\s+", " ", text)                          # 공백 여러 개 → 하나
    return text.strip().strip("\"'")                         # 양 끝 따옴표 지우기


# ── 번역① : 일반 번역 (예시 3개를 보여 주고 따라 하게) ──
REWRITE_TEMPLATE = """Convert the user's question about D&D 5.2 rules into a short English search query.
Use the official rule terms that the rulebook would use.
Output only the query, nothing else.

Question: 숙련 보너스는 레벨마다 얼마야?
Query: Proficiency Bonus by character level

Question: 레벨업하려면 경험치가 얼마나 필요해?
Query: Experience Points needed to advance level

Question: 방어도는 어떻게 계산해?
Query: Armor Class calculation

Question: {question}
Query:"""


def translate1(question):           # 번역① 일반 번역
    prompt = REWRITE_TEMPLATE.format(question=question)
    return clean(ask(prompt))


# ── 번역② : 규칙 용어 27개 중에 맞는 게 있으면 그대로 쓰게 ──
# 용어 목록(RULE_TERMS)은 Ex05 용어표에서 만들어요
REWRITE2 = """Translate the user's D&D question into a short English search query for the D&D rules.
If the question is about one of these official rule terms, use that term exactly as written.
Official terms: {terms}
Output only the query, nothing else.

Question: {q}
Query:"""


def translate2(question):           # 번역② 규칙 용어 선택지
    terms = ", ".join(RULE_TERMS)
    prompt = REWRITE2.format(terms=terms, q=question)
    return clean(ask(prompt))


# ======================================================================
# Ex08. 키워드 검색

# 불용어: 너무 흔해서 검색에 도움이 안 되는 단어
STOP = set("a an the of to in on and or is are be can you your it its as at by for with that this if when how what do does".split())


def tokenize(text):                 # 글 → 단어 목록 (소문자, 영어·숫자만, 불용어 제외)
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOP]


class BM25:
    # 점수가 높아지는 경우: 단어가 많이 나옴(tf) + 드문 단어(idf) + 짧은 청크(len)

    def __init__(self, docs, k1=1.5, b=0.75):      # 청크들의 단어 통계 미리 계산
        self.k1 = k1                               # 빈도가 늘 때 점수가 오르는 정도
        self.b = b                                 # 길이 보정을 얼마나 할지
        self.tf = [Counter(d) for d in docs]       # 청크마다 단어별 등장 횟수
        self.len = np.array([len(d) for d in docs], dtype="float32")   # 청크마다 단어 수
        self.avg = self.len.mean()                 # 평균 단어 수

        # df = 그 단어가 나오는 청크 수 (한 청크에서 여러 번 나와도 1번으로)
        df = Counter()
        for d in docs:
            for w in set(d):
                df[w] += 1

        # idf = 드문 단어일수록 큰 값
        total = len(docs)                          # 전체 청크 수
        self.idf = {}
        for w, n in df.items():
            self.idf[w] = math.log(1 + (total - n + 0.5) / (n + 0.5))

    def scores(self, query):                       # 질문 → 청크마다 점수
        s = np.zeros(len(self.tf), dtype="float32")
        for w in set(tokenize(query)):
            if w not in self.idf:                  # 룰북에 없는 단어는 건너뜀
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)                   # 이 청크에서 단어가 나온 횟수
                if f:
                    scale = 1 - self.b + self.b * self.len[i] / self.avg   # 길이 보정 (긴 청크일수록 큼)
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * scale)
                    # 희귀도 × 빈도 효과 / 길이 보정
        return s


def keyword(question, k=10):        # 키워드 검색 → 점수 높은 청크 k개
    order = np.argsort(-bm25.scores(question))            # 점수 큰 순서의 번호 (-를 붙여 내림차순)
    return [chunks[i] for i in order[:k]]


# ======================================================================
# Ex09. 최종 검색

def semantic(texts, k=5):           # 임베딩 검색 (뜻으로 찾기: 글 여러 개 → 비슷한 청크 k개)
    vectors = embed(texts)
    res = collection.query(query_embeddings=vectors, n_results=k)

    # 같은 청크가 여러 번 나오면 가장 높은 점수만 남김
    best = {}
    for ids, dists in zip(res["ids"], res["distances"]):     # 글마다 결과 k개
        for key, dist in zip(ids, dists):
            i = int(key)                        # Chroma id = 청크 순서 번호 (문자열)
            score = 1 - dist                    # 코사인 거리 → 유사도 (클수록 비슷)
            best[i] = max(best.get(i, -1.0), score)

    order = sorted(best, key=best.get, reverse=True)   # 점수 높은 순
    return [chunks[i] for i in order[:k]]

def semantic2(texts, k=5):      # 임베딩 검색을 글마다 따로 (임베딩 호출은 한 번에)
    vectors = embed(texts)       # 글 여러 개를 한 번의 호출로 벡터로
    res = collection.query(query_embeddings=vectors, n_results=k)

    lists = []
    for ids in res["ids"]:                  # 글마다 결과 k개 (가까운 순)
        found = []
        for key in ids:
            found.append(chunks[int(key)])  # Chroma id = 청크 순서 번호
        lists.append(found)
    return lists

def merge(lists, weights, k=TOP_K, c0=RRF_C):     # 여러 검색 결과를 순위 점수(RRF)로 합치기
    # 점수 = 비중 / (c0 + 순위) → 여러 목록에서 위에 있을수록 점수가 높아짐
    score, lookup = {}, {}
    for items, w in zip(lists, weights):
        for r, c in enumerate(items, 1):           # r = 순위 (1부터)
            score[c["id"]] = score.get(c["id"], 0) + w / (c0 + r)
            lookup[c["id"]] = c                    # id → 청크 (마지막에 되찾으려고)

    order = sorted(score, key=score.get, reverse=True)
    return [lookup[i] for i in order[:k]]


def retrieve(question, k=TOP_K):                        # 최종 검색
    job1 = pool.submit(translate1, question)            # 번역1
    job2 = pool.submit(translate2, question)            # 번역2
    job0 = pool.submit(semantic, [question], POOL_K)    # 임베딩 : 원문 질문

    # 끝날 때까지 기다렸다가 결과 받기
    en1 = job1.result()
    en2 = job2.result()
    first = job0.result()

    pair = semantic2([en1, en2], POOL_K)             # 임베딩: 번역1, 2를 한 번에

    lists = [
        first,                                       # 임베딩: 원문 질문
        pair[0],                                     # 임베딩: 번역1
        pair[1],                                     # 임베딩: 번역2
        keyword(en1, POOL_K),                        # 키워드: 번역1
        keyword(en2, POOL_K),                        # 키워드: 번역2
    ]
    return merge(lists, SEARCH_WEIGHTS, k), [question, en1, en2]    # (청크 5개, 검색에 쓴 글 3개)


# ======================================================================
# Ex11. LLM 호출

# 시스템 프롬프트
SYSTEM = ("You are a D&D rules assistant.")

def chat(prompt, model=None, system=SYSTEM, temperature=0.0, seed=GEN_SEED, limit=None):   # LLM 호출 (기본)
    options = {"num_ctx": NUM_CTX, "temperature": temperature}
    if seed is not None:
        options["seed"] = seed                     # seed가 있으면 같은 결과로 고정
    if limit:
        options["num_predict"] = limit             # 출력 길이 제한 (짧은 답용)

    res = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model or GEN_MODEL,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": prompt}],
              "stream": False,
              "think": False,                      # 생각 모드 끄기 (qwen3 이상용)
              "options": options},
        timeout=600,
    )
    res.raise_for_status()
    return res.json()["message"]["content"]


def ask3(prompt, model=None, system=SYSTEM):         # 짧은 답용 호출 (번호, 예/아니오) — 최대 20토큰
    return chat(prompt, model, system, 0.0, GEN_SEED, limit=20)


# ======================================================================
# Ex12. 근거 선택
SELECT_TEMPLATE = """다음은 D&D 규칙 질문과 후보 근거입니다. 질문에 직접 답하는 근거의 번호 하나만 고르세요.
- 질문이 묻는 규칙 자체를 설명하는 근거를 고르세요. 이름만 비슷하거나 예시로 지나가듯 언급되는 근거는 고르지 마세요.
- 숫자 하나만 출력하세요.

### 질문
{question}

### 후보 근거
{cands}

### 번호"""


def select(question, found, model=None):   # 후보 중 질문에 맞는 근거 번호 고르기 (1부터)
    # 후보마다 "[번호] 제목: 본문 앞부분 300자" 한 줄씩
    lines = []
    for n, c in enumerate(found, 1):
        preview = c["text"][:SELECT_CHARS].replace("\n", " ")   # 줄바꿈 → 공백 (한 줄로)
        lines.append(f"[{n}] {c['meta']['title']}: {preview}")
    cands = "\n".join(lines)

    out = ask3(SELECT_TEMPLATE.format(question=question, cands=cands), model)

    # 답에서 첫 숫자 꺼내기. 숫자가 없거나 범위 밖이면 1번
    m = re.search(r"\d+", out)
    n = int(m.group()) if m else 1
    if 1 <= n <= len(found):
        return n
    return 1


# ======================================================================
# Ex13. 원본 전체 읽기
# 검색은 나뉜 조각으로, GM에게 넘길 근거는 원본 전체로

def heading(chunk):                 # 청크 머리줄 (장 > 절 > 제목)
    return chunk["text"].split("\n", 1)[0]


def content(chunk):                 # 청크 본문 (둘째 줄부터)
    return chunk["text"].split("\n", 1)[1]


def expand(chunk):                  # 나뉜 조각이면 원본 전체(모든 조각)를 이어서 돌려줌
    meta = chunk["meta"]
    if "part" not in meta:                      # 나뉘지 않은 청크는 그대로
        return chunk

    group = parts[(meta["title"], meta["page"])]

    texts = []
    for k in range(1, meta["parts"] + 1):       # 1번 조각부터 마지막 조각까지 순서대로
        texts.append(content(group[k]))
    whole = " ".join(texts)

    return {"id": chunk["id"], "text": heading(chunk) + "\n" + whole, "meta": meta}


# ======================================================================
# Ex14. 근거 검증
VERIFY_TEMPLATE = """다음 D&D 규칙 질문과 근거입니다. 이 근거가 질문에 직접 답하는 규칙 내용을 담고 있으면 '예', 이름만 비슷하거나 다른 주제이면 '아니오'라고만 답하세요.
질문은 한국어, 근거는 영어예요. 표현이 달라도 같은 규칙을 다루면 '예'예요.

### 질문
{question}
(영어 검색어: {en})

### 근거
{title}: {text}

### 답 (예/아니오)"""

def verify(question, chunk, english=""):     # 근거가 질문에 맞는지 LLM에게 묻기 (예 → True)
    prompt = VERIFY_TEMPLATE.format(question=question, en=english or "-", title=chunk["meta"]["title"],
                                    text=chunk["text"][:VERIFY_CHARS])
    out = ask3(prompt)
    out = re.sub(r"[\s\*'\"]", "", out)               # 공백·별표·따옴표 지우기 ("**예**" → "예")
    return out.startswith(("예", "네"))

def match(texts, chunk):            # 근거 제목이 번역①② 안에 그대로 있는지 (있으면 확실)
    t = norm(chunk["meta"]["title"])
    for x in texts[1:]:                                # texts = [원문, 번역①, 번역②]
        if re.search(r"\b" + re.escape(t) + r"\b", x.lower()):
            return True
    return False

def verify2(question, chunk, texts):    # 근거 통과 여부 (제목이 맞으면 바로 통과, 아니면 LLM 검증)
    return match(texts, chunk) or verify(question, chunk, texts[1])

def pick(question, found, texts, n):    # 고른 근거부터 검증해서 처음 통과한 번호 (모두 실패면 None)
    # 1단계 : 고른 근거만 먼저 검증 (대부분 여기서 끝남)
    if verify2(question, found[n - 1], texts):
        return n

    # 2단계 : 나머지 후보를 검색 순위대로 모아서 동시에 검증
    others = []
    for i in range(1, len(found) + 1):
        if i != n:                                  # 1단계에서 실패한 번호는 빼고
            others.append(i)                        # 나머지는 검색 순위대로

    jobs = []
    for i in others:
        jobs.append(pool.submit(verify2, question, found[i - 1], texts))

    # 순위가 높은 후보부터 결과 확인 -> 처음 통과한 번호 (예전과 같은 규칙)
    for i, job in zip(others, jobs):
        if job.result():
            return i
    return None


# ======================================================================
# Ex15. 최종 함수 (LangGraph)
# 질문 → 검색 → 근거 선택 → 검증 → 근거 or 거절

class State(TypedDict, total=False):    # 노드끼리 주고받는 값
    question: str                       # 질문
    found: list                         # 검색 후보 5개
    texts: list                         # [원문, 번역1, 번역2]
    n: int                              # 고른 번호 → 검증 후 통과 (모두 실패면 None)
    result: dict                        # 최종 결과

def search(state):          # 노드: 검색
    found, texts = retrieve(state["question"])
    return {"found": found, "texts": texts}

def choose(state):          # 노드: 후보 중 근거 번호 고르기
    n = select(state["question"], state["found"])
    return {"n": n}

def check(state):           # 노드: 고른 근거부터 검증 (실패하면 나머지 동시 검증)
    n = pick(state["question"], state["found"], state["texts"], state["n"])
    return {"n": n}

def route(state):           # 갈림길: 통과 → accept, 모두 실패 → refuse
    if state["n"] is None:
        return "refuse"
    return "accept"

def accept(state):          # 노드: 검증된 근거를 GM에게 넘길 모양으로
    c = expand(state["found"][state["n"] - 1])     # 나뉜 청크면 원본 전체
    result = {
        "status": "ok",
        "title": c["meta"]["title"],
        "page": c["meta"]["page"],
        "path": heading(c),                        # 장 > 절 > 제목
        "text": content(c),                        # 영어 본문 (GM 프롬프트에 넣을 근거)
        "terms": glossary(c["text"]),              # 근거에 나온 용어 "영어=한국어"
        "picked": state["n"],                      # 후보 중 몇 번째였는지
        "queries": state["texts"][1:],             # 번역1, 2
    }
    return {"result": result}

def refuse(state):
    candidates = []
    for x in state["found"][:3]:
        candidates.append((x["meta"]["title"], x["meta"]["page"]))
    result = {
        "status": "refuse",
        "candidates": candidates,
        "queries": state["texts"][1:],
    }
    return {"result": result}

# 그래프 만들기 노드등록 -> 연결 -> 컴파일
graph = StateGraph(State)
graph.add_node("search", search)
graph.add_node("choose", choose)
graph.add_node("check", check)
graph.add_node("accept", accept)
graph.add_node("refuse", refuse)

graph.add_edge(START, "search")
graph.add_edge("search", "choose")
graph.add_edge("choose", "check")
graph.add_conditional_edges("check", route, ["accept", "refuse"])   # 검증 결과에 따라 갈림
graph.add_edge("accept", END)
graph.add_edge("refuse", END)

app = graph.compile()

def evidence(question):         # 질문에 답하는 룰북 근거 찾기 (GM이 부르는 함수)
    state = app.invoke({"question": question})
    return state["result"]


# ======================================================================
# Ex16. 화면 출력

def show(question):                 # 근거를 화면에 출력 (개발 확인용)
    e = evidence(question)

    if e["status"] == "refuse":
        print("🙅 맞는 규칙 근거를 찾지 못했어요. 후보:", e["candidates"])
        print("   번역:", e["queries"])
        return e

    print(f"출처: {e['title']} p.{e['page']} ({e['path']})")
    print("용어:", e["terms"])
    print("근거:", e["text"][:600])
    return e


# ======================================================================
# 실험 (주석) — 노트북 셀에 붙여 넣고, 셀 전체 선택(Ctrl + A) → Ctrl + / 로 주석 해제
# 먼저: import rule_rag, json, time / from collections import Counter / from tqdm.auto import tqdm / rule_rag.load()
# 기준값 (2026-11-13, v1.5, 개발 149문항: 정답 있는 137 / 거절 12)
#   실험1 (93, 119, 123) / 실험2 정답 근거 114 · 다른 근거 13 · 거절 10, 거절 문항 거절(맞음) 10 · 통과(틀림) 2 / 질문당 약 1.7초

# ── 실험0. 질문 세트 받기 (실험1·2 전에 한 번)
# rule_rag.download("golden_dev2.json")
# devset = json.load(open(rule_rag.DEV_PATH, encoding="utf-8"))
# print("devset:", len(devset), "문항")

# ── 실험1. 검색 점수 (정답 있는 문항만: 1등 / 3등 안 / 5등 안)
# answerable = []
# for g in devset:
#     if g["want"] is not None:                      # 정답 없는 거절 문항은 검색 채점에서 뺌
#         answerable.append(g)
# print("채점 문항:", len(answerable))
# print(rule_rag.evaluate(lambda q: rule_rag.retrieve(q)[0], answerable, misses=False))

# ── 실험2. 근거 정확도 · 거절
# # 정답 근거 = 통과한 근거의 제목이 정답과 같음 / 다른 근거 = 통과했지만 정답이 아님 (GM이 틀린 근거로 답함)
# rule_rag.ask3("안녕")                              # 워밍업: 14B를 먼저 올려 둠 (시간 측정에서 제외)
# normal = Counter()        # 정답 있는 문항: 정답 근거 / 다른 근거 / 거절
# refusal = Counter()       # 정답 없는 문항: 거절(맞음) / 통과(틀림)
# problems = []             # 직접 볼 문항
# t0 = time.time()
# for g in tqdm(devset):
#     e = rule_rag.evidence(g["q"])
#     if g["want"] is None:                          # 거절해야 맞는 문항
#         if e["status"] == "refuse":
#             result = "거절(맞음)"
#         else:
#             result = "통과(틀림)"
#             problems.append((result, g["q"], e["title"]))
#         refusal[result] += 1
#     else:
#         if e["status"] == "refuse":
#             result = "거절"
#             problems.append((result, g["q"], g["want"], e["candidates"]))
#         elif any(rule_rag.hit2(e["title"], w) for w in rule_rag.wants(g)):
#             result = "정답 근거"
#         else:
#             result = "다른 근거"
#             problems.append((result, g["q"], g["want"], e["title"]))
#         normal[result] += 1
# print(f"[정답 있는 문항] 정답 근거 {normal['정답 근거']} / 다른 근거 {normal['다른 근거']} / 거절 {normal['거절']}")
# print(f"[거절 문항]     거절(맞음) {refusal['거절(맞음)']} / 통과(틀림) {refusal['통과(틀림)']}")
# print("질문당", round((time.time() - t0) / len(devset), 1), "초")
# for p in problems:
#     print(p)

# ── 실험3. 단계별 시간 (질문 하나)
# question = "넘어졌을 때 일어나려면 이동력이 얼마나 들어?"
# rule_rag.ask3("안녕")                              # 워밍업
# t = time.time()
# found, texts = rule_rag.retrieve(question)
# print("검색 (번역 2 동시 + 검색):", round(time.time() - t, 1), "초")
# t = time.time()
# n = rule_rag.select(question, found)
# print("근거 선택:", round(time.time() - t, 1), "초")
# t = time.time()
# n = rule_rag.pick(question, found, texts, n)
# print("검증:", round(time.time() - t, 1), "초")
