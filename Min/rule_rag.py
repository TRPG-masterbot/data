"""rule_rag.py — 룰북 RAG (RuleRAG_basic_v1.2)를 다른 노트북에서 import해서 쓰는 파일

사용법 (Ollama 서버와 모델 bge-m3, qwen2.5:14b는 부르는 노트북에서 먼저 준비):
    import rule_rag
    rule_rag.load()                                   # 룰북 받기 + 임베딩 + 검색 준비 (한 번만)
    r = rule_rag.answer("넘어졌을 때 일어나려면 이동력이 얼마나 들어?")
    r["status"]   # "ok" / "refuse" / "hidden"
    r["explain"]  # 설명 (ok일 때)

- import만 하면 아무것도 실행되지 않아요. 데이터 읽기와 임베딩 계산은 load()에서만 해요.
- 룰북 청크는 GitHub(RULES_URL)에서 받아 현재 폴더에 저장하고, 임베딩도 현재 폴더에 저장해요.
- 노트북 RuleRAG_basic_v1.2의 Ex05~Ex10-c를 그대로 옮겼어요 (질문 해 보기, 평가, 캐시, 실험 기록은 뺌).
- 노트북을 고치면 이 파일도 같이 고쳐야 해요.
"""

# ── Ex00. 설정값 (바꿀 값은 여기만)

# ── 모델 ──
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"                    # 임베딩 모델 (글 → 숫자 벡터)
GEN_MODEL = "qwen2.5:14b"                 # 답변 생성 모델
TRANSLATE_MODEL = GEN_MODEL

# ── 경로 ──
# GitHub에 올린 룰북 청크 파일의 raw 주소 (저장소 이름이 다르면 여기만 고치세요)
RULES_URL = "https://raw.githubusercontent.com/TRPG-masterbot/TRPG-masterbot/main/Min/rules_new.jsonl"
OUT_PATH = "rules_new.jsonl"              # 받은 룰북 청크 파일 (현재 폴더)
EMB_PATH = "rules_new_emb.npy"            # 임베딩 저장 파일 (현재 폴더)
COLLECTION = "rule_rag"                   # Chroma 컬렉션 이름 (다른 노트북의 컬렉션과 겹치지 않게)

# ── 검색 ──
TOP_K = 5                                 # 최종으로 가져올 청크 수
POOL_K = 10                               # 검색 하나당 후보 수 (합치기 전)
SEARCH_WEIGHTS = [1, 1, 1, 0.5, 0.5]      # 검색 비중: 임베딩 3개, 키워드 2개
RRF_C = 60                                # RRF 점수 = 비중 / (RRF_C + 순위)
NUM_CTX = 4096

# ── 답변 ──
GEN_SEED = 42
SELECT_CHARS = 300                        # 근거 고를 때 후보마다 보여 줄 글자 수
VERIFY_CHARS = 800                        # 근거 검증 때 보여 줄 글자 수
CTX_CHARS = 2000                          # 설명 쓸 때 보여 줄 글자 수
EXPLAIN_TRIES = 3                         # 설명이 깨지면 다시 쓰는 최대 횟수
CHAT_ATTEMPTS = 3                         # 챗봇 모드에서 검사에 걸리면 다시 하는 최대 횟수


# ── import
import os, re, json, math
from collections import Counter
import numpy as np, requests, chromadb
from tqdm.auto import tqdm


# ── load()가 채우는 값 (import 직후에는 비어 있음)
chunks = None        # 룰북 청크 목록
emb = None           # 청크 임베딩 (청크 수 × 1024)
collection = None    # Chroma 컬렉션
bm25 = None          # 키워드 검색기


# ── Ex02. 임베딩 함수

def embed_batch(texts, model=EMBED_MODEL):      # 글 여러 개를 한 번에 벡터로 바꾸기
    res = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": texts, "options": {"num_ctx": 8192}},
        timeout=300,
    )
    res.raise_for_status()                       # 서버 오류면 여기서 멈춤
    return res.json()["embeddings"]




# ── Ex05-a. 번역 호출

def ask_translate(prompt, model=None, temperature=0.0):    # 번역용 LLM 호출 (항상 같은 결과)
    options = {
        "temperature": temperature,        # 0 = 가장 확실한 답만
        "seed": 42,                        # 같은 입력 → 같은 출력
        "num_ctx": NUM_CTX,                # 4096 고정
    }
    res = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model or TRANSLATE_MODEL,       # model을 안 주면 기본 번역 모델
              "messages": [{"role": "user", "content": prompt}],
              "stream": False,
              "options": options},
        timeout=300,
    )
    res.raise_for_status()
    return res.json()["message"]["content"]


# Ex05-b. 번역 프롬프트 두 가지

def clean_query(s):                 # 번역 결과 다듬기 (첫 줄만, "Query:"·따옴표 제거)
    s = s.strip().split("\n")[0]                       # 첫 줄만
    s = re.sub(r"^\s*query\s*:\s*", "", s, flags=re.I) # 앞의 "Query:" 지우기 (대소문자 무시)
    return s.strip().strip("\"'")                      # 양 끝 따옴표 지우기


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


def translate1(q):                  # 일반 번역
    prompt = REWRITE_TEMPLATE.format(question=q)
    return clean_query(ask_translate(prompt))


# ── 번역② : 규칙 용어 27개 중에 맞는 게 있으면 그대로 쓰게 ──
# 상태이상 15개 + 행동 12개 (룰북 전부)
RULE_TERMS = ["Blinded", "Charmed", "Deafened", "Exhaustion", "Frightened", "Grappled", "Incapacitated",
              "Invisible", "Paralyzed", "Petrified", "Poisoned", "Prone", "Restrained", "Stunned", "Unconscious",
              "Attack action", "Dash action", "Disengage action", "Dodge action", "Help action", "Hide action",
              "Influence action", "Magic action", "Ready action", "Search action", "Study action", "Utilize action"]

REWRITE2 = """Translate the user's D&D question into a short English search query for the D&D rules.
If the question is about one of these official rule terms, use that term exactly as written.
Official terms: {terms}
Output only the query, nothing else.

Question: {q}
Query:"""


def translate2(q):                  # 규칙 용어 선택지
    terms = ", ".join(RULE_TERMS)
    prompt = REWRITE2.format(terms=terms, q=q)
    return clean_query(ask_translate(prompt))


# Ex06. BM25 직접 만들기 (글자 그대로의 단어로 찾는 키워드 검색)

# 불용어: 너무 흔해서 검색에 도움이 안 되는 단어
STOP = set("a an the of to in on and or is are be can you your it its as at by for with that this if when how what do does".split())


def tokenize(s):                    # 글 → 단어 목록 (소문자, 영어·숫자만, 불용어 제외)
    words = re.findall(r"[a-z0-9]+", s.lower())
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
        N = len(docs)
        self.idf = {}
        for w, n in df.items():
            self.idf[w] = math.log(1 + (N - n + 0.5) / (n + 0.5))

    def scores(self, query):                       # 질문 → 청크마다 점수
        s = np.zeros(len(self.tf), dtype="float32")
        for w in set(tokenize(query)):
            if w not in self.idf:                  # 룰북에 없는 단어는 건너뜀
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)                   # 이 청크에서 단어가 나온 횟수
                if f:
                    length_norm = 1 - self.b + self.b * self.len[i] / self.avg   # 긴 청크일수록 큼
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * length_norm)
                    # 희귀도 × 빈도 효과 / 길이 보정
        return s


def bm25_search(q, k=10):           # 키워드 검색 → 점수 높은 청크 k개
    order = np.argsort(-bm25.scores(q))            # 점수 큰 순서의 번호 (-를 붙여 내림차순)
    return [chunks[i] for i in order[:k]]


# Ex07. 검색: 질문 하나 → 청크 5개
def dense_search(texts, k=5):       # 임베딩 검색 (글 여러 개 → 비슷한 청크 k개)
    query_vecs = embed_batch(texts)
    res = collection.query(query_embeddings=query_vecs, n_results=k)

    # 같은 청크가 여러 번 나오면 가장 높은 점수만 남김
    best = {}
    for id_row, dist_row in zip(res["ids"], res["distances"]):    # 글마다 결과 k개
        for chunk_id, dist in zip(id_row, dist_row):
            i = int(chunk_id)
            score = 1 - dist                    # 코사인 거리 → 유사도 (클수록 비슷)
            best[i] = max(best.get(i, -1.0), score)

    order = sorted(best, key=best.get, reverse=True)   # 점수 높은 순
    return [chunks[i] for i in order[:k]]

def rrf_merge(lists, weights, k=TOP_K, c0=RRF_C):  # 여러 검색 결과를 순위 점수로 합치기
    # 점수 = 비중 / (c0 + 순위) → 여러 목록에서 위에 있을수록 점수가 높아짐
    score, byid = {}, {}
    for lst, w in zip(lists, weights):
        for r, c in enumerate(lst, 1):             # r = 순위 (1부터)
            score[c["id"]] = score.get(c["id"], 0) + w / (c0 + r)
            byid[c["id"]] = c

    order = sorted(score, key=score.get, reverse=True)
    return [byid[i] for i in order[:k]]

def retrieve(q, k=TOP_K):           # 최종 검색 (임베딩 3갈래 + 키워드 2갈래)
    en1 = translate1(q)
    en2 = translate2(q)

    lists = [
        dense_search([q], POOL_K),                 # 임베딩: 원문 질문
        dense_search([en1], POOL_K),
        dense_search([en2], POOL_K),
        bm25_search(en1, POOL_K),
        bm25_search(en2, POOL_K),
    ]
    return rrf_merge(lists, SEARCH_WEIGHTS, k), [q, en1, en2]   # (청크 5개, 검색에 쓴 글 3개)


# Ex09-a. 용어표 (한국어 = 룰북 영어 용어) — 답변의 한국어 표기 통일용
TERM_V1 = {
    # 판정·전투
    "이점": "Advantage", "불리점": "Disadvantage", "능력 판정": "Ability Check",
    "공격 굴림": "Attack Roll", "내성 굴림": "Saving Throw", "기회 공격": "Opportunity Attack",
    "반응": "Reaction", "추가 행동": "Bonus Action", "집중": "Concentration",
    "이니셔티브": "Initiative", "저항": "Resistance", "취약": "Vulnerability",
    "엄폐": "Cover", "치명타": "Critical Hit", "숙련": "Proficiency", "전문화": "Expertise",
    "긴 휴식": "Long Rest", "짧은 휴식": "Short Rest",
    # 능력치
    "근력": "Strength", "민첩": "Dexterity", "건강": "Constitution",
    "지능": "Intelligence", "지혜": "Wisdom", "카리스마": "Charisma",
    # 피해 종류 13
    "산성": "Acid", "타격": "Bludgeoning", "냉기": "Cold", "화염": "Fire", "역장": "Force", "번개": "Lightning",
    "사령": "Necrotic", "관통": "Piercing", "독": "Poison", "정신": "Psychic", "광휘": "Radiant", "참격": "Slashing", "천둥": "Thunder",
    # 클래스 12
    "바바리안": "Barbarian", "바드": "Bard", "클레릭": "Cleric", "드루이드": "Druid", "파이터": "Fighter", "몽크": "Monk",
    "팔라딘": "Paladin", "레인저": "Ranger", "로그": "Rogue", "소서러": "Sorcerer", "워록": "Warlock", "위저드": "Wizard",
    # 주문 학파 8 + 주문 정보 칸
    "방호": "Abjuration", "소환": "Conjuration", "예지": "Divination", "환혹": "Enchantment", "방출": "Evocation",
    "환영": "Illusion", "강령": "Necromancy", "변환": "Transmutation",
    "시전 시간": "Casting Time", "사거리": "Range", "구성요소": "Components", "지속 시간": "Duration",
    "음성": "Verbal", "동작": "Somatic", "물질": "Material",
    # 룰북 용어집 -> 의미있나?
    "능력치와 수정치": "Ability Score and Modifier", "행동": "Action", "모험": "Adventure", "성향": "Alignment", "아군": "Ally",
    "효과 범위": "Area of Effect", "방어도": "Armor Class", "방어구 숙련": "Armor Training", "태도": "Attitude", "조율": "Attunement",
    "맹시": "Blindsight", "피투성이": "Bloodied", "밝은 빛": "Bright Light", "굴착 이동 속도": "Burrow Speed", "캠페인": "Campaign",
    "캔트립": "Cantrip", "운반 용량": "Carrying Capacity", "도전 지수": "Challenge Rating", "등반 속도": "Climb Speed",
    "상태": "Condition", "생물": "Creature", "생물 유형": "Creature Type", "원뿔": "Cone", "정육면체": "Cube", "원기둥": "Cylinder",
    "저주": "Curses", "D20 판정": "D20 Test", "피해": "Damage", "피해 굴림": "Damage Roll", "피해 임계값": "Damage Threshold",
    "피해 유형": "Damage Types", "어둠": "Darkness", "암시야": "Darkvision", "질주": "Dash", "사망": "Dead",
    "죽음 내성 굴림": "Death Saving Throw", "탈수": "Dehydration", "험지": "Difficult Terrain", "난이도 등급": "Difficulty Class",
    "희미한 빛": "Dim Light", "이탈": "Disengage", "회피": "Dodge", "발산": "Emanation", "조우": "Encounter", "적": "Enemy",
    "경험치": "Experience Points", "낙하": "Falling", "비행 속도": "Fly Speed", "비행": "Flying", "우호적": "Friendly",
    "위험 요소": "Hazard", "치유": "Healing", "심하게 가려짐": "Heavily Obscured", "도움": "Help", "영웅적 영감": "Heroic Inspiration",
    "은신": "Hide", "높이뛰기": "High Jump", "체력 주사위": "Hit Point Dice", "체력": "Hit Points", "적대적": "Hostile",
    "부양": "Hover", "면역": "Immunity", "임시변통 무기": "Improvised Weapons", "무관심": "Indifferent", "영향": "Influence",
    "도약": "Jumping", "약간 가려짐": "Lightly Obscured", "직선": "Line", "멀리뛰기": "Long Jump", "마법": "Magic",
    "마법 효과": "Magical Effect", "영양실조": "Malnutrition", "몬스터": "Monster", "비플레이어 캐릭터": "Nonplayer Character",
    "물체": "Object", "점유 공간": "Occupied Space", "수동 감지": "Passive Perception", "하루당": "Per Day",
    "플레이어 캐릭터": "Player Character", "소유": "Possession", "도달 거리": "Reach", "대기": "Ready", "의식": "Ritual",
    "내림": "Round Down", "내성": "Save", "수색": "Search", "변신": "Shape-Shifting", "동시 효과": "Simultaneous Effects",
    "크기": "Size", "기술": "Skill", "속도": "Speed", "주문": "Spell", "주문 공격": "Spell Attack", "주문 시전 매개체": "Spellcasting Focus",
    "구": "Sphere", "안정": "Stable", "스탯 블록": "Stat Block", "연구": "Study", "질식": "Suffocation", "수영 속도": "Swim Speed",
    "수영": "Swimming", "대상": "Target", "텔레파시": "Telepathy", "순간이동": "Teleportation", "임시 체력": "Temporary Hit Points",
    "진동감지": "Tremorsense", "진시": "Truesight", "맨손 타격": "Unarmed Strike", "빈 공간": "Unoccupied Space", "무기": "Weapon",
    "무기 공격": "Weapon Attack",
}

TERM_MAP = dict(TERM_V1)

# 프롬프트에 넣을 글: "Advantage=이점, Disadvantage=불리점, ..."
TERMS = ", ".join(f"{en}={ko}" for ko, en in TERM_MAP.items())

# 거꾸로 찾기용: 영어(소문자) → 한국어
TERM_REV = {en.lower(): ko for ko, en in TERM_MAP.items()}


def terms_for(text):                # 근거 글에 나오는 용어만 골라 "영어=한국어" 목록 만들기
    out = []
    for ko, en in TERM_MAP.items():
        if " " in en:
            # 두 단어 이상 (예: Attack Roll): 복수형 s 허용, 대소문자 무시
            pattern = rf"\b{re.escape(en)}s?\b"     # 앞뒤에 \b를 붙이면 단어 전체가 일치할 때만 찾음
            found = re.search(pattern, text, re.I)
        else:
            # 한 단어 (예: Dash): 대소문자까지 정확히 (dash 같은 일반 단어와 구분)
            pattern = rf"\b{re.escape(en)}\b"
            found = re.search(pattern, text, 0)
        if found:
            out.append(f"{en}={ko}")
    return ", ".join(out) or "(없음)"      # 하나도 없으면 "(없음)"

# 전체 제목에 대한 용어표 사용 검토 -> 의미가 있나? --> 코드 내 삽입이 아닌 파일로 저장


# Ex09-b. 호출 함수 + 깨짐/혼합/분수 검사

# 시스템 프롬프트: 한국어로만 답하고, 한자·일본어 등은 쓰지 말 것
SYSTEM = ("You are a D&D rules assistant. Write every answer in Korean (Hangul). "
          "Never use Chinese characters or Chinese words. "
          "Also never use Japanese, Vietnamese, Russian, Thai, or any other language or script. "
          "Exceptions: English D&D terms in parentheses only when the instructions ask for them, "
          "and the English source quote when asked to copy it.")


def _chat(prompt, model=None, system=SYSTEM, temperature=0.0, seed=GEN_SEED, max_tokens=None):   # LLM 호출 (기본)
    options = {"num_ctx": NUM_CTX, "temperature": temperature}
    if seed is not None:
        options["seed"] = seed                     # seed가 있으면 같은 결과로 고정
    if max_tokens:
        options["num_predict"] = max_tokens        # 출력 길이 제한 (짧은 답용)

    res = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model or GEN_MODEL,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": prompt}],
              "stream": False,
              "options": options},
        timeout=600,
    )
    res.raise_for_status()
    return res.json()["message"]["content"]


def ask_explain(prompt, model=None, temperature=0.1, seed=GEN_SEED):   # 설명 쓰기용 호출
    return _chat(prompt, model, SYSTEM, temperature, seed)


def ask_short(prompt, model=None, system=SYSTEM):    # 짧은 답용 호출 (번호, 예/아니오) — 최대 20토큰
    return _chat(prompt, model, system, 0.0, GEN_SEED, max_tokens=20)


# ── 출력 검사용 패턴 ──
# 이 글자들 중 하나라도 있으면 깨짐
#   一-鿿 한자 / ぀-ヿ 일본어 / Ạ-ỹ 및 ăâêôơưđ 베트남어
#   ̀-ͯ 글자 위에 붙는 기호 / � 깨진 글자(�)
BROKEN = re.compile(r"[一-鿿぀-ヿẠ-ỹ̀-ͯăâêôơưđĐ�]")


def is_broken(text):                # 깨진 글자가 있는지
    return bool(BROKEN.search(text))


LEAK = re.compile(r"(^|\n)\s*(user|assistant|system)\s*(\n|:)", re.I)
# 줄 처음에 "user" 같은 말이 새어 나왔는지

MIXED = re.compile(r"[가-힣][A-Za-z]")         # 한글 바로 뒤에 영어 철자가 붙음 (예: "이점Advantage", "스티일")
FRACTION = re.compile(r"\d+분의\s*\d+")        # 분수 뒤집는 오류 잡기


def bad_output(text):               # 다시 써야 하는 출력인지 (깨짐 / 혼합 철자 / 분수)
    if is_broken(text):
        return True
    if MIXED.search(text):
        return True
    if FRACTION.search(text):
        return True
    return False


# Ex09-c. 검사 도구

def expl_part(text):                # 답변에서 "설명:" 뒤의 글만 꺼내기 (없으면 전체)
    m = re.search(r"설명\s*[:：]\s*(.*)", text, flags=re.S)   # re.S = 줄바꿈까지 포함해서 끝까지
    if m:
        return m.group(1).strip()
    return text


NUM = re.compile(r"\d+(?:d\d+)?", re.I)            # 숫자 또는 주사위 표기 (예: 30, 2d6)


def numbers(text):                  # 글에 나온 숫자·주사위 표기 목록 (소문자)
    return [n.lower() for n in NUM.findall(text)]


# 영어 숫자 단어 → 숫자 (근거에 "two"라고 적혀 있어도 설명의 "2"를 인정하려고)
W2N = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
       "eight": "8", "nine": "9", "ten": "10", "half": "1 2", "quarter": "1 4", "quarters": "4"}
W2N.update({"twice": "2", "double": "2", "doubled": "2"})


def norm_txt(s):                    # 비교용으로 글 다듬기 (따옴표 제거, 공백 하나로, 소문자)
    s = re.sub(r"[\"'“”‘’]", "", s)                # 따옴표 종류 전부 지우기
    s = re.sub(r"\s+", " ", s)                     # 공백·줄바꿈 여러 개 → 공백 하나
    return s.strip().lower()


# Ex10-a. 근거 선택: 후보 5개 중 번호 하나 고르기
SELECT_TEMPLATE = """다음은 D&D 규칙 질문과 후보 근거입니다. 질문에 직접 답하는 근거의 번호 하나만 고르세요.
- 질문이 묻는 규칙 자체를 설명하는 근거를 고르세요. 이름만 비슷하거나 예시로 지나가듯 언급되는 근거는 고르지 마세요.
- 숫자 하나만 출력하세요.

### 질문
{question}

### 후보 근거
{cands}

### 번호"""


def select_chunk(q, found, model=None):    # 후보 중 질문에 맞는 근거 번호 고르기 (1부터)
    # 후보마다 "[번호] 제목: 본문 앞부분 300자" 한 줄씩
    lines = []
    for n, c in enumerate(found, 1):
        preview = c["text"][:SELECT_CHARS].replace("\n", " ")   # 줄바꿈 → 공백 (한 줄로)
        lines.append(f"[{n}] {c['meta']['title']}: {preview}")
    cands = "\n".join(lines)

    out = ask_short(SELECT_TEMPLATE.format(question=q, cands=cands), model)

    # 답에서 첫 숫자 꺼내기. 숫자가 없거나 범위 밖이면 1번
    m = re.search(r"\d+", out)
    n = int(m.group()) if m else 1
    if 1 <= n <= len(found):
        return n
    return 1


# Ex10-b. 설명하기 : 챗봇이 답을 쓰고, 코드가 그 답을 확인

# 설명 프롬프트
EXPLAIN_TEMPLATE = """당신은 D&D 규칙 안내자입니다. 아래 [근거] 하나만 사용해 질문에 답하세요.
1. 근거에 적힌 내용만 사용하세요. 숫자, 조건, 이점/불리점을 빼거나 바꾸지 마세요.
2. 용어는 이 표의 한국어를 쓰고 영어 원문을 괄호로 함께 적어. 같은 용어는 항상 같은 한국어로 써: {terms}
   '이점'은 Advantage에만 쓰고, bonus는 '보너스', roll은 '굴림', check는 '판정'으로 써.
3. 답변은 "원문:", "설명:" 두 줄로 쓰세요. 원문은 질문에 직접 답하는 문장 1~3개를 영어 그대로, 끝까지, 한 줄로 복사하세요.
   설명은 근거에 있는 내용만 한국어로 쓰세요. 질문이 묻는 내용은 빠짐없이 쓰고(목록이면 항목 전부, 이점/불리점은 누구의 굴림인지 구분), 묻지 않은 다른 규칙이나 예시는 쓰지 마세요.
   인사, 서론, 되묻기는 쓰지 마세요. "원문:"과 "설명:"은 각각 한 번만 쓰세요.

### 근거
{context}

### 질문
{question}

### 답변"""

# 2번 규칙을 바꿔 끼우기용: 영어 괄호 표기 있음(EN) / 없음(KO, 챗봇 화면용)
RULE_EN = "용어는 이 표의 한국어를 쓰고 영어 원문을 괄호로 함께 적어. 같은 용어는 항상 같은 한국어로 써: {terms}"
RULE_KO = "용어는 이 표의 한국어만 쓰세요. 영어는 괄호로도 적지 마세요. 같은 용어는 항상 같은 한국어로 써: {terms}"
assert RULE_EN in EXPLAIN_TEMPLATE, "2번 규칙 문구가 달라요"


def explain_chunk(question, k=TOP_K, tries=EXPLAIN_TRIES, temp0=0.1, seed0=40, pre=None, en_terms=True):   # 근거 하나로 원문+설명 쓰기
    # pre = (검색 결과, 검색어, 고른 번호)를 미리 받으면 검색·선택을 다시 안 해요 (answer에서 사용)
    # seed0 / temp0 = 챗봇 재시도 때 바꿔 쓰는 값 (기본값이면 이전과 같아요)

    # 1. 검색 + 번호 고르기
    if pre:
        found, texts, n = pre[0], pre[1], pre[2]
    else:
        found, texts = retrieve(question, k)
        n = select_chunk(question, found)
    c = found[n - 1]                                   # 고른 근거 (번호는 1부터라 -1)

    # 2. 프롬프트 만들기
    ctx = f"({c['meta']['title']} p.{c['meta']['page']}) {c['text'][:CTX_CHARS]}"
    if en_terms:
        tmpl = EXPLAIN_TEMPLATE
    else:
        tmpl = EXPLAIN_TEMPLATE.replace(RULE_EN, RULE_KO)
    prompt = tmpl.format(terms=terms_for(c["text"]), context=ctx, question=question)

    # 3. 설명 쓰기: 깨지면 온도·seed를 바꿔 최대 tries번
    for t in range(1, tries + 1):
        temperature = temp0 + 0.3 * (t - 1)           # 1번째 0.1 → 2번째 0.4 → 3번째 0.7
        out = ask_explain(prompt, temperature=temperature, seed=seed0 + t)
        if not bad_output(out):
            break

    # 4. "원문:" 꺼내서 근거에 진짜 있는 문장인지 확인
    m = re.search(r"원문\s*[:：]\s*(.+)", out)
    orig = m.group(1).strip() if m else ""
    verified = bool(orig) and norm_txt(orig) in norm_txt(c["text"])

    first_line, _, rest = c["text"].partition("\n")    # 청크 첫 줄 = 경로, 나머지 = 본문
    body = rest if "\n" in c["text"] else first_line

    return {
        "title": c["meta"]["title"],
        "page": c["meta"]["page"],
        "text": c["text"],
        "orig_ok": verified,
        "orig": orig if verified else body[:300],      # 검증 실패면 본문 앞 300자로 대신
        "path": first_line,
        "explain": expl_part(out),
        "picked": n,                                   # 고른 번호
        "queries": texts[1:],                          # 번역
        "tries": t,                                    # 설명을 몇 번 만에 썼는지
    }


def check_explain(r):               # 설명을 코드로 검사 (숫자·형식·철자 등)
    ok = r["orig_ok"]

    # 원문이 문장 끝(. ! ? ")으로 끝나는지
    complete = ok and r["orig"].rstrip().endswith((".", "!", "?", '"'))

    # 근거에 있는 숫자 모음: 숫자 그대로 + 영어 숫자 단어(two → 2)
    src = set(numbers(r["text"]))
    for w in re.findall(r"[a-z]+", r["text"].lower()):
        if w in W2N:
            for d in W2N[w].split():                   # "half" → "1 2" → {"1", "2"}
                src.add(d)

    # 근거에 없는 숫자 찾기: 설명에 나온 숫자를 하나씩 보고, 근거에 없으면 모아 두기
    bad_nums = []
    for num in numbers(r["explain"]):      # 설명에 나온 숫자들 (예: "30", "2d6")
        if num not in src:                 # src = 근거에 나온 숫자 모음
            bad_nums.append(num)           # 근거에 없음 → 지어낸 숫자일 수 있음

    # 원문에 나온 용어인데 설명에 한국어가 없는 것 (원문 검증 실패면 검사 안 함)
    term_miss = []
    if ok:
        for ko, en in TERM_MAP.items():
            if re.search(rf"\b{re.escape(en)}\b", r["orig"]) and ko not in r["explain"]:
                term_miss.append(ko)

    return {
        "원문_검증": ok,
        "원문_완결": complete,
        "근거에_없는_숫자": bad_nums,
        "용어_누락": term_miss,
        "혼합_철자": bool(MIXED.search(r["explain"])),
        "분수": FRACTION.findall(r["explain"]),
        "형식_누출": bool(re.search(r"(원문|설명)\s*[:：]", r["explain"])),   # 설명 안에 "원문:"이 또 나옴
    }


# Ex10-c. 근거 검증 · 충실성 검사 · 최종 함수 `answer` / `show_answer`

# ── 1. 근거 검증: 고른 근거가 질문에 진짜 답이 되는지 LLM에게 예/아니오로 묻기 ──
VERIFY_TEMPLATE = """다음 D&D 규칙 질문과 근거입니다. 이 근거가 질문에 직접 답하는 규칙 내용을 담고 있으면 '예', 이름만 비슷하거나 다른 주제이면 '아니오'라고만 답하세요.
질문은 한국어, 근거는 영어예요. 표현이 달라도 같은 규칙을 다루면 '예'예요.

### 질문
{question}
(영어 검색어: {en})

### 근거
{title}: {text}

### 답 (예/아니오)"""


def verify_chunk(q, c, en=""):      # 근거가 질문에 맞는지 LLM에게 묻기 (예 → True)
    prompt = VERIFY_TEMPLATE.format(question=q, en=en or "-", title=c["meta"]["title"],
                                    text=c["text"][:VERIFY_CHARS])
    out = ask_short(prompt)
    out = re.sub(r"[\s\*'\"]", "", out)               # 공백·별표·따옴표 지우기 ("**예**" → "예")
    return out.startswith(("예", "네"))


# ── 2. 충실성 검사: 설명에 근거에 없는 내용이 있는지 LLM에게 한 번 더 묻기 ──
FAITH_TEMPLATE = """다음은 D&D 규칙 근거(영어)와 그것을 한국어로 설명한 글입니다.
설명에 근거에 없는 내용(지어낸 사실, 근거와 다른 숫자·조건)이 있으면 '있음', 없으면 '없음'이라고만 답하세요.

### 근거
{src}

### 설명
{exp}

### 답 (있음/없음)"""


def check_faithful(r):              # 설명이 근거대로 썼는지 검사 (True = 통과)
    src = r["text"][:CTX_CHARS]     # 근거 원문 (앞 2000자)
    exp = r["explain"]              # 챗봇이 쓴 설명

    prompt = FAITH_TEMPLATE.format(src=src, exp=exp)        # format() : 문자열 안의 {이름}을 값으로 바꿔 넣는 기능
    reply = ask_short(prompt).strip()      # LLM 답: "없음" 또는 "있음"

    if reply.startswith("없음"):    # 지어낸 내용 없음 → 통과
        return True
    return False                    # "있음"이거나 엉뚱한 답 → 걸림


# ── 3. 경고 모으기 ──
def collect_warnings(r, chk, check_faith=True, show_orig=True):   # 검사 결과 → 경고 목록 (빈 목록 = 통과)
    w = []
    if not chk["원문_검증"]:
        w.append("원문 검증 실패")
    elif show_orig and not chk["원문_완결"]:          # 원문을 보여 줄 때만 끊김을 따짐
        w.append("원문이 문장 중간에서 끊김")
    if chk["근거에_없는_숫자"]:
        w.append(f"근거에 없는 숫자 {chk['근거에_없는_숫자']}")
    if chk["분수"]:
        w.append(f"분수 표현 {chk['분수']}")
    if chk["형식_누출"]:
        w.append("형식 누출")
    if chk["혼합_철자"]:
        w.append("혼합 철자")
    if is_broken(r["explain"]):
        w.append("깨진 글자")

    # 코드 검사를 다 통과했을 때만 LLM 충실성 검사 (느려서 마지막에)
    if not w and check_faith and not check_faithful(r):
        w.append("근거에 없는 내용")
    return w


# ── 4. 검증 생략 조건 ──
def norm(title):                    # 제목 다듬기 (꼬리표 제거 + 소문자) — 노트북 Ex04-a
    title = re.sub(r"\s*\[[^\]]*\]$", "", title)    # 예: "Prone [Condition]" → "Prone"
    title = re.sub(r"\s*\([^)]*\)$", "", title)     # 예: "Hide (Action)"     → "Hide"
    return title.strip().lower()


def sure_match(texts, c):           # 근거 제목이 번역1,2 안에 그대로 있는지 (있으면 확실)
    t = norm(c["meta"]["title"])
    for x in texts[1:]:                                # texts = [원문, 번역①, 번역②]
        if re.search(rf"\b{re.escape(t)}\b", x.lower()):
            return True
    return False


def passes_verify(q, c, texts):     # 근거 통과 여부 (제목이 맞으면 바로 통과, 아니면 LLM 검증)
    return sure_match(texts, c) or verify_chunk(q, c, texts[1])


# ── 5. 최종 함수 ──
def pipeline(question, attempts=1, show_orig=True):  # 질문 → 검색 → 선택 → 검증 → 설명 → 검사 (전체 결과)
    # 검색 + 번호 고르기
    found, texts = retrieve(question)
    n = select_chunk(question, found)
    c = found[n - 1]

    # 검증 실패 → 거절
    if not passes_verify(question, c, texts):

        # 후보 3개의 (제목, 페이지)만 모아 두기 → 개발 화면에서 "후보: ..."로 보여 줌
        candidates = []
        for x in found[:3]:
            title = x["meta"]["title"]
            page = x["meta"]["page"]
            candidates.append((title, page))

        return {
            "status": "refuse",          # 거절했다는 표시
            "candidates": candidates,    # 후보 3개
            "warn": [],                  # 경고 없음 (설명을 안 썼으니까)
            "first_flagged": False,      # 통계용: 첫 시도에 검사에 걸렸는지
            "attempts_used": 0,          # 통계용: 설명을 몇 번 썼는지 (거절이라 0)
        }

    # 설명 쓰기 + 검사: 경고가 있으면 seed를 바꿔 최대 attempts번
    first_flagged = None                               # 첫 시도에 경고가 있었는지 (통계용)
    r, warn = None, []
    for a in range(attempts):
        temp0 = 0.0 if a == 0 else 0.3                 # 첫 시도는 0, 재시도부터 0.3
        r = explain_chunk(question, temp0=temp0, seed0=40 + 10 * a,
                          pre=(found, texts, n), en_terms=show_orig)
        warn = collect_warnings(r, check_explain(r), show_orig=show_orig)
        if first_flagged is None:
            first_flagged = bool(warn)
        if not warn:
            break

    first_line, _, rest = r["text"].partition("\n")
    body = rest if "\n" in r["text"] else first_line   # 청크에서 경로(첫 줄)를 뺀 본문

    return {
        "status": "hidden" if warn else "ok",          # 경고가 남아 있으면 설명을 숨김
        "warn": warn,
        "first_flagged": first_flagged,
        "attempts_used": a + 1,
        "title": r["title"],
        "page": r["page"],
        "path": r["path"],
        "orig": r["orig"],
        "body": body[:600],
        "explain": r["explain"],
    }


# ── 준비: 룰북 받기 → 청크 읽기 → 임베딩 → Chroma → BM25 (load를 부를 때만 실행)

def download(url=RULES_URL, path=OUT_PATH):     # GitHub에서 룰북 청크 파일 받기 (이미 있으면 건너뜀)
    if os.path.exists(path):
        return
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")      # 비공개 저장소면 토큰이 필요해요
    if token:
        headers["Authorization"] = f"token {token}"
    res = requests.get(url, headers=headers, timeout=120)
    res.raise_for_status()                       # 404면 RULES_URL(저장소 이름·브랜치)을 확인
    with open(path, "w", encoding="utf-8") as f:
        f.write(res.text)
    print("룰북 받음:", path)


def load(url=RULES_URL, force=False):           # 검색 준비 (한 번만 부르면 돼요)
    global chunks, emb, collection, bm25
    if chunks is not None and not force:         # 이미 준비됐으면 다시 안 함
        return

    # 0. Ollama 서버 확인
    try:
        requests.get(OLLAMA_URL, timeout=5)
    except Exception:
        raise RuntimeError(f"Ollama 서버에 연결할 수 없어요: {OLLAMA_URL} (노트북에서 서버를 먼저 켜 주세요)")

    # 1. 룰북 받기
    if force and os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)
    download(url)

    # 2. 청크 읽기 (규칙 청크만)
    chunks = []
    for line in open(OUT_PATH, encoding="utf-8"):
        c = json.loads(line)
        if c["meta"]["type"] == "rule":
            chunks.append(c)
    print("검색 대상 청크:", len(chunks))

    # 3. 임베딩: 저장된 파일이 있고 청크 수가 같으면 다시 계산하지 않음
    saved = None
    if os.path.exists(EMB_PATH):
        saved = np.load(EMB_PATH)
    if saved is not None and saved.shape[0] == len(chunks):
        emb = saved
        print("임베딩 불러옴:", emb.shape)
    else:
        vecs = []
        for i in tqdm(range(0, len(chunks), 32)):           # 32개씩 묶어서 임베딩
            batch = []
            for c in chunks[i:i + 32]:
                batch.append(c["text"])
            vecs += embed_batch(batch)
        emb = np.array(vecs, dtype="float32")
        np.save(EMB_PATH, emb)
        print("임베딩 새로 계산:", emb.shape)

    # 4. Chroma 컬렉션 (메모리에만, 매번 새로 채움)
    client = chromadb.Client()
    try:
        client.delete_collection(COLLECTION)     # load를 다시 부르면 예전 것을 지우고 새로
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine", "hnsw:search_ef": 200},
    )
    ids = []                                     # Chroma id는 문자열: 청크 순서 번호
    metadatas = []                               # 장 이름 (나중에 거르기용)
    for i, c in enumerate(chunks):
        ids.append(str(i))
        metadatas.append({"h1": c["meta"]["h1"]})
    collection.add(ids=ids, embeddings=emb.tolist(), metadatas=metadatas)

    # 5. BM25 (제목을 두 번 넣어서 제목 단어에 가중치)
    docs = []
    for c in chunks:
        title = c["meta"]["title"]
        docs.append(tokenize(title + " " + title + " " + c["text"]))
    bm25 = BM25(docs)
    print("준비 완료")


# ── 밖에서 쓰는 함수

def answer(question, attempts=1, show_orig=False):   # 질문 → {"status", "explain", "title", "warn"}
    if chunks is None:
        raise RuntimeError("먼저 rule_rag.load()를 불러 주세요")
    r = pipeline(question, attempts=attempts, show_orig=show_orig)
    if r["status"] == "refuse":
        return {"status": "refuse", "explain": "", "title": "", "warn": []}
    return {"status": r["status"], "explain": r["explain"], "title": r["title"], "warn": r["warn"]}
