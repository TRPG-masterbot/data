"""sce_rag.py - 잠긴 종 시나리오 RAG (Kim)

시나리오 카드(cards.json)에서 "지금 플레이어가 볼 수 있는 카드"만 검색해 주는 모듈.

사용법 (통합 노트북에서)
    import sce_rag
    sce_rag.load()                                   # 카드 읽기 + 임베딩 + Chroma 저장 (처음 한 번)
    state = sce_rag.make_state(1, "fighter")         # 시험용 상태 (실제 게임에서는 엔진이 만들어 줌)
    hits = sce_rag.search("우물 뚜껑을 살핀다", state, k=2)    # [(카드 id, 유사도), ...]
    text = sce_rag.card_text(hits[0][0])             # 카드의 플레이어용 글

state 모양: {"scene": 장면 번호, "flags": 켜진 플래그 집합, "visited": 가 본 방 집합, "cls": 직업}
"""

import json

import requests
import chromadb
from sentence_transformers import SentenceTransformer

CARDS_URL = "https://raw.githubusercontent.com/TRPG-masterbot/data/main/Kim/TheDrownedBell/game/data/cards.json"
EMBED_MODEL = "intfloat/multilingual-e5-small"       # 글 앞에 "passage: ", 질문 앞에 "query: "를 붙여 쓰는 모델

# load()가 채우는 값들
cards = []              # cards.json의 카드 전체 (166장)
rooms = {}              # 방 id -> 한글 이름
card_by_id = {}         # 카드 id -> 카드 원본
chunks = []             # 검색 대상 청크 (player_text가 있는 카드만)
_model = None           # 임베딩 모델
_collection = None      # Chroma 컬렉션


# ---------- 1. 지금 볼 수 있는 카드인가? ----------

def can_search(card, state):
    if state["scene"] < card["reveal_after"]:
        return False        # 아직 공개 장면이 아님
    if card["req_flag"] and card["req_flag"] not in state["flags"]:
        return False        # 필요한 플래그가 없음
    if card["room"] and card["room"] not in state["visited"]:
        return False        # 아직 가 보지 않은 방
    if card["owner"] != "all" and card["owner"] != state["cls"]:
        return False        # 다른 직업 전용 카드
    return True


# ---------- 2. 카드 -> 청크 ----------

def _make_chunk_text(card):
    # "[종류] 방이름" 머리말 + player_text (GM 전용 문장은 넣지 않는다)
    room_name = rooms.get(card["room"], "")
    header = f"[{card['label']}]"
    if room_name:
        header += f" {room_name}"
    return header + "\n" + card["player_text"]


def _make_metadata(card):
    return {
        "card_id": card["id"],
        "scene_id": card["scene_id"],
        "reveal_after": card["reveal_after"],
        "owner": card["owner"],
        "kind": card["kind"],
        "req_flag": card["req_flag"],
        "room": card["room"],
    }


# ---------- 3. 준비 (처음 한 번) ----------

def load():
    global cards, rooms, card_by_id, chunks, _model, _collection

    response = requests.get(CARDS_URL, timeout=60)
    response.raise_for_status()
    data = json.loads(response.content)
    cards = data["cards"]
    rooms = data["rooms"]
    card_by_id = {c["id"]: c for c in cards}

    # player_text가 있는 카드만 검색 대상 (GM 전용 카드는 player_text가 비어 있음)
    search_cards = [c for c in cards if c["player_text"]]
    for banned in ("secret", "check", "monster"):
        assert all(c["kind"] != banned for c in search_cards), banned     # 비밀 종류가 섞이면 멈춤

    chunks = [{"id": c["id"], "text": _make_chunk_text(c), "metadata": _make_metadata(c)}
              for c in search_cards]
    for c, ch in zip(search_cards, chunks):                               # GM 문장이 청크에 없는지 확인
        if c["gm_only_text"]:
            assert c["gm_only_text"] not in ch["text"], c["id"]

    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    vectors = _model.encode(["passage: " + ch["text"] for ch in chunks],
                            normalize_embeddings=True, show_progress_bar=False)

    client = chromadb.Client()                      # 메모리에만 저장 (런타임을 끄면 사라짐)
    _collection = client.get_or_create_collection(name="scenario_cards", metadata={"hnsw:space": "cosine"})
    _collection.upsert(
        ids=[ch["id"] for ch in chunks],
        documents=[ch["text"] for ch in chunks],
        metadatas=[ch["metadata"] for ch in chunks],
        embeddings=vectors.tolist(),
    )
    print(f"sce_rag 준비 완료: 카드 {len(cards)}장 중 검색 대상 {len(chunks)}장")


# ---------- 4. 검색 ----------

def make_state(scene, cls):
    # 시험용: 그 장면까지 열린 방을 모두 가 본 상태로 만든다 (플래그는 없음)
    visited = {c["room"] for c in cards if c["reveal_after"] <= scene and c["room"]}
    return {"scene": scene, "flags": set(), "visited": visited, "cls": cls}


def search(question, state, k=3, use_filter=True):
    # Chroma에서 유사도 순으로 전부 가져온 뒤, 지금 볼 수 있는 카드만 남기고 앞에서 k개를 돌려준다
    q_vec = _model.encode("query: " + question, normalize_embeddings=True)
    res = _collection.query(query_embeddings=[q_vec.tolist()], n_results=len(chunks))

    hits = []
    for cid, dist in zip(res["ids"][0], res["distances"][0]):
        if use_filter and not can_search(card_by_id[cid], state):
            continue
        hits.append((cid, round(1 - dist, 3)))      # (카드 id, 유사도)
    return hits[:k]


def card_text(card_id):
    # 서술 모델에게 줄 글: 플레이어에게 보여도 되는 문장만
    return card_by_id[card_id]["player_text"]
