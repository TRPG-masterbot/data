<h3>rule RAG_5.ipynb    -- (2026_11_05 업로드)</h3>

<ul>
  <li>
    <small>RuleRAG 수정본</small>
  </li>
</ul>

<h3>TRPG_Rule.ipynb    -- (2026_11_03 업로드)</h3>

<ul>
  <li>
    <small>RuleRAG 초기 버전</small>
  * Qwen2.5-7B AI 모델 / bge-m3 임베딩 모델 / FAISS 벡터 인덱스 사용 (Ollama 로컬 실행, Colab T4)
* 룰북 원본 PDF(SRD 5.2, 영어)를 PyMuPDF로 직접 파싱
* 글꼴 크기로 제목 단계 판별 → 규칙 단위 청크 분할 → JSONL 저장
* 임베딩 .npy 캐시 / 코사인 유사도 검색
* 원문 복사 → 한국어 설명 프롬프트 + 예시 1개 (In-Context Learning)
* 중국어 섞임 시 재생성</li>
</ul>

<h3>‎Untitled5.ipynb    -- (2026_10_30 업로드)</h3>

<ul>
  <li>
    <small>AI생성 RuleRAG_폐기본</small>
  </li>
</ul>
