<h3>rule RAG_5.ipynb    -- (2026_11_05 업로드)</h3>

<ul>
  <li><small>TRPG_Rule 기반 개선 버전</small></li>
  <li>Qwen2.5-14B로 모델 변경 (번역·근거 선택·설명 모두 단일 모델)</li>
  <li>PDF 파싱 분리 → 저장된 JSONL 청크를 바로 불러와 시작</li>
  <li>한국어 질문 → 영어 검색어 번역 2종 (일반 번역 / 규칙 용어 선택지 번역)</li>
  <li>BM25 키워드 검색 직접 구현 + 임베딩 검색 결합 (가중 RRF, 후보 5개)</li>
  <li>번역·검색 결과 캐시 (드라이브 저장, 재실행 시 불러오기)</li>
  <li>2단계 답변: 근거 번호 선택 → 근거 검증(거절) → 원문 + 한국어 설명</li>
  <li>자동 검사 추가 (원문 확인, 근거에 없는 숫자, 혼합 철자, 깨진 글자, 충실성 검사)</li>
  <li>답변용 용어표로 한국어 용어 통일</li>
  <li>개발셋(38문항) / 최종셋(60문항) 분리 평가</li>
  <li>재순위(reranker) 실험 → 성능 하락으로 미채택</li>
</ul>

<h3>TRPG_Rule.ipynb    -- (2026_11_03 업로드)</h3>

<ul>
  <li><small>RuleRAG 초기 버전</small></li>
  <li>Qwen2.5-7B AI 모델 / bge-m3 임베딩 모델 / FAISS 벡터 인덱스 사용 (Ollama 로컬 실행, Colab T4)</li>
  <li>룰북 원본 PDF(SRD 5.2, 영어)를 PyMuPDF로 직접 파싱</li>
  <li>글꼴 크기로 제목 단계 판별 → 규칙 단위 청크 분할 → JSONL 저장</li>
  <li>임베딩 .npy 캐시 / 코사인 유사도 검색</li>
  <li>원문 복사 → 한국어 설명 프롬프트 + 예시 1개 (In-Context Learning)</li>
  <li>중국어 섞임 시 재생성</li>
</ul>

<h3>‎Untitled5.ipynb    -- (2026_10_30 업로드)</h3>

<ul>
  <li>
    <small>AI생성 RuleRAG_폐기본</small>
  </li>
</ul>
