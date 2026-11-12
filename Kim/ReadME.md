<h3>Kim/sce_RAG_시나리오전용.ipynb    -- (2026_11_12 업로드)</h3>

<ul>
  <li>
    <small>시나리오 RAG Colab 노트북 (룰북 제외). 지금 플레이어가 볼 수 있는 카드만 검색</small>
  </li>
  <li>
    <small>can_search (장면·방·직업 필터) / 카드 1장 = 청크 1개 / Chroma 저장 / search (상태 필터 검색)<br>
    골든셋 평가: 필터 적용 시 Hit@1 0.43 → 0.64, GM 문장 누설 0건</small>
  </li>
  <li>
    <small>narrate_scene (Qwen 장면 서술) 포함. 서술 품질 개선·룰북 RAG와 통합은 미구현</small>
  </li>
</ul>

<h3>TheDrownedBell/game/    -- (2026_11_06 업로드)</h3>

<ul>
  <li>
    <small>잠긴 종 게임 로직 (장면 1~2 구현, 파이썬만 사용·외부 라이브러리 없음). 가능하면 건드리지 않고 블랙박스로 쓸 예정</small>
  </li>
  <li>
    <small>engine.py (진행·선택지·상태 처리) / combat.py (전투) / dice.py (주사위)<br>
    model.py (상태 데이터) / content.py (JSON 불러오기) / events.py (선택지·이벤트 형식)</small>
  </li>
  <li>
    <small>narrator.py (문장 출력, 템플릿 또는 Ollama) / cli.py (터미널 플레이)</small>
  </li>
  <li>
    <small>data/ : 장면·몬스터·아이템·규칙 JSON과 RAG용 카드(cards.json, scenario.jsonl). 자유 입력 RAG는 아직 미구현</small>
  </li>
</ul>

<h3>Prototype_Code_0.1.ipynb    -- (2026_11_05 업로드)</h3>

<ul>
  <li>
    <small>잠긴종_Prototype_Code_0.1.ipynb</small>
  </li>
</ul>

<h3>rag_test.ipynb    -- (2026_10_30 업로드)</h3>

<ul>
  <li>
    <small>Qwen3.5 4B 모델 적용</small>
  </li>
  <li>
    <small>룰북, 시나리오 RAG 파일 텍스트 출력 완료</small>
  </li>
  <li>
    <small>룰북, 시나리오 RAG 파일 청킹 완료</small>
  </li>
    <li>
    <small>RAG 검색 유사도 결과 40%</small>
  </li>
</ul>
