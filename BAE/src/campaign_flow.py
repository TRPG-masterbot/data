"""공개된 소개와 선택 방문을 구분합니다. 원본 카드·엔진 효과는 바꾸지 않습니다."""

FLOW_STAGES = ("sera_request", "sera_doll", "halvik_visit", "halvik_request",
               "optional_preparation", "well_exploration", "exploration")
NPC_STAGE = {"scn-q1-npc-sera": "sera_request", "scn-s1-npc-halvik": "halvik_visit",
             "scn-m-npc-smith": "optional_preparation", "scn-m-npc-innkeeper": "optional_preparation",
             "scn-s1-npc-mera": "optional_preparation"}
ROOM_STAGE = {"village_square": "sera_request", "halvik_house": "halvik_visit",
              "halvik_study": "halvik_visit", "village_market": "optional_preparation",
              "well_side": "optional_preparation", "shaft_upper": "well_exploration"}
CONVERSATION_MILESTONES = {"scn-m-npc-smith", "scn-m-npc-innkeeper", "scn-s1-npc-mera"}
EARLY_AMBIENT_ACTIONS = {"sq_rumors"}

# 소개의 출처만 정합니다. 세 시설 사이에 선행 방문·구매·숙박 조건은 없습니다.
INTRODUCTION_SOURCES = {"scn-s1-npc-halvik": "scn-q1-npc-sera",
    "scn-m-npc-smith": "scn-s1-npc-halvik", "scn-m-npc-innkeeper": "scn-s1-npc-halvik",
    "scn-s1-npc-mera": "scn-s1-npc-halvik"}
ROOM_INTRODUCTIONS = {"halvik_house": ("scn-s1-npc-halvik",), "halvik_study": ("scn-s1-npc-halvik",),
    "village_market": ("scn-m-npc-smith", "scn-m-npc-innkeeper"),
    "well_side": ("scn-s1-npc-mera",), "shaft_upper": ("scn-s1-npc-mera",)}
INTRODUCTION_CONTENT = {
    "scn-s1-npc-halvik": "실종자들의 행방뿐 아니라 이 마을에 무슨 일이 있었는지 더 알고 싶다면 촌장 할빅과 이야기해 보세요.",
    "scn-m-npc-smith": "탐사 장비와 수리가 필요하면 대장장이 그란에게 알아볼 수 있습니다. 장비 구매는 선택입니다.",
    "scn-m-npc-innkeeper": "쉬어 가거나 숙박이 필요하면 여관 주인 오단에게 알아볼 수 있습니다. 숙박은 선택입니다.",
    "scn-s1-npc-mera": "옛 우물에 관한 이야기가 궁금하면 우물가의 메라에게 물어볼 수 있습니다. 대화는 선택입니다.",
}
# 눈앞의 우물을 실제 조사한 결과로 준비 장소를 알아내는 별도 공개 경로입니다.
# NPC의 소개나 의뢰 수락을 만들어 내지는 않습니다.
OBSERVATION_INTRODUCTIONS = {"ws_open_well": ("scn-m-npc-smith",)}


def reached(stage, minimum):
    # 이전 저장·연결 코드의 단계 이름은 선택 준비 단계와 같은 위치로 읽습니다.
    aliases = {"smith_visit": "optional_preparation", "inn_visit": "optional_preparation", "mera_visit": "optional_preparation"}
    stage, minimum = aliases.get(stage, stage), aliases.get(minimum, minimum)
    return FLOW_STAGES.index(stage) >= FLOW_STAGES.index(minimum)


def pending_stage(state, conversations):
    """수락·전달·실제로 공개한 사정 설명만 읽고 시설 방문 순서는 강제하지 않습니다."""
    if state.scene > 1:
        return "exploration"
    if not ("sq_accept_help" in state.done or state.vars.get("sera_help_accepted")):
        return "sera_request"
    if not ("sq_take_doll" in state.done or "got_rina_doll" in state.flags):
        return "sera_doll"
    if state.room == "well_side":
        return "well_exploration"
    if any(isinstance(record, dict) and record.get("source_kind") == "engine_observation"
           and record.get("origin") == "controller_disclosure"
           for record in state.vars.get("campaign_introductions", {}).values()):
        return "optional_preparation"
    if "scn-s1-npc-halvik" not in state.shown:
        return "halvik_visit"
    if not (state.vars.get("halvik_briefing_completed")
            or "hh_accept" in state.vars.get("request_disclosures", {})
            or "hh_accept" in state.done or state.vars.get("quest_accepted")):
        return "halvik_request"
    return "well_exploration" if state.room == "well_side" else "optional_preparation"
