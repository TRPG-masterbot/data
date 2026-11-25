"""사용자가 지정한 초반 방문 순서. 원본 카드·엔진 효과는 바꾸지 않습니다."""

FLOW_STAGES = ("sera_request", "sera_doll", "halvik_visit", "halvik_request",
               "smith_visit", "inn_visit", "mera_visit", "well_exploration", "exploration")
NPC_STAGE = {"scn-q1-npc-sera": "sera_request", "scn-s1-npc-halvik": "halvik_visit",
             "scn-m-npc-smith": "smith_visit", "scn-m-npc-innkeeper": "inn_visit",
             "scn-s1-npc-mera": "mera_visit"}
ROOM_STAGE = {"village_square": "sera_request", "halvik_house": "halvik_visit",
              "halvik_study": "halvik_visit", "village_market": "smith_visit",
              "well_side": "mera_visit", "shaft_upper": "well_exploration"}
CONVERSATION_MILESTONES = {"scn-m-npc-smith", "scn-m-npc-innkeeper", "scn-s1-npc-mera"}
EARLY_AMBIENT_ACTIONS = {"sq_rumors"}


def reached(stage, minimum):
    return FLOW_STAGES.index(stage) >= FLOW_STAGES.index(minimum)


def pending_stage(state, conversations):
    """수락·전달은 원본 엔진 기록, 방문은 실제 표시한 발언으로 판별합니다."""
    if state.scene > 1:
        return "exploration"
    if not ("sq_accept_help" in state.done or state.vars.get("sera_help_accepted")):
        return "sera_request"
    if not ("sq_take_doll" in state.done or "got_rina_doll" in state.flags):
        return "sera_doll"
    if "scn-s1-npc-halvik" not in state.shown:
        return "halvik_visit"
    if not ("hh_accept" in state.done or state.vars.get("quest_accepted")):
        return "halvik_request"
    for identifier, stage in (("scn-m-npc-smith", "smith_visit"),
                              ("scn-m-npc-innkeeper", "inn_visit"),
                              ("scn-s1-npc-mera", "mera_visit")):
        if identifier not in conversations:
            return stage
    return "well_exploration"
