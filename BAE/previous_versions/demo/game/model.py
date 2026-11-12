"""캐릭터와 게임 상태 (런타임). 설계: analysis/05_시스템설계/05-2 7장.

여기 있는 값은 게임 중에 바뀌는 값이다. 바꾸는 쪽은 engine.py·combat.py뿐이고,
서버(server/webapp.py)와 CLI는 읽어서 화면용으로 정리만 한다.
체크포인트는 GameState 전체를 deepcopy한 것이므로, 새 상태값은 반드시 GameState 안에 두어야 되돌리기에 포함된다.
"""
from dataclasses import dataclass, field

ABILITIES = ["str", "dex", "con", "int", "wis", "cha"]
ABILITY_KO = {"str": "근력", "dex": "민첩", "con": "건강", "int": "지능", "wis": "지혜", "cha": "매력"}


def ability_mod(score):
    """SRD 5.2 p.21 Ability Scores and Modifiers 표와 같은 값."""
    return (score - 10) // 2


@dataclass
class Character:
    """플레이 캐릭터 1명. characters.json의 항목 하나가 그대로 이 필드들로 들어온다 (Content.new_party).

    conditions(상태)의 값 형식은 상태마다 다르다:
      정수 = 남은 시간(분, _pass_time이 줄임) · None = 무기한 · dict = 추가 정보
      (dying: {"success", "fail"} 죽음 구원 횟수, grappled: {몬스터 id: {"dc", "restrained"}}) · "grapple" = 붙잡혀서 생긴 Restrained
    """
    id: str
    name: str
    cls: str
    cls_ko: str
    species: str
    background: str
    level: int
    abilities: dict
    hp_max: int
    save_profs: list
    skill_profs: list
    tool_profs: list
    traits: list
    tools: list
    darkvision: int = 0
    proficiency_bonus: int = 2
    hp: int = None
    conditions: dict = field(default_factory=dict)  # 이름 -> 남은 시간(분), None이면 무기한
    hit_die: int = 8                                # Hit Point Die 면 수 [SRD 클래스 표]
    hd_left: int = None                             # 남은 Hit Point Dice (최대 = 레벨) [SRD Rules Glossary]
    heroic_inspiration: bool = False                # [SRD Rules Glossary: Heroic Inspiration]
    # --- 전투 (W6) ---
    armor: str = "none"
    shield: bool = False
    fighting_style: str = ""
    weapons: list = field(default_factory=list)
    weapon_profs: list = field(default_factory=list)
    spellcasting: str = ""                          # 주문 시전 능력치 (int / wis)
    cantrips: list = field(default_factory=list)
    spells: list = field(default_factory=list)
    features: list = field(default_factory=list)
    resources: dict = field(default_factory=dict)   # 남은 사용 횟수: slots, second_wind, action_surge 등
    armor_penalty: int = 0                          # Gray Ooze 부식 [SRD p.290]
    weapon_penalty: dict = field(default_factory=dict)
    ability_loss: dict = field(default_factory=dict)  # Shadow 근력 흡수 등. 긴 휴식에서 복구 [SRD Long Rest]
    hp_max_loss: int = 0                            # Specter 생명 흡수
    concentration: str = ""                         # 집중 중인 주문 id

    def __post_init__(self):
        if self.hp is None:
            self.hp = self.hp_max
        if self.hd_left is None:
            self.hd_left = self.hd_max

    @property
    def hd_max(self):
        return self.level

    def mod(self, ability):
        """능력 수정치 (판정·공격·구원에 더하는 값). 줄어든 능력치(ability_loss)를 반영한다."""
        return ability_mod(self.score(ability))

    def score(self, ability):
        """현재 능력치 = 원래 값 − 흡수로 줄어든 값."""
        return self.abilities[ability] - self.ability_loss.get(ability, 0)

    @property
    def max_hp(self):
        """현재 최대 HP (생명 흡수로 줄어든 값). hp_max는 원래 값."""
        return max(0, self.hp_max - self.hp_max_loss)

    @property
    def down(self):
        return self.hp <= 0

    def can_act(self):
        return not self.down


@dataclass
class GameState:
    """게임 한 판의 바뀌는 상태 전부. Engine.s로 접근한다.

    flags는 원본 시나리오의 선택 flag(카드 req_flag와 연결), vars는 그 밖의 진행 상태
    (예: rope_anchored, shrieker_alerted, combat:rats). 장면 JSON의 {"flag": …}/{"var": …} 효과가 채운다.
    """
    party: list
    room: str
    scene: int = 1
    flags: set = field(default_factory=set)
    vars: dict = field(default_factory=dict)          # flag 표에 없는 상태값 (05-2 7-3)
    inventory: dict = field(default_factory=dict)     # 파티 공동 소지품: item_id -> 개수
    gold_cp: int = 0                                  # 돈은 cp 단위로 저장
    visited: set = field(default_factory=set)
    shown: dict = field(default_factory=dict)         # card_id -> 알게 된 캐릭터 id 목록 (플레이어 축)
    failed: set = field(default_factory=set)          # (choice_id, char_id): M3 재시도 규칙
    done: set = field(default_factory=set)            # 한 번만 하는 선택지 중 끝난 것
    minutes: int = 0                                  # 경과 시간
    ended: str = ""                                   # "scene1_complete" / "game_over" 등
    stock: dict = field(default_factory=dict)         # 상점 재고 (게임 중 바뀌는 값)
    deferred: dict = field(default_factory=dict)      # 나중에 공개할 카드·문장: key -> {"room", "card", "part"}
    once_done: set = field(default_factory=set)       # 한 번만 일어나는 방 진입·구매 반응
    traps: dict = field(default_factory=dict)         # trap_id -> {"count": 발동 횟수, "disarmed": bool} (W5)
    pending: dict = None                              # 플레이어 결정을 기다리는 일: 영감 재굴림 / 짧은 휴식 주사위 (W5·W6)
    combat: dict = None                               # 진행 중인 전투 (W6, game/combat.py)
    log: list = field(default_factory=list)

    def member(self, char_id):
        for c in self.party:
            if c.id == char_id:
                return c
        raise KeyError(char_id)

    def has_item(self, item_id):
        return self.inventory.get(item_id, 0) > 0
