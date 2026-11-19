"""캐릭터 한 명과 게임 한 판의 상태를 담는 자료형을 정의합니다.

입력: characters.json에서 읽은 캐릭터 항목과 엔진이 관리할 초기 위치·파티입니다.
처리: dataclass가 필드를 묶고, 수정치·현재 최대 HP처럼 원래 값에서 구하는 수치를 제공합니다.
출력: Character와 GameState 객체이며, 게임 진행에 따라 이 객체의 값이 갱신됩니다.

content.py가 캐릭터를 만들고 engine.py·combat.py·social.py가 상태를 사용합니다.
gameplay.py는 저장·복원 자료를 구성하며 game_ui.py는 이 값을 화면에 표시합니다.
체크포인트는 GameState의 복사본이므로 진행 중 보존할 상태는 이 구조 안에 둡니다."""
from dataclasses import dataclass, field

ABILITIES = ["str", "dex", "con", "int", "wis", "cha"]
ABILITY_KO = {"str": "근력", "dex": "민첩", "con": "건강", "int": "지능", "wis": "지혜", "cha": "매력"}


def ability_mod(score):
    """SRD 5.2 p.21 Ability Scores and Modifiers 표와 같은 값."""
    return (score - 10) // 2


@dataclass
class Character:
    """플레이어 캐릭터 한 명의 기본 수치와 진행 중 상태를 담습니다.

    Content.new_party가 characters.json의 항목을 필드로 전달합니다. hp_max는 원래 최대 HP,
    max_hp는 감소 효과를 반영한 현재 최대 HP이며 hp는 지금 남은 HP입니다.
    resources에는 주문 슬롯·직업 기능 횟수, conditions에는 현재 상태를 보관합니다.

    conditions의 값은 남은 분을 나타내는 정수, 무기한인 None, 추가 자료인 사전 등으로
    다를 수 있습니다. 시간 처리와 전투 함수가 이 형식에 맞게 읽고 갱신합니다."""
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
    # default_factory는 캐릭터마다 새 사전을 만들어 상태 목록을 공유하지 않게 합니다.
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
        """초기 HP·회복 주사위가 생략됐으면 최대값으로 준비합니다."""
        if self.hp is None:
            self.hp = self.hp_max
        if self.hd_left is None:
            self.hd_left = self.hd_max

    @property
    def hd_max(self):
        """현재 레벨에 해당하는 회복 주사위의 최대 개수입니다."""
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
        """현재 HP가 0 이하인지 반환하여 쓰러진 상태를 확인합니다."""
        return self.hp <= 0

    def can_act(self):
        """현재 HP를 기준으로 행동할 수 있는 상태인지 반환합니다."""
        return not self.down


@dataclass
class GameState:
    """게임 한 판에서 달라지는 위치·수치·진행 기록을 함께 보관합니다.

    party는 캐릭터, inventory와 gold_cp는 공동 소지품·금액입니다.
    flags·vars는 진행 조건, shown은 공개한 카드와 알게 된 캐릭터, done·failed는
    선택지의 완료·실패 기록입니다. pending과 combat에는 기다리는 판정·현재 전투를 둡니다.

    Engine은 이 객체를 바꾸고 체크포인트로 복사합니다. 화면·대화 자료는 현재 값을 읽어
    만들므로, 장소 묘사와 실제 위치 또는 제안과 물품 이전 기록을 구별할 수 있습니다."""
    party: list
    room: str
    scene: int = 1
    flags: set = field(default_factory=set)
    vars: dict = field(default_factory=dict)          # flag 표에 없는 상태값 (05-2 7-3)
    # 이번 판에서 바뀌는 가방·재고·진행 기록은 각 GameState가 별도로 보관합니다.
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
        """캐릭터 ID로 현재 파티원을 찾습니다. 없는 ID면 KeyError를 냅니다."""
        for c in self.party:
            if c.id == char_id:
                return c
        raise KeyError(char_id)

    def has_item(self, item_id):
        """현재 가방에 해당 물품이 1개 이상 있는지 확인합니다."""
        return self.inventory.get(item_id, 0) > 0
