"""주사위의 난수 생성과 주사위 식 계산을 담당합니다.

입력: 주사위 면 수, 이점·불리 모드, 또는 2d6+3 같은 문자열입니다.
처리: die가 한 번의 굴림을 만들고 d20과 roll이 필요한 횟수만큼 이를 호출합니다.
출력: 선택된 값이나 합계와, 설명에 사용할 개별 굴림 목록입니다.

engine.py와 combat.py가 이 객체를 받아 판정·피해·우선권에 사용합니다.
seed는 게임 주사위의 난수 순서를 정합니다. FixedDice는 같은 계산 경로에
미리 정한 굴림을 공급하여 특정 성공·실패 상황을 확인할 때 사용합니다."""
import random
import re

DICE_EXPR = re.compile(r"^\s*(\d*)d(\d+)\s*([+-]\s*\d+)?\s*$")


class Dice:
    """한 게임에서 사용할 주사위 난수 생성기를 보관합니다.

    seed가 같으면 같은 호출 순서에서 같은 굴림이 나옵니다.
    각 판정은 공유한 객체에서 다음 난수를 꺼내므로 굴림마다 seed를 다시 정하지 않습니다."""

    def __init__(self, seed=None):
        """이 Dice 객체가 사용할 독립적인 random.Random 생성기를 준비합니다."""
        self.rng = random.Random(seed)

    def die(self, sides):
        """1부터 sides까지 중 하나를 뽑아 주사위 한 번의 값을 반환합니다."""
        return self.rng.randint(1, sides)

    def d20(self, mode="normal"):
        """d20 판정용 값과 실제 굴림 목록을 반환합니다.

        일반 굴림은 한 번, 이점·불리는 두 번 굴립니다. 이점은 큰 값, 불리는 작은 값을
        선택하되 두 숫자를 함께 남겨 화면에서 판정 근거를 설명할 수 있게 합니다."""
        if mode == "normal":
            r = self.die(20)
            return r, [r]
        rolls = [self.die(20), self.die(20)]
        return (max(rolls) if mode == "advantage" else min(rolls)), rolls

    def roll(self, expr):
        """주사위 문자열을 개수·면 수·보정치로 나누어 합계와 개별 굴림을 반환합니다.

        예를 들어 2d6+3은 d6 두 개를 굴리고 보정치 3을 합계에 한 번 더합니다.
        지원하는 문자열 형식이 아니면 ValueError로 알려 줍니다."""
        m = DICE_EXPR.match(expr)
        if not m:
            raise ValueError(f"주사위 식을 읽을 수 없음: {expr}")
        count = int(m.group(1) or 1)
        sides = int(m.group(2))
        bonus = int(m.group(3).replace(" ", "")) if m.group(3) else 0
        # 보정치는 주사위마다 더하지 않고, 모든 굴림을 합친 뒤 한 번만 더합니다.
        rolls = [self.die(sides) for _ in range(count)]
        return sum(rolls) + bonus, rolls


class FixedDice(Dice):
    """미리 준 값을 순서대로 사용하는 확인용 주사위입니다.

    상위 Dice의 d20·roll 계산을 그대로 사용하고 한 번의 값 공급만 바꾸므로,
    같은 판정 흐름에서 성공·실패·치명타 같은 상황을 재현할 수 있습니다."""

    def __init__(self, values):
        super().__init__(0)
        self.values = list(values)

    def die(self, sides):
        """정해진 다음 값을 꺼내고 주사위 면 수 안의 값인지 확인합니다."""
        if not self.values:
            raise RuntimeError("FixedDice 값이 바닥남")
        v = self.values.pop(0)
        assert 1 <= v <= sides, (v, sides)
        return v
