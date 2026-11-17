"""주사위. 시드를 고정하거나 정해진 값을 넣어 테스트할 수 있다.

모든 무작위 값은 Dice.die() 하나를 거친다. 그래서 FixedDice가 die()만 바꿔도
d20·피해·낙하·우선권까지 모두 정해진 값으로 굴러간다 (tests/ 의 대부분이 이 방식).
"""
import random
import re

DICE_EXPR = re.compile(r"^\s*(\d*)d(\d+)\s*([+-]\s*\d+)?\s*$")


class Dice:
    """seed가 같으면 같은 순서로 같은 값이 나온다 (화면의 '주사위 시드' 입력이 여기로 온다)."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def die(self, sides):
        return self.rng.randint(1, sides)

    def d20(self, mode="normal"):
        """mode: normal / advantage / disadvantage. (선택된 값, 굴린 값 목록)을 돌려준다."""
        if mode == "normal":
            r = self.die(20)
            return r, [r]
        rolls = [self.die(20), self.die(20)]
        return (max(rolls) if mode == "advantage" else min(rolls)), rolls

    def roll(self, expr):
        """'1d10', '2d6+3' 같은 식. (합계, 개별 값)을 돌려준다."""
        m = DICE_EXPR.match(expr)
        if not m:
            raise ValueError(f"주사위 식을 읽을 수 없음: {expr}")
        count = int(m.group(1) or 1)
        sides = int(m.group(2))
        bonus = int(m.group(3).replace(" ", "")) if m.group(3) else 0
        rolls = [self.die(sides) for _ in range(count)]
        return sum(rolls) + bonus, rolls


class FixedDice(Dice):
    """테스트용: 미리 정한 값을 차례로 돌려준다 (d20, 피해 주사위 모두)."""

    def __init__(self, values):
        super().__init__(0)
        self.values = list(values)

    def die(self, sides):
        if not self.values:
            raise RuntimeError("FixedDice 값이 바닥남")
        v = self.values.pop(0)
        assert 1 <= v <= sides, (v, sides)
        return v
