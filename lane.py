from enum import Enum


class Lane(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2
    MERGING = 3

    def __str__(self):
        return {Lane.LEFT: 'left lane',
                Lane.CENTER: 'center lane',
                Lane.RIGHT: 'right lane',
                Lane.MERGING: 'merging lane'}[self]
