import enum


class ManeuverState(enum.Enum):
    NO_LANE_CHANGE = 0
    BEFORE_LANE_CHANGE = 1
    AFTER_LANE_CHANGE = 2

    def __str__(self):
        return {ManeuverState.NO_LANE_CHANGE: 'No lane change',
                ManeuverState.BEFORE_LANE_CHANGE: 'Before lane change',
                ManeuverState.AFTER_LANE_CHANGE: 'After lane change'}[self]
