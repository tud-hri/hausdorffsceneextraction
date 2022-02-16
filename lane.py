"""
Copyright 2022, Olger Siebinga (o.siebinga@tudelft.nl)
This file is part of the module hausdorffsceneextraction.

hausdorffsceneextraction is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

hausdorffsceneextraction is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with hausdorffsceneextraction.  If not, see <https://www.gnu.org/licenses/>.
"""

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
