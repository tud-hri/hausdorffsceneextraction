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

import tqdm
import multiprocessing as mp


class ProgressProcess(mp.Process):
    """
    Helper process used to track the progress of other workers. If a worker finishes it's calculations for a dataset, it puts a message in the queue to this
    process. This process will then update the progressbar for the user.
    """
    def __init__(self, total, manager: mp.Manager):
        super().__init__()

        self.tqdm = None
        self._total = total
        self._counter = 0
        self.queue = manager.Queue()

    def run(self):
        self.tqdm = tqdm.tqdm(total=self._total)
        while self._counter < self._total:
            increment = self.queue.get()
            self._counter += increment
            self.tqdm.update(n=increment)

        self.tqdm.close()
