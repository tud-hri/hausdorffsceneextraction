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
