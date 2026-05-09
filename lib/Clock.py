import time

class Clock:
    def __init__(self, dt):
        self.dt = dt
        self.next_time = time.perf_counter()

    def wait(self):
        self.next_time += self.dt
        remain = self.next_time - time.perf_counter()
        if remain > 0:
            time.sleep(remain)