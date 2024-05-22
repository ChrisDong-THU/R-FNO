import torch
import time


class Timer:
    def __init__(self):
        self.enabled = False
        self.timing_stat = {}

    def timer_func(self, func):
        # This function shows the execution time of the function object passed
        def wrap_func(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            t1 = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            t2 = time.time()

            # 结果保存为ms
            if func.__qualname__ in self.timing_stat.keys():
                self.timing_stat[func.__qualname__] += (t2 - t1) * 1000
            else:
                self.timing_stat[func.__qualname__] = (t2 - t1) * 1000

            return result

        return wrap_func

    def clear_timing_stat(self):
        self.timing_stat = {}

    def get_timing_stat(self):
        return self.timing_stat

    def set_enabled(self, enabled):
        self.enabled = enabled
        
timer = Timer()