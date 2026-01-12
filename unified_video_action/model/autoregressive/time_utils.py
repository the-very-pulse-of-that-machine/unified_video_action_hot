# timing_utils.py
import torch
from collections import defaultdict

class CUDATimer:
    def __init__(self):
        self.records = defaultdict(list)

    def start(self, name):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return name, start, end

    def stop(self, handle):
        name, start, end = handle
        end.record()
        torch.cuda.synchronize()
        self.records[name].append(start.elapsed_time(end))  # ms

    def summary(self):
        print("\n====== Timing Summary (ms) ======")
        for k, v in self.records.items():
            avg = sum(v) / len(v)
            print(f"{k:30s}: {avg:.3f} ms ({len(v)} iters)")
