import os
from torch.profiler import profile, record_function, ProfilerActivity

def start_profiling(func, log_dir="./profile"):
    os.makedirs(log_dir, exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        with record_function("model_training"):
            func()
    prof.export_chrome_trace(f"{log_dir}/trace.json")
