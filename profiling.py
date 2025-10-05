from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import os

def start_profiling(func, log_dir="./profile"):
    os.makedirs(log_dir, exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    ) as prof:
        with record_function("model_training"):
            func()
        prof.step()
