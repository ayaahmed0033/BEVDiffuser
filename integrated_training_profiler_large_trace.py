from pathlib import Path

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)


def create_training_profiler(
    output_directory="/home/aya/BEVDiffuser/profiler_traces",
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]

    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    return profile(
        activities=activities,

        # Total: 1 ignored + 1 warm-up + 3 measured = 5 iterations
        schedule=schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1,
        ),

        on_trace_ready=tensorboard_trace_handler(
            str(output_directory)
        ),

        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    )