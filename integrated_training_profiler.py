# """Small PyTorch Profiler configuration for BEVDiffuser training.

# The semantic record_function labels remain in train_bev_diffuser.py. This
# module only controls how much profiling detail is saved to disk.
# """

# import os

# import torch


# TRACE_DIRECTORY = "/home/aya/BEVDiffuser/profiler_traces_small"


# def create_training_profiler():
#     """Create a short trace that TensorBoard can load reliably."""
#     os.makedirs(TRACE_DIRECTORY, exist_ok=True)

#     return torch.profiler.profile(
#         activities=[
#             torch.profiler.ProfilerActivity.CPU,
#             torch.profiler.ProfilerActivity.CUDA,
#         ],
#         schedule=torch.profiler.schedule(
#             wait=1,
#             warmup=1,
#             active=1,
#             repeat=1,
#         ),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler(
#             TRACE_DIRECTORY
#         ),
#         record_shapes=False,
#         profile_memory=False,
#         with_stack=False,
#         with_flops=False,
#     )

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
            active=1,
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