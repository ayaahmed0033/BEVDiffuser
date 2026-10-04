import os
from collections import Counter

import mmcv
import numpy as np
from mmcv import Config


BEVFORMER_ROOT = "/home/aya/BEVDiffuser/BEVFormer"

CONFIG_PATH = os.path.join(
    BEVFORMER_ROOT,
    "projects/configs/bevdiffuser/layout_tiny.py"
)

cfg = Config.fromfile(CONFIG_PATH)
class_names = list(cfg.class_names)


def resolve_path(path):
    """Convert a configuration-relative path into an absolute path."""
    if os.path.isabs(path):
        return path

    return os.path.join(BEVFORMER_ROOT, path)


TRAIN_ANN_FILE = resolve_path(cfg.data.train.ann_file)
VAL_ANN_FILE = resolve_path(cfg.data.test.ann_file)


def count_annotations(annotation_file, class_names, filter_mode):
    data = mmcv.load(annotation_file)

    if "infos" not in data:
        raise KeyError(
            f"The annotation file has no 'infos' field: {annotation_file}"
        )

    infos = data["infos"]

    object_counts = Counter()
    frame_counts = Counter()
    unknown_counts = Counter()

    for info in infos:
        names = np.asarray(
            info.get("gt_names", []),
            dtype=object
        )

        number_of_objects = len(names)

        if filter_mode == "all":
            mask = np.ones(number_of_objects, dtype=bool)

        elif filter_mode == "valid_flag":
            if "valid_flag" in info:
                mask = np.asarray(
                    info["valid_flag"],
                    dtype=bool
                )
            else:
                print(
                    "Warning: valid_flag was missing. "
                    "Using all annotations for this frame."
                )
                mask = np.ones(number_of_objects, dtype=bool)

        elif filter_mode == "lidar_visible":
            if "num_lidar_pts" in info:
                mask = (
                    np.asarray(info["num_lidar_pts"]) > 0
                )
            else:
                print(
                    "Warning: num_lidar_pts was missing. "
                    "Using all annotations for this frame."
                )
                mask = np.ones(number_of_objects, dtype=bool)

        else:
            raise ValueError(
                f"Unknown filter mode: {filter_mode}"
            )

        if len(mask) != number_of_objects:
            raise ValueError(
                "The number of filtering flags does not match "
                "the number of ground-truth names."
            )

        selected_names = names[mask]
        classes_in_frame = set()

        for name in selected_names:
            name = str(name)

            if name in class_names:
                object_counts[name] += 1
                classes_in_frame.add(name)
            else:
                unknown_counts[name] += 1

        for name in classes_in_frame:
            frame_counts[name] += 1

    return {
        "objects": object_counts,
        "frames": frame_counts,
        "unknown": unknown_counts,
        "number_of_frames": len(infos),
    }


def print_results(title, results, class_names):
    object_counts = results["objects"]
    frame_counts = results["frames"]
    unknown_counts = results["unknown"]
    number_of_frames = results["number_of_frames"]

    total_objects = sum(object_counts.values())

    print()
    print("=" * 88)
    print(title)
    print("=" * 88)

    print(f"Frames: {number_of_frames}")
    print(f"Counted objects: {total_objects}")

    if unknown_counts:
        print(f"Unknown categories: {dict(unknown_counts)}")
    else:
        print("Unknown categories: none")

    print()
    print(
        f"{'Class':25s}"
        f"{'Objects':>12s}"
        f"{'Object %':>13s}"
        f"{'Frames':>12s}"
        f"{'Frame %':>13s}"
    )
    print("-" * 88)

    for class_name in class_names:
        objects = object_counts[class_name]
        frames = frame_counts[class_name]

        object_percentage = (
            100.0 * objects / total_objects
            if total_objects > 0
            else 0.0
        )

        frame_percentage = (
            100.0 * frames / number_of_frames
            if number_of_frames > 0
            else 0.0
        )

        print(
            f"{class_name:25s}"
            f"{objects:12d}"
            f"{object_percentage:12.3f}%"
            f"{frames:12d}"
            f"{frame_percentage:12.3f}%"
        )

    print()
    print("Frequency ranking:")

    ranking = sorted(
        class_names,
        key=lambda name: object_counts[name],
        reverse=True
    )

    for rank, class_name in enumerate(ranking, start=1):
        print(
            f"{rank:2d}. "
            f"{class_name:25s} "
            f"{object_counts[class_name]:8d}"
        )


print(f"Training annotations:   {TRAIN_ANN_FILE}")
print(f"Validation annotations: {VAL_ANN_FILE}")


# Training configuration has use_valid_flag=True.
train_results = count_annotations(
    TRAIN_ANN_FILE,
    class_names,
    filter_mode="valid_flag"
)

# Raw validation distribution.
val_all_results = count_annotations(
    VAL_ANN_FILE,
    class_names,
    filter_mode="all"
)

# Validation objects with at least one LiDAR point.
val_visible_results = count_annotations(
    VAL_ANN_FILE,
    class_names,
    filter_mode="lidar_visible"
)


print_results(
    "TRAIN: OBJECTS USED WITH valid_flag",
    train_results,
    class_names
)

print_results(
    "VALIDATION: ALL ANNOTATED OBJECTS",
    val_all_results,
    class_names
)

print_results(
    "VALIDATION: OBJECTS WITH num_lidar_pts > 0",
    val_visible_results,
    class_names
)