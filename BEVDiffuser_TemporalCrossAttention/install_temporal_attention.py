#!/usr/bin/env python3
"""Safely install the optional BEVDiffuser temporal-attention experiment.

Usage:
  python install_temporal_attention.py --repo ~/BEVDiffuser --apply

The installer adds one module and makes small, anchored edits to the trainer,
evaluator, and launchers. It creates a timestamped backup before writing.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


PACKAGE = Path(__file__).resolve().parent
REL = Path("BEVFormer/projects/bevdiffuser")
MARKER = "# TEMPORAL CONFIDENCE ATTENTION integration v1"
IMPORT = """from projects.bevdiffuser.temporal_cross_attention import (
    add_temporal_arguments, restore_temporal_options,
    configure_temporal_cross_attention, load_pretrained_with_temporal,
    register_temporal_metadata, get_temporal_runtime, set_temporal_trainable)"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise ValueError("Expected one {} anchor, found {}: {}".format(label, count, old[:100]))
    return text.replace(old, new, 1)


def _add_import(text):
    if "projects.bevdiffuser.temporal_cross_attention" in text:
        raise ValueError("Unmarked temporal-attention import found; inspect the existing edit")
    matches = list(re.finditer(r"^from model_utils import .+$", text, re.M))
    if len(matches) != 1:
        raise ValueError("Could not identify the single model_utils import")
    end = matches[0].end()
    return text[:end] + "\n" + IMPORT + text[end:]


def _add_parser(text):
    if "add_temporal_arguments(parser)" in text:
        raise ValueError("Unmarked temporal-attention parser edit found")
    matches = list(re.finditer(
        r"^(\s*)parser\s*=\s*argparse\.ArgumentParser\([^\n]*\)\s*$", text, re.M
    ))
    if len(matches) != 1:
        raise ValueError("Could not identify one argparse parser construction")
    m = matches[0]
    indent = m.group(1)
    return text[:m.end()] + "\n" + indent + "add_temporal_arguments(parser)" + text[m.end():]


def _add_restore(text, script):
    if script == "train_bev_diffuser.py":
        pattern = r"(?m)^(\s*)args\s*=\s*parse_args\(\)"
        replacement = (
            r"\1args = parse_args()\n"
            r"\1restore_temporal_options("
            r"args, args.resume_from_checkpoint or args.pretrained_unet_checkpoint, "
            r"exact=bool(args.resume_from_checkpoint))"
        )
    else:
        pattern = r"(?m)^(\s*)args\s*=\s*parse_args\(\)"
        replacement = (
            r"\1args = parse_args()\n"
            r"\1restore_temporal_options(args, args.checkpoint_dir, exact=True)"
        )
    if len(re.findall(pattern, text)) != 1:
        raise ValueError("Could not locate the single parse_args call in " + script)
    return re.sub(pattern, replacement, text, count=1)


def _insert_unet_config(text, script):
    if "configure_temporal_cross_attention(unet, bev_model, bev_cfg, args)" in text:
        raise ValueError("Unmarked temporal U-Net configuration found")
    builds = list(re.finditer(r"(?m)^([ \t]*)unet\s*=\s*build_unet\(bev_cfg\.unet\)\s*$", text))
    if len(builds) != 1:
        raise ValueError("Could not locate one U-Net construction in " + script)
    build = builds[0]
    indent = build.group(1)
    line_end = text.find("\n", build.end())
    if line_end < 0:
        line_end = len(text)
    else:
        line_end += 1
    # If the RBF adapter is already present, install temporal attention after
    # that setup call so the temporal hook runs after the bottleneck hook.
    window_end = min(len(text), line_end + 1600)
    window = text[line_end:window_end]
    config_match = re.search(
        r"(?m)^([ \t]*)(?:\w+\s*=\s*)?configure_unet\(unet,\s*args\)\s*$", window
    )
    if config_match:
        insertion = line_end + config_match.end()
        setup_indent = config_match.group(1)
        addition = (
            "\n" + setup_indent + "temporal_runtime = configure_temporal_cross_attention("
            "unet, bev_model, bev_cfg, args)"
        )
    else:
        addition = indent + "temporal_runtime = configure_temporal_cross_attention("
        addition += "unet, bev_model, bev_cfg, args)\n"
        return text[:line_end] + addition + text[line_end:]
    return text[:insertion] + addition + text[insertion:]


def _wrap_call(text, pattern, replacement, label, required=False):
    updated, count = re.subn(pattern, replacement, text)
    if required and count == 0:
        raise ValueError("Could not locate required U-Net load call: " + label)
    return updated


def _wrap_loaders(text, script):
    if script == "train_bev_diffuser.py":
        replacements = [
            (
                r"load_shape_unet\(unet,\s*args\.pretrained_unet_checkpoint,\s*allow_baseline=True\)",
                "load_pretrained_with_temporal(unet, args.pretrained_unet_checkpoint, "
                "lambda: load_shape_unet(unet, args.pretrained_unet_checkpoint, allow_baseline=True))",
                "shape-aware pretrained load",
            ),
            (
                r"strict_load_unet\(unet,\s*args\.pretrained_unet_checkpoint\)",
                "load_pretrained_with_temporal(unet, args.pretrained_unet_checkpoint, "
                "lambda: strict_load_unet(unet, args.pretrained_unet_checkpoint))",
                "RAF pretrained load",
            ),
            (
                r"unet\.from_pretrained\(args\.pretrained_unet_checkpoint,\s*subfolder=\"unet\"\)",
                "load_pretrained_with_temporal(unet, args.pretrained_unet_checkpoint, "
                "lambda: unet.from_pretrained(args.pretrained_unet_checkpoint, subfolder=\"unet\"))",
                "baseline pretrained load",
            ),
            (
                r"load_shape_unet\(model,\s*input_dir\)",
                "load_pretrained_with_temporal(model, input_dir, lambda: load_shape_unet(model, input_dir), required=True)",
                "shape-aware resume load",
            ),
            (
                r"strict_load_unet\(model,\s*input_dir\)",
                "load_pretrained_with_temporal(model, input_dir, lambda: strict_load_unet(model, input_dir), required=True)",
                "RAF resume load",
            ),
            (
                r"model\.from_pretrained\(os\.path\.join\(input_dir,\s*\"unet\"\)\)",
                "load_pretrained_with_temporal(model, input_dir, lambda: model.from_pretrained(os.path.join(input_dir, \"unet\")), required=True)",
                "baseline resume load",
            ),
        ]
    else:
        replacements = [
            (
                r"load_shape_unet\(unet,\s*args\.checkpoint_dir\)",
                "load_pretrained_with_temporal(unet, args.checkpoint_dir, lambda: load_shape_unet(unet, args.checkpoint_dir), required=True)",
                "shape-aware evaluation load",
            ),
            (
                r"strict_load_unet\(unet,\s*args\.checkpoint_dir\)",
                "load_pretrained_with_temporal(unet, args.checkpoint_dir, lambda: strict_load_unet(unet, args.checkpoint_dir), required=True)",
                "RAF evaluation load",
            ),
            (
                r"unet\.from_pretrained\(args\.checkpoint_dir,\s*subfolder=\"unet\"\)",
                "load_pretrained_with_temporal(unet, args.checkpoint_dir, lambda: unet.from_pretrained(args.checkpoint_dir, subfolder=\"unet\"), required=True)",
                "baseline evaluation load",
            ),
        ]
    for pattern, replacement, label in replacements:
        text = _wrap_call(text, pattern, replacement, label)
    return text


def _insert_train_runtime(text):
    if "temporal_runtime.begin_bev_call()" in text:
        raise ValueError("Unmarked temporal runtime hooks found in the trainer")
    lines = text.splitlines(True)
    only = [i for i, line in enumerate(lines) if "only_bev=True" in line]
    reshape = [i for i, line in enumerate(lines) if "latents = latents.reshape(-1, bev_cfg.bev_h_" in line]
    if not only or len(reshape) != 1 or only[0] > reshape[0] or any(i > reshape[0] for i in only):
        raise ValueError("Could not identify the trainer's BEV creation block")
    only_index, reshape_index = only[0], reshape[0]
    start = only_index
    lower = max(0, only_index - 8)
    # In optional RAF variants, clear context before the RAF branch. Otherwise
    # clear it directly before the no-grad BEV call.
    raf_parent = next((i for i in range(only_index - 1, lower - 1, -1)
                       if lines[i].lstrip().startswith("if ") and
                       ("raf_controller" in lines[i] or "use_raf" in lines[i])), None)
    no_grad = next((i for i in range(only_index - 1, lower - 1, -1)
                    if lines[i].lstrip().startswith("with torch.no_grad()")), None)
    if raf_parent is not None:
        start = raf_parent
    elif no_grad is not None:
        start = no_grad
    indent = re.match(r"\s*", lines[start]).group(0)
    lines.insert(start, indent + "if temporal_runtime is not None:\n")
    lines.insert(start + 1, indent + "    temporal_runtime.begin_bev_call()\n")
    reshape_index += 2
    permute = next((i for i in range(start, min(len(lines), reshape_index + 10))
                    if "latents = latents.permute(0, 3, 1, 2)" in lines[i]), None)
    if permute is None:
        raise ValueError("Could not locate the trainer BEV-to-U-Net layout conversion")
    indent = re.match(r"\s*", lines[permute]).group(0)
    lines.insert(permute + 1, indent + "if temporal_runtime is not None:\n")
    lines.insert(permute + 2, indent + "    temporal_runtime.use_captured_context(latents)\n")
    # Activation-checkpointed middle blocks may be recomputed during backward,
    # so retain history until backward has completed.
    backward = [i for i, line in enumerate(lines) if "accelerator.backward(loss)" in line]
    if len(backward) != 1:
        raise ValueError("Could not locate the trainer backward call")
    i = backward[0]
    indent = re.match(r"\s*", lines[i]).group(0)
    lines.insert(i + 1, indent + "if temporal_runtime is not None:\n")
    lines.insert(i + 2, indent + "    temporal_runtime.clear_context()\n")
    return "".join(lines)


def _insert_eval_runtime(text):
    if "temporal_runtime = get_temporal_runtime(unet)" in text:
        raise ValueError("Unmarked temporal runtime hooks found in the evaluator")
    marker = "    for step, batch in enumerate(dataloader):"
    if text.count(marker) != 1:
        raise ValueError("Could not locate the evaluation dataloader loop")
    text = _once(
        text, marker,
        "    temporal_runtime = get_temporal_runtime(unet)\n\n" + marker,
        "evaluation loop",
    )
    lines = text.splitlines(True)
    only = [i for i, line in enumerate(lines) if "only_bev=True" in line]
    permute = [i for i, line in enumerate(lines) if "latents = latents.permute(0, 3, 1, 2)" in line]
    det = [i for i, line in enumerate(lines) if "det_result = bev_model(return_loss=False, only_bev=False, given_bev=latents" in line]
    if not only or len(permute) != 1 or len(det) != 1 or any(i > permute[0] for i in only):
        raise ValueError("Could not identify evaluation BEV, denoising, and detection calls")
    only_index = only[0]
    start = only_index
    lower = max(0, only_index - 8)
    raf_parent = next((i for i in range(only_index - 1, lower - 1, -1)
                       if lines[i].lstrip().startswith("if ") and "raf_controller" in lines[i]), None)
    no_grad = next((i for i in range(only_index - 1, lower - 1, -1)
                    if lines[i].lstrip().startswith("with torch.no_grad()")), None)
    if raf_parent is not None:
        start = raf_parent
    elif no_grad is not None:
        start = no_grad
    indent = re.match(r"\s*", lines[start]).group(0)
    lines.insert(start, indent + "if temporal_runtime is not None:\n")
    lines.insert(start + 1, indent + "    temporal_runtime.begin_bev_call()\n")
    permute_index = permute[0] + (2 if start <= permute[0] else 0)
    indent = re.match(r"\s*", lines[permute_index]).group(0)
    lines.insert(permute_index + 1, indent + "if temporal_runtime is not None:\n")
    lines.insert(permute_index + 2, indent + "    temporal_runtime.use_captured_context(latents)\n")
    det_index = det[0] + 4
    indent = re.match(r"\s*", lines[det_index]).group(0)
    lines.insert(det_index + 1, indent + "if temporal_runtime is not None:\n")
    lines.insert(det_index + 2, indent + "    temporal_runtime.clear_context()\n")
    return "".join(lines)


def patch_python(text, script):
    if MARKER in text:
        return text
    if "temporal_cross_attention" in text:
        raise ValueError("Unmarked temporal-attention code found in " + script)
    text = _add_import(text)
    text = _add_parser(text)
    text = _add_restore(text, script)
    text = _insert_unet_config(text, script)
    text = _wrap_loaders(text, script)
    if script == "train_bev_diffuser.py":
        text = _insert_train_runtime(text)
        run_setup = (
            "    if args.use_temporal_cross_attention:\n"
            "        if args.output_dir and not args.output_dir.rstrip('/').endswith('-temporal'):\n"
            "            args.output_dir = args.output_dir.rstrip('/') + '-temporal'\n"
            "        if not (args.tracker_run_name or '').endswith('-temporal'):\n"
            "            args.tracker_run_name = (args.tracker_run_name or 'BEVDiffuser') + '-temporal'\n\n"
            "    bev_cfg = Config.fromfile(args.bev_config)"
        )
        text = _once(text, "    bev_cfg = Config.fromfile(args.bev_config)", run_setup, "temporal run isolation")
        anchor = '    assert version.parse(accelerate.__version__) >= version.parse("0.16.0")'
        text = _once(
            text, anchor,
            "    set_temporal_trainable(unet, args)\n\n" + anchor,
            "training trainability setup",
        )
        gradient_anchor = "    if args.gradient_checkpointing:"
        text = _once(
            text, gradient_anchor,
            "    register_temporal_metadata(accelerator, args)\n\n" + gradient_anchor,
            "checkpoint metadata hook",
        )
    else:
        text = _insert_eval_runtime(text)
    text = MARKER + "\n" + text
    ast.parse(text, filename=script)
    return text


def _forward_shell_args(text, is_train):
    if MARKER in text:
        return text
    if is_train:
        if not re.search(r"^\s*shift(?:\s|$)", text, re.M):
            if not re.search(r"^GPUS=\$1\s*$", text, re.M):
                raise ValueError("Cannot locate the training GPU argument in train.sh")
            text = re.sub(r"^(GPUS=\$1\s*)$", r"\1\nshift", text, count=1, flags=re.M)
        gpu_var = "GPUS"
    else:
        argument_parser = re.search(
            r"(?m)^\s*if\s+\[\[\s*\$\{1:-\}\s*=~\s*\^\[0-9\]\+\$\s*\]\];\s*then\s*(\w+)\s*=\$1;\s*shift;\s*fi",
            text,
        )
        if not argument_parser:
            shebang = re.match(r"^#![^\n]*\n", text)
            prefix = "TEMPORAL_TEST_GPUS=1\nif [[ ${1:-} =~ ^[0-9]+$ ]]; then TEMPORAL_TEST_GPUS=$1; shift; fi\n"
            text = (shebang.group(0) if shebang else "") + prefix + text[len(shebang.group(0)) if shebang else 0:]
            gpu_var = "TEMPORAL_TEST_GPUS"
        else:
            gpu_var = argument_parser.group(1)
        if "--nproc_per_node" not in text:
            pattern = r"(python\s+-m\s+torch\.distributed\.launch\s+--master_port=?(?:\s+)?\$?\w+\s+)"
            if re.search(pattern, text):
                text = re.sub(pattern, r'\1--nproc_per_node="$' + gpu_var + r'" ', text, count=1)
            else:
                pattern = r"(python\s+-m\s+torch\.distributed\.launch\s+--master_port\s+[^\s]+\s+)"
                if not re.search(pattern, text):
                    raise ValueError("Cannot locate test launcher port for GPU-count insertion")
                text = re.sub(pattern, r'\1--nproc_per_node="$' + gpu_var + r'" ', text, count=1)
        elif not argument_parser:
            # Preserve the existing launcher form while making its positional
            # GPU count available and consuming it before Python sees "$@".
            text = re.sub(
                r"--nproc_per_node=(?:\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|[0-9]+)",
                '--nproc_per_node="$' + gpu_var + '"',
                text,
                count=1,
            )
    if '"$@"' not in text and '"${@}"' not in text:
        lines = text.splitlines()
        start = next((i for i, line in enumerate(lines)
                      if "train_bev_diffuser.py" in line or "test_bev_diffuser.py" in line), None)
        if start is None:
            raise ValueError("Cannot find Python training/evaluation command in launcher")
        # Append after the final live argument, before any commented examples.
        active = [i for i in range(start, len(lines))
                  if lines[i].strip() and not lines[i].lstrip().startswith("#")]
        if not active:
            raise ValueError("Python launcher command has no arguments")
        end = active[-1]
        if not lines[end].rstrip().endswith("\\"):
            lines[end] = lines[end].rstrip() + " \\"  # continue into forwarded CLI arguments
        lines.insert(end + 1, '    "$@"')
        text = "\n".join(lines) + "\n"
    return "#!/usr/bin/env bash\n" + MARKER + "\n" + re.sub(r"^#!/[^\n]*\n", "", text, count=1)


def patch_shell(text, is_train):
    return _forward_shell_args(text, is_train)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    if not (repo / REL).is_dir():
        raise SystemExit("Expected repository with BEVFormer/projects/bevdiffuser/")
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=repo, text=True).strip()).resolve()
    if root != repo:
        raise SystemExit("--repo must be the Git repository root")
    for state in ("rebase-merge", "rebase-apply", "MERGE_HEAD", "CHERRY_PICK_HEAD"):
        git_path = Path(subprocess.check_output(["git", "rev-parse", "--git-path", state], cwd=repo, text=True).strip())
        if not git_path.is_absolute():
            git_path = repo / git_path
        if git_path.exists():
            raise SystemExit("Finish the active Git operation before installing: " + state)
    if subprocess.check_output(["git", "ls-files", "-u"], cwd=repo, text=True).strip():
        raise SystemExit("Resolve unmerged files before installing")

    manifest = json.loads((PACKAGE / "package_manifest.json").read_text())
    for item in manifest["package_files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit("Invalid package path")
        source = PACKAGE / relative
        if source.is_symlink() or not source.is_file() or sha(source.read_bytes()) != item["sha256"]:
            raise SystemExit("Package checksum mismatch: " + str(relative))

    writes = []
    try:
        for name in ("train_bev_diffuser.py", "test_bev_diffuser.py", "train.sh", "test.sh"):
            path = repo / REL / name
            if path.is_symlink() or not path.is_file() or repo not in path.resolve().parents:
                raise ValueError("Expected regular repository file: " + str(path))
            old = path.read_text()
            new = patch_shell(old, name == "train.sh") if name.endswith(".sh") else patch_python(old, name)
            mode = path.stat().st_mode & 0o777
            writes.append((path, new.encode(), mode))
        for relative_text in manifest["install_files"]:
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid install path")
            destination = repo / relative
            source = PACKAGE / "files" / relative
            if destination.is_symlink() or repo not in destination.resolve().parents:
                raise ValueError("Install path escapes the repository: " + str(relative))
            if destination.exists() and (not destination.is_file() or destination.read_bytes() != source.read_bytes()):
                raise ValueError("Existing file differs; compare before replacing: " + str(relative))
            writes.append((destination, source.read_bytes(), 0o644))
        for path, content, _mode in writes:
            if path.suffix == ".py":
                ast.parse(content.decode("utf-8"), filename=str(path))
            elif path.suffix == ".sh":
                result = subprocess.run(["bash", "-n"], input=content, capture_output=True)
                if result.returncode:
                    raise ValueError("Invalid shell syntax in " + str(path) + ": " + result.stderr.decode())
    except (ValueError, SyntaxError, UnicodeError) as error:
        raise SystemExit(str(error) + "\nNothing installed; existing files are unchanged.")

    changed = [(p, data, mode) for p, data, mode in writes
               if not p.exists() or p.read_bytes() != data or (p.stat().st_mode & 0o777) != mode]
    for path, _content, _mode in changed:
        print("INSTALL " + str(path))
    if not args.apply:
        print("Preview only. Add --apply to back up and install.")
        return
    if not changed:
        print("Temporal cross-attention integration is already installed.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = Path(tempfile.mkdtemp(prefix=repo.name + "_before_temporal_attention_" + stamp + "_", dir=repo.parent))
    manifest_rows = []
    for path, _content, _mode in changed:
        relative = path.relative_to(repo)
        manifest_rows.append({"path": str(relative), "existed": path.exists()})
        if path.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, saved)
    (backup / "backup_manifest.json").write_text(json.dumps(manifest_rows, indent=2) + "\n")
    attempted = []
    try:
        for path, content, mode in changed:
            attempted.append(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(mode)
            if sha(path.read_bytes()) != sha(content):
                raise RuntimeError("Installed checksum mismatch: " + str(path))
    except Exception:
        for path in reversed(attempted):
            saved = backup / path.relative_to(repo)
            if saved.exists():
                shutil.copy2(saved, path)
            elif path.is_file():
                path.unlink()
        raise
    print("Backup: " + str(backup))
    print("Installed. Train with: bash train.sh 1 --use_temporal_cross_attention")
    print("Evaluate with: bash test.sh 1 (temporal settings restore from the checkpoint)")


if __name__ == "__main__":
    main()
