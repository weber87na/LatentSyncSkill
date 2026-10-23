#!/usr/bin/env python3
"""Colab-only bootstrap and LatentSync worker; media helpers also run without ML.

The adapter deliberately specifies dtype instead of copying upstream's T4->fp32
heuristic. Upstream files are never patched. Every upstream invocation has an
isolated ASCII working directory because upstream uses unquoted shell commands
and recursively replaces its own ``temp`` directory.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import uuid
import zipfile

from config import resolve_options

UPSTREAM_URL = "https://github.com/bytedance/LatentSync.git"
UPSTREAM_SHA = "a229c3948406bc2cf6eaf4873e662e70c6a04746"
VAE_REPO = "stabilityai/sd-vae-ft-mse"
VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
MODELS = {
    "1.5": {"repo": "ByteDance/LatentSync-1.5", "revision": "32a20d29aead0498e3e885e90dbbe8027da1b61b",
            "config": "configs/unet/stage2.yaml", "resolution": 256, "minimum_gib": 8},
    "1.6": {"repo": "ByteDance/LatentSync-1.6", "revision": "c42c7e6c8e9c213626389fa7d9a3c444b8536353",
            "config": "configs/unet/stage2_512.yaml", "resolution": 512, "minimum_gib": 18},
}
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
FACE_FILES = ("det_10g.onnx", "2d106det.onnx")
# All environment/bootstrap pins live here. Increment INSTALLER_REVISION if the
# installation algorithm changes without changing a version below.
INSTALLER_REVISION = 2
PYTHON_VERSION = "3.10"
UV_REQUIREMENT = "uv==0.8.22"
BUILD_REQUIREMENTS = [
    "pip==25.0.1", "setuptools==75.8.0", "wheel==0.45.1",
    "Cython==3.0.12", "numpy==1.26.4",
]
TORCH_REQUIREMENTS = ["torch==2.5.1", "torchvision==0.20.1"]
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"
INSIGHTFACE_REQUIREMENTS = ["insightface==0.7.3"]
INSIGHTFACE_SUPPORT_REQUIREMENTS = [
    "tqdm==4.67.1", "requests==2.32.3", "easydict==1.13", "prettytable==3.12.0",
]
# Inference-only requirements. Training/evaluation/Gradio are intentionally not
# installed. Pin OpenCV/albumentations to avoid silently reintroducing NumPy 2.
REQUIREMENTS = [
    "diffusers==0.32.2", "transformers==4.48.0", "decord==0.6.0",
    "accelerate==0.26.1", "einops==0.7.0", "omegaconf==2.3.0",
    "opencv-python-headless==4.9.0.80", "ffmpeg-python==0.2.0",
    "imageio==2.31.1", "imageio-ffmpeg==0.5.1", "huggingface-hub==0.30.2",
    "kornia==0.8.0", "onnxruntime-gpu==1.21.0",
    "onnx==1.17.0", "albumentations==1.4.3", "scipy==1.12.0",
    "scikit-image==0.22.0", "scikit-learn==1.4.2", "matplotlib==3.8.4",
    "soundfile==0.12.1", "more-itertools==10.2.0", "DeepCache==0.1.1",
    "protobuf==4.25.6", "packaging==24.2", "safetensors==0.5.2",
    "tokenizers==0.21.0", "python-dateutil==2.9.0.post0", "Pillow==10.4.0",
]


def environment_manifest():
    """Complete direct installation contract, including bootstrap and indexes."""
    return {
        "installer_revision": INSTALLER_REVISION,
        "python": PYTHON_VERSION,
        "uv": UV_REQUIREMENT,
        "build": BUILD_REQUIREMENTS,
        "torch": TORCH_REQUIREMENTS,
        "torch_index": TORCH_INDEX_URL,
        "inference": REQUIREMENTS,
        "insightface": INSIGHTFACE_REQUIREMENTS,
        "insightface_support": INSIGHTFACE_SUPPORT_REQUIREMENTS,
    }


def environment_fingerprint():
    encoded = json.dumps(environment_manifest(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


ENV_VERSION = environment_fingerprint()


def run_cmd(args, *, cwd=None, env=None, capture=False):
    """Never pass user input through a shell; propagate every process failure."""
    args = [str(arg) for arg in args]
    if not capture:
        print("+ " + " ".join(args[:6]) + (" ..." if len(args) > 6 else ""), flush=True)
    result = subprocess.run(args, cwd=cwd, env=env, check=True, text=True,
                            encoding="utf-8", errors="replace", capture_output=capture)
    return result.stdout if capture else result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path):
    """Return stream metadata; never use container duration when stream is known."""
    data = json.loads(run_cmd(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                               "-of", "json", path], capture=True))
    streams = data.get("streams", [])
    result = {"path": str(path), "streams": streams}
    container_duration = float(data.get("format", {}).get("duration", 0) or 0)
    for kind in ("video", "audio"):
        stream = next((s for s in streams if s.get("codec_type") == kind), None)
        if stream:
            result[kind] = stream
            result[kind + "_duration"] = float(stream.get("duration", container_duration) or 0)
    return result


def select_duration(video_duration, audio_duration, options):
    video_left = video_duration - options["video_start"]
    raw_audio_left = audio_duration - options["audio_start"]
    audio_left = raw_audio_left + options["audio_offset"]
    if video_left <= 0 or raw_audio_left <= 0 or audio_left <= 0:
        raise ValueError("Start time / negative audio offset removes the entire video or audio.")
    mode = options["length_mode"]
    duration = {"shortest": min(video_left, audio_left), "video": video_left,
                "audio": audio_left}[mode]
    if options["duration"] is not None:
        duration = min(duration, options["duration"])
    # All cut positions use exact 25 FPS frame boundaries. Do not generate a
    # partial frame or let rounding extend the requested source duration.
    frames = int(math.floor(duration * 25 + 1e-7))
    if frames < 16:
        raise ValueError("The selected duration must contain at least 0.64 seconds (16 frames at 25 FPS).")
    return frames / 25


def segment_ranges(duration, segment_seconds):
    total = int(round(duration * 25))
    chunk = max(1, int(math.floor(segment_seconds * 25))) if segment_seconds else total
    ranges = [(start / 25, min(chunk, total - start) / 25) for start in range(0, total, chunk)]
    if len(ranges) > 1 and ranges[-1][1] < 0.64:
        tail = ranges.pop()
        previous = ranges.pop()
        ranges.append((previous[0], round(previous[1] + tail[1], 8)))
    return ranges


def prepare_media(video, audio, workspace, options):
    """Trim, normalize and align media without ever modifying source files."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    vmeta, ameta = probe(video), probe(audio)
    if "video" not in vmeta or "audio" not in ameta:
        raise ValueError("Video input needs a video stream; audio input needs an audio stream.")
    duration = select_duration(vmeta["video_duration"], ameta["audio_duration"], options)
    # Stage a trimmed video before looping, so video_start applies to every loop.
    filters = ["fps=25"]
    if options["max_side"]:
        size = options["max_side"]
        filters.append(f"scale=w='trunc(iw*min(1,{size}/max(iw,ih))/2)*2':h='trunc(ih*min(1,{size}/max(iw,ih))/2)*2'")
    else:
        filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    base = workspace / "base.mp4"
    run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-ss", options["video_start"], "-i", video, "-t",
             min(duration, vmeta["video_duration"] - options["video_start"]),
             "-map", "0:v:0", "-an", "-vf", ",".join(filters), "-c:v", "libx264",
             "-crf", "13", "-preset", "fast", "-pix_fmt", "yuv420p", base])
    video_out = workspace / "input.mp4"
    # Explicit forward looping differs from upstream's hidden ping-pong loop.
    run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-stream_loop", "-1", "-i", base, "-frames:v", round(duration * 25), "-map", "0:v:0",
             "-an", "-c:v", "libx264", "-crf", "13", "-preset", "fast", video_out])
    afilters = []
    if options["audio_offset"] < 0:
        afilters += [f"atrim=start={-options['audio_offset']}", "asetpts=PTS-STARTPTS"]
    elif options["audio_offset"] > 0:
        # Sample units preserve sub-millisecond offsets at the target rate.
        afilters += ["aresample=16000", f"adelay={round(options['audio_offset'] * 16000)}S:all=1"]
    afilters += [f"volume={options['audio_gain_db']}dB", "apad", f"atrim=duration={duration}"]
    audio_out = workspace / "audio.wav"
    run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-ss", options["audio_start"], "-i", audio, "-map", "0:a:0", "-vn",
             "-af", ",".join(afilters), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_out])
    return {"duration": duration, "video": str(video_out), "audio": str(audio_out),
            "source_video": vmeta, "source_audio": ameta,
            "segments": segment_ranges(duration, options["segment_seconds"])}


def validate_output(path, expected_duration=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 256:
        raise RuntimeError("Inference did not produce a nonempty video.")
    metadata = probe(path)
    if "video" not in metadata or "audio" not in metadata:
        raise RuntimeError("Output must contain both video and audio streams.")
    if expected_duration is not None:
        for kind in ("video", "audio"):
            if abs(metadata[kind + "_duration"] - expected_duration) > 0.16:
                raise RuntimeError(f"Output {kind} duration differs from expected duration: {metadata[kind + '_duration']} vs {expected_duration}")
    run_cmd(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", path,
             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
    return metadata


def managed_root(root):
    root = Path(root).expanduser().resolve()
    if sys.platform != "linux" or not Path("/content").is_dir():
        raise RuntimeError("Setup/inference runs inside a Linux Colab runtime, not on your local computer.")
    # Keep mutable environment/locks away from Drive, and enforce safe shell paths.
    if not str(root).startswith("/content/") or not all(c.isascii() and (c.isalnum() or c in "/_-.") for c in str(root)):
        raise ValueError("--root must be an ASCII path below /content (letters/digits/_/-/. only).")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".latentsync-skill-managed"
    if not marker.exists() and any(root.iterdir()):
        raise ValueError("Refusing to adopt a nonempty directory without the LatentSync managed marker.")
    marker.touch(exist_ok=True)
    return root


@contextlib.contextmanager
def root_lock(root):
    import fcntl
    with (root / ".runtime.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another setup/job is using this runtime root. Wait for it to finish.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def runtime_python(root):
    return root / ("venv-" + ENV_VERSION) / "bin" / "python"


def child_env(root):
    env = os.environ.copy()
    env["HF_HOME"] = str(root / "hf-cache")
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["UV_PYTHON_INSTALL_DIR"] = str(root / "python")
    env["UV_CACHE_DIR"] = str(root / "uv-cache")
    env.pop("PYTHONPATH", None)
    return env


def check_dependencies(python, root, env):
    """Check requirements and actually decode H.264 before marking ready.

    Decord 0.6.0's Linux wheel has a stale internal cp36 tag even though its
    filename advertises py3-none. pip >=24.2 reports this metadata defect.
    Only that exact warning is eligible for the functional decode check;
    missing/conflicting requirements and decode failures remain fatal.
    See https://github.com/dmlc/decord/issues/313.
    """
    command = [str(python), "-m", "pip", "check"]
    checked = subprocess.run(command, env=env, text=True, encoding="utf-8",
                             errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    diagnostics = checked.stdout or ""
    print(diagnostics, end="" if diagnostics.endswith("\n") else "\n", flush=True)
    known = "decord 0.6.0 is not supported on this platform"
    lines = [line.strip() for line in diagnostics.splitlines() if line.strip()]
    if checked.returncode and not (checked.returncode == 1 and lines == [known]):
        raise subprocess.CalledProcessError(checked.returncode, command, output=diagnostics)
    with tempfile.TemporaryDirectory(prefix="decode-check-", dir=root) as temporary:
        video = Path(temporary) / "probe.mp4"
        run_cmd(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                 "testsrc2=size=64x64:rate=25:duration=0.64", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", video], env=env)
        smoke = (
            "import platform, sys, decord, importlib.metadata\n"
            "assert sys.platform == 'linux' and platform.machine() == 'x86_64'\n"
            "assert decord.__version__ == '0.6.0'\n"
            + ("assert 'Tag: cp36-cp36m-manylinux2010_x86_64' in importlib.metadata.distribution('decord').read_text('WHEEL').splitlines()\n" if checked.returncode else "") +
            "reader = decord.VideoReader(sys.argv[1], ctx=decord.cpu(0))\n"
            "assert len(reader) == 16\n"
            "frames = reader.get_batch([0, 15]).asnumpy()\n"
            "assert frames.shape == (2, 64, 64, 3) and frames.max() > 0\n"
            "print('LATENTSYNC_DECORD_DECODE_OK', flush=True)\n"
        )
        run_cmd([python, "-c", smoke, video], env=env)
    if checked.returncode:
        print("Decord 0.6.0 wheel metadata warning accepted after successful Linux H.264 decode.", flush=True)


def setup(root, model_version):
    root = managed_root(root)
    with root_lock(root):
        env = child_env(root)
        if not shutil.which("ffmpeg") or not shutil.which("git") or not shutil.which("g++"):
            run_cmd(["apt-get", "update", "-qq"])
            run_cmd(["apt-get", "install", "-y", "ffmpeg", "git", "build-essential", "libgl1", "libglib2.0-0"])
        python = runtime_python(root)
        ready = python.parent.parent / ".dependencies-ready"
        if not ready.exists():
            # A distinct bootstrap directory avoids importing a stale uv after
            # changing its pin. --upgrade also repairs interrupted installations.
            bootstrap = root / ("bootstrap-" + ENV_VERSION)
            run_cmd([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                     "--upgrade", "--target", bootstrap, UV_REQUIREMENT], env=env)
            uv = [sys.executable, "-m", "uv"]
            uv_env = dict(env, PYTHONPATH=str(bootstrap))
            if not python.exists():
                run_cmd(uv + ["venv", "--python", PYTHON_VERSION, "--python-preference", "only-managed",
                              "--seed", python.parent.parent], env=uv_env)
            pip = [python, "-m", "pip", "install", "--disable-pip-version-check"]
            run_cmd(pip + BUILD_REQUIREMENTS, env=env)
            run_cmd(pip + TORCH_REQUIREMENTS + ["--index-url", TORCH_INDEX_URL], env=env)
            # Include build pins again so the resolver cannot upgrade NumPy or
            # setuptools while satisfying a later inference dependency.
            run_cmd(pip + REQUIREMENTS + BUILD_REQUIREMENTS, env=env)
            run_cmd(pip + ["--no-build-isolation", "--no-deps"] + INSIGHTFACE_REQUIREMENTS, env=env)
            # insightface requirements not already provided by the inference set.
            run_cmd(pip + INSIGHTFACE_SUPPORT_REQUIREMENTS, env=env)
            check_dependencies(python, root, env)
            write_json(python.parent.parent / "environment.manifest.json", environment_manifest())
            ready.write_text(ENV_VERSION, encoding="ascii")
        repo = root / "upstream"
        if not (repo / ".git").is_dir():
            if repo.exists():
                raise RuntimeError("Incomplete upstream directory exists. Use a fresh --root to recover.")
            run_cmd(["git", "clone", "--no-checkout", "--filter=blob:none", UPSTREAM_URL, repo])
            run_cmd(["git", "-C", repo, "checkout", "--detach", UPSTREAM_SHA])
        revision = run_cmd(["git", "-C", repo, "rev-parse", "HEAD"], capture=True).strip()
        if revision != UPSTREAM_SHA or run_cmd(["git", "-C", repo, "status", "--porcelain"], capture=True).strip():
            raise RuntimeError("Upstream checkout changed; use a fresh --root instead of overwriting edits.")
        run_cmd([python, Path(__file__).resolve(), "_download", "--root", root,
                 "--model-version", model_version], env=env)
        frozen = run_cmd([python, "-m", "pip", "freeze"], env=env, capture=True)
        (root / "environment.freeze.txt").write_text(frozen, encoding="utf-8")
        print("LATENTSYNC_SETUP_OK", flush=True)
    return root


def download_models(root, version):
    # Check the real isolated torch/CUDA combination before downloading ~5 GB
    # of weights. A mismatched GPU is an error, never an implicit downgrade.
    gpu_info(version, "auto")
    from huggingface_hub import hf_hub_download, snapshot_download
    model = MODELS[version]
    folder = root / "models" / version
    for name in ("latentsync_unet.pt", "whisper/tiny.pt"):
        hf_hub_download(model["repo"], name, revision=model["revision"], local_dir=str(folder))
    snapshot_download(VAE_REPO, revision=VAE_REVISION, local_dir=str(root / "vae"),
                      allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
    face_dir = folder / "auxiliary" / "models" / "buffalo_l"
    if not all((face_dir / name).is_file() and (face_dir / name).stat().st_size > 1024 * 1024 for name in FACE_FILES):
        archive = root / "buffalo_l.zip"
        if not archive.exists():
            temporary = archive.with_suffix(".part")
            urllib.request.urlretrieve(BUFFALO_URL, temporary)
            temporary.replace(archive)
        face_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            # Extract only known basenames, never trust archive traversal paths.
            for name in FACE_FILES:
                member = next((m for m in handle.namelist() if Path(m).name == name), None)
                if member is None:
                    raise RuntimeError(f"Official face archive is missing {name}.")
                with handle.open(member) as source, (face_dir / name).open("wb") as dest:
                    shutil.copyfileobj(source, dest)
    write_json(folder / "provenance.json", {"upstream": UPSTREAM_SHA, "model": model,
               "vae": {"repo": VAE_REPO, "revision": VAE_REVISION},
               "face_source": BUFFALO_URL,
               "face_sha256": {name: sha256(face_dir / name) for name in FACE_FILES}})


def gpu_info(version, precision):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU. In Colab choose Runtime > Change runtime type > GPU.")
    free, total = torch.cuda.mem_get_info()
    minimum = MODELS[version]["minimum_gib"]
    if free / 1024 ** 3 < minimum:
        raise RuntimeError(f"LatentSync {version} needs at least {minimum} GiB free VRAM; found {free / 1024 ** 3:.1f}. Select 1.5 for T4 or a larger GPU for 1.6.")
    capability = torch.cuda.get_device_capability()
    dtype = precision if precision != "auto" else ("fp16" if capability[0] >= 7 else "fp32")
    return {"name": torch.cuda.get_device_name(), "free_gib": free / 1024 ** 3,
            "total_gib": total / 1024 ** 3, "capability": list(capability), "precision": dtype,
            "torch": torch.__version__, "cuda": torch.version.cuda}


def make_stage(directory, root, version):
    directory.mkdir(parents=True, exist_ok=True)
    for name, target in {"configs": root / "upstream" / "configs",
                         "latentsync": root / "upstream" / "latentsync",
                         "checkpoints": root / "models" / version}.items():
        (directory / name).symlink_to(target, target_is_directory=True)


def infer_segments(root, workspace, prepared, options, gpu):
    # Import torch first: ONNX Runtime then finds CUDA/cuDNN shipped with torch.
    import torch
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from accelerate.utils import set_seed
    sys.path.insert(0, str(root / "upstream"))
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature
    version = options["model_version"]
    model = MODELS[version]
    config = OmegaConf.load(root / "upstream" / model["config"])
    dtype = torch.float16 if gpu["precision"] == "fp16" else torch.float32
    audio_encoder = Audio2Feature(model_path=str(root / "models" / version / "whisper" / "tiny.pt"),
                                 device="cuda", num_frames=16, audio_feat_length=config.data.audio_feat_length)
    vae = AutoencoderKL.from_pretrained(str(root / "vae"), torch_dtype=dtype, local_files_only=True)
    vae.config.scaling_factor, vae.config.shift_factor = 0.18215, 0
    unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(config.model),
                str(root / "models" / version / "latentsync_unet.pt"), device="cpu")
    pipeline = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet.to(dtype=dtype),
                scheduler=DDIMScheduler.from_pretrained(str(root / "upstream" / "configs"))).to("cuda")
    if options["vae_slicing"]:
        pipeline.enable_vae_slicing()
    helper = None
    if options["enable_deepcache"]:
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
    # Generate and report the concrete random seed even for seed=-1.
    seed = options["seed"] if options["seed"] >= 0 else int.from_bytes(os.urandom(4), "big")
    outputs = []
    previous_cwd = Path.cwd()
    try:
        for index, (start, duration) in enumerate(prepared["segments"]):
            stage = workspace / f"segment_{index:05d}"
            make_stage(stage, root, version)
            run_cmd(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", start,
                     "-i", prepared["video"], "-t", duration, "-an", "-c:v", "libx264",
                     "-crf", "13", "-preset", "fast", stage / "input.mp4"])
            run_cmd(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", start,
                     "-i", prepared["audio"], "-t", duration, "-c:a", "pcm_s16le", stage / "audio.wav"])
            os.chdir(stage)
            set_seed((seed + index) % (2 ** 32))
            pipeline(video_path="input.mp4", audio_path="audio.wav", video_out_path="generated.mp4",
                     num_frames=16, num_inference_steps=options["inference_steps"],
                     guidance_scale=options["guidance_scale"], weight_dtype=dtype,
                     width=model["resolution"], height=model["resolution"],
                     mask_image_path="latentsync/utils/mask.png", temp_dir="temp")
            # Upstream does not check its final FFmpeg exit status. This check is
            # mandatory. Audio features may add ~2 frames; trim before concat.
            validate_output(stage / "generated.mp4")
            run_cmd(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", "generated.mp4",
                     "-t", duration, "-an", "-c:v", "libx264", "-crf", "13", "-preset", "fast", "trimmed.mp4"])
            outputs.append(stage / "trimmed.mp4")
            torch.cuda.empty_cache()
    finally:
        os.chdir(previous_cwd)
        if helper is not None:
            helper.disable()
    return outputs, seed


def finalize_media(outputs, prepared, workspace, options):
    concat = workspace / "concat.txt"
    # Paths are generated by us, contain no quote/backslash, and are under /content.
    concat.write_text("".join(f"file '{path.as_posix()}'\n" for path in outputs), encoding="utf-8")
    final = workspace / "final.mp4"
    run_cmd(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
             "-i", concat, "-i", prepared["audio"], "-map", "0:v:0", "-map", "1:a:0",
             "-t", prepared["duration"], "-r", "25", "-c:v", "libx264", "-crf", options["crf"],
             "-preset", options["preset"], "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-b:a", options["audio_bitrate"], "-movflags", "+faststart", final])
    return final


def run_job(root, job_path):
    root = managed_root(root)
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    options = resolve_options(job.get("options", {}))
    video, audio, output = (Path(job[key]).expanduser().resolve() for key in ("video", "audio", "output"))
    if not video.is_file() or not audio.is_file():
        raise FileNotFoundError("Video/audio input does not exist in this Colab runtime.")
    if output in (video, audio) or output.exists():
        raise ValueError("Output already exists or points at an input; choose a new filename.")
    with root_lock(root):
        gpu = gpu_info(options["model_version"], options["precision"])
        workspace = root / "work" / ("job_" + uuid.uuid4().hex)
        workspace.mkdir(parents=True)
        started = time.time()
        report = {"status": "running", "job_id": job.get("id"), "options": options,
                  "gpu": gpu, "upstream_sha": UPSTREAM_SHA, "model": MODELS[options["model_version"]],
                  "workspace": str(workspace), "input_video": str(video), "input_audio": str(audio)}
        report_path = Path(str(output) + ".report.json")
        try:
            prepared = prepare_media(video, audio, workspace, options)
            outputs, seed = infer_segments(root, workspace, prepared, options, gpu)
            final = finalize_media(outputs, prepared, workspace, options)
            metadata = validate_output(final, prepared["duration"])
            output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation avoids overwriting a file that appeared mid-run.
            with final.open("rb") as source, output.open("xb") as dest:
                shutil.copyfileobj(source, dest)
            report.update(status="completed", effective_seed=seed,
                          duration=prepared["duration"], segments=prepared["segments"],
                          output=str(output), output_sha256=sha256(output), output_metadata=metadata,
                          elapsed_seconds=round(time.time() - started, 3),
                          provenance=json.loads((root / "models" / options["model_version"] / "provenance.json").read_text()))
            write_json(report_path, report)
            print("LATENTSYNC_JOB_OK:" + str(job.get("id", "completed")), flush=True)
        except BaseException as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                          elapsed_seconds=round(time.time() - started, 3))
            write_json(report_path, report)
            print(f"Job failed; intermediates retained at {workspace}", file=sys.stderr, flush=True)
            raise
        else:
            if not options["keep_intermediates"]:
                # Only delete the uniquely generated child, never arbitrary input.
                if workspace.parent == root / "work" and workspace.name.startswith("job_"):
                    shutil.rmtree(workspace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "run", "_download", "_worker"):
        command = commands.add_parser(name)
        command.add_argument("--root", default="/content/latentsync-skill")
        if name in ("setup", "_download"):
            command.add_argument("--model-version", choices=list(MODELS), default="1.6")
        else:
            command.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    if args.command == "setup":
        setup(args.root, args.model_version)
    elif args.command == "_download":
        download_models(Path(args.root), args.model_version)
    elif args.command == "_worker":
        run_job(args.root, args.job)
    else:
        job_path = Path(args.job).resolve()
        job = json.loads(job_path.read_text(encoding="utf-8"))
        options = resolve_options(job.get("options", {}))
        root = setup(args.root, options["model_version"])
        run_cmd([runtime_python(root), Path(__file__).resolve(), "_worker", "--root", root,
                 "--job", job_path], env=child_env(root))


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        traceback.print_exc()
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
