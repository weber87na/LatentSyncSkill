"""Shared, dependency-free LatentSync job options for local and Colab execution."""
from __future__ import annotations

import argparse
import math

DEFAULTS = {
    "model_version": "1.6",
    "inference_steps": 20,
    "guidance_scale": 1.5,
    "seed": 1247,
    "precision": "auto",
    "enable_deepcache": False,
    "video_start": 0.0,
    "audio_start": 0.0,
    "duration": None,
    "length_mode": "shortest",
    "audio_offset": 0.0,
    "audio_gain_db": 0.0,
    "max_side": 1280,
    "segment_seconds": 0.0,
    "crf": 18,
    "preset": "medium",
    "audio_bitrate": "192k",
    "vae_slicing": False,
    "keep_intermediates": False,
}

PROFILES = {
    "preview": {"model_version": "1.6", "inference_steps": 10},
    "balanced": {"model_version": "1.6", "inference_steps": 20},
    "quality": {"model_version": "1.6", "inference_steps": 30},
}

CHOICES = {
    "model_version": ("1.6", "1.5"),
    "precision": ("auto", "fp16", "fp32"),
    "length_mode": ("shortest", "audio", "video"),
    "preset": ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"),
    "audio_bitrate": ("96k", "128k", "160k", "192k", "256k", "320k"),
}

HELP = {
    "model_version": "Matched upstream checkpoint/config version (default 1.6; never auto-downgraded).",
    "inference_steps": "Diffusion steps 1..100; usual 20..50, preview profile 10.",
    "guidance_scale": "Audio guidance 1..5; upstream recommends 1..3 (default 1.5).",
    "seed": "Random seed 0..4294967295; -1 chooses a fresh seed and records it.",
    "precision": "auto uses fp16 on supported CUDA GPUs; fp32 needs more VRAM.",
    "enable_deepcache": "Enable upstream DeepCache acceleration; may change output quality.",
    "video_start": "Trim this many seconds from the source video (>=0).",
    "audio_start": "Trim this many seconds from the source audio (>=0).",
    "duration": "Maximum output seconds after alignment; omitted uses available media.",
    "length_mode": "shortest: trim both; audio: loop video to audio; video: pad/trim audio.",
    "audio_offset": "Seconds: positive delays audio with silence; negative trims more audio.",
    "audio_gain_db": "Gain applied to replacement audio, -60..24 dB; 0 preserves level.",
    "max_side": "Maximum input long edge (default1280); 0 preserves; otherwise even256..4096.",
    "segment_seconds": "0 processes as one clip; >=1 splits on25fps boundaries (possible seams).",
    "crf": "Final H.264 CRF 0..51; smaller means larger, cleaner files (default18).",
    "preset": "Final x264 encoding speed/compression tradeoff, not diffusion quality.",
    "audio_bitrate": "Final AAC bitrate (default192k). Model conditioning remains16kHz mono.",
    "vae_slicing": "Enable diffusers VAE slicing; effect on peak memory depends on workload.",
    "keep_intermediates": "Retain normalized/segment files in the remote job directory.",
}


def resolve_options(raw: dict | None = None, profile: str = "balanced") -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile!r}; choose {', '.join(PROFILES)}")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("options must be a JSON object")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown options: {', '.join(sorted(map(str, unknown)))}")
    result = {**DEFAULTS, **PROFILES[profile], **raw}
    for key, choices in CHOICES.items():
        if not isinstance(result[key], str) or result[key] not in choices:
            raise ValueError(f"{key} must be one of {', '.join(choices)}")
    for key in ("enable_deepcache", "vae_slicing", "keep_intermediates"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be a JSON boolean (true/false)")
    for key, lower, upper in (("inference_steps", 1, 100), ("seed", -1, 2**32-1), ("crf", 0, 51), ("max_side", 0, 4096)):
        value = result[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{key} must be an integer in {lower}..{upper}")
    if result["max_side"] != 0 and (result["max_side"] < 256 or result["max_side"] % 2):
        raise ValueError("max_side must be 0 or an even integer in 256..4096")
    for key in ("guidance_scale", "video_start", "audio_start", "duration", "audio_offset", "audio_gain_db", "segment_seconds"):
        value = result[key]
        if key == "duration" and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        result[key] = float(value)
    if not 1 <= result["guidance_scale"] <= 5:
        raise ValueError("guidance_scale must be in 1..5; upstream recommends 1..3")
    if result["video_start"] < 0 or result["audio_start"] < 0:
        raise ValueError("video_start and audio_start must be >= 0")
    if result["duration"] is not None and result["duration"] < 0.64:
        raise ValueError("duration must be >= 0.64 seconds (16 frames at25fps)")
    if not -60 <= result["audio_gain_db"] <= 24:
        raise ValueError("audio_gain_db must be in -60..24")
    if result["segment_seconds"] != 0 and result["segment_seconds"] < 1:
        raise ValueError("segment_seconds must be 0 (disabled) or >= 1")
    return result


def add_option_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model, media preparation and encoding options")
    for key, default in DEFAULTS.items():
        kwargs = {"dest": key, "default": None, "help": HELP[key]}
        if type(default) is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif key in CHOICES:
            kwargs["choices"] = CHOICES[key]
        else:
            kwargs["type"] = float if default is None else type(default)
        group.add_argument("--" + key.replace("_", "-"), **kwargs)


def options_from_namespace(args: argparse.Namespace) -> dict:
    return {key: getattr(args, key) for key in DEFAULTS if getattr(args, key, None) is not None}
