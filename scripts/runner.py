#!/usr/bin/env python3
"""Validate local jobs and orchestrate the official Google Colab CLI.

Only the Python standard library is required locally. No model is loaded here.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from config import add_option_arguments, options_from_namespace, resolve_options, PROFILES

HERE = Path(__file__).resolve().parent
SESSION_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
JOB_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_log(text):
    text = ANSI_RE.sub("", str(text)).replace("\r", "")
    text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)((?:access_token|refresh_token|id_token|colab-runtime-proxy-token|authorization)\s*[=:]\s*)[^\s&\"',]+", r"\1[REDACTED]", text)
    return text


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex[:8] + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    def reject_nonfinite(value):
        raise ValueError(f"Non-finite JSON number is not allowed: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_nonfinite)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("File paths must be non-empty strings.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Timeout must be finite and greater than zero.")
    return result


def validate_session(name):
    if not isinstance(name, str) or not SESSION_RE.fullmatch(name):
        raise ValueError("Session must contain 1–64 ASCII letters, digits, underscores or hyphens.")
    return name


def validate_input_media(video, audio, options):
    executable = shutil.which("ffprobe")
    if not executable:
        return {"verification": "size_only", "note": "Install ffprobe for stream/duration validation before GPU allocation."}
    durations = {}
    for kind, path in (("video", video), ("audio", audio)):
        result = subprocess.run([executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if result.returncode:
            raise ValueError(f"Cannot inspect {kind} input {path}: {safe_log(result.stderr[-1000:])}")
        data = json.loads(result.stdout)
        stream = next((entry for entry in data.get("streams", []) if entry.get("codec_type") == kind), None)
        if stream is None:
            raise ValueError(f"{kind} input does not have a {kind} stream: {path}")
        try:
            duration = float(stream.get("duration", data.get("format", {}).get("duration", 0)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot determine {kind} duration: {path}") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid {kind} duration: {path}")
        durations[kind] = duration
    from remote_runtime import select_duration
    selected = select_duration(durations["video"], durations["audio"], options)
    return {"verification": "ffprobe", "video_seconds": durations["video"], "audio_seconds": durations["audio"], "selected_seconds": selected}


def resolve_jobs(manifest, base, *, profile="balanced", overrides=None, output_dir=None):
    """Validate every job before spending compute; resolve paths relative to manifest."""
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be an object with defaults and jobs.")
    unknown = set(manifest) - {"defaults", "jobs"}
    if unknown:
        raise ValueError(f"Unknown manifest fields: {', '.join(sorted(unknown))}")
    defaults = manifest.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("Manifest defaults must be an object of model/media options.")
    raw_jobs = manifest.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("Manifest jobs must be a non-empty array.")
    result, used_ids, used_outputs = [], set(), set()
    base = Path(base).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else base / "outputs"
    for index, raw in enumerate(raw_jobs, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"Job {index} must be an object.")
        unknown = set(raw) - {"id", "video", "audio", "output", "options", "profile"}
        if unknown:
            raise ValueError(f"Job {index}: unknown fields {', '.join(sorted(unknown))}")
        job_id = raw.get("id", f"job-{index:03d}")
        if not isinstance(job_id, str) or not JOB_RE.fullmatch(job_id) or job_id in used_ids:
            raise ValueError(f"Job {index}: id must be unique, 1–64 ASCII letters/digits/_/-.")
        used_ids.add(job_id)
        job_options = raw.get("options", {})
        if not isinstance(job_options, dict):
            raise ValueError(f"Job {job_id}: options must be an object.")
        chosen_profile = raw.get("profile", profile)
        if not isinstance(chosen_profile, str):
            raise ValueError(f"Job {job_id}: profile must be a string.")
        options = resolve_options({**defaults, **job_options, **(overrides or {})}, profile=chosen_profile)
        video = local_path(raw.get("video"), base)
        audio = local_path(raw.get("audio"), base)
        for path in (video, audio):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty input: {path}")
        output = local_path(raw["output"], base) if "output" in raw else output_dir / f"{job_id}.mp4"
        if output.suffix.lower() != ".mp4":
            raise ValueError(f"Output must end in .mp4: {output}")
        if output in used_outputs or output in (video, audio):
            raise ValueError(f"Output collides with another output or an input: {output}")
        used_outputs.add(output)
        preflight = validate_input_media(video, audio, options)
        result.append({"id": job_id, "video": str(video), "audio": str(audio), "output": str(output), "options": options, "preflight": preflight})
    inputs = {Path(job[key]).resolve() for job in result for key in ("video", "audio")}
    outputs_and_reports = {Path(path).resolve() for job in result for path in (job["output"], job["output"] + ".report.json")}
    if inputs & outputs_and_reports:
        raise ValueError("An output or report would overwrite a job's input.")
    return result


def fingerprint(job):
    data = {"options": job["options"], "video_sha256": sha256(job["video"]), "audio_sha256": sha256(job["audio"]), "adapter_sha256": {name: sha256(HERE / name) for name in ("config.py", "remote_runtime.py")}}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_video(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Output is missing or empty: {path}")
    probe = shutil.which("ffprobe")
    if not probe:
        return {"verification": "size_only", "bytes": path.stat().st_size}
    checked = subprocess.run([probe, "-v", "error", "-show_entries", "stream=codec_type:format=duration", "-of", "json", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if checked.returncode:
        raise ValueError(f"ffprobe rejected output: {safe_log(checked.stderr[-1000:])}")
    info = json.loads(checked.stdout)
    streams = {stream.get("codec_type") for stream in info.get("streams", [])}
    if not {"video", "audio"} <= streams:
        raise ValueError(f"Output must contain video and audio streams: {path}")
    duration = float(info.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Output has invalid duration: {path}")
    return {"verification": "ffprobe", "duration": duration, "bytes": path.stat().st_size}


class CommandError(RuntimeError):
    pass


class RemoteTimeout(CommandError):
    pass


class ColabClient:
    def __init__(self, auth="oauth2", executable=None, logger=None):
        self.auth = auth
        command = executable or os.environ.get("COLAB_BIN") or shutil.which("colab")
        if not command:
            raise CommandError("找不到 colab。請在 Linux/macOS 或 WSL 執行 uv tool install --python 3.12 google-colab-cli==0.7.4。")
        self.command = command if isinstance(command, list) else [str(command)]
        self.logger = logger

    def call(self, arguments, timeout=120):
        """Stream bounded, redacted diagnostics without a shell or interactive stdin."""
        command = [*self.command, f"--auth={self.auth}", *map(str, arguments)]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, start_new_session=(os.name != "nt"))
        lines = queue.Queue()
        tail = deque(maxlen=200)

        def reader():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        deadline, ended = time.monotonic() + timeout, False
        try:
            while not ended or process.poll() is None:
                if time.monotonic() >= deadline:
                    raise RemoteTimeout(f"colab {arguments[0]} 超過 {timeout:g} 秒；遠端工作可能仍在執行，請勿直接重送。")
                try:
                    line = lines.get(timeout=min(.2, max(.001, deadline - time.monotonic())))
                except queue.Empty:
                    continue
                if line is None:
                    ended = True
                else:
                    line = safe_log(line.rstrip("\n"))[:8000]
                    tail.append(line)
                    if self.logger:
                        self.logger(line)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            raise
        finally:
            thread.join(timeout=1)
            if process.stdout and not thread.is_alive():
                process.stdout.close()
        output = "\n".join(tail)
        if code:
            raise CommandError(f"colab {arguments[0]} failed (exit {code}):\n" + "\n".join(list(tail)[-12:]))
        return output


def parse_usage(output):
    patterns = {"balance": r"^Current balance:\s*([\d,]+(?:\.\d+)?)\s+compute units\s*$", "rate_per_hour": r"^Usage rate:\s*([\d,]+(?:\.\d+)?)\s*/hr\s*$", "active_assignments": r"^Active assignments:\s*(\d+)\s*$"}
    parsed = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, safe_log(output), re.MULTILINE)
        if match is None:
            raise ValueError("Unrecognized colab usage output; check the installed CLI version.")
        parsed[name] = float(match.group(1).replace(",", ""))
    parsed["active_assignments"] = int(parsed["active_assignments"])
    return parsed


def execute_driver(client, session, work, code, timeout):
    """A sentinel is mandatory: Colab CLI may return exit 0 on kernel errors."""
    token = "LATENTSYNC_DRIVER_OK_" + uuid.uuid4().hex
    driver = Path(work) / (uuid.uuid4().hex + ".py")
    driver.write_text(code + "\nprint(" + repr(token) + ", flush=True)\n", encoding="utf-8")
    output = client.call(["exec", "--session", session, "--timeout", str(timeout), "--file", str(driver)], timeout=timeout + 30)
    if token not in output.splitlines():
        raise CommandError("遠端程式沒有回傳成功標記；即使 Colab CLI 回傳 0，也不能視為成功。請檢查 log。")


def validate_gpu(jobs, gpu, reused=False):
    if not reused and gpu == "T4" and any(job["options"]["model_version"] == "1.6" for job in jobs):
        raise ValueError("LatentSync 1.6 官方最低需求為 18 GB VRAM；T4 只有 16 GB。請使用 --gpu L4 或 --gpu A100，或改 --model-version 1.5。")


def validate_report(report, job, output_hash):
    if not isinstance(report, dict) or report.get("status") != "completed":
        raise ValueError("Remote report does not confirm a completed job.")
    if report.get("job_id") != job["id"]:
        raise ValueError("Remote report job_id does not match this job.")
    if report.get("options") != job["options"]:
        raise ValueError("Remote report options/model do not match this job.")
    if report.get("output_sha256") != output_hash:
        raise ValueError("Downloaded video SHA-256 does not match the remote report.")


def validate_local_paths(jobs, args):
    """Protect every input (including the manifest) from every local write."""
    protected = {Path(job[key]).resolve() for job in jobs for key in ("video", "audio")}
    if getattr(args, "manifest", None):
        protected.add(Path(args.manifest).expanduser().resolve())
    progress = Path(args.progress).expanduser().resolve()
    log = Path(args.log).expanduser().resolve() if args.log else progress.with_suffix(".log").resolve()
    lock = progress.with_name(progress.name + ".lock").resolve()
    writes = [("progress", progress), ("log", log), ("lock", lock)]
    for job in jobs:
        writes.extend(((f"{job['id']} output", Path(job["output"]).resolve()),
                       (f"{job['id']} report", Path(job["output"] + ".report.json").resolve())))
    seen = {}
    for label, path in writes:
        if path in protected:
            raise ValueError(f"{label} would overwrite an input or manifest: {path}")
        if path in seen:
            raise ValueError(f"{label} collides with {seen[path]}: {path}")
        if path.is_dir():
            raise ValueError(f"{label} must be a file, not an existing directory: {path}")
        seen[path] = label
    return progress, log, lock


def run_jobs(jobs, args, client=None):
    """One writer per progress file. A stale lock requires deliberate user cleanup."""
    validate_gpu(jobs, args.gpu, bool(args.session))
    _, _, lock_path = validate_local_paths(jobs, args)
    if args.dry_run:
        return _run_jobs(jobs, args, client)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(f"Progress is locked: {lock_path}。請勿同時執行相同 progress；僅在確認原工作已結束後手動刪除殘留 lock。") from exc
    try:
        with lock:
            json.dump({"pid": os.getpid(), "started_at": utc_now()}, lock)
        return _run_jobs(jobs, args, client)
    finally:
        lock_path.unlink(missing_ok=True)


def _run_jobs(jobs, args, client=None):
    validate_gpu(jobs, args.gpu, bool(args.session))
    if args.session:
        validate_session(args.session)
    progress_path, log_path, _ = validate_local_paths(jobs, args)
    if progress_path.exists() and not args.resume and not args.overwrite and not args.dry_run:
        raise ValueError(f"Progress already exists: {progress_path}。續跑請使用 --resume；新工作請指定不同 --progress，或用 --overwrite 明確替換舊紀錄。")
    old = read_json(progress_path) if args.resume and progress_path.is_file() else {}
    if not isinstance(old, dict):
        raise ValueError("Progress state must be a JSON object.")
    previous = {item["id"]: item for item in old.get("jobs", []) if isinstance(item, dict) and "id" in item}
    state = {"schema_version": 1, "status": "validated", "started_at": utc_now(), "jobs": [], "log": str(log_path), "session": args.session, "session_status": "not_started"}
    for job in jobs:
        item = {"id": job["id"], "output": job["output"], "fingerprint": fingerprint(job), "status": "queued"}
        output = Path(job["output"])
        prior = previous.get(job["id"], {})
        if args.resume and prior.get("status") == "completed" and prior.get("fingerprint") == item["fingerprint"] and prior.get("output") == job["output"] and output.is_file():
            try:
                if sha256(output) != prior.get("output_sha256"):
                    raise ValueError("hash changed")
                verify_video(output)
                report_path = Path(str(output) + ".report.json")
                validate_report(read_json(report_path), job, prior["output_sha256"])
                item.update(prior, skipped=True)
            except (ValueError, OSError, subprocess.SubprocessError):
                pass
        if item["status"] != "completed" and (output.exists() or Path(str(output) + ".report.json").exists()) and not args.overwrite:
            raise ValueError(f"Output/report already exists: {output}。使用 --resume 續跑已驗證工作，或明確傳入 --overwrite。")
        state["jobs"].append(item)
    if args.dry_run:
        return {"status": "dry_run", "jobs": jobs, "will_skip": [j["id"] for j in state["jobs"] if j["status"] == "completed"], "gpu": args.gpu, "profile": args.profile, "progress": str(progress_path)}
    if all(item["status"] == "completed" for item in state["jobs"]):
        state.update(status="completed", session_status="not_used", updated_at=utc_now())
        write_json(progress_path, state)
        return state
    if os.name == "nt" and client is None:
        raise CommandError("Google Colab CLI 不支援原生 Windows。請在 WSL 執行；D:\\ 對應 /mnt/d/。dry-run 與本機驗證仍可在 Windows 使用。")
    for script in (HERE / "config.py", HERE / "remote_runtime.py"):
        if not script.is_file():
            raise FileNotFoundError(f"Missing bundled script: {script}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    session = args.session or "latentsync-" + uuid.uuid4().hex[:16]
    state.update(session=session, status="running")
    owned = args.session is None
    run_id = uuid.uuid4().hex
    remote_dir = "/content/ls_" + run_id
    cleanup_needed, uncertain, interrupted = False, False, False
    current = None
    with log_path.open("a", encoding="utf-8") as log, tempfile.TemporaryDirectory(prefix="latentsync-") as work:
        def logger(line):
            line = safe_log(line)
            log.write(line + "\n")
            log.flush()
            print(line, file=sys.stderr, flush=True)

        client = client or ColabClient(args.auth, args.colab_bin, logger)
        if isinstance(client, ColabClient):
            client.logger = logger

        def save():
            state["updated_at"] = utc_now()
            write_json(progress_path, state)

        def remote_command(arguments, timeout):
            code = "import subprocess\nsubprocess.run(" + repr(arguments) + ", check=True)"
            execute_driver(client, session, work, code, timeout)

        save()
        try:
            if owned:
                cleanup_needed = True  # An allocation timeout can still have created a VM.
                command = ["new", "--session", session, "--gpu", args.gpu]
                if args.high_mem:
                    command.append("--high-mem")
                client.call(command, args.setup_timeout)
            else:
                status = client.call(["status", "--session", session])
                if "not found" in status.lower() or "no active sessions" in status.lower():
                    raise CommandError(f"Session does not exist: {session}")
                if "BUSY" in status:
                    raise CommandError(f"Session is busy: {session}; wait for its existing work.")
            state["session_status"] = "active"
            execute_driver(client, session, work, "import os\nos.makedirs(" + repr(remote_dir) + ", exist_ok=True)", 60)
            for name in ("config.py", "remote_runtime.py"):
                client.call(["upload", "--session", session, str(HERE / name), remote_dir + "/" + name], args.transfer_timeout)
            prepared = set()
            for job, item in zip(jobs, state["jobs"]):
                if item["status"] == "completed":
                    continue
                current = item
                item.update(status="setup", started_at=utc_now())
                save()
                model = job["options"]["model_version"]
                if model not in prepared:
                    remote_command(["python3", remote_dir + "/remote_runtime.py", "setup", "--root", args.remote_root, "--model-version", model], args.setup_timeout)
                    prepared.add(model)
                prefix = remote_dir + "/" + job["id"]
                def safe_suffix(path):
                    suffix = Path(path).suffix.lower()
                    return suffix if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix) else ".bin"
                remote = {"id": job["id"], "video": prefix + "_video" + safe_suffix(job["video"]), "audio": prefix + "_audio" + safe_suffix(job["audio"]), "output": prefix + "_output.mp4", "options": job["options"]}
                item.update(status="uploading", remote_output=remote["output"])
                save()
                for kind in ("video", "audio"):
                    client.call(["upload", "--session", session, job[kind], remote[kind]], args.transfer_timeout)
                local_job = Path(work) / (job["id"] + ".json")
                write_json(local_job, remote)
                remote_job = prefix + ".json"
                client.call(["upload", "--session", session, str(local_job), remote_job], args.transfer_timeout)
                item["status"] = "generating"
                save()
                remote_command(["python3", remote_dir + "/remote_runtime.py", "run", "--root", args.remote_root, "--job", remote_job], args.timeout)
                item["status"] = "downloading"
                save()
                output = Path(job["output"])
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name("." + output.stem + "." + run_id[:8] + ".partial.mp4")
                temporary_report = temporary.with_suffix(".report.json")
                try:
                    client.call(["download", "--session", session, remote["output"], str(temporary)], args.transfer_timeout)
                    client.call(["download", "--session", session, remote["output"] + ".report.json", str(temporary_report)], args.transfer_timeout)
                    report = read_json(temporary_report)
                    verified = verify_video(temporary)
                    output_hash = sha256(temporary)
                    validate_report(report, job, output_hash)
                    if (output.exists() or Path(str(output) + ".report.json").exists()) and not args.overwrite:
                        raise ValueError(f"Output/report appeared during execution; refusing to overwrite: {output}")
                    os.replace(temporary, output)
                    os.replace(temporary_report, Path(str(output) + ".report.json"))
                    item.update(status="completed", output_sha256=output_hash, finished_at=utc_now(), **verified)
                finally:
                    temporary.unlink(missing_ok=True)
                    temporary_report.unlink(missing_ok=True)
                save()
            state["status"] = "completed"
        except (Exception, KeyboardInterrupt) as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            uncertain = isinstance(exc, RemoteTimeout) or interrupted or "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
            state.update(status="partial" if any(item["status"] == "completed" for item in state["jobs"]) else "failed", error=safe_log(str(exc)) or "Interrupted by user")
            if current and current["status"] != "completed":
                current.update(status="uncertain" if uncertain else "failed", error=state["error"])
            for item in state["jobs"]:
                if item["status"] == "queued":
                    item["status"] = "cancelled"
        finally:
            should_stop = (owned and cleanup_needed and (not args.keep_session or uncertain)) or (not owned and args.stop_on_complete and state["session_status"] == "active")
            if should_stop:
                try:
                    client.call(["stop", "--session", session], 120)
                    state["session_status"] = "stopped"
                except Exception as exc:
                    state.update(session_status="stop_failed", cleanup_error=safe_log(str(exc)))
                    if state["status"] == "completed":
                        state["status"] = "cleanup_failed"
            elif state["session_status"] == "active":
                state["session_status"] = "unknown_busy" if uncertain else "kept"
            save()
    return state


def build_parser():
    parser = argparse.ArgumentParser(description="LatentSync：本機準備素材，使用 Google Colab GPU 做嘴型同步。")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("infer", "batch", "doctor", "usage", "status", "stop"):
        sub = commands.add_parser(name)
        sub.add_argument("--auth", choices=("oauth2", "adc"), default=os.environ.get("COLAB_AUTH", "oauth2"))
        sub.add_argument("--colab-bin", default=os.environ.get("COLAB_BIN"), help="Colab CLI executable path; default searches PATH")
        if name in ("infer", "batch"):
            if name == "infer":
                sub.add_argument("--video", required=True)
                sub.add_argument("--audio", required=True)
                sub.add_argument("--output", required=True)
            else:
                sub.add_argument("--manifest", required=True)
                sub.add_argument("--output-dir", help="Default destination for jobs without output")
            sub.add_argument("--profile", choices=tuple(PROFILES), default="balanced")
            add_option_arguments(sub)
            sub.add_argument("--gpu", choices=("T4", "L4", "A100", "H100", "G4"), default="A100")
            sub.add_argument("--high-mem", action="store_true", help="Request system RAM, not GPU VRAM")
            sub.add_argument("--session", help="Reuse an existing session; do not create it")
            sub.add_argument("--keep-session", action="store_true", help="Keep a newly created session; timeout/interruption still stops it")
            sub.add_argument("--stop-on-complete", action="store_true", help="Stop a reused session even when a job fails")
            sub.add_argument("--timeout", type=positive_seconds, default=7200, help="Maximum seconds for each inference job")
            sub.add_argument("--setup-timeout", type=positive_seconds, default=3600, help="Maximum seconds for allocation / environment setup")
            sub.add_argument("--transfer-timeout", type=positive_seconds, default=900)
            sub.add_argument("--remote-root", default="/content/latentsync-skill", help="Persistent cache directory within the current runtime")
            sub.add_argument("--progress", default="outputs/latentsync-progress.json")
            sub.add_argument("--log", help="Local log path; defaults next to progress JSON")
            sub.add_argument("--dry-run", action="store_true", help="Validate and print resolved jobs; no authentication/upload/GPU")
            sub.add_argument("--resume", action="store_true", help="Skip matching completed outputs from --progress")
            sub.add_argument("--overwrite", action="store_true", help="Replace prior progress; replace output/report only after successful generation")
        elif name in ("status", "stop"):
            sub.add_argument("--session", required=(name == "stop"))
        elif name == "doctor":
            sub.add_argument("--check-auth", action="store_true", help="Also query usage (no GPU allocation); requires prior interactive login")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command in ("infer", "batch"):
            if not re.fullmatch(r"/content/[A-Za-z0-9_/-]+", args.remote_root) or ".." in args.remote_root.split("/"):
                raise ValueError("remote-root must be a safe absolute directory under /content.")
            if args.keep_session and args.stop_on_complete:
                raise ValueError("--keep-session and --stop-on-complete cannot be combined.")
            if args.command == "infer":
                manifest = {"jobs": [{"id": "single", "video": args.video, "audio": args.audio, "output": args.output}]}
                base = Path.cwd()
            else:
                manifest_path = Path(args.manifest).expanduser().resolve()
                manifest, base = read_json(manifest_path), manifest_path.parent
            jobs = resolve_jobs(manifest, base, profile=args.profile, overrides=options_from_namespace(args), output_dir=getattr(args, "output_dir", None))
            result = run_jobs(jobs, args)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] in ("completed", "dry_run") else 1
        if args.command == "doctor":
            result = {"python": platform.python_version(), "platform": platform.system(), "colab_cli_supported_platform": os.name != "nt", "ffprobe": shutil.which("ffprobe"), "scripts_present": all((HERE / name).is_file() for name in ("config.py", "remote_runtime.py")), "gpu_allocated": False}
            try:
                client = ColabClient(args.auth, args.colab_bin)
                result["colab_version"] = client.call(["version"], 30)
                if args.check_auth:
                    result["usage"] = parse_usage(client.call(["usage"], 60))
            except Exception as exc:
                result["error"] = safe_log(str(exc))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if "error" not in result and result["colab_cli_supported_platform"] else 1
        if getattr(args, "session", None):
            validate_session(args.session)
        client = ColabClient(args.auth, args.colab_bin)
        arguments = [args.command]
        if getattr(args, "session", None):
            arguments.extend(["--session", args.session])
        output = client.call(arguments, 120)
        print(json.dumps(parse_usage(output), ensure_ascii=False, indent=2) if args.command == "usage" else output)
        return 0
    except (OSError, ValueError, CommandError, subprocess.SubprocessError) as exc:
        print("Error: " + safe_log(str(exc)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
