"""Offline orchestration checks: fake Colab transport, no account or GPU access."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runner


class FakeColab:
    """Imitate CLI file transfer and exec, including exit-0 kernel failures."""
    def __init__(self, fail_run=None, timeout=False, stop_fail=False):
        self.calls = []
        self.files = {}
        self.run_count = 0
        self.fail_run = fail_run
        self.timeout = timeout
        self.stop_fail = stop_fail

    def call(self, arguments, timeout=120):
        self.calls.append(list(arguments))
        operation = arguments[0]
        if operation == "upload":
            self.files[arguments[-1]] = Path(arguments[-2]).read_bytes()
        elif operation == "download":
            Path(arguments[-1]).write_bytes(self.files[arguments[-2]])
        elif operation == "exec":
            code = Path(arguments[arguments.index("--file") + 1]).read_text(encoding="utf-8")
            token = re.search(r"LATENTSYNC_DRIVER_OK_[0-9a-f]+", code).group(0)
            parsed = ast.parse(code)
            for node in ast.walk(parsed):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr not in ("run", "Popen"):
                    continue
                command = ast.literal_eval(node.args[0])
                if "--job" in command:
                    self.run_count += 1
                    if self.run_count == self.fail_run:
                        if self.timeout:
                            raise runner.RemoteTimeout("Fake timeout: remote may still execute")
                        return "Traceback: kernel error (CLI exit 0)"
                    config = json.loads(self.files[command[command.index("--job") + 1]])
                    self.files[config["output"]] = b"fake rendered video with audio"
                    self.files[config["output"] + ".report.json"] = json.dumps({"status": "completed", "job_id": config["id"], "options": config["options"], "output_sha256": hashlib.sha256(self.files[config["output"]]).hexdigest()}).encode()
            return token
        elif operation == "stop" and self.stop_fail:
            raise runner.CommandError("fake cleanup failed")
        elif operation == "status":
            return "[existing] example | Hardware: A100 | Status: IDLE"
        return "ok"


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        (self.base / "video with space.mp4").write_bytes(b"source video")
        (self.base / "audio.wav").write_bytes(b"source audio")
        preflight = mock.patch.object(runner, "validate_input_media", return_value={"verification": "fake"})
        preflight.start()
        self.addCleanup(preflight.stop)
        self.manifest = {"jobs": [self.raw_job("first"), self.raw_job("second")]}
        self.jobs = runner.resolve_jobs(self.manifest, self.base)
        self.args = runner.build_parser().parse_args(["batch", "--manifest", str(self.base / "jobs.json"), "--progress", str(self.base / "progress.json")])
        # Tests mock media inspection, while real output checks are tested below.
        self.verify = mock.patch.object(runner, "verify_video", side_effect=lambda p: {"verification": "fake", "bytes": Path(p).stat().st_size})
        self.verify.start()
        self.addCleanup(self.verify.stop)
        # Runtime file may still be authored by a parallel agent; no script is executed.
        self.original_here = runner.HERE
        self.script_dir = self.base / "scripts"
        self.script_dir.mkdir()
        for name in ("config.py", "remote_runtime.py"):
            (self.script_dir / name).write_text("# fake bundled file\n")
        self.patcher = mock.patch.object(runner, "HERE", self.script_dir)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def raw_job(self, name):
        return {"id": name, "video": "video with space.mp4", "audio": "audio.wav"}

    def test_defaults_are_16_a100(self):
        self.assertEqual(self.args.gpu, "A100")
        self.assertTrue(all(j["options"]["model_version"] == "1.6" for j in self.jobs))

    def test_option_precedence(self):
        manifest = {"defaults": {"inference_steps": 21}, "jobs": [{**self.raw_job("job"), "profile": "quality", "options": {"inference_steps": 25}}]}
        jobs = runner.resolve_jobs(manifest, self.base, overrides={"inference_steps": 31})
        self.assertEqual(jobs[0]["options"]["inference_steps"], 31)

    def test_paths_are_relative_to_manifest_not_cwd(self):
        self.assertEqual(self.jobs[0]["video"], str(self.base / "video with space.mp4"))

    def test_invalid_later_job_rejected_before_client(self):
        self.manifest["jobs"][1]["audio"] = "missing.wav"
        with self.assertRaises(ValueError):
            runner.resolve_jobs(self.manifest, self.base)

    def test_unknown_keys_and_bad_options_rejected(self):
        for manifest in ({**self.manifest, "typo": 1}, {"jobs": [{**self.raw_job("a"), "typo": 1}]}, {"jobs": [{**self.raw_job("a"), "options": {"steps": 20}}]}, {"jobs": [{**self.raw_job("a"), "profile": []}]}):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                runner.resolve_jobs(manifest, self.base)

    def test_duplicate_ids_and_outputs_rejected(self):
        for field, value in (("id", "first"), ("output", "one.mp4")):
            manifest = {"jobs": [self.raw_job("first"), self.raw_job("second")]}
            manifest["jobs"][0][field] = value
            manifest["jobs"][1][field] = value
            with self.assertRaises(ValueError):
                runner.resolve_jobs(manifest, self.base)

    def test_output_cannot_overwrite_input(self):
        with self.assertRaises(ValueError):
            runner.resolve_jobs({"jobs": [{**self.raw_job("a"), "output": "video with space.mp4"}]}, self.base)

    def test_report_cannot_overwrite_any_job_input(self):
        unusual_audio = self.base / "result.mp4.report.json"
        unusual_audio.write_bytes(b"audio content despite extension")
        manifest = {"jobs": [{**self.raw_job("first"), "output": "result.mp4"},
                             {**self.raw_job("second"), "audio": unusual_audio.name}]}
        with self.assertRaisesRegex(ValueError, "report"):
            runner.resolve_jobs(manifest, self.base)
        self.assertEqual(unusual_audio.read_bytes(), b"audio content despite extension")

    def test_report_input_collision_blocked_even_overwrite(self):
        unusual_audio = Path(self.jobs[0]["output"] + ".report.json")
        unusual_audio.parent.mkdir()
        unusual_audio.write_bytes(b"protected input")
        self.jobs[1]["audio"] = str(unusual_audio)
        self.args.overwrite = True
        client = FakeColab()
        with self.assertRaisesRegex(ValueError, "input"):
            runner.run_jobs(self.jobs, self.args, client)
        self.assertEqual(client.calls, [])
        self.assertEqual(unusual_audio.read_bytes(), b"protected input")

    def test_manifest_protected_against_every_write_even_overwrite(self):
        for target in ("progress", "log", "output", "report", "lock"):
            with self.subTest(target=target):
                self.args.overwrite = True
                self.args.progress = str(self.base / "progress.json")
                self.args.log = None
                self.jobs[0]["output"] = str(self.base / "result.mp4")
                paths = {"progress": self.args.progress, "log": str(self.base / "custom.log"),
                         "output": self.jobs[0]["output"], "report": self.jobs[0]["output"] + ".report.json",
                         "lock": self.args.progress + ".lock"}
                manifest_path = Path(paths[target])
                manifest_path.write_text('{"jobs": []}', encoding="utf-8")
                self.args.manifest = str(manifest_path)
                if target == "log":
                    self.args.log = str(manifest_path)
                client = FakeColab()
                with self.assertRaisesRegex(ValueError, "manifest"):
                    runner.run_jobs(self.jobs, self.args, client)
                self.assertEqual(client.calls, [])
                self.assertEqual(manifest_path.read_text(), '{"jobs": []}')
                manifest_path.unlink()

    def test_main_rejects_progress_pointing_at_manifest(self):
        manifest_path = self.base / "jobs.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        original = manifest_path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            code = runner.main(["batch", "--manifest", str(manifest_path), "--progress", str(manifest_path), "--overwrite", "--dry-run"])
        self.assertEqual(code, 1)
        self.assertEqual(manifest_path.read_bytes(), original)
        self.assertFalse(Path(str(manifest_path) + ".lock").exists())

    def test_dry_run_makes_no_cli_calls_or_progress(self):
        self.args.dry_run = True
        client = FakeColab()
        result = runner.run_jobs(self.jobs, self.args, client)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(client.calls, [])
        self.assertFalse(Path(self.args.progress).exists())

    def test_16_t4_rejected_without_allocation(self):
        self.args.gpu = "T4"
        client = FakeColab()
        with self.assertRaises(ValueError):
            runner.run_jobs(self.jobs, self.args, client)
        self.assertEqual(client.calls, [])

    def test_owned_batch_single_allocation_stop_and_reports(self):
        client = FakeColab()
        result = runner.run_jobs(self.jobs, self.args, client)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["session_status"], "stopped")
        self.assertEqual(sum(c[0] == "new" for c in client.calls), 1)
        self.assertEqual(sum(c[0] == "stop" for c in client.calls), 1)
        self.assertEqual(client.run_count, 2)
        for job in self.jobs:
            self.assertTrue(Path(job["output"]).exists())
            self.assertTrue(Path(job["output"] + ".report.json").exists())

    def test_kernel_error_exit_zero_keeps_first_result_and_stops(self):
        client = FakeColab(fail_run=2)
        result = runner.run_jobs(self.jobs, self.args, client)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["jobs"][0]["status"], "completed")
        self.assertEqual(result["jobs"][1]["status"], "failed")
        self.assertTrue(Path(self.jobs[0]["output"]).exists())
        self.assertFalse(Path(self.jobs[1]["output"]).exists())
        self.assertEqual(result["session_status"], "stopped")

    def test_timeout_forces_owned_stop_even_keep(self):
        self.args.keep_session = True
        client = FakeColab(fail_run=1, timeout=True)
        result = runner.run_jobs(self.jobs, self.args, client)
        self.assertEqual(result["jobs"][0]["status"], "uncertain")
        self.assertEqual(result["session_status"], "stopped")
        self.assertEqual(client.run_count, 1)

    def test_reused_session_never_stopped_by_default(self):
        self.args.session = "existing"
        client = FakeColab(fail_run=1, timeout=True)
        result = runner.run_jobs(self.jobs, self.args, client)
        self.assertEqual(result["session_status"], "unknown_busy")
        self.assertFalse(any(c[0] in ("new", "stop") for c in client.calls))

    def test_stop_on_complete_applies_to_reused_session(self):
        self.args.session = "existing"
        self.args.stop_on_complete = True
        client = FakeColab()
        result = runner.run_jobs(self.jobs, self.args, client)
        self.assertEqual(result["session_status"], "stopped")
        self.assertTrue(any(c[0] == "stop" for c in client.calls))

    def test_stop_error_is_not_reported_as_full_success(self):
        result = runner.run_jobs(self.jobs, self.args, FakeColab(stop_fail=True))
        self.assertEqual(result["status"], "cleanup_failed")
        self.assertEqual(result["session_status"], "stop_failed")
        self.assertTrue(Path(self.jobs[0]["output"]).exists())

    def test_resume_skips_only_unchanged_completed_work(self):
        runner.run_jobs(self.jobs, self.args, FakeColab(fail_run=2))
        self.args.resume = True
        client = FakeColab()
        result = runner.run_jobs(self.jobs, self.args, client)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(client.run_count, 1)
        self.assertTrue(result["jobs"][0]["skipped"])
        all_complete = FakeColab()
        runner.run_jobs(self.jobs, self.args, all_complete)
        self.assertEqual(all_complete.calls, [])

    def test_resume_rejects_changed_input_or_corrupt_output(self):
        runner.run_jobs(self.jobs, self.args, FakeColab())
        self.args.resume = True
        Path(self.jobs[0]["video"]).write_bytes(b"different input")
        with self.assertRaises(ValueError):
            runner.run_jobs(self.jobs, self.args, FakeColab())

    def test_resume_rejects_changed_adapter(self):
        runner.run_jobs(self.jobs, self.args, FakeColab())
        self.args.resume = True
        (self.script_dir / "remote_runtime.py").write_text("# changed adapter\n")
        with self.assertRaises(ValueError):
            runner.run_jobs(self.jobs, self.args, FakeColab())

    def test_existing_output_preserved_on_failed_overwrite(self):
        output = Path(self.jobs[0]["output"])
        output.parent.mkdir()
        output.write_bytes(b"old video")
        self.args.overwrite = True
        runner.run_jobs(self.jobs, self.args, FakeColab(fail_run=1))
        self.assertEqual(output.read_bytes(), b"old video")

    def test_progress_log_collision_rejected(self):
        self.args.progress = self.jobs[0]["video"]
        with self.assertRaises(ValueError):
            runner.run_jobs(self.jobs, self.args, FakeColab())

    def test_existing_progress_requires_explicit_resume_or_overwrite(self):
        Path(self.args.progress).write_text('{"jobs": []}')
        with self.assertRaises(ValueError):
            runner.run_jobs(self.jobs, self.args, FakeColab())
        self.assertEqual(json.loads(Path(self.args.progress).read_text()), {"jobs": []})

    def test_progress_lock_prevents_concurrent_writer(self):
        lock = Path(self.args.progress + ".lock")
        lock.write_text('{"pid": 12345}')
        with self.assertRaises(ValueError):
            runner.run_jobs(self.jobs, self.args, FakeColab())
        self.assertTrue(lock.exists())

    def test_failed_or_corrupt_report_not_published(self):
        for field, value in (("status", "failed"), ("output_sha256", "bad hash"), ("job_id", "wrong job"), ("options", {})):
            with self.subTest(field=field):
                client = FakeColab()
                original = client.call
                def call(arguments, timeout=120):
                    result = original(arguments, timeout)
                    if arguments[0] == "download" and arguments[-2].endswith(".report.json"):
                        path = Path(arguments[-1])
                        report = json.loads(path.read_text())
                        report[field] = value
                        path.write_text(json.dumps(report))
                    return result
                client.call = call
                self.args.overwrite = True
                result = runner.run_jobs(self.jobs, self.args, client)
                self.assertEqual(result["status"], "failed")
                self.assertFalse(Path(self.jobs[0]["output"]).exists())
                self.assertFalse(Path(self.jobs[0]["output"] + ".report.json").exists())

    def test_report_appearing_mid_download_not_overwritten(self):
        client = FakeColab()
        original = client.call
        target = Path(self.jobs[0]["output"] + ".report.json")
        def call(arguments, timeout=120):
            result = original(arguments, timeout)
            if arguments[0] == "download" and arguments[-2].endswith(".report.json"):
                target.write_text("unrelated report")
            return result
        client.call = call
        result = runner.run_jobs(self.jobs, self.args, client)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(target.read_text(), "unrelated report")

    def test_busy_reused_session_not_stopped_when_rejected(self):
        self.args.session = "existing"
        self.args.stop_on_complete = True
        client = FakeColab()
        original = client.call
        client.call = lambda args, timeout=120: "Status: BUSY" if args[0] == "status" else original(args, timeout)
        runner.run_jobs(self.jobs, self.args, client)
        self.assertFalse(any(c[0] == "stop" for c in client.calls))


class UtilityTests(unittest.TestCase):
    def test_driver_relays_os_stdout_and_stderr_into_kernel_stream(self):
        # Jupyter captures Python's stdout object, not inherited child FDs.
        kernel_stream = io.StringIO()
        argument = "space ' quote $() literal"
        code = runner.subprocess_driver([sys.executable, "-c",
                "import os, sys; os.write(1, b'child stdout\\n'); os.write(2, b'child stderr\\n'); print(sys.argv[1])",
                argument])
        with contextlib.redirect_stdout(kernel_stream):
            exec(code, {})
        self.assertIn("child stdout\n", kernel_stream.getvalue())
        self.assertIn("child stderr\n", kernel_stream.getvalue())
        self.assertIn(argument, kernel_stream.getvalue())

    def test_driver_keeps_error_details_and_never_emits_success_on_failure(self):
        kernel_stream = io.StringIO()
        code = runner.subprocess_driver([sys.executable, "-c",
                "import os, sys; os.write(2, b'install failed: exact cause\\n'); sys.exit(7)"])
        code += "print('UNEXPECTED_SUCCESS')\n"
        with contextlib.redirect_stdout(kernel_stream), self.assertRaises(subprocess.CalledProcessError) as caught:
            exec(code, {})
        self.assertEqual(caught.exception.returncode, 7)
        self.assertIn("install failed: exact cause", kernel_stream.getvalue())
        self.assertNotIn("UNEXPECTED_SUCCESS", kernel_stream.getvalue())

    def test_input_preflight_rejects_bad_trim_before_gpu(self):
        result = subprocess.CompletedProcess([], 0, json.dumps({"streams": [{"codec_type": "video", "duration": "2"}, {"codec_type": "audio", "duration": "2"}]}), "")
        with mock.patch.object(runner.shutil, "which", return_value="ffprobe"), mock.patch.object(runner.subprocess, "run", return_value=result):
            with self.assertRaises(ValueError):
                runner.validate_input_media("input.mp4", "audio.wav", runner.resolve_options({"video_start": 3}))
            metadata = runner.validate_input_media("input.mp4", "audio.wav", runner.resolve_options())
            self.assertEqual(metadata["selected_seconds"], 2)

    def test_usage_schema_and_unknown_format(self):
        output = "Current balance: 1,234.50 compute units\nUsage rate: 4.25/hr\nActive assignments: 2"
        self.assertEqual(runner.parse_usage(output), {"balance": 1234.5, "rate_per_hour": 4.25, "active_assignments": 2})
        with self.assertRaises(ValueError):
            runner.parse_usage("unknown")

    def test_credentials_redacted(self):
        output = runner.safe_log("Authorization: Bearer secretABC\nhttps://example?colab-runtime-proxy-token=secretDEF&x=1\naccess_token=secretXYZ")
        for secret in ("secretABC", "secretDEF", "secretXYZ"):
            self.assertNotIn(secret, output)

    def test_exec_without_sentinel_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = mock.Mock()
            client.call.return_value = "Traceback..."
            with self.assertRaises(runner.CommandError):
                runner.execute_driver(client, "session", temporary, "raise RuntimeError()", 1)

    def test_ffprobe_rejects_video_without_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.mp4"
            path.write_bytes(b"video")
            probe = subprocess.CompletedProcess([], 0, json.dumps({"streams": [{"codec_type": "video"}], "format": {"duration": "1"}}), "")
            with mock.patch.object(runner.shutil, "which", return_value="ffprobe"), mock.patch.object(runner.subprocess, "run", return_value=probe):
                with self.assertRaises(ValueError):
                    runner.verify_video(path)

    def test_real_subprocess_cli_argv_and_redaction_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "fake_cli.py"
            fake.write_text("import sys\nprint(repr(sys.argv[1:]))\nprint('access_token=SECRET')\n", encoding="utf-8")
            client = runner.ColabClient(executable=[sys.executable, str(fake)])
            output = client.call(["upload", "--session", "s", "space name.mp4", "/content/in.mp4"], 5)
            self.assertIn("'--auth=oauth2'", output)
            self.assertIn("'space name.mp4'", output)
            self.assertNotIn("SECRET", output)

    def test_real_subprocess_timeout_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "slow_cli.py"
            fake.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
            client = runner.ColabClient(executable=[sys.executable, str(fake)])
            with self.assertRaises(runner.RemoteTimeout):
                client.call(["exec"], .1)


if __name__ == "__main__":
    unittest.main()
