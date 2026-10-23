"""CPU tests for duration alignment and actual FFmpeg media I/O.

No GPU/model download is implied by passing these tests.
"""
import array
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from config import resolve_options
import remote_runtime as runtime


class EnvironmentTests(unittest.TestCase):
    def test_known_decord_metadata_warning_requires_real_decode(self):
        result = subprocess.CompletedProcess([], 1, "decord 0.6.0 is not supported on this platform\n")
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime.subprocess, "run", return_value=result), patch.object(runtime, "run_cmd", side_effect=[None, subprocess.CalledProcessError(1, ["python", "decode"])]) as command:
            with self.assertRaises(subprocess.CalledProcessError):
                runtime.check_dependencies("python", Path(directory), {})
            self.assertEqual(command.call_count, 2)

    def test_dependency_conflicts_are_never_ignored_with_decord_warning(self):
        for diagnostic in ("decord 0.6.0 is not supported on this platform\nother package requires missing\n",
                           "decord 0.7.0 is not supported on this platform\n", ""):
            result = subprocess.CompletedProcess([], 1, diagnostic)
            with patch.object(runtime.subprocess, "run", return_value=result), patch.object(runtime, "run_cmd") as command:
                with self.assertRaises(subprocess.CalledProcessError):
                    runtime.check_dependencies("python", Path("unused"), {})
                command.assert_not_called()

    def test_every_direct_pin_invalidates_environment_cache(self):
        baseline = runtime.environment_fingerprint()
        self.assertEqual(runtime.ENV_VERSION, baseline)
        # Exercise each individual package, not only one representative in each
        # group: future additions automatically get the same cache-key check.
        for constant in ("BUILD_REQUIREMENTS", "TORCH_REQUIREMENTS", "REQUIREMENTS",
                         "INSIGHTFACE_REQUIREMENTS", "INSIGHTFACE_SUPPORT_REQUIREMENTS"):
            pins = getattr(runtime, constant)
            for index, pin in enumerate(pins):
                changed = list(pins)
                changed[index] = pin + ".changed"
                with self.subTest(pin=pin), patch.object(runtime, constant, changed):
                    self.assertNotEqual(runtime.environment_fingerprint(), baseline)
        for constant in ("PYTHON_VERSION", "UV_REQUIREMENT", "TORCH_INDEX_URL", "INSTALLER_REVISION"):
            with self.subTest(constant=constant), patch.object(runtime, constant, str(getattr(runtime, constant)) + "-changed"):
                self.assertNotEqual(runtime.environment_fingerprint(), baseline)

    def test_manifest_has_no_conflicting_duplicate_package_pins(self):
        pins = {}
        for values in runtime.environment_manifest().values():
            if isinstance(values, list):
                for requirement in values:
                    name, version = requirement.split("==", 1)
                    normalized = name.lower().replace("_", "-")
                    self.assertNotIn(normalized, pins, f"Define {name} in only one central group")
                    pins[normalized] = version
        self.assertEqual(pins["numpy"], "1.26.4")
        self.assertIn("torch", pins)
        self.assertIn("insightface", pins)


class DurationTests(unittest.TestCase):
    def test_trim_offsets_and_length_policies(self):
        options = resolve_options({"video_start": 1, "audio_start": 2, "audio_offset": 0.5})
        self.assertEqual(runtime.select_duration(6, 5, options), 3.48)
        options["length_mode"] = "video"
        self.assertEqual(runtime.select_duration(6, 5, options), 5)
        options["length_mode"] = "audio"
        options["audio_offset"] = -0.5
        self.assertEqual(runtime.select_duration(6, 5, options), 2.48)
        options["duration"] = 1.07
        self.assertEqual(runtime.select_duration(6, 5, options), 1.04)

    def test_empty_and_very_short_media_rejected(self):
        for video, audio, overrides in [(1, 2, {"video_start": 1}), (2, 1, {"audio_offset": -1}),
                                        (0.5, 2, {}), (2, 1, {"audio_start": 1})]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                runtime.select_duration(video, audio, resolve_options(overrides))

    def test_segments_are_frame_aligned_without_tiny_tail(self):
        segments = runtime.segment_ranges(3.08, 1)
        self.assertEqual(segments, [(0.0, 1.0), (1.0, 1.0), (2.0, 1.08)])
        self.assertAlmostEqual(sum(length for _, length in segments), 3.08)
        self.assertEqual(runtime.segment_ranges(3.08, 0), [(0.0, 3.08)])

    def test_t4_auto_dtype_and_no_silent_model_downgrade(self):
        class Cuda:
            @staticmethod
            def is_available(): return True
            @staticmethod
            def mem_get_info(): return (14 * 1024**3, 15 * 1024**3)
            @staticmethod
            def get_device_capability(): return (7, 5)
            @staticmethod
            def get_device_name(): return "Tesla T4"
        class Torch:
            cuda = Cuda
            __version__ = "2.5.1"
            class version:
                cuda = "12.1"
        with patch.dict(sys.modules, {"torch": Torch}):
            self.assertEqual(runtime.gpu_info("1.5", "auto")["precision"], "fp16")
            with self.assertRaisesRegex(RuntimeError, "18 GiB"):
                runtime.gpu_info("1.6", "auto")

    def test_commands_propagate_errors(self):
        import subprocess
        with self.assertRaises(subprocess.CalledProcessError):
            runtime.run_cmd([sys.executable, "-c", "raise SystemExit(7)"], capture=True)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe required")
class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="latentsync-media-")
        cls.root = Path(cls.temp.name)
        cls.video = cls.root / "source video 中文 ' ; test.mp4"
        cls.audio = cls.root / "source audio 中文 ' ; test.wav"
        runtime.run_cmd(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                         "testsrc2=size=640x360:rate=30:duration=1.2", "-c:v", "libx264", cls.video])
        runtime.run_cmd(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                         "sine=frequency=440:sample_rate=48000:duration=1.6", "-ac", "2", cls.audio])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_shortest_normalizes_frame_rate_resolution_and_audio(self):
        options = resolve_options({"max_side": 320})
        prepared = runtime.prepare_media(self.video, self.audio, self.root / "shortest", options)
        metadata = runtime.probe(prepared["video"])
        self.assertEqual(metadata["video"]["width"], 320)
        self.assertEqual(metadata["video"]["height"], 180)
        self.assertEqual(metadata["video"]["avg_frame_rate"], "25/1")
        self.assertAlmostEqual(prepared["duration"], 1.2)
        with wave.open(prepared["audio"], "rb") as handle:
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getnframes(), 19200)

    def test_positive_offset_prepends_silence_and_forward_loop_extends_video(self):
        options = resolve_options({"length_mode": "audio", "audio_offset": 0.2})
        prepared = runtime.prepare_media(self.video, self.audio, self.root / "loop", options)
        self.assertAlmostEqual(prepared["duration"], 1.8)
        metadata = runtime.probe(prepared["video"])
        self.assertAlmostEqual(metadata["video_duration"], 1.8, delta=0.04)
        with wave.open(prepared["audio"], "rb") as handle:
            samples = array.array("h", handle.readframes(handle.getnframes()))
        self.assertEqual(max(abs(v) for v in samples[:3000]), 0)
        self.assertGreater(max(abs(v) for v in samples[3300:4000]), 100)

    def test_negative_offset_and_video_mode_pad_audio(self):
        options = resolve_options({"length_mode": "video", "audio_offset": -1.0})
        prepared = runtime.prepare_media(self.video, self.audio, self.root / "pad", options)
        with wave.open(prepared["audio"], "rb") as handle:
            samples = array.array("h", handle.readframes(handle.getnframes()))
        self.assertEqual(len(samples), 19200)
        self.assertGreater(max(abs(v) for v in samples[:4000]), 100)
        self.assertEqual(max(abs(v) for v in samples[-4000:]), 0)

    def test_final_mp4_decodes_and_duration_failure_is_detected(self):
        prepared = runtime.prepare_media(self.video, self.audio, self.root / "final", resolve_options())
        # Test packaging independently of ML by treating normalized video as the
        # worker's output. Apostrophes are deliberately absent from stage paths.
        final = runtime.finalize_media([Path(prepared["video"])], prepared, self.root / "final", resolve_options())
        metadata = runtime.validate_output(final, 1.2)
        self.assertEqual(metadata["video"]["codec_name"], "h264")
        self.assertEqual(metadata["audio"]["codec_name"], "aac")
        with self.assertRaisesRegex(RuntimeError, "duration differs"):
            runtime.validate_output(final, 2.0)
        with self.assertRaisesRegex(RuntimeError, "both video and audio"):
            runtime.validate_output(prepared["video"])


if __name__ == "__main__":
    unittest.main()
