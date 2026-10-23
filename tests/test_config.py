import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from config import add_option_arguments, options_from_namespace, resolve_options


class ConfigTests(unittest.TestCase):
    def test_user_requested_version_across_profiles(self):
        for profile in ("preview", "balanced", "quality"):
            self.assertEqual(resolve_options(profile=profile)["model_version"], "1.6")

    def test_bool_and_cli_precedence(self):
        parser = argparse.ArgumentParser()
        add_option_arguments(parser)
        options = options_from_namespace(parser.parse_args(["--no-enable-deepcache", "--inference-steps", "33"]))
        self.assertEqual(options, {"enable_deepcache": False, "inference_steps": 33})
        self.assertEqual(resolve_options(options, "preview")["inference_steps"], 33)

    def test_invalid_options_fail_before_compute(self):
        for invalid in ({"seed": True}, {"inference_steps": 2.5}, {"guidance_scale": float("nan")},
                        {"audio_offset": float("inf")}, {"model_version": 1.6}, {"duration": 0},
                        {"unknown": 1}, {"max_side": 511}, {"enable_deepcache": "false"},
                        {"segment_seconds": 0.5}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_options(invalid)


if __name__ == "__main__":
    unittest.main()
