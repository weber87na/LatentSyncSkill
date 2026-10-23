"""Installer safety tests; fixtures never touch the user's installed skills."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


WORKSPACE = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("latentsync_skill_installer", WORKSPACE / "scripts" / "install.py")
installer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = installer
SPEC.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        temporary_parent = WORKSPACE / ".test-tmp"
        temporary_parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="installer-", dir=temporary_parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.destination = self.root / "skills"
        self.source.mkdir()
        self.write("SKILL.md", "---\nname: latentsync-colab\n---\nA test fixture.\n")
        self.write("README.md", "Installation instructions.\n")
        self.write("scripts/run.py", "print('fixture')\n")
        self.write("assets/default.json", '{"steps": 20}\n')
        self.write("references/options.md", "Options.\n")
        self.write("agents/openai.yaml", "interface: {}\n")
        self.write("examples/jobs.json", '{"jobs": []}\n')

    def write(self, relative, content):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    @property
    def target(self):
        return self.destination / installer.SKILL_NAME

    def assert_no_staging(self):
        if self.destination.exists():
            self.assertFalse(list(self.destination.glob(".latentsync-colab-install-*")))

    def test_new_install_copies_only_allowlisted_clean_content(self):
        for relative in (
            ".research/private.txt", "tests/test_secret.py", "outputs/movie.mp4",
            ".venv/secret.txt", "unrelated.txt", "scripts/__pycache__/run.pyc",
            "scripts/session.log", "scripts/data.partial", "assets/outputs/tmp.txt",
        ):
            self.write(relative, "excluded")
        result = installer.install(self.source, self.destination)
        self.assertEqual(result.status, "installed")
        self.assertEqual(result.target, self.target)
        self.assertIsNone(result.backup)
        files = {str(path.relative_to(self.target)).replace("\\", "/") for path in self.target.rglob("*") if path.is_file()}
        self.assertEqual(files, {
            "SKILL.md", "README.md", "scripts/run.py", "assets/default.json",
            "references/options.md", "agents/openai.yaml",
            "examples/jobs.json",
        })
        self.assert_no_staging()

    def test_unchanged_install_does_not_rewrite_files_or_make_backup(self):
        installer.install(self.source, self.destination)
        old_time = (self.target / "SKILL.md").stat().st_mtime_ns
        result = installer.install(self.source, self.destination, force=True)
        self.assertEqual(result.status, "unchanged")
        self.assertEqual((self.target / "SKILL.md").stat().st_mtime_ns, old_time)
        self.assertEqual(list(self.destination.glob("*.backup-*")), [])
        self.assert_no_staging()

    def test_changed_install_is_retained_without_force(self):
        installer.install(self.source, self.destination)
        self.write("scripts/run.py", "print('updated')\n")
        result = installer.install(self.source, self.destination)
        self.assertEqual(result.status, "exists")
        self.assertEqual((self.target / "scripts/run.py").read_text(), "print('fixture')\n")
        self.assert_no_staging()

    def test_forced_update_preserves_all_existing_files_in_backup(self):
        installer.install(self.source, self.destination)
        (self.target / "personal.txt").write_text("user settings", encoding="utf-8")
        self.write("scripts/run.py", "print('updated')\n")
        result = installer.install(self.source, self.destination, force=True)
        self.assertEqual(result.status, "updated")
        self.assertTrue(result.backup.is_dir())
        self.assertEqual((result.backup / "personal.txt").read_text(), "user settings")
        self.assertEqual((result.backup / "scripts/run.py").read_text(), "print('fixture')\n")
        self.assertEqual((self.target / "scripts/run.py").read_text(), "print('updated')\n")
        self.assertFalse((self.target / "personal.txt").exists())
        self.assert_no_staging()

    def test_default_destination_honors_codex_home(self):
        with mock.patch.dict(os.environ, {"CODEX_HOME": str(self.root / "custom")}, clear=False):
            self.assertEqual(installer.default_destination(), self.root / "custom/skills")
            result = installer.install(self.source)
        self.assertEqual(result.target, self.root / "custom/skills/latentsync-colab")

    def test_default_destination_falls_back_to_user_home(self):
        with mock.patch.dict(os.environ, {"CODEX_HOME": ""}, clear=False), mock.patch.object(Path, "home", return_value=self.root):
            self.assertEqual(installer.default_destination(), self.root / ".codex/skills")

    def test_running_from_installed_location_is_a_noop(self):
        installer.install(self.source, self.destination)
        result = installer.install(self.target, self.destination, force=True)
        self.assertEqual(result.status, "same-location")
        self.assert_no_staging()

    def test_install_inside_source_is_refused_before_writing(self):
        with self.assertRaisesRegex(installer.InstallError, "must not contain"):
            installer.install(self.source, self.source / "nested", force=True)
        self.assertFalse((self.source / "nested").exists())

    def test_install_over_source_ancestor_is_refused(self):
        outer = self.root / installer.SKILL_NAME
        source = outer / "nested"
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text("test", encoding="utf-8")
        with self.assertRaisesRegex(installer.InstallError, "must not contain"):
            installer.install(source, self.root, force=True)
        self.assertTrue((source / "SKILL.md").is_file())

    def test_file_destination_is_refused_even_with_force(self):
        self.destination.mkdir()
        self.target.write_text("do not replace", encoding="utf-8")
        with self.assertRaisesRegex(installer.InstallError, "not a directory"):
            installer.install(self.source, self.destination, force=True)
        self.assertEqual(self.target.read_text(), "do not replace")

    def test_missing_skill_definition_is_refused(self):
        (self.source / "SKILL.md").unlink()
        with self.assertRaisesRegex(installer.InstallError, "SKILL.md"):
            installer.install(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_failed_final_rename_restores_existing_installation(self):
        installer.install(self.source, self.destination)
        self.write("scripts/run.py", "print('updated')\n")
        original_rename = Path.rename

        def fail_stage(path, target):
            if path.name.startswith(".latentsync-colab-install-"):
                raise PermissionError("simulated file lock")
            return original_rename(path, target)

        with mock.patch.object(Path, "rename", fail_stage):
            with self.assertRaisesRegex(installer.InstallError, "previous installation was restored"):
                installer.install(self.source, self.destination, force=True)
        self.assertEqual((self.target / "scripts/run.py").read_text(), "print('fixture')\n")
        self.assertEqual(list(self.destination.glob("*.backup-*")), [])
        self.assert_no_staging()

    def test_failed_rollback_preserves_backup_for_manual_recovery(self):
        installer.install(self.source, self.destination)
        self.write("scripts/run.py", "print('updated')\n")
        original_rename = Path.rename

        def fail_stage_and_rollback(path, target):
            if path.name.startswith(".latentsync-colab-install-") or ".backup-" in path.name:
                raise PermissionError("simulated file lock")
            return original_rename(path, target)

        with mock.patch.object(Path, "rename", fail_stage_and_rollback):
            with self.assertRaisesRegex(installer.InstallError, "previous installation is preserved at"):
                installer.install(self.source, self.destination, force=True)
        backups = list(self.destination.glob("*.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "scripts/run.py").read_text(), "print('fixture')\n")
        self.assert_no_staging()

    def test_copy_failure_leaves_previous_install_untouched(self):
        installer.install(self.source, self.destination)
        self.write("scripts/run.py", "print('updated')\n")
        with mock.patch.object(installer.shutil, "copy2", side_effect=OSError("simulated disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                installer.install(self.source, self.destination, force=True)
        self.assertEqual((self.target / "scripts/run.py").read_text(), "print('fixture')\n")
        self.assert_no_staging()

    def test_source_symlink_is_refused(self):
        external = self.root / "external.txt"
        external.write_text("outside", encoding="utf-8")
        try:
            (self.source / "assets/link.txt").symlink_to(external)
        except OSError as error:
            self.skipTest(f"Symlink creation is unavailable: {error}")
        with self.assertRaisesRegex(installer.InstallError, "symbolic link or junction"):
            installer.install(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_link_destination_is_refused(self):
        actual = self.root / "actual-skills"
        actual.mkdir()
        try:
            self.destination.symlink_to(actual, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlink creation is unavailable: {error}")
        with self.assertRaisesRegex(installer.InstallError, "symbolic link or junction"):
            installer.install(self.source, self.destination, force=True)
        self.assertEqual(list(actual.iterdir()), [])

    def test_windows_junction_destination_is_refused_without_symlink_privileges(self):
        original_lstat = Path.lstat

        def junction_stat(path, *args, **kwargs):
            if path == self.destination:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original_lstat(path, *args, **kwargs)

        with mock.patch.object(Path, "lstat", junction_stat):
            with self.assertRaisesRegex(installer.InstallError, "symbolic link or junction"):
                installer.install(self.source, self.destination, force=True)
        self.assertFalse(self.destination.exists())

    def test_source_changed_during_copy_is_refused_before_replacing_install(self):
        installer.install(self.source, self.destination)
        self.write("scripts/run.py", "print('updated')\n")
        original_copy = installer.shutil.copy2

        def mutate_copy(source, destination, **kwargs):
            result = original_copy(source, destination, **kwargs)
            if source.name == "run.py":
                Path(destination).write_text("unexpected changed content", encoding="utf-8")
            return result

        with mock.patch.object(installer.shutil, "copy2", mutate_copy):
            with self.assertRaisesRegex(installer.InstallError, "Source changed"):
                installer.install(self.source, self.destination, force=True)
        self.assertEqual((self.target / "scripts/run.py").read_text(), "print('fixture')\n")
        self.assert_no_staging()

    def test_cli_prints_retained_status_and_passes_destination(self):
        output = io.StringIO()
        result = installer.InstallResult("exists", self.target)
        with mock.patch.object(installer, "install", return_value=result) as call, redirect_stdout(output):
            self.assertEqual(installer.main(["--dest", str(self.destination)]), 0)
        self.assertEqual(call.call_args.args[1], self.destination)
        self.assertIn("retained", output.getvalue())
        self.assertIn("--force", output.getvalue())


if __name__ == "__main__":
    unittest.main()
