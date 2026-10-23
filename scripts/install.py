#!/usr/bin/env python3
"""Install this skill without third-party dependencies or destructive overwrites."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import uuid


SKILL_NAME = "latentsync-colab"
INCLUDED_FILES = ("SKILL.md", "README.md")
INCLUDED_DIRS = ("scripts", "assets", "references", "agents", "examples")
EXCLUDED_NAMES = frozenset({
    ".git", ".research", "research", "test", "tests", "output", "outputs",
    ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", ".runtime", ".progress", "logs",
    ".DS_Store", "Thumbs.db", "desktop.ini",
})
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".partial", ".log")


class InstallError(RuntimeError):
    """An install cannot be completed safely."""


@dataclass(frozen=True)
class InstallResult:
    status: str
    target: Path
    backup: Path | None = None


def default_destination() -> Path:
    """Return the parent skills directory, honoring CODEX_HOME."""
    codex_home = os.environ.get("CODEX_HOME")
    return (Path(codex_home).expanduser() if codex_home else Path.home() / ".codex") / "skills"


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _is_link(path: Path) -> bool:
    """Include Windows junctions/reparse points on Python 3.10."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _reject_link_components(path: Path) -> None:
    for component in (path, *path.parents):
        if _is_link(component):
            raise InstallError(f"Refusing a symbolic link or junction: {component}")


def _excluded(path: Path) -> bool:
    return path.name in EXCLUDED_NAMES or path.name.lower().endswith(EXCLUDED_SUFFIXES)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path, *, source: bool) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}

    def visit(path: Path) -> None:
        if _excluded(path):
            return
        if _is_link(path):
            raise InstallError(f"Refusing a symbolic link or junction: {path}")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            result[relative] = ("directory", "")
            for child in sorted(path.iterdir()):
                visit(child)
        elif path.is_file():
            result[relative] = ("file", _digest(path))
        else:
            raise InstallError(f"Refusing a non-regular file: {path}")

    if source:
        skill_file = root / "SKILL.md"
        if not skill_file.is_file() or _is_link(skill_file):
            raise InstallError(f"Source must contain a regular SKILL.md file: {root}")
        for name in INCLUDED_FILES:
            path = root / name
            if path.exists() or _is_link(path):
                if not path.is_file() or _is_link(path):
                    raise InstallError(f"Expected a regular file: {path}")
                visit(path)
        for name in INCLUDED_DIRS:
            path = root / name
            if path.exists() or _is_link(path):
                if not path.is_dir() or _is_link(path):
                    raise InstallError(f"Expected a regular directory: {path}")
                visit(path)
    else:
        for child in sorted(root.iterdir()):
            visit(child)
    return result


def _copy_manifest(source: Path, stage: Path, manifest: dict[str, tuple[str, str]]) -> None:
    for relative, (kind, _) in manifest.items():
        original = source / relative
        _reject_link_components(original)
        destination = stage / relative
        if kind == "directory":
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination, follow_symlinks=False)
    if _manifest(stage, source=False) != manifest:
        raise InstallError("Source changed while it was copied; run the installer again.")


def install(source: Path, destination: Path | None = None, *, force: bool = False) -> InstallResult:
    """Install into destination/SKILL_NAME, preserving an existing install by default.

    Replacement stages a complete copy on the destination filesystem. An existing
    installation is renamed to a unique backup before the final atomic rename.
    If that final rename fails, the previous installation is restored.
    """
    source = _absolute(Path(source))
    parent = _absolute(Path(destination) if destination is not None else default_destination())
    target = parent / SKILL_NAME
    _reject_link_components(source)
    _reject_link_components(target)
    if not source.is_dir():
        raise InstallError(f"Source directory does not exist: {source}")
    if target == source:
        return InstallResult("same-location", target)
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise InstallError("Source and installation must not contain each other.")
    if target.exists() and not target.is_dir():
        raise InstallError(f"Installation path exists but is not a directory: {target}")

    manifest = _manifest(source, source=True)
    if target.exists():
        if _manifest(target, source=False) == manifest:
            return InstallResult("unchanged", target)
        if not force:
            return InstallResult("exists", target)

    parent.mkdir(parents=True, exist_ok=True)
    _reject_link_components(parent)
    stage = Path(tempfile.mkdtemp(prefix=f".{SKILL_NAME}-install-", dir=parent))
    backup: Path | None = None
    try:
        _copy_manifest(source, stage, manifest)
        # Recheck just before changing any existing installation.
        _reject_link_components(target)
        if target.exists():
            if not force:
                return InstallResult("exists", target)
            if not target.is_dir():
                raise InstallError(f"Installation path is not a directory: {target}")
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = parent / f"{SKILL_NAME}.backup-{timestamp}-{uuid.uuid4().hex[:8]}"
            target.rename(backup)
        try:
            stage.rename(target)
        except OSError as install_error:
            if backup is not None:
                try:
                    backup.rename(target)
                except OSError as rollback_error:
                    raise InstallError(
                        f"Installation failed and automatic rollback failed. "
                        f"Your previous installation is preserved at {backup}. "
                        f"Install error: {install_error}; rollback error: {rollback_error}"
                    ) from rollback_error
            detail = "the previous installation was restored" if backup is not None else "no installation was created"
            raise InstallError(f"Installation failed; {detail}: {install_error}") from install_error
        return InstallResult("updated" if backup is not None else "installed", target, backup)
    finally:
        # Only our own temporary staging directory is eligible for deletion.
        if stage.exists():
            if stage.parent != parent or not stage.name.startswith(f".{SKILL_NAME}-install-") or _is_link(stage):
                raise InstallError(f"Refusing to clean an unexpected staging path: {stage}")
            shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the LatentSync Colab Codex skill (Python 3.10+).")
    parser.add_argument("--dest", type=Path, help="Parent skills directory (default: $CODEX_HOME/skills or ~/.codex/skills).")
    parser.add_argument("--force", action="store_true", help="Update a changed installation, retaining a timestamped backup.")
    args = parser.parse_args(argv)
    try:
        result = install(Path(__file__).resolve().parent.parent, args.dest, force=args.force)
    except (InstallError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    messages = {
        "installed": "Installed",
        "updated": "Updated",
        "unchanged": "Unchanged; this version is already installed",
        "same-location": "Already running from the installation directory; no files changed",
        "exists": "An existing installation was retained. Use --force to update it with a backup",
    }
    print(f"{messages[result.status]}: {result.target}")
    if result.backup is not None:
        print(f"Previous installation backup: {result.backup}")
    if result.status in {"installed", "updated"}:
        print("Start a new Codex chat to load the skill, then ask to use $latentsync-colab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
