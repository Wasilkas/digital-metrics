"""Create checked local version commits and tags after contributor commits."""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

RELEASE_VERSION = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)")
RELEASE_FILES = ("pyproject.toml", "uv.lock", "CHANGELOG.md")
RETRY = "uv run --locked --no-sync python scripts/local_release.py"
GUARD = "DIGITAL_METRICS_RELEASE_ACTIVE"
HOOK_MARKER = "# digital-metrics: local-release post-commit launcher"


class ReleaseError(Exception):
    """An operation cannot safely finish without contributor intervention."""


class Attempt(BaseModel):
    """Persist ownership before modifying tracked files, for tag-only retries."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    version: str = Field(pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
    changelog_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @property
    def tag(self) -> str:
        return f"v{self.version}"

    @property
    def subject(self) -> str:
        return f"chore(release): {self.version}"


def run(root: Path, *args: str) -> str:
    env = os.environ.copy()
    env.pop("PSR_BUILD_METADATA", None)
    env[GUARD] = "1"
    result = subprocess.run(  # noqa: S603 - argument lists, never a shell
        args, cwd=root, env=env, text=True, capture_output=True, check=False
    )
    if result.returncode:
        detail = (result.stderr + result.stdout).strip()
        raise ReleaseError(f"{args[0]} {' '.join(args[1:])} failed: {detail}")
    return result.stdout.strip()


def git(root: Path, *args: str) -> str:
    return run(root, "git", *args)


def git_path(root: Path, name: str) -> Path:
    return Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", name))


def skip_reason(root: Path, *, post_commit: bool) -> str | None:
    if git(root, "rev-parse", "--abbrev-ref", "HEAD") != "main":
        return "only main receives automatic release tags"
    operations = (
        "rebase-merge",
        "rebase-apply",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "MERGE_HEAD",
        "sequencer",
    )
    if any(git_path(root, name).exists() for name in operations):
        return "a Git history operation is in progress"
    if post_commit:
        action = git(root, "reflog", "-1", "--format=%gs")
        if action.startswith(("commit (amend):", "cherry-pick:", "rebase", "revert:")):
            return "history rewrite commits require an explicit release retry"
    return None


def install_hooks(root: Path) -> None:
    """Install a post-commit launcher that does not stash away dirty tracked files."""
    hook = git_path(root, "hooks/post-commit")
    if hook.is_symlink() or (hook.exists() and HOOK_MARKER not in hook.read_text().splitlines()):
        raise ReleaseError(
            f"Existing post-commit hook at {hook} is not owned by this project. "
            "Preserve it and integrate its behavior manually before installing this launcher."
        )
    run(root, sys.executable, "-m", "pre_commit", "install", "--hook-type", "pre-commit")
    launcher = (
        f"#!/usr/bin/env bash\n{HOOK_MARKER}\n"
        f"exec {shlex.quote(sys.executable)} -m pre_commit run "
        "--hook-stage post-commit --all-files\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", dir=hook.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(launcher)
    try:
        temporary.chmod(0o755)
        temporary.replace(hook)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Installed pre-commit checks and local post-commit release hook.")


@contextmanager
def release_lock(root: Path) -> Iterator[None]:
    common = Path(git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    lock = common / "digital-metrics-release.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ReleaseError(
            f"Release lock exists at {lock}. Inspect owner.json and verify that no release "
            "process is running before manually removing a stale lock."
        ) from exc
    owner = lock / "owner.json"
    try:
        owner.write_text(json.dumps({"pid": os.getpid(), "worktree": str(root)}) + "\n")
        yield
    finally:
        owner.unlink(missing_ok=True)
        lock.rmdir()


def require_clean(root: Path) -> None:
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ReleaseError(
            "Release deferred: dirty tracked worktree or index. Inspect retained changes; "
            "if a release attempt failed, finish only its pyproject.toml, uv.lock and "
            "CHANGELOG.md edits "
            "and commit them with the recorded chore(release): VERSION subject before retrying."
        )


def validate_metadata(root: Path, attempt: Attempt) -> None:
    """Allow only the root project version to change in either parsed TOML document."""
    project_before = tomllib.loads(git(root, "show", f"{attempt.source}:pyproject.toml"))
    project_after = tomllib.loads((root / "pyproject.toml").read_text())
    if project_after["project"]["version"] != attempt.version:
        raise ReleaseError("Package version does not match the release attempt.")
    project_after["project"]["version"] = project_before["project"]["version"]
    if project_after != project_before:
        raise ReleaseError("Release changed project metadata beyond the version.")

    lock_before = tomllib.loads(git(root, "show", f"{attempt.source}:uv.lock"))
    lock_after = tomllib.loads((root / "uv.lock").read_text())
    name = project_before["project"]["name"]
    packages = [entry for entry in lock_after["package"] if entry["name"] == name]
    if len(packages) != 1 or packages[0]["version"] != attempt.version:
        raise ReleaseError("Lockfile root package version does not match the release attempt.")
    packages[0]["version"] = project_before["project"]["version"]
    if lock_after != lock_before:
        raise ReleaseError("Lock refresh changed dependencies or sources; inspect retained files.")


def validate_release_commit(root: Path, attempt: Attempt) -> None:
    if git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()[1:] != [attempt.source]:
        raise ReleaseError("HEAD is not the direct release child of the recorded source commit.")
    if git(root, "log", "-1", "--format=%s") != attempt.subject:
        raise ReleaseError(f"Expected release commit subject: {attempt.subject}")
    paths = git(root, "diff", "--name-only", attempt.source, "HEAD").splitlines()
    if set(paths) != set(RELEASE_FILES):
        raise ReleaseError(
            "Release commit must change only pyproject.toml, uv.lock and CHANGELOG.md."
        )
    require_clean(root)
    validate_metadata(root, attempt)
    if (
        attempt.changelog_sha256 is None
        or hashlib.sha256((root / "CHANGELOG.md").read_bytes()).hexdigest()
        != attempt.changelog_sha256
    ):
        raise ReleaseError(
            "Changelog does not match the prepared release; inspect recovery metadata."
        )


def tag_exists(root: Path, tag: str) -> bool:
    return tag in git(root, "tag", "--list", tag).splitlines()


def finish_tag(root: Path, attempt: Attempt, *, retrying: bool) -> None:
    validate_release_commit(root, attempt)
    head = git(root, "rev-parse", "HEAD")
    if tag_exists(root, attempt.tag):
        if (
            git(root, "rev-parse", f"{attempt.tag}^{{commit}}") == head
            and git(root, "cat-file", "-t", attempt.tag) == "tag"
        ):
            return
        raise ReleaseError(f"Tag conflict: {attempt.tag} already exists; it will not be replaced.")
    if retrying:
        run(
            root,
            sys.executable,
            "-m",
            "pre_commit",
            "run",
            "--all-files",
            "--hook-stage",
            "pre-commit",
        )
        validate_release_commit(root, attempt)
        if git(root, "rev-parse", "HEAD") != head:
            raise ReleaseError("HEAD changed while validating the release commit.")
    git(root, "tag", "-a", attempt.tag, head, "-m", attempt.tag)


def semantic_release(root: Path, *options: str) -> str:
    return run(
        root,
        sys.executable,
        "-m",
        "semantic_release",
        "version",
        "--changelog",
        "--skip-build",
        "--no-commit",
        "--no-tag",
        "--no-push",
        "--no-vcs-release",
        *options,
    )


def next_attempt(root: Path) -> Attempt | None:
    reachable = git(root, "tag", "--merged", "HEAD", "--list", "v*").splitlines()
    if not any(RELEASE_VERSION.fullmatch(tag[1:]) for tag in reachable):
        raise ReleaseError("No reachable vX.Y.Z baseline tag; restore release history first.")
    version = semantic_release(root, "--print")
    if not RELEASE_VERSION.fullmatch(version):
        raise ReleaseError(f"Unexpected semantic-release version: {version!r}")
    attempt = Attempt(source=git(root, "rev-parse", "HEAD"), version=version)
    if attempt.tag in reachable:
        return None
    if tag_exists(root, attempt.tag):
        raise ReleaseError(f"Tag conflict: {attempt.tag} exists outside current release history.")
    return attempt


def refresh_changelog(root: Path) -> bool:
    """Generate committed history without replacing a prepared, untagged release."""
    if git_path(root, "digital-metrics-release.json").exists():
        logger.info("Preserving changelog for the recorded release attempt.")
        return False
    if git(root, "rev-parse", "--is-shallow-repository") == "true":
        raise ReleaseError("Shallow history cannot generate a complete changelog.")
    if git(root, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD":
        logger.info("Changelog refresh skipped for detached HEAD.")
        return False
    path = root / "CHANGELOG.md"
    before = path.read_bytes() if path.exists() else None
    config = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["semantic_release"]
    # PSR requires a release branch even for changelog-only rendering. Override that
    # restriction in a temporary config; the project's main-only version policy stays intact.
    config["branches"] = {"changelog": {"match": ".*", "prerelease": False}}
    with tempfile.TemporaryDirectory(prefix="digital-metrics-changelog-") as directory:
        temporary = Path(directory) / "config.json"
        temporary.write_text(json.dumps({"semantic_release": config}))
        run(
            root,
            sys.executable,
            "-m",
            "semantic_release",
            "--strict",
            "--config",
            str(temporary),
            "changelog",
        )
    return path.read_bytes() != before


def prepare_commit(root: Path, attempt: Attempt) -> Attempt:
    semantic_release(root)
    run(root, "uv", "lock", "--offline")
    validate_metadata(root, attempt)
    paths = git(root, "diff", "HEAD", "--name-only").splitlines()
    if not git(root, "ls-files", "--", "CHANGELOG.md"):
        paths.append("CHANGELOG.md")
    if set(paths) != set(RELEASE_FILES) or git(root, "rev-parse", "HEAD") != attempt.source:
        raise ReleaseError("Source commit or files changed during release preparation.")
    attempt = attempt.model_copy(
        update={
            "changelog_sha256": hashlib.sha256((root / "CHANGELOG.md").read_bytes()).hexdigest()
        }
    )
    git_path(root, "digital-metrics-release.json").write_text(attempt.model_dump_json() + "\n")
    git(root, "add", "--", *RELEASE_FILES)
    # A contributor can stage unrelated work after our preflight. A path-limited
    # commit preserves it in the index rather than including it in our release.
    git(root, "commit", "--only", "-m", attempt.subject, "--", *RELEASE_FILES)
    return attempt


def release(root: Path) -> None:
    if git(root, "rev-parse", "--is-shallow-repository") == "true":
        raise ReleaseError("Shallow history cannot be released locally; obtain full history first.")
    with release_lock(root):
        require_clean(root)
        record = git_path(root, "digital-metrics-release.json")
        if record.exists():
            attempt = Attempt.model_validate_json(record.read_text())
            if git(root, "rev-parse", "HEAD") != attempt.source:
                finish_tag(root, attempt, retrying=True)
                record.unlink()
                logger.info("Recovered local release {}", attempt.tag)
                return
            # A failed attempt with restored source files can be restarted, but never
            # reuse its version blindly after history or release tags have changed.
            if next_attempt(root) != attempt.model_copy(update={"changelog_sha256": None}):
                raise ReleaseError("Recorded release candidate changed; inspect recovery metadata.")
        else:
            candidate = next_attempt(root)
            if candidate is None:
                logger.info("No unreleased semantic changes.")
                return
            attempt = candidate
            with record.open("x") as stream:
                stream.write(attempt.model_dump_json() + "\n")
        attempt = prepare_commit(root, attempt)
        finish_tag(root, attempt, retrying=False)
        record.unlink()
        logger.info("Created local release {}.", attempt.tag)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--install-hooks", action="store_true", help="install both local Git hooks")
    mode.add_argument("--post-commit", action="store_true", help="invoked automatically by Git")
    mode.add_argument(
        "--changelog", action="store_true", help="refresh committed changelog history"
    )
    args = parser.parse_args()
    if os.environ.get(GUARD) == "1":
        return 0
    try:
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
        if args.changelog:
            if refresh_changelog(root):
                logger.info(
                    "Changelog updated. Review and stage CHANGELOG.md, then retry the commit."
                )
                return 1
        elif args.install_hooks:
            install_hooks(root)
        else:
            reason = skip_reason(root, post_commit=args.post_commit)
            if reason:
                logger.info("Release skipped: {}.", reason)
            else:
                release(root)
    except (ReleaseError, OSError, ValueError, KeyError, ValidationError) as exc:
        logger.error(
            "{}. The original commit is retained. Retry after resolving this: {}", exc, RETRY
        )
        return 1
    except KeyboardInterrupt:
        logger.error("Release interrupted. The original commit is retained. Retry: {}", RETRY)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
