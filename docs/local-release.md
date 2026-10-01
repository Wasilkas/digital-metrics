# Local semantic releases

Python Semantic Release analyzes Conventional Commits on `main`. The local
release helper follows the parent project's workflow, using this repository's
checks. No remote operations, CI/CD, package builds or publishing are involved.

## Setup and commands

```bash
uv sync --locked --dev
uv run --locked --no-sync python scripts/local_release.py --install-hooks
```

Installation enables `pre-commit` checks and a `post-commit` release hook. It
does not create a release and refuses to overwrite a foreign `post-commit` hook.
Hooks are local to the clone and must be installed in each checkout.

```bash
# Create a release manually, including after a fast-forward merge.
uv run --locked --no-sync python scripts/local_release.py
# Refresh the changelog without making a commit or tag.
uv run --locked --no-sync python scripts/local_release.py --changelog
```

`fix:` and `perf:` bump PATCH; `feat:` bumps MINOR. Breaking changes (`!` or
`BREAKING CHANGE:`) bump MINOR in the `0.x` series (`major_on_zero = false`).
Other categories, including `docs:`, `chore:` and `test:`, do not trigger a
version bump alone. All commits since the last reachable `vX.Y.Z` tag contribute
to the next release. A reachable baseline tag and complete history are required;
the helper never fetches missing history. The migration baseline `v0.5.3`
identifies the existing 0.5.3 version commit, not the new tooling commit.

The helper updates only `pyproject.toml`, the root package version in `uv.lock`,
and `CHANGELOG.md`. It verifies that dependency versions and sources did not
change, then creates `chore(release): X.Y.Z` through the normal checks and an
annotated `vX.Y.Z` tag. The release hook prevents recursion; checks remain enabled.

Releases skip other branches, detached HEAD, and active Git history operations.
Automatic releases also skip amend, cherry-pick, revert and rebase commits.
Dirty tracked files or staged changes defer the release without discarding them.
Unrelated untracked files are retained and excluded from the release commit.

## Automatic changelog

The pre-commit hook regenerates `CHANGELOG.md` from committed history through
[the template](../templates/CHANGELOG.md.j2). It shows pending commits under
`Unreleased`, groups released commits by version and category, and excludes
generated release commits. The curated notes through 0.5.3 are appended verbatim
from [changelog history](changelog-history.md). Edit that source to correct older
notes; manual changes to generated entries are overwritten.

If generation changes the file, the hook stops without staging it. Review the
result, stage `CHANGELOG.md`, and retry. The current commit is not in history
until committed; its entry appears during release preparation or the next
refresh. `--changelog` returns 1 when it updates the file or encounters an error,
and 0 when unchanged. Refresh works on named feature branches and skips detached
HEAD. During an unfinished release it preserves the prepared notes for recovery.

## Recovery

A post-commit failure retains the original commit; Git may still report that
commit as successful. Read the hook's output. If release checks fail, inspect
and repair the retained metadata changes, then complete the exact release commit:

```bash
git diff HEAD -- pyproject.toml uv.lock CHANGELOG.md
# Replace X.Y.Z with the version reported by the failed attempt.
git add -- pyproject.toml uv.lock CHANGELOG.md
git commit -m "chore(release): X.Y.Z"
uv run --locked --no-sync python scripts/local_release.py
```

If the release commit already exists but tag creation failed, rerunning the
helper validates that commit, reruns checks and adds the missing tag without
another commit. It refuses conflicting tags, unrelated follow-up commits,
dependency drift and changes to the prepared changelog's recorded SHA-256.

Attempt state is stored at `git rev-parse --git-path digital-metrics-release.json`.
A shared `digital-metrics-release.lock` directory under
`git rev-parse --git-common-dir` prevents concurrent releases across worktrees.
After a crash, inspect its `owner.json` and confirm that the recorded process
has stopped before manually removing a stale lock. Do not remove attempt state
until its changes have been inspected and completed or canceled manually.
