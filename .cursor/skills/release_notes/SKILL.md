---
name: release-notes
description: Generate Trisynapse Memory release notes from git tags, commits, diffs, and working-tree changes. Use when the user asks for release notes, a GitHub release body, a changelog, or a summary of changes between versions.
---

# Trisynapse Memory Release Notes

Write concise, user-facing GitHub release notes from the actual code changes. Do not repeat commit messages or list files.

Save the result to `docs/releases/vX.Y.Z.md`. Add or update the version row in the README **Release notes** table. Also return the Markdown in the reply. Do not bump versions or create tags unless the user asks.

## Project

| Item | Value |
|---|---|
| Product | Trisynapse Memory — local-first Trace & Recall memory for agents |
| Repo | https://github.com/shanmukh05/trisynapse-memory |
| Tags | `vX.Y.Z` (example: `v0.1.1`) |
| Canonical version | `pyproject.toml` → `[project].version` |
| Version script | `uv run python scripts/version.py current` / `check --tag vX.Y.Z` |
| Python package | `trisynapse-memory` on PyPI |
| CLI | `trisynapse-memory` |
| TypeScript package | `@trisynapse/trisynapse-memory` on npm |
| Other surfaces | REST API, Memory Studio, macOS/Linux `install.sh`, Windows `install.ps1` |

Use the project's terms: **Trace**, **Recall**, **Formation**, **Query Run**, **MemoryEngine**, **Studio**, **namespace**.

## Workflow

### 1. Determine the range

```bash
git tag --sort=-version:refname | head -10
git describe --tags --abbrev=0
```

| User asked for | Range |
|---|---|
| Explicit tags (`v0.1.0` → `v0.1.1`) | `v0.1.0..v0.1.1` |
| Published version | previous tag → that tag |
| Upcoming / unreleased version | previous tag → `HEAD`, **plus the working tree** |
| No tags and no clear boundary | inspect recent commits; ask before summarizing the whole repo |

Unreleased work in this repo is often still uncommitted. Include `git diff HEAD` and untracked engine/docs/package files. Ignore `.cursor/`, `docs/local/`, caches, and generated build artifacts.

### 2. Gather evidence

```bash
git log <range> --pretty=format:"%h %s"
git diff <range> --stat
git diff <range> --name-status
gh release view vX.Y.Z
```

Commit messages and existing GitHub release bodies are hints only. Verify important claims against the diff and tests.

### 3. Inspect surfaces in this order

1. Public Python API (`src/trisynapse_memory/__init__.py`, `engine/memory.py`, `engine/models.py`)
2. CLI (`cli.py`) and terminal (`terminal.py`)
3. REST (`api.py`) and `docs/api.md`
4. TypeScript SDK (`packages/js-sdk/`)
5. Studio (`packages/studio/src/`)
6. Config, store schema, and migrations
7. Installers and `scripts/version.py`
8. Release workflows (only if they change what users download)
9. Docs and tests (as evidence, not as bullets)

Read targeted file diffs. Do not dump the whole repository diff.

### 4. Classify, then write

Use only the sections that have real changes. Merge small releases into fewer sections.

Treat these as breaking when the diff shows them:

- Renamed or removed CLI commands/flags
- Renamed or removed Python/TS exports
- Changed REST JSON fields
- Changed store schema or installer defaults
- Renamed published package names

Existing stores may migrate automatically. Still call out the API/field change and the migration path.

## Ignore

Formatting, lint, comment-only edits, test refactors, CI cleanup, lockfile churn, generated files, `docs/local/**`, and internal module moves — unless they change a public import, CLI, REST field, or published package.

Do not list every dependency bump. Mention a dependency only when it enables a feature, changes compatibility, or has a security impact.

## Output

Write one file per version:

```text
docs/releases/v0.1.2.md
```

```markdown
# vX.Y.Z

One to three sentences on the release theme.

## ✨ Highlights

- User-facing change

## 🚀 New Features

- Added ...

## 🔧 Improvements

- Improved ...

## 🐛 Bug Fixes

- Fixed ...

## ⚠️ Breaking Changes

- Changed ...
  - **Migration:** ...

## 📦 Other Changes

- Updated ...
```

For a first public release, describe the product users received — not “initial commit.”

End with a real compare URL. Never invent a repository:

```markdown
**Full Changelog:** https://github.com/shanmukh05/trisynapse-memory/compare/v0.1.0...v0.1.1
```

For an unreleased version, use `v<previous>...HEAD`.

## Writing

Prefer what a user can do:

- “The CLI now shows progress while validating a store.”
- “Python clients configure retrieval routes through `RouteRegistry`.”

Avoid implementation narration:

- “Refactored `engine/retrieval.py` into a package.”

Rules:

1. Never fabricate changes.
2. Prefer the diff when a commit message disagrees.
3. Do not describe unfinished or dead code as shipped.
4. Combine commits that implement one feature.
5. Do not modify the project version unless asked.
6. After writing `docs/releases/vX.Y.Z.md`, confirm every major bullet against the diff.
