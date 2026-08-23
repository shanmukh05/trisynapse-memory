# Production release runbook

This runbook takes Trisynapse Memory from a finished source tree to a tested GitHub release, verified installers, and a production service. It is intentionally operational: follow it in order for every release.

Shell examples use `RELEASE_VERSION=0.2.0` as a placeholder. Set it to the version being released before running a section. Versions in files do **not** include the leading `v`; Git tags do.

```bash
RELEASE_VERSION=0.2.0
```

## 1. Know what the release produces

Pushing a tag such as `v0.2.0` starts [`.github/workflows/release.yml`](../../.github/workflows/release.yml). A tag points to an existing commit; it does not create a branch. The workflow:

1. tests Python 3.11 and 3.12 on Linux, macOS, and Windows;
2. checks the TypeScript client and checks, unit-tests, and builds Studio;
3. builds and publishes `@trisynapse/trisynapse-memory` to npm using trusted publishing;
4. installs the published npm version and imports its public client;
5. builds the Python wheel and source distribution;
6. publishes `trisynapse-memory` to PyPI using trusted publishing;
7. creates a GitHub Release and uploads the distributions, installers, release metadata, and checksums;
8. installs the published PyPI version on Linux, macOS, and Windows and runs `check`.

The expected GitHub Release assets are:

```text
trisynapse_memory-VERSION-py3-none-any.whl
trisynapse_memory-VERSION.tar.gz
install.sh
install.ps1
release.json
SHA256SUMS
```

The installers are small wrappers around `uv tool install`. They do not contain Python, models, or the application itself. The version must already exist on PyPI before an installer can install it.

The Python and TypeScript versions are synchronized by `scripts/version.py`. A tag is considered successful only when both registries publish the same version and their smoke tests pass.

## 2. Production gates

Do not push a release tag until every required gate below passes.

| Gate | Required result |
|---|---|
| Version consistency | Tag, Python package, engine, Studio, TypeScript, installers, and README agree |
| Offline tests | Ruff, the complete Python suite, TypeScript check, Studio unit tests, and Studio browser tests pass locally |
| Production build | Studio is rebuilt before the wheel; wheel contains hashed Studio assets and the logo |
| npm package | Packed SDK contains only the compiled client, declarations, package metadata, and README |
| Store validation | `trisynapse-memory validate` reports a consistent store and all retained source blobs are present |
| Retrieval benchmark | Current-schema LoCoMo and LongMemEval production artifacts pass `bench gate --mode retrieval` |
| Live provider smoke | Every provider/model intended for the deployment has an explicit successful connection test |
| Installer smoke | Published installer works in clean Linux, macOS, and Windows environments |
| Restore drill | A backup restores into an empty directory and the restored store validates |
| Security | No `.env`, provider key, API token, store, benchmark cache, or private source is committed or included in an image |

At the time this runbook was added, `bench gate --mode retrieval --data-root data` reports that no current production retrieval artifacts exist. Generate and pass those artifacts before calling a build production-ready. The tag workflow does not run the benchmark gate automatically.

End-to-end benchmarks are recommended before a public release that changes extraction, synthesis, prompts, provider transports, or answer generation. They make billable model calls and must be kept separate from retrieval-only artifacts.

## 3. One-time GitHub, PyPI, and npm setup

### GitHub repository

Use the canonical repository:

```text
https://github.com/shanmukh05/trisynapse-memory
```

In GitHub repository settings:

1. Allow GitHub Actions to run.
2. Protect `main` and require the `test` workflow before merge.
3. Disable force pushes and branch deletion for `main`.
4. Require pull requests for changes to release workflows and installers.
5. Create an Actions environment named exactly `pypi`.
6. Create an Actions environment named exactly `npm`.
7. Add required reviewers to the registry environments if releases should need manual approval.

The release workflow already requests `contents: write` and `id-token: write`. It does not need a long-lived PyPI API token.

### PyPI trusted publisher

On PyPI, create or configure the `trisynapse-memory` project and add a trusted publisher with these exact values:

| Field | Value |
|---|---|
| PyPI project | `trisynapse-memory` |
| Owner | `shanmukh05` |
| Repository | `trisynapse-memory` |
| Workflow | `release.yml` |
| Environment | `pypi` |

For the first publication, use PyPI's pending trusted publisher if the project does not exist yet. Confirm that the GitHub repository is public or otherwise meets PyPI's trusted-publishing requirements.

### npm package and trusted publisher

The public SDK package is the scoped package `@trisynapse/trisynapse-memory`. The scope marker is part of its name; `trisynapse-memory`, `trisynapse/trisynapse-memory`, and `@trisynapse/memory` are different package specifications.

The canonical package already exists. Verify the currently published baseline before configuring automation:

```bash
npm view '@trisynapse/trisynapse-memory@0.1.1' version
```

Do not try to republish `0.1.1`: npm versions are immutable. A new release must use a new version. If setting up a fork whose package has never been published, publish its first version manually from a clean checkout, then configure trusted publishing; this bootstrap step is not needed for the canonical repository.

Create or sign in to an npm account that has write access to `@trisynapse/trisynapse-memory`, has account-level two-factor authentication enabled, and is authorized to manage package settings. Configure one GitHub Actions trusted publisher using either the npm website or npm CLI.

In the npm website, open the package, select **Settings**, and add a trusted publisher with these exact values:

| Field | Value |
|---|---|
| GitHub organization or user | `shanmukh05` |
| Repository | `trisynapse-memory` |
| Workflow filename | `release.yml` |
| Environment | `npm` |
| Allowed action | `npm publish` |

The release workflow uses a GitHub-hosted runner, Node.js 24, npm's public registry, and the existing `id-token: write` permission. npm exchanges the workflow identity for a short-lived publishing credential and generates provenance automatically; no `NPM_TOKEN` is used.

To inspect or create the same relationship from the CLI, use npm 11.15.0 or newer:

```bash
npm install --global npm@latest
npm --version
npm login
npm whoami
npm trust list '@trisynapse/trisynapse-memory'
npm trust github '@trisynapse/trisynapse-memory' \
  --repo shanmukh05/trisynapse-memory \
  --file release.yml \
  --env npm \
  --allow-publish
npm trust list '@trisynapse/trisynapse-memory'
```

Run `npm trust github` only when the package does not already have the intended relationship. A package can have only one trusted publisher relationship. To replace one, get its ID from `npm trust list` and revoke that exact relationship before creating the replacement:

```bash
npm trust revoke '@trisynapse/trisynapse-memory' --id TRUST_ID
```

The workflow value is the filename `release.yml`, not `.github/workflows/release.yml`. Since npm requires an allowed action, include `--allow-publish` when configuring through the CLI.

Once a tagged release has published successfully, set the package's publishing access to require two-factor authentication and disallow traditional tokens. Keep the trusted publisher enabled. Every later `v*` tag will build, publish, and install-test the matching npm version before PyPI and the GitHub Release are published.

Before creating an automated tag, verify the existing package and trust relationship:

```bash
npm view '@trisynapse/trisynapse-memory@0.1.1' version
npm trust list '@trisynapse/trisynapse-memory'
```

After the release workflow publishes the new version, verify it separately:

```bash
npm view "@trisynapse/trisynapse-memory@${RELEASE_VERSION}" version
```

If `npm trust` reports `Unknown command`, upgrade to npm 11.15.0 or newer. The npm website can be used instead. If publishing reports `ENEEDAUTH`, recheck the authenticated account, case-sensitive workflow filename, repository owner, repository name, GitHub environment, `id-token: write` permission, and the package's exact `repository.url`. See npm's official [trusted publishing guide](https://docs.npmjs.com/trusted-publishers/) and [`npm trust` reference](https://docs.npmjs.com/cli/v11/commands/npm-trust/).

### Maintainer workstation

Install:

- Git and access to the GitHub repository;
- Python 3.11 or newer;
- [`uv`](https://docs.astral.sh/uv/);
- Node.js 24 and npm 11.15.0 or newer;
- pnpm 10;
- Docker with Compose, if the container deployment will be tested;
- GitHub CLI (`gh`), optional but useful for watching workflows and downloading assets.

Check the toolchain:

```bash
git --version
uv --version
node --version
npm --version
pnpm --version
docker --version
gh auth status
```

## 4. Put the current source under Git

First check whether the working directory already has Git metadata:

```bash
git rev-parse --show-toplevel
git remote -v
```

If the first command fails and the GitHub repository is empty, initialize this directory:

```bash
git init
git branch -M main
git remote add origin git@github.com:shanmukh05/trisynapse-memory.git
```

Use the HTTPS remote instead if SSH is not configured:

```bash
git remote add origin https://github.com/shanmukh05/trisynapse-memory.git
```

If the GitHub repository already contains commits, do not force-push this directory over it. Clone the repository, copy the source into that clone while preserving `.git`, review the diff, and commit normally.

Before the first commit, verify that local secrets and generated data are ignored:

```bash
git check-ignore .env.local .venv dist var node_modules
git status --short --ignored
```

Never commit:

- `.env`, `.env.local`, or real provider credentials;
- a memory store, retained source blobs, backups, or `.api-key`;
- `dist/`, `build/`, `.venv/`, `node_modules/`, caches, or generated `*.egg-info`;
- benchmark datasets or artifacts whose license does not permit redistribution.

Develop changes on normal short-lived feature branches and merge them through pull requests. Do not create a `release/VERSION` branch for pre-1.0 releases. The release itself is the tag applied to the reviewed commit on `main`, not a new branch. If the release changes are already merged, skip this example and continue on `main`:

```bash
git switch -c codex/release-preparation
git add .
git status --short
git diff --cached --check
git commit -m "Prepare Trisynapse Memory ${RELEASE_VERSION}"
git push -u origin codex/release-preparation
```

Open a pull request, let the entire `test` workflow pass, review the built Studio visually, then merge into `main`.

## 5. Choose and apply the version

Trisynapse Memory is pre-1.0. Use semantic versioning with this interpretation:

- patch (`0.1.1`) for compatible fixes;
- minor (`0.2.0`) for features or intentional pre-1.0 breaking changes;
- prerelease (`0.2.0rc1` with tag `v0.2.0rc1`) for a release candidate.

`pyproject.toml` is the canonical release version. Do not edit every version file manually. Set the new version once from the repository root:

```bash
uv run python scripts/version.py set "$RELEASE_VERSION"
```

The command validates the version, updates all duplicated release metadata, refreshes `uv.lock`, and then checks the result. If updating or locking fails, it restores every touched file. It updates:

| Surface | How it gets its version |
|---|---|
| `pyproject.toml` | Canonical `[project].version` edited by the command |
| Python runtime and `MemoryEngine.VERSION` | Resolved from `pyproject.toml` in a source tree or installed distribution metadata in a wheel |
| `uv.lock` | Refreshed by `uv lock` inside the command |
| `packages/js-sdk/package.json` | Updated by the command |
| `packages/studio/package.json` | Updated by the command |
| `install.sh` | Default pinned version updated by the command |
| `install.ps1` | Default pinned version updated by the command |
| README status | Version-independent pre-1.0 wording; no release edit required |

`pnpm-lock.yaml` does not store workspace package versions, so changing only the release version does not require regenerating it. Do not edit generated `dist/` or `src/trisynapse_memory.egg-info/` metadata by hand.

Stable versions are identical on PyPI and npm. For prereleases, the command converts the canonical PEP 440 spelling into npm SemVer: for example, `0.2.0rc1` and tag `v0.2.0rc1` publish as `0.2.0-rc.1` on npm. Prerelease SDKs use the npm `next` dist-tag, so they do not replace the stable `latest` version.

Inspect and verify the result:

```bash
uv run python scripts/version.py current
uv run python scripts/version.py check
uv run python scripts/version.py check --tag "v${RELEASE_VERSION}"
```

`check` compares the canonical version with the Python lock, both JavaScript manifests, and both installers. The optional `--tag` additionally requires the exact `v<version>` Git tag. CI runs the ordinary check on every push, and the release workflow runs the tag-aware check before testing or publishing.

Confirm installed Python metadata and runtime reporting explicitly when desired:

```bash
uv run python -c 'from importlib.metadata import version; from trisynapse_memory import __version__, MemoryEngine; expected = version("trisynapse-memory"); assert __version__ == expected == MemoryEngine.VERSION; print(expected)'
```

After changing the version, build Studio and run the normal test/build sequence. The tag must be `v${RELEASE_VERSION}`. The installers are pinned automatically because the uploaded scripts are used unchanged.

### Prepare release notes

Prepare the release notes before tagging even though the workflow creates the GitHub Release automatically. Use this structure:

```markdown
## Trisynapse Memory VERSION

### Highlights
- User-visible outcome, not an internal implementation detail.

### Compatibility and migrations
- Python/Node/OS requirements.
- Store migration and rollback constraints.
- Breaking Python, REST, TypeScript, CLI, or Studio changes.

### Providers and ingestion
- New providers, model behavior, source types, limits, or security changes.

### Verification
- Test matrix and passing benchmark artifact IDs.

### Installation
- Versioned macOS/Linux and Windows commands.

### Known limitations
- Production-relevant issues users must understand before upgrading.
```

After the workflow creates the release, edit its description in GitHub and paste the reviewed notes. The current workflow does not supply an authored release body. If the tag is a release candidate, also mark the GitHub Release as a prerelease; the workflow does not infer that policy for you.

## 6. Run the complete local test suite

Start from a clean dependency state:

```bash
uv sync --extra dev --extra all
pnpm install --frozen-lockfile
```

### Python engine, REST contracts, CLI, and migrations

```bash
uv run ruff check src tests scripts
uv run --extra dev --extra all python -m pytest -q
```

For faster diagnosis, the main surfaces map to these tests:

```bash
# Trace, Recall, retrieval, and synchronous REST
uv run --extra dev --extra all python -m pytest -q \
  tests/test_trace_recall_engine.py

# Unified sources, CLI, terminal, archives, code, images, and removal
uv run --extra dev --extra all python -m pytest -q \
  tests/test_ingestion.py tests/test_productization.py

# Providers, model configuration, rebuild safety, and model CLI/REST
uv run --extra dev --extra all python -m pytest -q \
  tests/test_model_configuration.py

# REST, Studio backend, query runs, source preview, graph, and retrieval settings
uv run --extra dev --extra all python -m pytest -q \
  tests/test_api.py

# Benchmark adapters, provenance, prompts, and release gate logic
uv run --extra dev --extra all python -m pytest -q \
  tests/test_benchmark_adapters.py tests/test_prompts_and_benchmarks.py
```

The automated provider tests use mocks and do not prove that a real credential, account, quota, or selected model works. Live provider testing is a separate step below.

### TypeScript client

```bash
pnpm --filter @trisynapse/trisynapse-memory check
pnpm --filter @trisynapse/trisynapse-memory build
(cd packages/js-sdk && npm pack --dry-run)
```

The dry run must show the compiled `dist/` client and declarations, `package.json`, and package README without source-tree secrets, caches, Studio assets, or repository-only files.

### Studio unit, production, responsive, and accessibility tests

```bash
pnpm --filter @trisynapse/studio check
pnpm --filter @trisynapse/studio test
pnpm --filter @trisynapse/studio build
pnpm --filter @trisynapse/studio exec playwright install chromium
pnpm --filter @trisynapse/studio test:e2e
```

The Playwright suite starts a real local Trisynapse server, tests desktop and narrow layouts, checks the five-view shell and source modal, and runs automated accessibility checks.

### Build the Python distributions

Studio must be built before Python packaging because the wheel embeds `src/trisynapse_memory/studio/dist`.

```bash
pnpm --filter @trisynapse/studio build
uv build
```

Inspect the artifacts:

```bash
ls -lh dist/
unzip -l "dist/trisynapse_memory-${RELEASE_VERSION}-py3-none-any.whl" | \
  rg 'trisynapse_memory/(studio/dist|prompts)'
```

The wheel must contain:

- `studio/dist/index.html`;
- hashed JavaScript and CSS assets;
- the canonical logo asset;
- every versioned prompt under `prompts/`.

### Smoke-test the wheel, not the source tree

Create an isolated environment and install the local wheel with all user-facing extras:

```bash
uv venv .release-smoke
uv pip install --python .release-smoke/bin/python \
  "dist/trisynapse_memory-${RELEASE_VERSION}-py3-none-any.whl[all]"

.release-smoke/bin/trisynapse-memory --version
.release-smoke/bin/trisynapse-memory --path .release-smoke/store init
.release-smoke/bin/trisynapse-memory --path .release-smoke/store add observation \
  "Production smoke memory"
.release-smoke/bin/trisynapse-memory --path .release-smoke/store search \
  "Production smoke"
.release-smoke/bin/trisynapse-memory --path .release-smoke/store validate
.release-smoke/bin/trisynapse-memory --path .release-smoke/store --json check
```

On Windows, use `.release-smoke\Scripts\python.exe` and `.release-smoke\Scripts\trisynapse-memory.exe`.

`check.ok` must be true. Review `missing_credentials`, `failed_jobs`, source extras, store permissions, and store validation rather than checking only the process exit code.

Smoke-test the installed Python API as well:

```bash
.release-smoke/bin/python - <<'PY'
from pathlib import Path
from trisynapse_memory import MemoryEngine

store = Path(".release-smoke/python-store")
memory = MemoryEngine.open(store)
try:
    added = memory.add("Python package smoke memory", process=False)
    found = memory.search("Python package smoke", top_k=3)
    assert added.id
    assert found.hits
    validation = memory.validate_store()
    assert validation.ok
    print({"version": memory.VERSION, "store_valid": True})
finally:
    memory.close()
PY
```

The first use of the default local embedding model can download model files. Run the smoke where that download is expected, cached, and allowed, then separately test offline startup using the prepared cache expected in production.

## 7. Run benchmark gates

Retrieval mode exercises the production ingestion, indexing, routing, grounding, and citation path without answer-generation or judge calls:

```bash
trisynapse-memory bench run locomo \
  --data-root data/locomo --max-questions 100 --mode retrieval

trisynapse-memory bench run longmemeval \
  --data-root data/longmemeval --max-questions 25 --mode retrieval

trisynapse-memory bench gate --data-root data --mode retrieval
```

The gate requires current artifact schema, production architecture identity, provider/prompt provenance, a valid benchmark Trace, and these minimums:

| Suite | Questions | Pre-answer evidence recall@k |
|---|---:|---:|
| LoCoMo | 100 | 0.55 |
| LongMemEval | 25 | 0.80 |

Retrieval mode does not gate on answer token F1: without a completion model its extractive output is intentionally a Trace excerpt, not a benchmark-style short answer. Token F1 and judge accuracy belong to the end-to-end gate.

For an end-to-end release gate, configure and test a completion provider first, then run:

```bash
trisynapse-memory bench run locomo \
  --data-root data/locomo --max-questions 100 --mode end-to-end

trisynapse-memory bench run longmemeval \
  --data-root data/longmemeval --max-questions 25 --mode end-to-end

trisynapse-memory bench gate --data-root data --mode end-to-end
```

End-to-end mode additionally requires mean token F1 of at least `0.20` and judge accuracy of at least `0.50`. Its artifacts must record completion provider/model plus extraction, Episode Recall, answer, and benchmark-judge prompt provenance. Budget and rate-limit this run; it can make many billable calls.

Archive the exact passing artifact files with the release evidence. Do not let an older artifact satisfy a new algorithm release without deliberately reviewing its engine version and provenance.

## 8. Test real providers and models

Credentials stay in environment variables. Use a staging account or restricted key, never a personal production key in shell history, CI logs, or committed files.

Select models in a disposable store, then explicitly test each active role:

```bash
PROVIDER_TEST_STORE=.release-provider-smoke

trisynapse-memory --path "$PROVIDER_TEST_STORE" models providers
trisynapse-memory --path "$PROVIDER_TEST_STORE" models list \
  --role completion --provider openai --refresh
trisynapse-memory --path "$PROVIDER_TEST_STORE" models set \
  completion openai MODEL_ID
trisynapse-memory --path "$PROVIDER_TEST_STORE" models test \
  --role completion
```

Repeat for the selected embedding provider and model. The connection test can be billable. For a non-empty store, an embedding change requires `--yes` and performs a staged rebuild; use an empty disposable store for a simple release smoke.

If the release claims image support for a selected model, ingest a small non-sensitive PNG/JPEG/WebP and verify that visible text, description, and provenance are correct. A successful text request does not establish vision support.

## 9. Manual CLI acceptance test

Use a fresh store rather than a developer store:

```bash
SMOKE_ROOT="$(mktemp -d)"
SMOKE_STORE="$SMOKE_ROOT/store"

trisynapse-memory --path "$SMOKE_STORE" init
trisynapse-memory --path "$SMOKE_STORE" ingest README.md
trisynapse-memory --path "$SMOKE_STORE" sources list
trisynapse-memory --path "$SMOKE_STORE" query \
  "What is Trisynapse Memory?"
trisynapse-memory --path "$SMOKE_STORE" history
trisynapse-memory --path "$SMOKE_STORE" jobs list
trisynapse-memory --path "$SMOKE_STORE" validate
trisynapse-memory --path "$SMOKE_STORE" --json check
```

Also run `trisynapse-memory` in a real TTY and verify:

- the wide and narrow logos retain their shape;
- typed command recommendations appear;
- `/model`, `/ingest`, `/sources`, `/search`, `/timeline`, `/history`, `/remove`, `/jobs`, `/check`, and `/help` open or execute correctly;
- Ctrl+C cancels the current interaction without corrupting the store;
- machine-readable `--json` output contains no logo, progress display, or ANSI escape sequences.

## 10. Manual REST and Studio acceptance test

Use two terminals. In terminal one:

```bash
REST_SMOKE_ROOT="$(mktemp -d)"
REST_SMOKE_STORE="$REST_SMOKE_ROOT/store"

trisynapse-memory --path "$REST_SMOKE_STORE" init
export TRISYNAPSE_MEMORY_API_KEY="$(cat "$REST_SMOKE_STORE/.api-key")"
trisynapse-memory --path "$REST_SMOKE_STORE" serve --studio
```

In terminal two, export the same token and test the public and authenticated routes:

```bash
export TRISYNAPSE_MEMORY_API_KEY="PASTE_THE_TEST_TOKEN"
API_ROOT=http://127.0.0.1:8765/api/v1

curl --fail-with-body "$API_ROOT/health"
curl --fail-with-body \
  -H "Authorization: Bearer $TRISYNAPSE_MEMORY_API_KEY" \
  "$API_ROOT/check"
curl --fail-with-body \
  -H "Authorization: Bearer $TRISYNAPSE_MEMORY_API_KEY" \
  "$API_ROOT/session?project_id=production-smoke"
```

Ingest a source:

```bash
curl --fail-with-body -X POST "$API_ROOT/sources/ingest" \
  -H "Authorization: Bearer $TRISYNAPSE_MEMORY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": {"project_id": "production-smoke"},
    "sources": [{
      "kind": "text",
      "source_key": "production-smoke",
      "text": "Production smoke tests must preserve citations."
    }]
  }'
```

Save the returned run ID, wait for it to complete, and verify source browsing and preview:

```bash
curl --fail-with-body \
  -H "Authorization: Bearer $TRISYNAPSE_MEMORY_API_KEY" \
  "$API_ROOT/ingestion-runs/RUN_ID"

curl --fail-with-body \
  -H "Authorization: Bearer $TRISYNAPSE_MEMORY_API_KEY" \
  "$API_ROOT/sources?project_id=production-smoke"
```

Create a durable query run:

```bash
curl --fail-with-body -X POST "$API_ROOT/query-runs" \
  -H "Authorization: Bearer $TRISYNAPSE_MEMORY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What must production smoke tests preserve?",
    "namespace": {"project_id": "production-smoke"}
  }'
```

Use its query ID to verify the saved run and SSE endpoint:

```bash
curl --fail-with-body \
  -H "Authorization: Bearer $TRISYNAPSE_MEMORY_API_KEY" \
  "$API_ROOT/query-runs/QUERY_ID?project_id=production-smoke"

curl -N --fail-with-body \
  -H "Authorization: Bearer $TRISYNAPSE_MEMORY_API_KEY" \
  "$API_ROOT/query-runs/QUERY_ID/events?project_id=production-smoke"
```

Confirm that the completed run contains an answer or explicit abstention, citations, effective retrieval configuration, ordered steps, durations, and bounded safe candidate excerpts. It must not contain API keys, embeddings, authorization headers, hidden reasoning, or full system prompts.

Open `http://127.0.0.1:8765/studio/` and test:

1. the canonical logo and five sidebar sections;
2. source card search, details, preview, metadata, and authenticated download;
3. batch text/link ingestion and one tailored single-source form;
4. a live query progressing across the workflow canvas;
5. reopening the same query from history and clicking a citation into its source;
6. Knowledge, Lineage, and Trace graph views, accessible list, and neighborhood expansion;
7. model and retrieval configuration revision handling;
8. Connection token/session behavior in a private browsing window;
9. narrow-screen navigation and drawer close controls;
10. browser console and network panel with no uncaught errors or failed same-origin requests.

Stop the server with Ctrl+C and rerun `validate` on the test store.

### TypeScript client against the live REST server

With the same REST smoke server still running, build the client and call the real API:

```bash
pnpm --filter @trisynapse/trisynapse-memory build
export TRISYNAPSE_API_KEY="PASTE_THE_TEST_TOKEN"

node --input-type=module <<'JS'
import { TrisynapseMemory } from "./packages/js-sdk/dist/index.js";

const memory = new TrisynapseMemory({
  baseUrl: "http://127.0.0.1:8765",
  apiKey: process.env.TRISYNAPSE_API_KEY,
  namespace: { project_id: "typescript-production-smoke" },
});

const health = await memory.health();
const added = await memory.add("The TypeScript client reached production REST.");
const result = await memory.query("What did the TypeScript client reach?");
if (health.status !== "ready" || !added.delta_id || !result.query_id) {
  throw new Error("TypeScript production smoke failed");
}
console.log({ health, queryId: result.query_id, citations: result.citations.length });
JS
```

This verifies the built SDK, authentication header, namespace payload, observation write, query response, and runtime Fetch integration—not only TypeScript compilation.

## 11. Backup and upgrade rehearsal

Before upgrading any real store:

```bash
trisynapse-memory --path /path/to/store validate
trisynapse-memory --path /path/to/store backup ./pre-release-backup.zip
trisynapse-memory restore ./pre-release-backup.zip ./restore-check
trisynapse-memory --path ./restore-check validate
```

Test the new wheel against a copy or restored store, never the only production store. Opening a store can run schema migrations. There is no supported schema downgrade, so preserve the pre-upgrade backup until the new version has been stable for the required retention period.

## 12. Commit, merge, tag, and publish

After all gates pass, commit the version, locks, source, tests, docs, and workflow changes. Do not commit local `dist/` output or built Studio assets; the release workflow rebuilds Studio before packaging the Python distributions.

```bash
git status --short
git diff --check
git diff --stat
uv run python scripts/version.py check --tag "v${RELEASE_VERSION}"
git add pyproject.toml uv.lock pnpm-lock.yaml \
  packages src README.md docs install.sh install.ps1 \
  .github Dockerfile docker-compose.yml .dockerignore
git commit -m "Release Trisynapse Memory ${RELEASE_VERSION}"
git push
```

Merge through the protected `main` branch and wait for the ordinary `test` workflow to be green. Then update a clean local `main`:

```bash
git switch main
git pull --ff-only origin main
git status --short
```

Create an annotated tag. Use a signed tag when signing is configured:

```bash
git tag -s "v${RELEASE_VERSION}" -m "Trisynapse Memory ${RELEASE_VERSION}"
git push origin "v${RELEASE_VERSION}"
```

Without signing:

```bash
git tag -a "v${RELEASE_VERSION}" -m "Trisynapse Memory ${RELEASE_VERSION}"
git push origin "v${RELEASE_VERSION}"
```

Pushing the tag is the publication trigger, not a preview. The workflow can publish to PyPI before a maintainer has time to intervene. Never use a tag merely to test the workflow.

Watch the release:

```bash
gh run list --workflow release.yml --limit 5
gh run watch RUN_ID --exit-status
```

If any job fails before PyPI publication, fix the cause, delete the failed remote/local tag, and retag the corrected commit only if no immutable artifact was published. If PyPI publication succeeded, never reuse the version for different code; fix forward with a new patch version.

## 13. Download and verify release assets

Download all assets with GitHub CLI:

```bash
mkdir "release-assets-${RELEASE_VERSION}"
gh release download "v${RELEASE_VERSION}" --dir "release-assets-${RELEASE_VERSION}"
cd "release-assets-${RELEASE_VERSION}"
```

Or download a specific asset directly:

```bash
curl -fLO "https://github.com/shanmukh05/trisynapse-memory/releases/download/v${RELEASE_VERSION}/install.sh"
curl -fLO "https://github.com/shanmukh05/trisynapse-memory/releases/download/v${RELEASE_VERSION}/SHA256SUMS"
```

On Linux:

```bash
sha256sum --check SHA256SUMS
```

On macOS:

```bash
shasum -a 256 -c SHA256SUMS
```

On PowerShell, inspect every hash and compare it with `SHA256SUMS`:

```powershell
Get-ChildItem -File | Where-Object Name -ne "SHA256SUMS" | Get-FileHash -Algorithm SHA256
```

Inspect `release.json`; its version must match the tag, PyPI, and wheel metadata. Verify the tag in GitHub and confirm that the release points to the intended commit.

Confirm the published SDK independently of the repository workspace:

```bash
npm view "@trisynapse/trisynapse-memory@${RELEASE_VERSION}" version
SDK_SMOKE_ROOT="$(mktemp -d)"
cd "$SDK_SMOKE_ROOT"
npm init --yes
npm install "@trisynapse/trisynapse-memory@${RELEASE_VERSION}"
node --input-type=module -e \
  'import { TrisynapseMemory } from "@trisynapse/trisynapse-memory"; if (typeof TrisynapseMemory !== "function") process.exit(1)'
```

This test must install from npm rather than resolving the local workspace package.

## 14. Test the published installers

Use clean disposable machines or VMs for all three supported operating systems. An existing `uv` tool environment can hide dependency and PATH errors.

### macOS and Linux

Test the version-specific release asset first:

```bash
curl -LsSf "https://github.com/shanmukh05/trisynapse-memory/releases/download/v${RELEASE_VERSION}/install.sh" | sh
trisynapse-memory --version
trisynapse-memory --json check
```

For a stable release, verify that the public latest-release command resolves to the same version:

```bash
curl -LsSf https://github.com/shanmukh05/trisynapse-memory/releases/latest/download/install.sh | sh
trisynapse-memory --version
```

### Windows PowerShell

```powershell
$Version = "0.2.0"
irm "https://github.com/shanmukh05/trisynapse-memory/releases/download/v$Version/install.ps1" | iex
trisynapse-memory --version
trisynapse-memory --json check
```

For a stable release, repeat with `/releases/latest/download/install.ps1`. Skip the `latest` check for a release candidate because GitHub's latest stable release should remain unchanged.

### Installer acceptance criteria

- detects the supported OS and CPU architecture;
- installs or reuses `uv`;
- installs `trisynapse-memory[all]==VERSION` from PyPI;
- records installer metadata under the user state directory;
- explains any required PATH change;
- exposes only the `trisynapse-memory` command;
- reports the requested release version;
- passes `--json check` without a logo or progress content;
- can launch the CLI, REST API, and packaged Studio;
- re-running the installer upgrades or leaves the same version healthy.

Test uninstall and reinstall once:

```bash
uv tool uninstall trisynapse-memory
```

Before publication, at minimum validate installer syntax:

```bash
sh -n install.sh
```

On a Windows runner:

```powershell
[void][ScriptBlock]::Create((Get-Content -Raw ./install.ps1))
```

A new pinned installer cannot complete its real installation test until that exact version is available from PyPI. The post-publication clean-machine tests and `installer-smoke` workflow are therefore mandatory.

## 15. Production deployment

### Deployment boundary

The current server is local-first. For production:

- run one Trisynapse server process per store;
- do not horizontally scale multiple application processes against the same store;
- persist the entire store directory, including SQLite, vector data, retained originals, and configuration;
- keep authentication enabled;
- terminate TLS and apply network/rate controls at a reverse proxy or platform edge;
- keep the service and store on a private network whenever possible;
- inject provider credentials from a secret manager as environment variables;
- never place provider keys or bearer tokens in SQLite, images, Git, logs, or release assets;
- remember that the store is permission-restricted but not encrypted at rest;
- use an encrypted disk/volume when memory content is sensitive.

The CLI-created server supports one administrator bearer token from `TRISYNAPSE_MEMORY_API_KEY` or `<store>/.api-key`. Programmatic `create_app(..., api_keys=...)` supports scoped keys. Do not expose an administrator token to untrusted browser users.

### Container deployment

The supplied Compose file publishes only on loopback and persists `/data` in a named volume. Build Studio immediately before building the image because the Dockerfile packages the existing Studio distribution:

```bash
pnpm --filter @trisynapse/studio build
docker build --pull -t "trisynapse-memory:${RELEASE_VERSION}" .
```

Create a deployment-only environment file outside Git with a strong random API key and only the provider credentials required by the selected models. The checked-in Compose file forwards the administrator API key; add explicit provider-variable mappings in a deployment override when external models are used.

Start and inspect:

```bash
docker compose --env-file /secure/path/trisynapse-production.env up -d
docker compose ps
docker compose logs --tail=200 memory
curl --fail-with-body http://127.0.0.1:8765/api/v1/health
```

The image runs as a non-root `trisynapse` user, stores data under `/data`, and has a health check. Do not publish port `8765` directly to the public internet.

### Direct service deployment

For a host installation, pin the released wheel/package in a dedicated environment or `uv tool`, use an absolute store path, and run:

```bash
trisynapse-memory --path /srv/trisynapse-memory/store \
  serve --host 127.0.0.1 --port 8765 --studio
```

Run it under the operating system's service manager with restart-on-failure, a dedicated unprivileged account, an explicit working directory, a restrictive umask, and environment variables loaded from a root-readable secret file. Keep Uvicorn bound to loopback and proxy HTTPS to it. Ensure the proxy does not buffer or prematurely time out `/api/v1/query-runs/*/events` SSE responses.

Do not add multiple Uvicorn workers for one store. Durable jobs and live query execution are coordinated by the single application process.

For a concrete Linux/systemd deployment, create a dedicated account and directories using your operating system's administration tools, then install the pinned release into `/opt/trisynapse-memory`:

```bash
uv venv /opt/trisynapse-memory
uv pip install --python /opt/trisynapse-memory/bin/python \
  "trisynapse-memory[all]==${RELEASE_VERSION}"
```

Create `/etc/systemd/system/trisynapse-memory.service`:

```ini
[Unit]
Description=Trisynapse Memory
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=trisynapse
Group=trisynapse
WorkingDirectory=/var/lib/trisynapse-memory
EnvironmentFile=/etc/trisynapse-memory/environment
ExecStart=/opt/trisynapse-memory/bin/trisynapse-memory --path /var/lib/trisynapse-memory/store serve --host 127.0.0.1 --port 8765 --studio
Restart=on-failure
RestartSec=5
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=/var/lib/trisynapse-memory

[Install]
WantedBy=multi-user.target
```

Store `TRISYNAPSE_MEMORY_API_KEY` and only the required provider variables in `/etc/trisynapse-memory/environment`. Make that file readable only by root and the service account. Then start and verify:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now trisynapse-memory
sudo systemctl status trisynapse-memory
sudo journalctl -u trisynapse-memory -n 200 --no-pager
curl --fail-with-body http://127.0.0.1:8765/api/v1/health
```

A minimal Caddy reverse proxy is:

```caddyfile
memory.example.com {
    reverse_proxy 127.0.0.1:8765 {
        flush_interval -1
    }
}
```

Restrict the hostname with firewall, VPN, identity-aware proxy, or another access layer appropriate to the deployment. Caddy can supply TLS, but TLS alone does not replace Trisynapse bearer authentication or network isolation.

### Production observability

Monitor:

```text
GET /api/v1/health
GET /api/v1/check
GET /api/v1/metrics
```

Alert on:

- liveness failure or invalid store validation;
- failed or persistently pending jobs;
- failed/interrupted ingestion and query runs;
- missing provider credentials or provider connection failures;
- low disk space or unwritable store;
- backup or restore-drill failure;
- unexpected model-configuration revision or embedding rebuild failure.

Treat logs as sensitive. Query diagnostics are designed not to persist keys, vectors, authorization headers, hidden reasoning, or full prompts, but source excerpts and identifiers can still be confidential.

## 16. Post-release verification

Within the release window:

1. confirm the GitHub Release is public and contains all expected assets;
2. confirm PyPI displays the same version, README, license, and supported Python versions;
3. install through both the versioned and `latest` installer URLs;
4. repeat the CLI, REST, Studio, and wheel smoke tests against published artifacts;
5. deploy first to staging with a restored production-like store;
6. run `validate`, `check`, one ingestion, one query, one query-history replay, and one backup;
7. canary production traffic before full rollout;
8. preserve the release evidence: commit SHA, tag signature, CI URLs, checksums, benchmark artifacts, restore result, and acceptance-test result.

## 17. Failure, rollback, and yanking

If application startup or migration fails, stop the new process and preserve the store unchanged. Restore the validated pre-upgrade backup into a new empty directory. Do not hand-edit SQLite rows, retained source blobs, or index files.

If the release is defective but safe to install, publish a patch release. PyPI versions are immutable: never overwrite a published version or move its tag to different code.

Yank a PyPI release only when users should not select it, and clearly explain the reason in the GitHub Release. Deleting a GitHub Release does not remove the PyPI package or installations already made. For a credential or supply-chain incident, also rotate affected credentials, invalidate tokens, preserve evidence, and follow [`SECURITY.md`](../../SECURITY.md).

## 18. Final sign-off checklist

```text
[ ] Version is consistent everywhere and matches the intended tag
[ ] `scripts/version.py check --tag v<version>` passes
[ ] No secrets, stores, private datasets, or generated local artifacts are staged
[ ] Ruff and the complete Python suite pass
[ ] TypeScript client checks and builds
[ ] Studio checks, unit tests, production build, Playwright, and accessibility pass
[ ] Wheel/sdist build and wheel-content inspection pass
[ ] Clean wheel smoke passes
[ ] Current retrieval benchmark gate passes
[ ] Required end-to-end benchmark gate passes
[ ] Real selected provider/model connection and vision tests pass
[ ] CLI manual acceptance passes
[ ] REST and SSE manual acceptance passes with authentication
[ ] Studio desktop and narrow-screen acceptance passes
[ ] Backup restore drill and restored store validation pass
[ ] Docker or host staging deployment passes health/check/validate
[ ] Protected-main CI is green
[ ] PyPI trusted publisher and GitHub pypi environment are configured
[ ] `@trisynapse/trisynapse-memory` trusted publisher and GitHub npm environment are configured
[ ] Annotated/signed tag points to the reviewed main commit
[ ] Release workflow, publication, checksums, and three-OS installer smoke pass
[ ] Published artifacts are canary-tested before full production rollout
```
