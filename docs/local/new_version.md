The current published version is `0.1.1`. Given the substantial feature and architecture changes, use `0.2.0`.

Run these commands from the repository root.

### 1. Set the version

```bash
RELEASE_VERSION=0.2.0

uv run python scripts/version.py set "$RELEASE_VERSION"
uv run python scripts/version.py check --tag "v${RELEASE_VERSION}"
```

### 2. Install dependencies and test everything

```bash
uv sync --extra dev --extra all
pnpm install --frozen-lockfile

uv run ruff check src tests scripts
uv run --extra dev --extra all python -m pytest -q

pnpm --filter @trisynapse/trisynapse-memory check
pnpm --filter @trisynapse/trisynapse-memory build

pnpm --filter @trisynapse/studio check
pnpm --filter @trisynapse/studio test
pnpm --filter @trisynapse/studio build
pnpm --filter @trisynapse/studio exec playwright install chromium
pnpm --filter @trisynapse/studio test:e2e

sh -n install.sh
```

### 3. Review and commit the changes

You are currently on `main`, so no release branch is required.

```bash
git status --short
git diff --check
git add -A
git diff --cached --check
git status --short

git commit -m "Release Trisynapse Memory ${RELEASE_VERSION}"
```

Bring in any remote changes before pushing:

```bash
git pull --rebase origin main
git push origin main
```

### 4. Wait for normal CI

```bash
gh run list --workflow test.yml --limit 5
gh run watch RUN_ID --exit-status
```

Replace `RUN_ID` with the newest run ID shown by the first command.

Do not create the release tag unless this workflow passes.

### 5. Create the release tag

```bash
git status --short
git pull --ff-only origin main

git tag -a "v${RELEASE_VERSION}" \
  -m "Trisynapse Memory ${RELEASE_VERSION}"

git push origin "v${RELEASE_VERSION}"
```

Pushing this tag automatically triggers `.github/workflows/release.yml`. It publishes:

- `trisynapse-memory==0.2.0` to PyPI
- `@trisynapse/trisynapse-memory@0.2.0` to npm
- GitHub Release assets
- macOS/Linux and Windows installers

Do not manually run `npm publish` or `uv publish`.

### 6. Watch the release

```bash
gh run list --workflow release.yml --limit 5
gh run watch RUN_ID --exit-status
```

### 7. Verify publication

```bash
gh release view "v${RELEASE_VERSION}"

npm view "@trisynapse/trisynapse-memory@${RELEASE_VERSION}" version

curl -LsSf \
  "https://pypi.org/pypi/trisynapse-memory/${RELEASE_VERSION}/json" \
  | python3 -m json.tool
```

### 8. Test the published installer

macOS/Linux:

```bash
curl -LsSf \
  "https://github.com/shanmukh05/trisynapse-memory/releases/download/v${RELEASE_VERSION}/install.sh" \
  | sh

trisynapse-memory --version
trisynapse-memory --json check
```

Windows PowerShell:

```powershell
$Version = "0.2.0"

irm "https://github.com/shanmukh05/trisynapse-memory/releases/download/v$Version/install.ps1" | iex

trisynapse-memory --version
trisynapse-memory --json check
```

If either PyPI or npm successfully publishes `0.2.0` but a later release job fails, do not delete and reuse the tag. Fix the problem and release `0.2.1`.