Test the latest source version from the repository so you don’t accidentally launch the older globally installed release.

### 1. Prepare the development environment

```bash
cd /Users/shanmukh/Desktop/Projects/trisynapse-memory

uv sync --extra dev --extra all
```

### 2. Create some test memory

Use a separate test store:

```bash
uv run trisynapse-memory \
  --path /tmp/trisynapse-cli-test \
  add observation "Project Atlas launches on Monday."
```

### 3. Open the new interactive CLI

```bash
uv run trisynapse-memory \
  --path /tmp/trisynapse-cli-test
```

You should see the Conversation pane permanently visible and the Sources, Trace, Jobs, and Config inspector beside it.

Enter a plain query:

```text
When does Project Atlas launch?
```

The expected behavior is:

```text
> When does Project Atlas launch?
Project Atlas launches on Monday.
[citation ID]
```

The answer should remain directly below the question without changing tabs.

### 4. Test the inspector

Try these commands:

```text
/sources
/search Project Atlas
/timeline
/jobs
/config
/check
/model
```

The relevant inspector tab should change, while the Conversation pane remains visible. `/sources` should no longer produce the `write_json` error.

Test ingestion:

```text
/ingest README.md
/sources
```

### 5. Test responsive behavior

Resize the terminal:

- Above 80 columns: conversation and inspector should appear side by side.
- Below 80 columns: the inspector should move underneath the conversation.

### 6. Run the automated CLI regression test

```bash
uv run pytest -q \
  tests/test_ingestion.py::test_interactive_terminal_tabs_and_ctrl_c
```

Run the complete suite:

```bash
uv run pytest -q
ruff check src tests scripts
```

If `uv run` encounters a local cache-permission issue, use the existing environment directly:

```bash
.venv/bin/trisynapse-memory --path /tmp/trisynapse-cli-test
.venv/bin/pytest -q
```

Running plain `trisynapse-memory` may still use the globally installed `0.1.1` release instead of your current source checkout.