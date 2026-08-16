# Contributing

Trisynapse Memory is Python-first: the canonical algorithm lives in `src/trisynapse_memory`; clients and Studio must call the same public API rather than reimplement retrieval.

Before submitting a change:

```bash
pip install -e '.[dev,server,files]'
python scripts/version.py check
ruff check src tests scripts
pytest -q
pnpm install --frozen-lockfile
pnpm --filter @trisynapse/memory check
```

Preserve these invariants:

- Trace evidence is append-only except for an explicit hard purge.
- Episode Recall may route retrieval but must never enter answer context.
- Embedding failures are explicit; never introduce substitute vectors.
- Every read and write honors its namespace.
- Benchmark inputs must never add gold answers to retrievable memory.

Behavior changes require tests and an update to the relevant file in `docs/`.

For a release, never edit version surfaces individually. Run:

```bash
uv run python scripts/version.py set 0.2.0
uv run python scripts/version.py check --tag v0.2.0
```

`pyproject.toml` is the canonical version; the release utility synchronizes package manifests, installers, and `uv.lock`.
