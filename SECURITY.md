# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately to the project maintainer before opening a public issue. Include the affected version, reproduction steps, impact, and any suggested mitigation. Do not include real credentials or private user data.

## Deployment boundary

The v0 server is local-first. It binds to `127.0.0.1` by default and creates a bearer token when started through the CLI. Use HTTPS through a trusted reverse proxy before exposing it to another network. Namespace-scoped API keys limit application access, but they are not a replacement for network isolation.

Accepted content and metadata are stored as supplied. Applications should remove credentials, personal data, and other sensitive values before sending them to the engine or service. The memory store uses restrictive filesystem permissions where supported, but it is not encrypted.

Logical forgetting appends a retraction. Physical removal is a separate, explicit operation that redacts selected content and identifying source metadata, clears affected Recall data, and uses SQLite secure deletion plus vacuum/WAL truncation. Storage-device snapshots and external backups are outside that operation; apply their own retention policy.

Trace is an ordered application data store, not a cryptographic audit ledger. `check` and `validate` detect SQLite problems, sequence gaps, broken evidence references, and missing or changed retained-source blobs. They cannot prove that an administrator with write access never changed the database. Use externally signed or append-only audit storage when that guarantee is required.
