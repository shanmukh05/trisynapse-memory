# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately to the project maintainer before opening a public issue. Include the affected version, reproduction steps, impact, and any suggested mitigation. Do not include real credentials or private user data.

## Deployment boundary

The v0 server is local-first. It binds to `127.0.0.1` by default and creates a bearer token when started through the CLI. Use HTTPS through a trusted reverse proxy before exposing it to another network. Namespace-scoped API keys limit application access, but they are not a replacement for network isolation.

Secrets are filtered before Trace append using high-confidence patterns. No pattern set is perfect: applications should avoid sending credentials to memory and use `<private>…</private>` for content that must never be persisted.

Logical forgetting appends a retraction. Hard purge is a separate, explicit operation that redacts content and identifying source metadata, clears derived caches, uses SQLite secure deletion plus vacuum/WAL truncation, and starts a new verifiable hash-chain epoch. Storage-device snapshots and external backups are outside that operation; apply their own retention policy.
