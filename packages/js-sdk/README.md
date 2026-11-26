# `@trisynapse/trisynapse-memory`

Typed, dependency-free client for a local or remote Trisynapse Memory server.
Use Node.js 18 or later and start the REST service with `trisynapse-memory serve --studio`.

```sh
npm install @trisynapse/trisynapse-memory
```

```ts
import { TrisynapseMemory } from "@trisynapse/trisynapse-memory";

const memory = new TrisynapseMemory({
  apiKey: process.env.TRISYNAPSE_MEMORY_API_KEY,
  namespace: { project_id: "docs", user_id: "alice" },
});

await memory.add("Alice prefers concise release notes.");
const result = await memory.search("How should release notes be written?");
console.log(result.hits.map(hit => ({ text: hit.text, locator: hit.locator })));
```

The package contains no memory algorithm. It is a thin client for the canonical Python engine and `/api/v1` REST contract.

The service creates its API key in `~/.trisynapse-memory/store/.api-key` by default. Set `TRISYNAPSE_MEMORY_API_KEY` on the client to that value or use the token configured on your service. Keep an administrator token on the server side.

Use `getRetrievalConfiguration()` to read the current revision before passing a changed configuration to `setRetrievalConfiguration()`. With a v0.3.0 server, `readout: "raw_sources"` limits the returned text to original observations while keeping the configured search routes.

Use `listProviders()` and `listModels()` to build a selector. `testModelConnection()` sends an explicit, potentially billable request. An embedding change must pass `{confirmEmbeddingRebuild: true}`; the response contains the durable rebuild job and pending configuration.
