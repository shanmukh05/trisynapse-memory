# `@trisynapse/trisynapse-memory`

Typed, dependency-free client for a local or remote Trisynapse Memory server.

```ts
import { TrisynapseMemory } from "@trisynapse/trisynapse-memory";

const memory = new TrisynapseMemory({
  apiKey: process.env.TRISYNAPSE_API_KEY,
  namespace: { project_id: "docs", user_id: "alice" },
});

await memory.add("Alice prefers concise release notes.");
const run = await memory.ingest([
  { kind: "url", url: "https://example.com/handbook" },
  { kind: "file", filename: "rules.md", content_base64: markdownBase64 },
]);
const result = await memory.query("How should release notes be written?");
console.log(result.answer, result.citations);

const state = await memory.getModelConfiguration();
state.configuration.completion = {
  provider: "anthropic",
  model: "claude-sonnet-4-5",
};
await memory.setModelConfiguration(state.configuration);
```

The package contains no memory algorithm. It is a thin client for the canonical Python engine and `/api/v1` REST contract.

Use `listProviders()` and `listModels()` to build a selector. `testModelConnection()` sends an explicit, potentially billable request. An embedding change must pass `{confirmEmbeddingRebuild: true}`; the response contains the durable rebuild job and pending configuration.
