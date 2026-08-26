import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";

const catalog = {
  helpers: [
    { id: "trace", title: "Trace", kind: "timeline", inspect_path: "/api/v1/memories", count: 4, health: { observations: 3 } },
    { id: "documents", title: "Documents", kind: "table", inspect_path: "/api/v1/memory/documents", count: 3, health: {} },
    { id: "bm25", title: "BM25", kind: "postings", inspect_path: "/api/v1/memory/terms", count: 12, health: {} },
    { id: "vectors", title: "Vectors", kind: "embedding", inspect_path: "/api/v1/memory/vectors/projection", count: 3, health: {} },
    { id: "episodes", title: "Episode Recall", kind: "cards", inspect_path: "/api/v1/episodes", count: 1, health: { stale: 0 } },
    { id: "claims", title: "Claims", kind: "table", inspect_path: "/api/v1/memory/claims", count: 2, health: { contested: 0 } },
    { id: "graph", title: "Graph", kind: "graph", inspect_path: "/api/v1/memory/retrieval-graph", count: 2, health: {} },
    { id: "temporal_index", title: "Temporal Index", kind: "cards", inspect_path: "/api/v1/memory/helpers/temporal_index", count: 1, health: {} },
  ],
  retrieval_routes: [
    { name: "bm25", title: "bm25", enabled: true, weight: 1 },
    { name: "semantic", title: "semantic", enabled: true, weight: 1.1 },
    { name: "custom_lex", title: "custom lex", enabled: true, weight: 0.8 },
  ],
};

const payloads: Record<string, unknown> = {
  "/api/v1/health": { status: "ready", version: "0.0.0", storage_ready: true, pending_jobs: 0 },
  "/api/v1/memory/catalog": catalog,
  "/api/v1/memory/helpers/temporal_index": {
    helper_id: "temporal_index", kind: "cards", truncated: false,
    items: [{ id: "t1", helper_id: "temporal_index", kind: "window", title: "Last week", subtitle: "7d", excerpt: "Atlas launch window", status: "fresh", data: {} }],
  },
  "/api/v1/memory/terms": { terms: [{ term: "atlas", document_frequency: 2, postings: [{ delta_id: "d1", term_frequency: 1, excerpt: "Project Atlas", token_count: 4 }] }], total: 1 },
  "/api/v1/memory/claims": { claims: [] },
  "/api/v1/memory/vectors/projection": { points: [], embedded: 0, searchable: 0, sampled: 0 },
  "/api/v1/sources": {
    sources: [{ id: "src_demo", source_key: "guide", kind: "file", title: "Product guide", content_hash: "abc", media_type: "text/markdown", byte_size: 1024, chunk_count: 4, skipped_count: 0, version: 1, status: "active", namespace: { project_id: "default" }, metadata: {}, delta_ids: [], created_at: "2026-08-11T00:00:00Z" }],
    total: 1, facets: { file: 1 }, next_cursor: null,
  },
  "/api/v1/ingestion-runs": { runs: [] },
};

let lastSearch: Record<string, unknown> | null = null;

function renderApp() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}><BrowserRouter basename="/studio"><App /></BrowserRouter></QueryClientProvider>);
}

describe("Memory Studio shell", () => {
  beforeEach(() => {
    lastSearch = null;
    window.history.pushState({}, "", "/studio/memory");
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), "http://studio.local");
      if (url.pathname === "/api/v1/search") {
        lastSearch = JSON.parse(String(init?.body || "{}")) as Record<string, unknown>;
        return new Response(JSON.stringify({ query_id: "q1", hits: [{ item_id: "d1", kind: "observation", text: "Atlas", score: 0.9, route: "bm25", source_delta_ids: [] }], stage: "fast", confident: true, retrieval_trace: { routes: { bm25: ["d1"], custom_lex: [] } } }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      const value = payloads[url.pathname];
      return new Response(JSON.stringify(value ?? {}), { status: value ? 200 : 404, headers: { "Content-Type": "application/json" } });
    }));
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("opens on Memory Viewer with five navigation sections", async () => {
    renderApp();
    const navigation = screen.getByRole("navigation", { name: "Studio sections" });
    expect(navigation.querySelectorAll("a")).toHaveLength(5);
    expect([...navigation.querySelectorAll("a")].map(item => item.textContent)).toEqual(["Memory Viewer", "Sources", "Queries", "Configuration", "Connection"]);
    expect(await screen.findByRole("heading", { name: "Memory Viewer" })).toBeInTheDocument();
  });

  it("builds helper tabs and playground chips from the catalog", async () => {
    renderApp();
    const tabs = await screen.findByRole("tablist", { name: "Recall helpers" });
    expect(within(tabs).getByRole("tab", { name: "Temporal Index" })).toBeInTheDocument();
    expect(within(tabs).getByRole("tab", { name: "BM25" })).toBeInTheDocument();
    expect(screen.getByText("custom lex")).toBeInTheDocument();
  });

  it("renders a specialized helper when its tab is selected", async () => {
    renderApp();
    const tabs = await screen.findByRole("tablist", { name: "Recall helpers" });
    fireEvent.click(within(tabs).getByRole("tab", { name: "BM25" }));
    expect(await screen.findByPlaceholderText("Look up a term")).toBeInTheDocument();
  });

  it("renders the family fallback for an unknown helper id", async () => {
    window.history.pushState({}, "", "/studio/memory?helper=temporal_index");
    renderApp();
    expect(await screen.findByRole("heading", { name: "Last week" })).toBeInTheDocument();
    expect(screen.getByText("Atlas launch window")).toBeInTheDocument();
  });

  it("runs playground search without persisting a Query Run", async () => {
    renderApp();
    await screen.findByRole("tablist", { name: "Recall helpers" });
    const box = await screen.findByPlaceholderText("Search from the current memory item…");
    fireEvent.change(box, { target: { value: "Who owns Atlas?" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(lastSearch).toMatchObject({ query: "Who owns Atlas?", persist: false }));
    expect(await screen.findByText(/1 hits/)).toBeInTheDocument();
  });
});
