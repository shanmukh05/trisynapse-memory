import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";

const payloads: Record<string, unknown> = {
  "/api/v1/health": { status: "ready", version: "0.0.0", trace_valid: true, pending_jobs: 0 },
  "/api/v1/sources": {
    sources: [{ id: "src_demo", source_key: "guide", kind: "file", title: "Product guide", content_hash: "abc", media_type: "text/markdown", byte_size: 1024, chunk_count: 4, skipped_count: 0, version: 1, status: "active", namespace: { project_id: "default" }, metadata: {}, delta_ids: [], created_at: "2026-08-11T00:00:00Z" }],
    total: 1, facets: { file: 1 }, next_cursor: null,
  },
  "/api/v1/ingestion-runs": { runs: [] },
};

describe("Memory Studio shell", () => {
  beforeEach(() => {
    window.history.pushState({}, "", "/studio/sources");
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const path = new URL(String(input), "http://studio.local").pathname;
      const value = payloads[path];
      return new Response(JSON.stringify(value ?? {}), { status: value ? 200 : 404, headers: { "Content-Type": "application/json" } });
    }));
  });
  afterEach(() => vi.unstubAllGlobals());

  it("shows exactly the five redesigned navigation sections", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><BrowserRouter basename="/studio"><App /></BrowserRouter></QueryClientProvider>);
    const navigation = screen.getByRole("navigation", { name: "Studio sections" });
    expect(navigation.querySelectorAll("a")).toHaveLength(5);
    for (const name of ["Sources", "Queries", "Memory Viewer", "Configuration", "Connection"]) expect(screen.getByRole("link", { name })).toBeInTheDocument();
    expect(await screen.findByText("Product guide")).toBeInTheDocument();
  });
});
