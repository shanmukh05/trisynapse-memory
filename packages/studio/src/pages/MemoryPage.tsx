import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { useApi } from "../context";
import { Empty, Loading, PageHeader } from "../ui";
import type { InspectorRecord } from "../types";
import { lookupHelper } from "../memory/helpers";
import { FallbackHelper, registerBuiltinHelpers } from "../memory/helpers/renderers";
import { Inspector } from "../memory/Inspector";
import { Overview } from "../memory/Overview";
import { Playground } from "../memory/Playground";

registerBuiltinHelpers();

export function MemoryPage() {
  const api = useApi();
  const [params, setParams] = useSearchParams();
  const helperId = params.get("helper") || "overview";
  const selectedId = params.get("id");
  const search = params.get("q") || "";
  const catalog = useQuery({ queryKey: ["memory-catalog", api.connection], queryFn: () => api.catalog() });
  const helpers = catalog.data?.helpers || [];
  const selectedHelper = helpers.find(item => item.id === helperId);
  const renderer = helperId === "overview" ? null : lookupHelper(helperId);
  const setHelper = (id: string, itemId?: string | null) => {
    const next = new URLSearchParams(params);
    if (id === "overview") next.delete("helper"); else next.set("helper", id);
    if (itemId) next.set("id", itemId); else next.delete("id");
    setParams(next, { replace: true });
  };
  const selected = useSelectedRecord(helperId, selectedId);
  useEffect(() => {
    if (helperId === "overview" || !catalog.data || selectedHelper || !helpers.length) return;
    setHelper("overview");
  }, [catalog.data, helperId, selectedHelper, helpers.length]);

  return <>
    <PageHeader eyebrow="TRACE + RECALL" title="Memory Viewer" description="Inspect every Recall helper, follow evidence back to Trace, and run live retrieval from the current item." />
    {catalog.isLoading ? <Loading label="Loading memory catalog" /> : catalog.isError ? <Empty title="Catalog unavailable" detail={(catalog.error as Error).message} /> : <>
      <div className="summary-strip wrap" role="group" aria-label="Memory health">
        {helpers.map(helper => (
          <button key={helper.id} className={helperId === helper.id ? "active" : ""} onClick={() => setHelper(helper.id)}>
            <strong>{helper.count}</strong> {helper.title}
          </button>
        ))}
      </div>
      <div className="graph-toolbar">
        <div className="segmented wrap" role="tablist" aria-label="Recall helpers">
          <button role="tab" aria-selected={helperId === "overview"} className={helperId === "overview" ? "active" : ""} onClick={() => setHelper("overview")}>Overview</button>
          {helpers.map(helper => (
            <button role="tab" aria-selected={helperId === helper.id} key={helper.id} className={helperId === helper.id ? "active" : ""} onClick={() => setHelper(helper.id)}>{helper.title}</button>
          ))}
        </div>
        {helperId !== "overview" && <label className="search-field"><Search size={15} /><input value={search} onChange={event => {
          const next = new URLSearchParams(params);
          if (event.target.value) next.set("q", event.target.value); else next.delete("q");
          setParams(next, { replace: true });
        }} placeholder="Filter this helper" /></label>}
      </div>
      <div className="memory-layout">
        <section className="graph-card memory-canvas">
          {helperId === "overview" && <Overview helpers={helpers} onOpen={id => setHelper(id)} />}
          {helperId !== "overview" && selectedHelper && (() => {
            const View = renderer?.render ?? FallbackHelper;
            return <View helper={selectedHelper} search={search} selectedId={selectedId} onSelect={record => setHelper(record.helperId, record.id)} />;
          })()}
        </section>
        <aside className="graph-detail memory-side" aria-label="Inspector and playground">
          <Inspector record={selected} helpers={helpers} onClose={() => setHelper(helperId)} onOpen={(id, itemId) => setHelper(id, itemId)} />
          <Playground routes={catalog.data?.retrieval_routes || []} record={selected} onJump={(id, itemId) => setHelper(id, itemId)} />
        </aside>
      </div>
    </>}
  </>;
}

function useSelectedRecord(helperId: string, selectedId: string | null): InspectorRecord | null {
  const api = useApi();
  const items = useQuery({
    queryKey: ["helper-items", helperId, selectedId, api.connection],
    queryFn: () => api.helperItems(helperId),
    enabled: Boolean(selectedId && helperId !== "overview"),
  });
  if (!selectedId || helperId === "overview") return null;
  const item = items.data?.items.find(entry => entry.id === selectedId);
  if (!item) return { id: selectedId, helperId, kind: helperId, title: selectedId, data: {} };
  return { id: item.id, helperId: item.helper_id, kind: item.kind, title: item.title, subtitle: item.subtitle, excerpt: item.excerpt, status: item.status, data: item.data };
}
