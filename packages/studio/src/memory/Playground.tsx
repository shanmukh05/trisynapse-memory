import { FormEvent, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronDown, LoaderCircle, Send, Sparkles } from "lucide-react";
import { useApi } from "../context";
import type { CatalogRoute, InspectorRecord, SearchResult } from "../types";
import { seedFromRecord } from "./helpers";

export function Playground({ routes, record, onJump }: { routes: CatalogRoute[]; record: InspectorRecord | null; onJump: (helperId: string, id: string) => void }) {
  const api = useApi();
  const navigate = useNavigate();
  const [open, setOpen] = useState(true);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<SearchResult | null>(null);
  const [error, setError] = useState("");
  const enabled = routes.filter(route => route.enabled);
  useEffect(() => { if (record) setQuery(seedFromRecord(record)); }, [record?.id]);
  const seed = () => { if (record) setQuery(seedFromRecord(record)); };
  const run = async (event: FormEvent) => {
    event.preventDefault();
    if (!query.trim()) return;
    setBusy(true); setError("");
    try { setResult(await api.search(query.trim(), false)); }
    catch (caught) { setError((caught as Error).message); }
    finally { setBusy(false); }
  };
  return <section className="playground">
    <header>
      <button type="button" className="quiet playground-toggle" onClick={() => setOpen(current => !current)} aria-expanded={open}>
        <ChevronDown size={14} style={{ transform: open ? "rotate(0deg)" : "rotate(-90deg)" }} />
        Retrieval playground
      </button>
      <div className="route-chips">{enabled.map(route => <span key={route.name} title={`weight ${route.weight}`}>{route.title}</span>)}</div>
    </header>
    {open && <>
      <form onSubmit={run}>
        <Sparkles size={16} />
        <textarea rows={2} value={query} onChange={event => setQuery(event.target.value)} placeholder="Search from the current memory item…" />
        <button type="button" className="quiet" disabled={!record} onClick={seed}>Seed</button>
        <button className="primary" disabled={busy || !query.trim()}>{busy ? <LoaderCircle className="spin" size={15} /> : <Send size={15} />} Search</button>
      </form>
      {error && <p className="form-status">{error}</p>}
      {result && <div className="playground-hits">
        <small>{result.hits.length} hits · {result.stage} · {result.confident ? "confident" : "low confidence"}</small>
        {result.retrieval_trace?.routes && <div className="route-results">{Object.entries(result.retrieval_trace.routes).map(([name, ids]) => <span key={name}>{name} {ids.length}</span>)}</div>}
        {result.hits.slice(0, 8).map(hit => (
          <button key={`${hit.route}:${hit.item_id}`} onClick={() => onJump("trace", hit.item_id)}>
            <header><span>{hit.route}</span><strong>{hit.score.toFixed(3)}</strong></header>
            <p>{hit.text.slice(0, 180)}</p>
          </button>
        ))}
        <button className="secondary" onClick={() => navigate("/queries", { state: { question: query } })}>Open as Query</button>
      </div>}
    </>}
  </section>;
}
