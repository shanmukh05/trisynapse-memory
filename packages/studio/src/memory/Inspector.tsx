import { ExternalLink, Network, X } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useApi } from "../context";
import type { CatalogHelper, InspectorRecord } from "../types";
import { json } from "../util";

export function Inspector({ record, helpers = [], onClose, onOpen }: {
  record: InspectorRecord | null;
  helpers?: CatalogHelper[];
  onClose: () => void;
  onOpen?: (helperId: string, id?: string | null) => void;
}) {
  const navigate = useNavigate();
  const api = useApi();
  const neighbors = useQuery({
    queryKey: ["vector-neighbors", record?.id, api.connection],
    queryFn: () => api.vectorNeighbors(record!.id),
    enabled: Boolean(record && record.helperId === "vectors"),
  });
  if (!record) {
    return <div className="detail-placeholder"><Network size={23} /><p>Select a memory item to inspect its role, evidence, and Recall links.</p></div>;
  }
  const sourceId = sourceFrom(record);
  const fields = Object.entries(record.data).filter(([key]) => !["text", "payload", "index_text", "postings"].includes(key)).slice(0, 12);
  const others = helpers.filter(item => item.id !== record.helperId).slice(0, 6);
  return <div className="inspector-panel">
    <button className="icon-button drawer-close" onClick={onClose} aria-label="Close inspector"><X size={16} /></button>
    <div className={`node-badge ${record.kind}`}>{record.kind}</div>
    <h2>{record.title}</h2>
    {record.subtitle && <p>{record.subtitle}</p>}
    {record.status && <span className={`status-chip ${record.status}`}>{record.status}</span>}
    {record.excerpt && <p className="inspector-excerpt">{record.excerpt}</p>}
    <dl className="metadata-list compact">
      <div><dt>Helper</dt><dd>{record.helperId}</dd></div>
      <div><dt>ID</dt><dd>{record.id}</dd></div>
      {fields.map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{stringify(value)}</dd></div>)}
    </dl>
    {neighbors.data?.neighbors.length ? <section className="neighbor-list"><h3>Nearest neighbors</h3>{neighbors.data.neighbors.map(item => (
      <button key={item.id} onClick={() => onOpen?.("vectors", item.id)}><strong>{item.score.toFixed(3)}</strong><span>{item.excerpt || item.id}</span></button>
    ))}</section> : null}
    {sourceId && <button className="secondary" onClick={() => navigate(`/sources?source=${encodeURIComponent(sourceId)}`)}><ExternalLink size={15} /> Open source</button>}
    {others.length > 0 && <div className="helper-jumps"><span>Show in other helpers</span>{others.map(item => (
      <button key={item.id} className="quiet" onClick={() => onOpen?.(item.id, record.id)}>{item.title}</button>
    ))}</div>}
    <details className="raw-json"><summary>Raw record</summary><pre>{json(record.data)}</pre></details>
  </div>;
}

function sourceFrom(record: InspectorRecord) {
  const data = record.data;
  const sourceRef = data.source_ref;
  if (sourceRef && typeof sourceRef === "object" && "id" in sourceRef && typeof sourceRef.id === "string") return sourceRef.id;
  const scope = data.scope;
  if (scope && typeof scope === "object" && "source_id" in scope && typeof scope.source_id === "string") return scope.source_id;
  return undefined;
}

function stringify(value: unknown) {
  if (value == null) return "—";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}
