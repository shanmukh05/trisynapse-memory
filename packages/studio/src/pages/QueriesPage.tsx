import { FormEvent, useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocation, useNavigate } from "react-router-dom";
import { LoaderCircle, PanelLeftClose, Search, Send, Sparkles, Trash2, X } from "lucide-react";
import { Background, Controls, Edge, Node, ReactFlow } from "@xyflow/react";
import { useApi } from "../context";
import { Empty, PageHeader } from "../ui";
import { classNames, formatDate, json } from "../util";
import type { QueryRun, QueryStep } from "../types";

export function QueriesPage() {
  const api = useApi(); const client = useQueryClient(); const [search, setSearch] = useState(""); const [question, setQuestion] = useState("");
  const navigate = useNavigate();
  const location = useLocation();
  const [selected, setSelected] = useState<QueryRun | null>(null); const [selectedStep, setSelectedStep] = useState<QueryStep | null>(null); const [running, setRunning] = useState(false);
  const history = useQuery({ queryKey: ["query-runs", search, api.connection], queryFn: () => api.listQueries(search), refetchInterval: 3_000 });
  useEffect(() => {
    const seeded = (location.state as { question?: string } | null)?.question;
    if (seeded) setQuestion(seeded);
  }, [location.state]);
  const open = async (run: QueryRun) => { const value = await api.getQuery(run.id); setSelected(value); setSelectedStep(value.steps.at(-1) || null); };
  const ask = async (event: FormEvent) => { event.preventDefault(); if (!question.trim()) return; setRunning(true); try { const run = await api.createQuery(question.trim()); setSelected(run); setQuestion(""); const controller = new AbortController(); try { await api.streamQuery(run.id, value => { setSelected(value); setSelectedStep(current => current ? value.steps.find(step => step.id === current.id) || value.steps.at(-1) || null : value.steps.at(-1) || null); }, controller.signal); } catch { let value = await api.getQuery(run.id); for (let i = 0; i < 240 && ["pending", "running"].includes(value.status); i++) { await new Promise(resolve => setTimeout(resolve, 500)); value = await api.getQuery(run.id); setSelected(value); } } await client.invalidateQueries({ queryKey: ["query-runs"] }); } finally { setRunning(false); } };
  const remove = async (id: string) => { if (!confirm("Remove this saved query workflow?")) return; await api.removeQuery(id); if (selected?.id === id) setSelected(null); client.invalidateQueries({ queryKey: ["query-runs"] }); };
  return <>
    <PageHeader eyebrow="GROUNDED RETRIEVAL" title="Queries" description="Ask memory, watch every retrieval stage, and reopen the same evidence path later." />
    <form className="query-composer" onSubmit={ask}><Sparkles size={19} /><textarea rows={2} value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask what memory knows…" /><button className="primary" disabled={running || !question.trim()}>{running ? <LoaderCircle className="spin" size={17} /> : <Send size={17} />} Ask</button></form>
    <div className="query-workspace">
      <aside className="query-history" aria-label="Past queries"><header><strong>Past queries</strong><button className="icon-button" title="Clear query history" onClick={async () => { if (confirm("Remove all saved query workflows in this namespace?")) { await api.clearQueries(); setSelected(null); client.invalidateQueries({ queryKey: ["query-runs"] }); } }}><Trash2 size={15} /></button></header><label className="search-field compact"><Search size={14} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Find a query" /></label><div>{history.data?.runs.map(run => <button key={run.id} className={classNames("history-item", selected?.id === run.id && "active")} onClick={() => open(run)}><span>{run.query}</span><small><i className={`run-dot ${run.status}`} />{formatDate(run.created_at)}</small></button>)}</div></aside>
      <section className="query-inspector">{selected ? <><QueryAnswer run={selected} onRemove={() => remove(selected.id)} onCitation={deltaId => navigate(`/memory?helper=trace&id=${encodeURIComponent(deltaId)}`)} /><Workflow run={selected} selected={selectedStep} onSelect={setSelectedStep} /></> : <Empty title="Run or select a query" detail="The retrieval workflow will appear here as a clickable diagram." icon={<Sparkles size={28} />} />}</section>
      <aside className="step-detail" aria-label="Step details">{selectedStep ? <StepDetail step={selectedStep} onJump={id => navigate(`/memory?helper=trace&id=${encodeURIComponent(id)}`)} onClose={() => setSelectedStep(null)} /> : <div className="detail-placeholder"><PanelLeftClose size={21} /><p>Select a workflow box to inspect its inputs and outputs.</p></div>}</aside>
    </div>
  </>;
}

function QueryAnswer({ run, onRemove, onCitation }: { run: QueryRun; onRemove: () => void; onCitation: (deltaId: string) => void }) {
  return <div className="answer-card"><div className="answer-top"><div><span className={`status-chip ${run.status}`}>{run.status}</span>{run.retrieval_trace && <span className="subtle-chip">{run.retrieval_trace.stage}</span>}</div><button className="icon-button" onClick={onRemove}><Trash2 size={15} /></button></div><h2>{run.query}</h2>{run.answer ? <p>{run.answer}</p> : <p className="muted">{run.error || "Retrieval is running…"}</p>}{run.citations.length > 0 && <div className="citations">{run.citations.map((citation, index) => <button key={citation.delta_id} title="Open this Trace record in Memory Viewer" onClick={() => onCitation(citation.delta_id)}><span>{index + 1}</span>{citation.source_ref && typeof citation.source_ref === "object" ? citation.source_ref.title || citation.delta_id : citation.delta_id}</button>)}</div>}</div>;
}

function Workflow({ run, selected, onSelect }: { run: QueryRun; selected: QueryStep | null; onSelect: (step: QueryStep) => void }) {
  const nodes: Node[] = run.steps.map((step, index) => ({ id: step.id, position: { x: index % 2 ? 320 : 40, y: index * 112 }, data: { label: <div className="flow-node"><span>{step.phase.replaceAll("_", " ")}</span><strong>{step.label}</strong>{step.duration_ms != null && <small>{step.duration_ms.toFixed(0)} ms</small>}</div> }, className: classNames("workflow-node", step.status, selected?.id === step.id && "selected") }));
  const edges: Edge[] = run.steps.flatMap((step, index) => { const parents = step.parent_ids.length ? step.parent_ids : index ? [run.steps[index - 1].id] : []; return parents.map(parent => ({ id: `${parent}:${step.id}`, source: parent, target: step.id, type: "smoothstep", animated: run.status === "running" && index === run.steps.length - 1, label: step.phase === "deep_recall" ? "low confidence" : undefined })); });
  return <div className="workflow-canvas" aria-label="Retrieval workflow"><ReactFlow nodes={nodes} edges={edges} fitView nodesDraggable={false} nodesConnectable={false} onNodeClick={(_, node) => { const step = run.steps.find(item => item.id === node.id); if (step) onSelect(step); }}><Background gap={18} size={1} color="#e9e9e7" /><Controls showInteractive={false} /></ReactFlow></div>;
}

function StepDetail({ step, onClose, onJump }: { step: QueryStep; onClose: () => void; onJump: (id: string) => void }) {
  return <div className="step-panel"><button className="icon-button drawer-close" onClick={onClose} aria-label="Close step details"><X size={16} /></button><div className="eyebrow">{step.phase}</div><h2>{step.label}</h2><div className="step-stats"><span>{step.status}</span>{step.duration_ms != null && <span>{step.duration_ms.toFixed(1)} ms</span>}</div><DataSection title="Input" value={step.input} /><DataSection title="Output" value={step.output} /><DataSection title="Metrics" value={step.metrics} />{step.candidates.length > 0 && <section><h3>Ranked candidates</h3><div className="candidate-list">{step.candidates.map(candidate => <article key={`${candidate.route}:${candidate.item_id}`}><header><span>{candidate.route} · #{candidate.rank}</span><strong>{candidate.score.toFixed(4)}</strong></header><p>{candidate.excerpt}</p><button className="quiet" onClick={() => onJump(candidate.item_id)}><code>{candidate.item_id}</code></button></article>)}</div></section>}</div>;
}
function DataSection({ title, value }: { title: string; value: Record<string, unknown> }) { if (!Object.keys(value).length) return null; return <section><h3>{title}</h3><pre>{json(value)}</pre></section>; }
