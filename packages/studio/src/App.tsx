import {
  createContext, FormEvent, ReactNode, useContext, useEffect, useMemo, useRef, useState,
} from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Navigate, NavLink, Route, Routes, useNavigate, useSearchParams } from "react-router-dom";
import {
  Archive, Box, Braces, ChevronRight, CircleCheck, Clock3, Code2, Database,
  Download, ExternalLink, File, FileImage, FileText, FolderGit2, GitBranch,
  Globe2, HelpCircle, Image, Inbox, Link2, LoaderCircle, MemoryStick, Network,
  PanelLeftClose, Plus, RefreshCw, Search, Send, Settings2, SlidersHorizontal,
  Sparkles, Trash2, Upload, Wifi, X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Background, Controls, Edge, Node, ReactFlow } from "@xyflow/react";
import cytoscape, { Core } from "cytoscape";
import logo from "@trisynapse-logo";
import { initialConnection, StudioApi } from "./api";
import type {
  Connection, GraphNode, GraphPage, IngestionRun, ModelConfigurationState,
  PreviewItem, QueryRun, QueryStep, RetrievalConfiguration, Source, SourceInput,
  SourcePreview,
} from "./types";

const ApiContext = createContext<StudioApi | null>(null);
const useApi = () => {
  const value = useContext(ApiContext);
  if (!value) throw new Error("Studio API is unavailable");
  return value;
};

function formatBytes(value = 0) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}
function formatDate(value?: string | null) {
  return value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "—";
}
function json(value: unknown) { return JSON.stringify(value, null, 2); }
function classNames(...values: Array<string | false | null | undefined>) { return values.filter(Boolean).join(" "); }

const sourceIcons: Record<string, typeof File> = {
  text: FileText, file: File, image: FileImage, git: FolderGit2,
  directory: FolderGit2, archive: Archive, url: Globe2,
};

export function App() {
  const [connection, setConnection] = useState<Connection>(initialConnection);
  const api = useMemo(() => new StudioApi(connection), [connection]);
  const health = useQuery({ queryKey: ["health", connection.baseUrl], queryFn: () => api.health(), refetchInterval: 5_000 });
  return (
    <ApiContext.Provider value={api}>
      <div className="app-shell">
        <aside className="sidebar">
          <div className="brand-block">
            <img src={logo} alt="Trisynapse" />
            <div><strong>Trisynapse</strong><span>Memory Studio</span></div>
          </div>
          <nav aria-label="Studio sections">
            <SideLink to="/sources" icon={Inbox}>Sources</SideLink>
            <SideLink to="/queries" icon={Sparkles}>Queries</SideLink>
            <SideLink to="/memory" icon={Network}>Memory Viewer</SideLink>
            <SideLink to="/configuration" icon={Settings2}>Configuration</SideLink>
            <SideLink to="/connection" icon={Wifi}>Connection</SideLink>
          </nav>
          <div className="sidebar-footer">
            <div className="namespace-dot"><span />{connection.namespace.project_id || "default"}</div>
            <small>{health.data?.trace_valid ? "Trace verified" : health.isError ? "Server unavailable" : "Checking Trace…"}</small>
          </div>
        </aside>
        <div className="workspace">
          <header className="topbar">
            <div className="mobile-brand"><img src={logo} alt="Trisynapse" /><strong>Memory Studio</strong></div>
            <div className="crumb"><Database size={15} /> {connection.namespace.project_id || "default"}</div>
            <div className={classNames("server-status", health.data?.status === "ready" && "ready", health.isError && "offline")}>
              <span /> {health.isError ? "Offline" : health.data ? `Ready · v${health.data.version}` : "Connecting"}
            </div>
          </header>
          <main>
            <Routes>
              <Route path="/sources" element={<SourcesPage />} />
              <Route path="/queries" element={<QueriesPage />} />
              <Route path="/memory" element={<MemoryPage />} />
              <Route path="/configuration" element={<ConfigurationPage />} />
              <Route path="/connection" element={<ConnectionPage value={connection} onSave={setConnection} />} />
              <Route path="*" element={<Navigate to="/sources" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </ApiContext.Provider>
  );
}

function SideLink({ to, icon: Icon, children }: { to: string; icon: typeof Inbox; children: ReactNode }) {
  return <NavLink to={to} className={({ isActive }) => classNames("side-link", isActive && "active")}><Icon size={17} /><span>{children}</span></NavLink>;
}

function PageHeader({ eyebrow, title, description, actions }: { eyebrow: string; title: string; description: string; actions?: ReactNode }) {
  return <div className="page-header"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{actions && <div className="page-actions">{actions}</div>}</div>;
}

function Modal({ title, subtitle, children, onClose, wide = false }: { title: string; subtitle?: string; children: ReactNode; onClose: () => void; wide?: boolean }) {
  useEffect(() => {
    const close = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", close); return () => window.removeEventListener("keydown", close);
  }, [onClose]);
  return <div className="modal-backdrop" role="presentation" onMouseDown={event => event.target === event.currentTarget && onClose()}>
    <section className={classNames("modal", wide && "wide")} role="dialog" aria-modal="true" aria-label={title}>
      <header><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div><button className="icon-button" onClick={onClose} aria-label="Close"><X size={19} /></button></header>
      <div className="modal-content">{children}</div>
    </section>
  </div>;
}

function SourcesPage() {
  const api = useApi(); const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const [search, setSearch] = useState(""); const [kind, setKind] = useState(""); const [sort, setSort] = useState("newest");
  const [selected, setSelected] = useState<Source | null>(null); const [adding, setAdding] = useState(false);
  const sources = useQuery({ queryKey: ["sources", search, kind, sort, api.connection], queryFn: () => api.listSources({ q: search, kind, sort }) });
  const runs = useQuery({ queryKey: ["ingestion-runs", api.connection], queryFn: () => api.listIngestionRuns(), refetchInterval: query => query.state.data?.runs.some(run => ["pending", "running"].includes(run.status)) ? 1_000 : false });
  const activeRuns = runs.data?.runs.filter(run => ["pending", "running"].includes(run.status)).length || 0;
  useEffect(() => {
    const sourceId = params.get("source");
    if (!sourceId || selected?.id === sourceId) return;
    void api.getSource(sourceId).then(value => setSelected(value as Source)).catch(() => setParams({}, { replace: true }));
  }, [api, params, selected?.id, setParams]);
  const closeDetails = () => { setSelected(null); if (params.has("source")) setParams({}, { replace: true }); };
  return <>
    <PageHeader eyebrow="SOURCE INBOX" title="Sources" description="Everything Trisynapse has accepted, retained, and turned into grounded memory." actions={<button className="primary" onClick={() => setAdding(true)}><Plus size={16} /> Add source</button>} />
    <div className="summary-strip">
      <span><strong>{sources.data?.total ?? "—"}</strong> sources</span>
      <span><strong>{Object.keys(sources.data?.facets || {}).length}</strong> source types</span>
      <span><strong>{activeRuns}</strong> active imports</span>
    </div>
    <div className="filterbar">
      <label className="search-field"><Search size={16} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search title, URL, key, or filename" /></label>
      <select value={kind} onChange={e => setKind(e.target.value)} aria-label="Source type"><option value="">All types</option>{["text", "file", "image", "url", "git", "archive", "directory"].map(value => <option key={value}>{value}</option>)}</select>
      <select value={sort} onChange={e => setSort(e.target.value)} aria-label="Sort sources"><option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="title">Title</option></select>
      <button className="quiet" onClick={() => { setSearch(""); setKind(""); setSort("newest"); }}><RefreshCw size={15} /> Reset</button>
    </div>
    {sources.isLoading ? <Loading label="Loading sources" /> : sources.isError ? <Empty title="Could not load sources" detail={(sources.error as Error).message} /> : !sources.data?.sources.length ? <Empty title="Your source inbox is empty" detail="Add text, documents, code, images, web pages, or a public repository." action={<button className="primary" onClick={() => setAdding(true)}><Plus size={16} /> Add the first source</button>} /> :
      <div className="source-grid">{sources.data.sources.map(source => <SourceCard key={source.id} source={source} onClick={() => setSelected(source)} />)}</div>}
    {selected && <SourceDetails source={selected} onClose={closeDetails} onChanged={() => { closeDetails(); client.invalidateQueries({ queryKey: ["sources"] }); }} />}
    {adding && <AddSourceModal onClose={() => setAdding(false)} onComplete={() => { setAdding(false); client.invalidateQueries({ queryKey: ["sources"] }); client.invalidateQueries({ queryKey: ["ingestion-runs"] }); }} />}
  </>;
}

function SourceCard({ source, onClick }: { source: Source; onClick: () => void }) {
  const Icon = sourceIcons[source.kind] || File;
  const origin = source.uri ? (() => { try { return new URL(source.uri).hostname; } catch { return source.uri; } })() : source.filename || source.source_key;
  return <button className="source-card" onClick={onClick}>
    <div className="source-card-top"><div className={`source-icon ${source.kind}`}><Icon size={20} /></div><span className={`status-chip ${source.status}`}>{source.status}</span></div>
    <div className="source-card-body"><h3>{source.title}</h3><p>{origin}</p></div>
    <div className="source-card-meta"><span>{source.preview_type || source.kind}</span><span>v{source.version}</span><span>{formatBytes(source.byte_size)}</span></div>
    <div className="source-card-foot"><span>{source.chunk_count} memory chunks</span><time>{formatDate(source.created_at)}</time></div>
  </button>;
}

function SourceDetails({ source, onClose, onChanged }: { source: Source; onClose: () => void; onChanged: () => void }) {
  const api = useApi(); const [tab, setTab] = useState("preview"); const [busy, setBusy] = useState(false);
  const preview = useQuery({ queryKey: ["source-preview", source.id, api.connection], queryFn: () => api.sourcePreview(source.id) });
  const remove = async () => { if (!confirm(`Remove ${source.title} and all memory derived from it?`)) return; setBusy(true); try { await api.removeSource(source.id); onChanged(); } finally { setBusy(false); } };
  const download = async () => { const blob = await api.sourceBlob(source.id, "attachment"); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = source.filename || source.title; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1_000); };
  return <Modal title={source.title} subtitle={`${source.preview_type || source.kind} · ${source.status} · version ${source.version}`} onClose={onClose} wide>
    <div className="detail-actions"><button className="secondary" onClick={() => navigator.clipboard.writeText(source.id)}><Box size={15} /> Copy ID</button><button className="secondary" onClick={download}><Download size={15} /> Download original</button><button className="danger" disabled={busy} onClick={remove}><Trash2 size={15} /> Remove source</button></div>
    <div className="tabs">{["preview", "derived memory", "metadata", "versions", "activity"].map(value => <button className={tab === value ? "active" : ""} key={value} onClick={() => setTab(value)}>{value}</button>)}</div>
    {preview.isLoading ? <Loading label="Preparing preview" /> : preview.isError ? <Empty title="Preview unavailable" detail={(preview.error as Error).message} /> : preview.data && <SourceTab tab={tab} source={source} preview={preview.data} />}
  </Modal>;
}

function SourceTab({ tab, source, preview }: { tab: string; source: Source; preview: SourcePreview }) {
  if (tab === "metadata") return <dl className="metadata-list"><Meta label="Source ID" value={source.id} /><Meta label="Source key" value={source.source_key} /><Meta label="Type" value={source.preview_type || source.kind} /><Meta label="Media type" value={source.media_type} /><Meta label="Size" value={formatBytes(source.byte_size)} /><Meta label="SHA-256" value={source.content_hash} /><Meta label="Created" value={formatDate(source.created_at)} /><Meta label="URI" value={source.uri || "Local upload"} /><Meta label="Extra metadata" value={json(source.metadata)} /></dl>;
  if (tab === "versions") return <div className="timeline"><div><CircleCheck size={17} /><span><strong>Version {source.version}</strong><small>{source.status} · {formatDate(source.created_at)}</small></span></div>{source.previous_source_id && <div><Clock3 size={17} /><span><strong>Previous version</strong><small>{source.previous_source_id}</small></span></div>}</div>;
  if (tab === "activity") return <div className="activity-panel"><h3>Ingestion activity</h3><p>{source.chunk_count} chunks created from this source. {source.skipped_count ? `${source.skipped_count} paths were safely skipped.` : "No paths were skipped."}</p>{source.ingestion_run_id && <code>{source.ingestion_run_id}</code>}</div>;
  if (tab === "derived memory") return <div className="preview-items">{preview.items.map(item => <PreviewBlock key={item.delta_id} item={item} showId />)}</div>;
  return <SourcePreviewView source={source} preview={preview} />;
}

function SourcePreviewView({ source, preview }: { source: Source; preview: SourcePreview }) {
  const api = useApi(); const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const visual = source.media_type.startsWith("image/") || source.media_type === "application/pdf";
  useEffect(() => {
    if (!visual) return; let active = true; let url = "";
    api.sourceBlob(source.id).then(blob => { if (active) { url = URL.createObjectURL(blob); setObjectUrl(url); } }).catch(() => undefined);
    return () => { active = false; if (url) URL.revokeObjectURL(url); };
  }, [api, source.id, visual]);
  return <div className="source-preview">
    {objectUrl && source.media_type.startsWith("image/") && <div className="image-preview"><img src={objectUrl} alt={source.title} /></div>}
    {objectUrl && source.media_type === "application/pdf" && <iframe src={objectUrl} title={`PDF preview of ${source.title}`} />}
    {!!preview.manifest.length && <div className="manifest"><h3>Accepted files</h3>{preview.manifest.map(path => <div key={path}><File size={14} />{path}</div>)}</div>}
    <div className="preview-items">{preview.items.map(item => <PreviewBlock key={item.delta_id} item={item} />)}</div>
  </div>;
}

function PreviewBlock({ item, showId = false }: { item: PreviewItem; showId?: boolean }) {
  const code = ["code_symbol", "code_lines", "notebook_cell"].includes(item.kind);
  const markdown = item.kind === "text" || item.kind === "paragraph";
  return <article className="preview-block"><header><span>{item.kind.replaceAll("_", " ")}</span><code>{locatorLabel(item.locator)}</code></header>{showId && <small>{item.delta_id}</small>}{code ? <pre><code>{item.text}</code></pre> : markdown ? <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{item.text}</ReactMarkdown></div> : <p className="preserve">{item.text}</p>}</article>;
}
function locatorLabel(value: PreviewItem["locator"]) { if (!value) return ""; if (typeof value === "string") return value; return Object.entries(value).filter(([key]) => key !== "metadata").map(([key, item]) => `${key}: ${String(item)}`).join(" · "); }
function Meta({ label, value }: { label: string; value: string }) { return <div><dt>{label}</dt><dd>{value}</dd></div>; }

const singleTypes = [
  ["text", "Text", FileText], ["document", "Document", File], ["code", "Code / Notebook", Code2],
  ["image", "Image", Image], ["url", "Web page", Globe2], ["git", "Public Git", GitBranch], ["archive", "Archive", Archive],
] as const;

function AddSourceModal({ onClose, onComplete }: { onClose: () => void; onComplete: () => void }) {
  const api = useApi(); const [mode, setMode] = useState<"batch" | "single">("batch"); const [singleType, setSingleType] = useState("text");
  const [files, setFiles] = useState<File[]>([]); const [links, setLinks] = useState(""); const [text, setText] = useState(""); const [title, setTitle] = useState(""); const [ref, setRef] = useState("");
  const [status, setStatus] = useState(""); const [run, setRun] = useState<IngestionRun | null>(null); const [busy, setBusy] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setStatus("Preparing sources in your browser…");
    try {
      let inputs: SourceInput[] = [];
      if (mode === "batch") {
        inputs = await Promise.all(files.map(fileInput));
        inputs.push(...links.split(/\r?\n/).map(value => value.trim()).filter(Boolean).map(url => ({ kind: looksGit(url) ? "git" : "url", url } as SourceInput)));
      } else if (singleType === "text") inputs = [{ kind: "text", text, title: title || "Pasted text", source_key: title ? `studio:${title}` : undefined }];
      else if (singleType === "url" || singleType === "git") inputs = [{ kind: singleType, url: links.trim(), title: title || undefined, ref: ref || undefined } as SourceInput];
      else if (files[0]) { const input = await fileInput(files[0]); input.title = title || undefined; if (singleType === "image") input.kind = "image"; if (singleType === "archive") input.kind = "archive"; inputs = [input]; }
      if (!inputs.length) throw new Error("Add at least one file, link, or text source.");
      if (inputs.length > 100) throw new Error("One run can contain at most 100 sources.");
      const accepted = await api.ingest(inputs); setRun(accepted); setStatus(`Run ${accepted.id} is processing…`);
      let current = accepted;
      for (let attempt = 0; attempt < 600 && ["pending", "running"].includes(current.status); attempt++) { await new Promise(resolve => setTimeout(resolve, 500)); current = await api.ingestionRun(current.id); setRun(current); }
      const failed = current.results.filter(item => item.status === "failed").length;
      setStatus(`${current.results.length - failed} ready${failed ? ` · ${failed} failed` : ""}`);
      if (!failed) setTimeout(onComplete, 600);
    } catch (error) { setStatus((error as Error).message); } finally { setBusy(false); }
  };
  return <Modal title="Add sources" subtitle="Import one tailored source or submit a mixed batch." onClose={onClose} wide>
    <div className="segmented"><button className={mode === "batch" ? "active" : ""} onClick={() => setMode("batch")}><Upload size={16} /> Batch import</button><button className={mode === "single" ? "active" : ""} onClick={() => setMode("single")}><Plus size={16} /> Single source</button></div>
    <form onSubmit={submit} className="ingest-form">
      {mode === "single" && <div className="source-type-tabs">{singleTypes.map(([value, label, Icon]) => <button type="button" key={value} className={singleType === value ? "active" : ""} onClick={() => { setSingleType(value); setFiles([]); setLinks(""); }}><Icon size={17} />{label}</button>)}</div>}
      {(mode === "batch" || !["text", "url", "git"].includes(singleType)) && <label className="dropzone"><Upload size={24} /><strong>{mode === "batch" ? "Drop or choose multiple files" : `Choose one ${singleType} source`}</strong><span>Documents, code, notebooks, images, archives, and structured data up to 25 MiB each.</span><input type="file" multiple={mode === "batch"} onChange={event => setFiles([...event.target.files || []])} />{files.length > 0 && <div className="file-queue">{files.map(file => <span key={`${file.name}:${file.size}`}><File size={14} />{file.name}<small>{formatBytes(file.size)}</small></span>)}</div>}</label>}
      {(mode === "batch" || ["url", "git"].includes(singleType)) && <label><span>{mode === "batch" ? "Links — one web page or public Git repository per line" : singleType === "git" ? "Public HTTPS Git URL" : "Web page URL"}</span><textarea rows={mode === "batch" ? 5 : 2} value={links} onChange={e => setLinks(e.target.value)} placeholder={mode === "batch" ? "https://example.com/guide\nhttps://github.com/org/repository" : "https://…"} /></label>}
      {singleType === "text" && mode === "single" && <label><span>Text to remember</span><textarea rows={10} value={text} onChange={e => setText(e.target.value)} placeholder="Paste notes, decisions, or reference material…" /></label>}
      {mode === "single" && <div className="form-grid"><label><span>Display title <em>optional</em></span><input value={title} onChange={e => setTitle(e.target.value)} /></label>{singleType === "git" && <label><span>Branch or tag <em>optional</em></span><input value={ref} onChange={e => setRef(e.target.value)} /></label>}</div>}
      {run && <div className="run-progress"><div><strong>{run.status}</strong><span>{run.id}</span></div>{run.results.map(item => <div className={`run-item ${item.status}`} key={item.index}>{item.status === "failed" ? <X size={15} /> : <CircleCheck size={15} />}<span>{item.source_key || item.kind}</span><small>{item.error}</small></div>)}</div>}
      <footer><p className="form-status">{status}</p><button type="button" className="quiet" onClick={onClose}>Cancel</button><button className="primary" disabled={busy}>{busy ? <LoaderCircle className="spin" size={16} /> : <Upload size={16} />} Ingest sources</button></footer>
    </form>
  </Modal>;
}

async function fileInput(file: File): Promise<SourceInput> {
  const bytes = new Uint8Array(await file.arrayBuffer()); let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  const image = /^image\/(png|jpeg|webp)$/.test(file.type); const archive = /\.(zip|tar|tgz|tar\.gz)$/i.test(file.name);
  return { kind: image ? "image" : archive ? "archive" : "file", filename: file.name, content_base64: btoa(binary), source_key: `studio:${file.name}` };
}
function looksGit(url: string) { return url.endsWith(".git") || /github\.com|gitlab\.com|bitbucket\.org/.test(url); }

function QueriesPage() {
  const api = useApi(); const client = useQueryClient(); const [search, setSearch] = useState(""); const [question, setQuestion] = useState("");
  const navigate = useNavigate();
  const [selected, setSelected] = useState<QueryRun | null>(null); const [selectedStep, setSelectedStep] = useState<QueryStep | null>(null); const [running, setRunning] = useState(false);
  const history = useQuery({ queryKey: ["query-runs", search, api.connection], queryFn: () => api.listQueries(search), refetchInterval: 3_000 });
  const open = async (run: QueryRun) => { const value = await api.getQuery(run.id); setSelected(value); setSelectedStep(value.steps.at(-1) || null); };
  const ask = async (event: FormEvent) => { event.preventDefault(); if (!question.trim()) return; setRunning(true); try { const run = await api.createQuery(question.trim()); setSelected(run); setQuestion(""); const controller = new AbortController(); try { await api.streamQuery(run.id, value => { setSelected(value); setSelectedStep(current => current ? value.steps.find(step => step.id === current.id) || value.steps.at(-1) || null : value.steps.at(-1) || null); }, controller.signal); } catch { let value = await api.getQuery(run.id); for (let i = 0; i < 240 && ["pending", "running"].includes(value.status); i++) { await new Promise(resolve => setTimeout(resolve, 500)); value = await api.getQuery(run.id); setSelected(value); } } await client.invalidateQueries({ queryKey: ["query-runs"] }); } finally { setRunning(false); } };
  const remove = async (id: string) => { if (!confirm("Remove this saved query workflow?")) return; await api.removeQuery(id); if (selected?.id === id) setSelected(null); client.invalidateQueries({ queryKey: ["query-runs"] }); };
  return <>
    <PageHeader eyebrow="GROUNDED RETRIEVAL" title="Queries" description="Ask memory, watch every retrieval stage, and reopen the same evidence path later." />
    <form className="query-composer" onSubmit={ask}><Sparkles size={19} /><textarea rows={2} value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask what memory knows…" /><button className="primary" disabled={running || !question.trim()}>{running ? <LoaderCircle className="spin" size={17} /> : <Send size={17} />} Ask</button></form>
    <div className="query-workspace">
      <aside className="query-history"><header><strong>Past queries</strong><button className="icon-button" title="Clear query history" onClick={async () => { if (confirm("Remove all saved query workflows in this namespace?")) { await api.clearQueries(); setSelected(null); client.invalidateQueries({ queryKey: ["query-runs"] }); } }}><Trash2 size={15} /></button></header><label className="search-field compact"><Search size={14} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Find a query" /></label><div>{history.data?.runs.map(run => <button key={run.id} className={classNames("history-item", selected?.id === run.id && "active")} onClick={() => open(run)}><span>{run.query}</span><small><i className={`run-dot ${run.status}`} />{formatDate(run.created_at)}</small></button>)}</div></aside>
      <section className="query-inspector">{selected ? <><QueryAnswer run={selected} onRemove={() => remove(selected.id)} onCitation={sourceId => navigate(`/sources?source=${encodeURIComponent(sourceId)}`)} /><Workflow run={selected} selected={selectedStep} onSelect={setSelectedStep} /></> : <Empty title="Run or select a query" detail="The retrieval workflow will appear here as a clickable diagram." icon={<Sparkles size={28} />} />}</section>
      <aside className="step-detail">{selectedStep ? <StepDetail step={selectedStep} onClose={() => setSelectedStep(null)} /> : <div className="detail-placeholder"><PanelLeftClose size={21} /><p>Select a workflow box to inspect its inputs and outputs.</p></div>}</aside>
    </div>
  </>;
}

function QueryAnswer({ run, onRemove, onCitation }: { run: QueryRun; onRemove: () => void; onCitation: (sourceId: string) => void }) {
  return <div className="answer-card"><div className="answer-top"><div><span className={`status-chip ${run.status}`}>{run.status}</span>{run.retrieval_trace && <span className="subtle-chip">{run.retrieval_trace.stage}</span>}</div><button className="icon-button" onClick={onRemove}><Trash2 size={15} /></button></div><h2>{run.query}</h2>{run.answer ? <p>{run.answer}</p> : <p className="muted">{run.error || "Retrieval is running…"}</p>}{run.citations.length > 0 && <div className="citations">{run.citations.map((citation, index) => { const sourceId = citation.source_ref && typeof citation.source_ref === "object" ? citation.source_ref.id : undefined; return <button key={citation.delta_id} disabled={!sourceId} title={sourceId ? "Open the supporting source" : "Trace citation"} onClick={() => sourceId && onCitation(sourceId)}><span>{index + 1}</span>{citation.source_ref && typeof citation.source_ref === "object" ? citation.source_ref.title || citation.delta_id : citation.delta_id}</button>; })}</div>}</div>;
}

function Workflow({ run, selected, onSelect }: { run: QueryRun; selected: QueryStep | null; onSelect: (step: QueryStep) => void }) {
  const nodes: Node[] = run.steps.map((step, index) => ({ id: step.id, position: { x: index % 2 ? 320 : 40, y: index * 112 }, data: { label: <div className="flow-node"><span>{step.phase.replaceAll("_", " ")}</span><strong>{step.label}</strong>{step.duration_ms != null && <small>{step.duration_ms.toFixed(0)} ms</small>}</div> }, className: classNames("workflow-node", step.status, selected?.id === step.id && "selected") }));
  const edges: Edge[] = run.steps.flatMap((step, index) => { const parents = step.parent_ids.length ? step.parent_ids : index ? [run.steps[index - 1].id] : []; return parents.map(parent => ({ id: `${parent}:${step.id}`, source: parent, target: step.id, type: "smoothstep", animated: run.status === "running" && index === run.steps.length - 1, label: step.phase === "deep_recall" ? "low confidence" : undefined })); });
  return <div className="workflow-canvas" aria-label="Retrieval workflow"><ReactFlow nodes={nodes} edges={edges} fitView nodesDraggable={false} nodesConnectable={false} onNodeClick={(_, node) => { const step = run.steps.find(item => item.id === node.id); if (step) onSelect(step); }}><Background gap={18} size={1} color="#e9e9e7" /><Controls showInteractive={false} /></ReactFlow></div>;
}

function StepDetail({ step, onClose }: { step: QueryStep; onClose: () => void }) {
  return <div className="step-panel"><button className="icon-button drawer-close" onClick={onClose} aria-label="Close step details"><X size={16} /></button><div className="eyebrow">{step.phase}</div><h2>{step.label}</h2><div className="step-stats"><span>{step.status}</span>{step.duration_ms != null && <span>{step.duration_ms.toFixed(1)} ms</span>}</div><DataSection title="Input" value={step.input} /><DataSection title="Output" value={step.output} /><DataSection title="Metrics" value={step.metrics} />{step.candidates.length > 0 && <section><h3>Ranked candidates</h3><div className="candidate-list">{step.candidates.map(candidate => <article key={`${candidate.route}:${candidate.item_id}`}><header><span>{candidate.route} · #{candidate.rank}</span><strong>{candidate.score.toFixed(4)}</strong></header><p>{candidate.excerpt}</p><code>{candidate.item_id}</code></article>)}</div></section>}</div>;
}
function DataSection({ title, value }: { title: string; value: Record<string, unknown> }) { if (!Object.keys(value).length) return null; return <section><h3>{title}</h3><pre>{json(value)}</pre></section>; }

function MemoryPage() {
  const api = useApi(); const navigate = useNavigate(); const [view, setView] = useState<GraphPage["view"]>("knowledge"); const [search, setSearch] = useState(""); const [selected, setSelected] = useState<GraphNode | null>(null); const [table, setTable] = useState(false); const [neighborhoods, setNeighborhoods] = useState<GraphPage[]>([]);
  const graph = useQuery({ queryKey: ["memory-graph", view, search, api.connection], queryFn: () => api.graph(view, search) });
  useEffect(() => setNeighborhoods([]), [view, search]);
  const visibleGraph = useMemo(() => {
    if (!graph.data) return undefined;
    const pages = [graph.data, ...neighborhoods];
    const nodes = [...new Map(pages.flatMap(page => page.nodes).map(node => [node.id, node])).values()];
    const edges = [...new Map(pages.flatMap(page => page.edges).map(edge => [edge.id, edge])).values()];
    const counts = nodes.reduce<Record<string, number>>((all, node) => ({ ...all, [node.type]: (all[node.type] || 0) + 1 }), {});
    return { ...graph.data, nodes, edges, counts };
  }, [graph.data, neighborhoods]);
  return <><PageHeader eyebrow="TRACE + RECALL" title="Memory Viewer" description="Explore knowledge, provenance, and the ordered evidence that supports every connection." actions={<button className="secondary" onClick={() => setTable(value => !value)}>{table ? <Network size={16} /> : <PanelLeftClose size={16} />}{table ? "Graph view" : "Accessible list"}</button>} />
    <div className="graph-toolbar"><div className="segmented">{(["knowledge", "lineage", "trace"] as const).map(value => <button key={value} className={view === value ? "active" : ""} onClick={() => { setView(value); setSelected(null); }}>{value}</button>)}</div><label className="search-field"><Search size={15} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Find a source, concept, claim, or Trace record" /></label></div>
    <div className="graph-layout"><section className="graph-card">{graph.isLoading ? <Loading label="Building graph" /> : graph.isError ? <Empty title="Graph unavailable" detail={(graph.error as Error).message} /> : visibleGraph && (table ? <GraphTable graph={visibleGraph} onSelect={setSelected} /> : <CytoscapeGraph graph={visibleGraph} onSelect={setSelected} />)}</section><aside className="graph-detail">{selected ? <><button className="icon-button drawer-close" onClick={() => setSelected(null)} aria-label="Close node details"><X size={16} /></button><div className={`node-badge ${selected.type}`}>{selected.type}</div><h2>{selected.label}</h2><p>{selected.subtitle}</p><pre>{json(selected.data)}</pre>{selected.type === "source" && <button className="secondary" onClick={() => navigate(`/sources?source=${encodeURIComponent(selected.id)}`)}><ExternalLink size={15} /> Open source</button>}<button className="secondary" onClick={async () => { const values = await api.graphNeighbors(selected.id, view); setNeighborhoods(current => [...current, values]); }}><Network size={15} /> Expand neighborhood</button></> : <div className="detail-placeholder"><Network size={23} /><p>Select a node to inspect its memory role and provenance.</p></div>}</aside></div>
    {visibleGraph && <div className="graph-legend">{Object.entries(visibleGraph.counts).map(([key, value]) => <span key={key}><i className={key} />{key} <strong>{value}</strong></span>)}{visibleGraph.truncated && <em>Overview is bounded; search or select a node to reach more memory.</em>}</div>}
  </>;
}

function CytoscapeGraph({ graph, onSelect }: { graph: GraphPage; onSelect: (node: GraphNode) => void }) {
  const container = useRef<HTMLDivElement>(null); const instance = useRef<Core | null>(null);
  useEffect(() => {
    if (!container.current) return;
    instance.current?.destroy();
    const cy = cytoscape({ container: container.current, elements: [...graph.nodes.map(node => ({ data: { ...node, label: node.label } })), ...graph.edges.map(edge => ({ data: edge }))], style: [
      { selector: "node", style: { "background-color": "#e9e9e7", "border-color": "#9b9a97", "border-width": "1px", label: "data(label)", color: "#37352f", "font-size": "10px", "text-wrap": "ellipsis", "text-max-width": "110px", width: "34px", height: "34px" } },
      { selector: "node[type = 'source']", style: { "background-color": "#d3e5ef", shape: "round-rectangle", width: "46px" } },
      { selector: "node[type = 'concept']", style: { "background-color": "#fdecc8", width: "42px", height: "42px" } },
      { selector: "node[type = 'claim']", style: { "background-color": "#e8deee", shape: "diamond" } },
      { selector: "node[type = 'trace']", style: { "background-color": "#dbeddb" } },
      { selector: "node[type = 'recall']", style: { "background-color": "#f5e0e9" } },
      { selector: "edge", style: { width: "1.2px", "line-color": "#c6c5c2", "target-arrow-color": "#c6c5c2", "target-arrow-shape": "triangle", "curve-style": "bezier", label: "data(label)", "font-size": "8px", color: "#787774", "text-background-color": "#fff", "text-background-opacity": 0.8, "text-background-padding": "2px" } },
      { selector: ":selected", style: { "border-color": "#2383e2", "border-width": "3px", "line-color": "#2383e2", "target-arrow-color": "#2383e2" } },
    ], layout: graph.view === "knowledge" ? { name: "cose", animate: false, fit: true, padding: 42 } : graph.view === "lineage" ? { name: "breadthfirst", directed: true, padding: 42, spacingFactor: 1.25 } : { name: "grid", padding: 42, avoidOverlap: true } });
    cy.on("tap", "node", event => { const found = graph.nodes.find(item => item.id === event.target.id()); if (found) onSelect(found); });
    instance.current = cy; return () => cy.destroy();
  }, [graph, onSelect]);
  return <div className="cytoscape" ref={container} />;
}
function GraphTable({ graph, onSelect }: { graph: GraphPage; onSelect: (node: GraphNode) => void }) { return <div className="graph-table" role="table"><div role="row" className="graph-table-head"><span>Type</span><span>Memory item</span><span>Status</span></div>{graph.nodes.map(node => <button role="row" key={node.id} onClick={() => onSelect(node)}><span><i className={node.type} />{node.type}</span><span><strong>{node.label}</strong><small>{node.subtitle}</small></span><span>{node.status || "—"}</span></button>)}</div>; }

function ConfigurationPage() {
  const api = useApi(); const client = useQueryClient(); const providers = useQuery({ queryKey: ["providers", api.connection], queryFn: () => api.providers() }); const models = useQuery({ queryKey: ["models", api.connection], queryFn: () => api.modelConfiguration() }); const retrieval = useQuery({ queryKey: ["retrieval-config", api.connection], queryFn: () => api.retrievalConfiguration() });
  return <><PageHeader eyebrow="STORE-WIDE SETUP" title="Configuration" description="Choose generation and embedding models, then tune how retrieval expands and abstains." />
    <div className="configuration-grid"><ModelSettings providers={providers.data?.providers || []} state={models.data} onSaved={() => client.invalidateQueries({ queryKey: ["models"] })} /><RetrievalSettings value={retrieval.data} onSaved={() => client.invalidateQueries({ queryKey: ["retrieval-config"] })} /></div>
  </>;
}

function ModelSettings({ providers, state, onSaved }: { providers: Array<{ id: string; display_name: string; roles: Array<"completion" | "embedding">; credential_env?: string | null; credential_configured: boolean }>; state?: ModelConfigurationState; onSaved: () => void }) {
  const api = useApi(); const [draft, setDraft] = useState<ModelConfigurationState | null>(null); const [confirmRebuild, setConfirmRebuild] = useState(false); const [message, setMessage] = useState("");
  useEffect(() => { if (state) setDraft(structuredClone(state)); }, [state]);
  if (!draft) return <section className="settings-card"><Loading label="Loading model configuration" /></section>;
  const change = (role: "completion" | "embedding", field: string, value: string) => setDraft(current => current ? ({ ...current, configuration: { ...current.configuration, [role]: { ...current.configuration[role], [field]: value || null } } }) : current);
  const save = async () => { try { setMessage("Saving…"); await api.saveModels(draft, confirmRebuild); setMessage("Model configuration saved."); onSaved(); } catch (error) { setMessage((error as Error).message); } };
  return <section className="settings-card"><header><div><span className="kicker">MODELS</span><h2>Completion and embeddings</h2></div><span className="revision">revision {draft.configuration.revision}</span></header>{(["completion", "embedding"] as const).map(role => <div className="model-role" key={role}><div><h3>{role === "completion" ? "Completion" : "Embedding"}</h3><p>{role === "completion" ? "Extraction, grounded answers, and image understanding." : "Semantic indexing and retrieval. A change may rebuild the index."}</p></div><div className="form-grid"><label><span>Provider</span><select value={draft.configuration[role].provider} onChange={e => change(role, "provider", e.target.value)}>{providers.filter(item => item.roles.includes(role)).map(item => <option value={item.id} key={item.id}>{item.display_name}{!item.credential_configured && item.credential_env ? ` · needs ${item.credential_env}` : ""}</option>)}</select></label><label><span>Model ID</span><input value={draft.configuration[role].model || ""} onChange={e => change(role, "model", e.target.value)} placeholder={role === "completion" ? "Provider model ID" : "Embedding model ID"} /></label><label className="wide-field"><span>Custom base URL <em>optional</em></span><input value={draft.configuration[role].base_url || ""} onChange={e => change(role, "base_url", e.target.value)} /></label></div><button className="quiet test-button" onClick={async () => { const value = await api.testModel(role, draft.configuration[role]); setMessage(value.message); }}><Wifi size={14} /> Test {role}</button></div>)}<label className="checkbox-row"><input type="checkbox" checked={confirmRebuild} onChange={e => setConfirmRebuild(e.target.checked)} /><span>Confirm a staged embedding rebuild if the embedding selection changes.</span></label><footer><p>{message || state?.message}</p><button className="primary" onClick={save}>Save models</button></footer></section>;
}

function RetrievalSettings({ value, onSaved }: { value?: RetrievalConfiguration; onSaved: () => void }) {
  const api = useApi(); const [draft, setDraft] = useState<RetrievalConfiguration | null>(null); const [message, setMessage] = useState(""); useEffect(() => { if (value) setDraft({ ...value }); }, [value]);
  if (!draft) return <section className="settings-card"><Loading label="Loading retrieval settings" /></section>;
  const number = (key: keyof RetrievalConfiguration, next: string) => setDraft(current => current ? { ...current, [key]: Number(next) } : current);
  const reset = () => setDraft(current => current ? { ...current, default_top_k: 12, max_context_items: 24, max_refinement_rounds: 2, graph_hops: 2, confidence_margin: .018, deep_recall_enabled: true, answer_abstain_threshold: .1 } : current);
  const save = async () => { try { const saved = await api.saveRetrieval(draft); setDraft(saved); setMessage("Retrieval settings saved."); onSaved(); } catch (error) { setMessage((error as Error).message); } };
  return <section className="settings-card"><header><div><span className="kicker">RETRIEVAL</span><h2>Search and grounding</h2></div><span className="revision">revision {draft.revision}</span></header><div className="setting-list"><NumberSetting label="Default results" detail="Evidence returned for a normal search." value={draft.default_top_k} min={1} max={100} onChange={value => number("default_top_k", value)} /><NumberSetting label="Maximum answer context" detail="Most Trace records allowed into answer context." value={draft.max_context_items} min={draft.default_top_k} max={100} onChange={value => number("max_context_items", value)} /><NumberSetting label="Refinement rounds" detail="How often a low-confidence query can be rewritten." value={draft.max_refinement_rounds} min={0} max={2} onChange={value => number("max_refinement_rounds", value)} /><NumberSetting label="Graph hops" detail="Connection depth used while grounding." value={draft.graph_hops} min={0} max={4} onChange={value => number("graph_hops", value)} /><NumberSetting label="Confidence margin" detail="Required score separation before accepting a route." value={draft.confidence_margin} min={0} max={1} step="0.001" onChange={value => number("confidence_margin", value)} /><NumberSetting label="Abstain threshold" detail="Minimum top score required to answer." value={draft.answer_abstain_threshold} min={0} max={1} step="0.01" onChange={value => number("answer_abstain_threshold", value)} /><label className="switch-row"><div><strong>Deep Recall</strong><span>Expand the route when normal retrieval remains uncertain.</span></div><input type="checkbox" checked={draft.deep_recall_enabled} onChange={e => setDraft(current => current ? { ...current, deep_recall_enabled: e.target.checked } : current)} /></label></div><footer><p>{message}</p><button className="quiet" onClick={reset}>Reset defaults</button><button className="primary" onClick={save}>Save retrieval</button></footer></section>;
}
function NumberSetting({ label, detail, value, onChange, ...input }: { label: string; detail: string; value: number; onChange: (value: string) => void; min: number; max: number; step?: string }) { return <label className="number-setting"><div><strong>{label}</strong><span>{detail}</span></div><input type="number" value={value} onChange={e => onChange(e.target.value)} {...input} /></label>; }

function ConnectionPage({ value, onSave }: { value: Connection; onSave: (value: Connection) => void }) {
  const api = useApi(); const client = useQueryClient(); const [draft, setDraft] = useState(value); const [message, setMessage] = useState(""); const session = useQuery({ queryKey: ["session", api.connection], queryFn: () => api.session() });
  const save = async (event: FormEvent) => { event.preventDefault(); sessionStorage.setItem("trisynapse.baseUrl", draft.baseUrl); sessionStorage.setItem("trisynapse.token", draft.token); localStorage.setItem("trisynapse.namespace", JSON.stringify(draft.namespace)); onSave(draft); setMessage("Connection saved for this browser tab."); await client.invalidateQueries(); };
  return <><PageHeader eyebrow="SERVER + ISOLATION" title="Connection" description="Connect Studio to a memory server and choose the namespace applied to every operation." />
    <div className="connection-layout"><form className="settings-card connection-card" onSubmit={save}><header><div><span className="kicker">SERVER</span><h2>API connection</h2></div><span className={`connection-role ${session.data?.role || "unknown"}`}>{session.data?.role || "not connected"}</span></header><label><span>API base URL</span><input value={draft.baseUrl} onChange={e => setDraft({ ...draft, baseUrl: e.target.value.replace(/\/$/, "") })} placeholder="Same origin" /><small>Leave empty when Studio is served by the Trisynapse server.</small></label><label><span>Bearer token</span><input type="password" autoComplete="off" value={draft.token} onChange={e => setDraft({ ...draft, token: e.target.value })} placeholder="Token from the memory store" /><small>Kept in session storage only and never written to the server database.</small></label><div className="divider" /><span className="kicker">NAMESPACE</span><div className="form-grid"><NamespaceField label="Project ID" value={draft.namespace.project_id} onChange={project_id => setDraft({ ...draft, namespace: { ...draft.namespace, project_id } })} required /><NamespaceField label="User ID" value={draft.namespace.user_id || ""} onChange={user_id => setDraft({ ...draft, namespace: { ...draft.namespace, user_id: user_id || null } })} /><NamespaceField label="Agent ID" value={draft.namespace.agent_id || ""} onChange={agent_id => setDraft({ ...draft, namespace: { ...draft.namespace, agent_id: agent_id || null } })} /><NamespaceField label="Session ID" value={draft.namespace.session_id || ""} onChange={session_id => setDraft({ ...draft, namespace: { ...draft.namespace, session_id: session_id || null } })} /></div><footer><p>{message}</p><button className="primary">Save and reconnect</button></footer></form><aside className="connection-info"><div className={classNames("connection-health", session.isSuccess && "ready", session.isError && "error")}><Wifi size={21} /><div><strong>{session.isSuccess ? "Connection ready" : session.isError ? "Connection failed" : "Checking connection"}</strong><p>{session.isError ? (session.error as Error).message : session.data ? `${session.data.role} access to ${session.data.effective_namespace.project_id}` : "Reading server capabilities…"}</p></div></div><article><HelpCircle size={18} /><div><strong>Custom servers and CORS</strong><p>A different API origin must explicitly allow the Studio origin. Same-origin serving needs no CORS setup.</p></div></article><article><Database size={18} /><div><strong>Namespace isolation</strong><p>Project, user, agent, and session identifiers scope sources, queries, Trace records, and graph views together.</p></div></article></aside></div>
  </>;
}
function NamespaceField({ label, value, onChange, required }: { label: string; value: string; onChange: (value: string) => void; required?: boolean }) { return <label><span>{label}</span><input value={value} onChange={e => onChange(e.target.value)} required={required} /></label>; }

function Loading({ label }: { label: string }) { return <div className="loading"><LoaderCircle className="spin" size={20} /><span>{label}…</span></div>; }
function Empty({ title, detail, action, icon }: { title: string; detail: string; action?: ReactNode; icon?: ReactNode }) { return <div className="empty-state">{icon || <Inbox size={28} />}<h2>{title}</h2><p>{detail}</p>{action}</div>; }
