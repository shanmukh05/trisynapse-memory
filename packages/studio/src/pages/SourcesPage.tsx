import { FormEvent, useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Archive, Box, CircleCheck, Clock3, Code2, Download, File, FileImage, FileText,
  FolderGit2, GitBranch, Globe2, Image, LoaderCircle, Plus, RefreshCw, Search, Trash2, Upload, X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useApi } from "../context";
import { Empty, Loading, Meta, Modal, PageHeader } from "../ui";
import { formatBytes, formatDate, json, locatorLabel } from "../util";
import type { IngestionRun, PreviewItem, Source, SourceInput, SourcePreview } from "../types";

const sourceIcons: Record<string, typeof File> = {
  text: FileText, file: File, image: FileImage, git: FolderGit2,
  directory: FolderGit2, archive: Archive, url: Globe2,
};

export function SourcesPage() {
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
    void api.getSource(sourceId).then(value => setSelected(value)).catch(() => setParams({}, { replace: true }));
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
  const navigate = useNavigate();
  const code = ["code_symbol", "code_lines", "notebook_cell"].includes(item.kind);
  const markdown = item.kind === "text" || item.kind === "paragraph";
  return <article className="preview-block">
    <header><span>{item.kind.replaceAll("_", " ")}</span><code>{locatorLabel(item.locator)}</code></header>
    {showId && <small>{item.delta_id} <button className="quiet" onClick={() => navigate(`/memory?helper=trace&id=${encodeURIComponent(item.delta_id)}`)}>View in Memory</button></small>}
    {code ? <pre><code>{item.text}</code></pre> : markdown ? <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{item.text}</ReactMarkdown></div> : <p className="preserve">{item.text}</p>}
  </article>;
}

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
