import { useQuery } from "@tanstack/react-query";
import { Background, Controls, Edge, Node, ReactFlow } from "@xyflow/react";
import { useEffect, useMemo, useRef } from "react";
import { useApi } from "../../context";
import { Empty, Loading } from "../../ui";
import { classNames } from "../../util";
import type { GraphNode, GraphPage, HelperItem, InspectorRecord } from "../../types";
import { itemToRecord, type HelperViewProps } from "./index";

export function GenericItemsView({ helper, search, selectedId, onSelect, kind }: HelperViewProps & { kind?: string }) {
  const api = useApi();
  const page = useQuery({
    queryKey: ["helper-items", helper.id, search, api.connection],
    queryFn: () => api.helperItems(helper.id, search),
  });
  if (page.isLoading) return <Loading label={`Loading ${helper.title}`} />;
  if (page.isError) return <Empty title={`${helper.title} unavailable`} detail={(page.error as Error).message} />;
  const items = page.data?.items || [];
  const family = kind || helper.kind;
  if (!items.length) return <Empty title={`No ${helper.title.toLowerCase()} yet`} detail="Ingest sources or wait for Recall to compile this helper." />;
  if (family === "cards") return <CardList items={items} selectedId={selectedId} onSelect={onSelect} />;
  if (family === "timeline") return <TimelineList items={items} selectedId={selectedId} onSelect={onSelect} />;
  if (family === "postings") return <PostingsList items={items} selectedId={selectedId} onSelect={onSelect} />;
  if (family === "graph") return <ItemTable items={items} selectedId={selectedId} onSelect={onSelect} />;
  if (family === "embedding") return <ItemTable items={items} selectedId={selectedId} onSelect={onSelect} />;
  return <ItemTable items={items} selectedId={selectedId} onSelect={onSelect} />;
}

export function ItemTable({ items, selectedId, onSelect }: { items: HelperItem[]; selectedId?: string | null; onSelect: (record: InspectorRecord) => void }) {
  return <div className="graph-table" role="table">
    <div role="row" className="graph-table-head"><span>Type</span><span>Memory item</span><span>Status</span></div>
    {items.map(item => (
      <button role="row" key={item.id} className={classNames(selectedId === item.id && "active")} onClick={() => onSelect(itemToRecord(item))}>
        <span><i className={item.kind} />{item.kind}</span>
        <span><strong>{item.title}</strong><small>{item.subtitle}</small></span>
        <span>{item.status || "—"}</span>
      </button>
    ))}
  </div>;
}

export function TimelineList({ items, selectedId, onSelect }: { items: HelperItem[]; selectedId?: string | null; onSelect: (record: InspectorRecord) => void }) {
  return <div className="timeline-list">
    {items.map(item => (
      <button key={item.id} className={classNames("timeline-item", selectedId === item.id && "active")} onClick={() => onSelect(itemToRecord(item))}>
        <span className={`node-badge ${item.kind}`}>{item.kind}</span>
        <div>
          <strong>{item.title}</strong>
          <small>{item.subtitle || item.status}</small>
          {item.excerpt && <p>{item.excerpt}</p>}
        </div>
      </button>
    ))}
  </div>;
}

export function CardList({ items, selectedId, onSelect }: { items: HelperItem[]; selectedId?: string | null; onSelect: (record: InspectorRecord) => void }) {
  return <div className="helper-cards">
    {items.map(item => (
      <button key={item.id} className={classNames("helper-card", selectedId === item.id && "active")} onClick={() => onSelect(itemToRecord(item))}>
        <header><span className={`status-chip ${item.status || ""}`}>{item.status || item.kind}</span><small>{item.subtitle}</small></header>
        <h3>{item.title}</h3>
        {item.excerpt && <p>{item.excerpt}</p>}
      </button>
    ))}
  </div>;
}

export function PostingsList({ items, selectedId, onSelect }: { items: HelperItem[]; selectedId?: string | null; onSelect: (record: InspectorRecord) => void }) {
  return <div className="postings-layout">
    <div className="postings-terms">
      {items.map(item => (
        <button key={item.id} className={classNames(selectedId === item.id && "active")} onClick={() => onSelect(itemToRecord(item))}>
          <strong>{item.title}</strong>
          <span>{item.subtitle}</span>
        </button>
      ))}
    </div>
    <div className="postings-detail">
      {items.filter(item => item.id === selectedId).map(item => {
        const postings = Array.isArray(item.data.postings) ? item.data.postings as Array<{ delta_id: string; term_frequency: number; excerpt: string }> : [];
        return <div key={item.id}>{postings.map(posting => (
          <article key={posting.delta_id}><header>{posting.delta_id}<strong>tf {posting.term_frequency}</strong></header><p>{posting.excerpt}</p></article>
        ))}</div>;
      })}
    </div>
  </div>;
}

export function MemoryFlowGraph({ graph, selectedId, onSelect }: { graph: GraphPage; selectedId?: string | null; onSelect: (node: GraphNode) => void }) {
  const columns = graph.view === "knowledge"
    ? ["concept", "claim"]
    : graph.view === "lineage"
      ? ["source", "episode", "trace", "claim", "recall"]
      : ["trace"];
  const positions = useMemo(() => layoutColumns(graph.nodes, columns), [graph.nodes, columns]);
  const nodes: Node[] = graph.nodes.map(node => ({
    id: node.id,
    position: positions[node.id] || { x: 0, y: 0 },
    data: { label: <div className="flow-node"><span>{node.type}</span><strong>{node.label}</strong></div> },
    className: classNames("workflow-node", "memory-node", node.type, selectedId === node.id && "selected"),
  }));
  const edges: Edge[] = graph.edges.map(edge => ({
    id: edge.id, source: edge.source, target: edge.target, label: edge.label || undefined, type: "smoothstep",
  }));
  return <div className="workflow-canvas memory-flow" aria-label={`${graph.view} graph`}>
    <ReactFlow nodes={nodes} edges={edges} fitView nodesDraggable={false} nodesConnectable={false} onNodeClick={(_, node) => {
      const found = graph.nodes.find(item => item.id === node.id);
      if (found) onSelect(found);
    }}>
      <Background gap={18} size={1} color="#e9e9e7" />
      <Controls showInteractive={false} />
    </ReactFlow>
  </div>;
}

function layoutColumns(nodes: GraphNode[], order: string[]) {
  const buckets: Record<string, GraphNode[]> = {};
  for (const node of nodes) (buckets[node.type] ||= []).push(node);
  const positions: Record<string, { x: number; y: number }> = {};
  const used = new Set<string>();
  order.forEach((type, column) => {
    (buckets[type] || []).forEach((node, row) => {
      positions[node.id] = { x: 40 + column * 250, y: 24 + row * 88 };
      used.add(node.id);
    });
  });
  let extra = 0;
  for (const node of nodes) {
    if (used.has(node.id)) continue;
    positions[node.id] = { x: 40 + order.length * 250, y: 24 + extra * 88 };
    extra += 1;
  }
  return positions;
}

export function VectorCanvas({ points, selectedId, onSelect }: { points: Array<{ id: string; x: number; y: number; modality: string; excerpt: string }>; selectedId?: string | null; onSelect: (id: string) => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const bounds = useMemo(() => {
    const xs = points.map(item => item.x);
    const ys = points.map(item => item.y);
    return {
      minX: Math.min(...xs, -1), maxX: Math.max(...xs, 1),
      minY: Math.min(...ys, -1), maxY: Math.max(...ys, 1),
    };
  }, [points]);
  const project = (x: number, y: number, width: number, height: number) => {
    const nx = bounds.maxX === bounds.minX ? 0.5 : (x - bounds.minX) / (bounds.maxX - bounds.minX);
    const ny = bounds.maxY === bounds.minY ? 0.5 : (y - bounds.minY) / (bounds.maxY - bounds.minY);
    return { x: nx * (width - 24) + 12, y: (1 - ny) * (height - 24) + 12 };
  };
  useEffect(() => {
    const node = canvas.current;
    if (!node) return;
    const context = node.getContext("2d");
    if (!context) return;
    const width = node.width = node.clientWidth * devicePixelRatio;
    const height = node.height = node.clientHeight * devicePixelRatio;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#fcfcfb";
    context.fillRect(0, 0, width, height);
    const colors: Record<string, string> = { document: "#7fb1d1", code: "#9b78b5", table: "#dda94f", image: "#c7779d", conversation: "#6fa47e", text: "#9b9a97" };
    for (const point of points) {
      const { x, y } = project(point.x, point.y, width, height);
      context.beginPath();
      context.arc(x, y, point.id === selectedId ? 8 : 5, 0, Math.PI * 2);
      context.fillStyle = point.id === selectedId ? "#2383e2" : colors[point.modality] || "#9b9a97";
      context.fill();
    }
  }, [points, selectedId, bounds]);
  return <canvas ref={canvas} className="vector-canvas" onClick={event => {
    const rect = event.currentTarget.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    let best: { id: string; dist: number } | null = null;
    for (const point of points) {
      const mapped = project(point.x, point.y, width, height);
      const dist = (mapped.x - (event.clientX - rect.left)) ** 2 + (mapped.y - (event.clientY - rect.top)) ** 2;
      if (!best || dist < best.dist) best = { id: point.id, dist };
    }
    if (best && best.dist < 400) onSelect(best.id);
  }} />;
}
