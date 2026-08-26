import type { ComponentType } from "react";
import type { CatalogHelper, HelperKind, InspectorRecord } from "../../types";

export type HelperViewProps = {
  helper: CatalogHelper;
  search: string;
  selectedId?: string | null;
  onSelect: (record: InspectorRecord) => void;
};

export type HelperRenderer = {
  id: string;
  kind?: HelperKind;
  render: ComponentType<HelperViewProps>;
  seedPlayground?: (record: InspectorRecord) => string;
};

const registry = new Map<string, HelperRenderer>();

export function registerHelper(renderer: HelperRenderer) {
  registry.set(renderer.id, renderer);
}

export function lookupHelper(id: string) {
  return registry.get(id);
}

export function seedFromRecord(record: InspectorRecord) {
  const renderer = lookupHelper(record.helperId);
  if (renderer?.seedPlayground) return renderer.seedPlayground(record);
  return record.excerpt || record.title;
}

export function itemToRecord(item: { id: string; helper_id: string; kind: string; title: string; subtitle?: string | null; excerpt?: string | null; status?: string | null; data: Record<string, unknown> }): InspectorRecord {
  return {
    id: item.id,
    helperId: item.helper_id,
    kind: item.kind,
    title: item.title,
    subtitle: item.subtitle,
    excerpt: item.excerpt,
    status: item.status,
    data: item.data,
  };
}
