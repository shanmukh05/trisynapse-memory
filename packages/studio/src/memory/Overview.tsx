import type { CatalogHelper } from "../types";

export function Overview({ helpers, onOpen }: { helpers: CatalogHelper[]; onOpen: (id: string) => void }) {
  return <div className="overview-grid">
    {helpers.map(helper => (
      <button key={helper.id} className="overview-card" onClick={() => onOpen(helper.id)}>
        <span className="eyebrow">{helper.kind}</span>
        <h3>{helper.title}</h3>
        <strong>{helper.count}</strong>
        <ul>
          {Object.entries(helper.health).filter(([key]) => key !== "count" && key !== "edges").slice(0, 4).map(([key, value]) => (
            <li key={key}>{key.replaceAll("_", " ")} <em>{formatHealth(value)}</em></li>
          ))}
        </ul>
      </button>
    ))}
  </div>;
}

function formatHealth(value: unknown) {
  if (typeof value === "number") return String(value);
  if (typeof value === "string" || typeof value === "boolean") return String(value);
  if (value && typeof value === "object") return Object.entries(value).map(([key, item]) => `${key} ${item}`).join(", ");
  return "—";
}
