export function formatBytes(value = 0) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

export function formatDate(value?: string | null) {
  return value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "—";
}

export function json(value: unknown) {
  return JSON.stringify(value, null, 2);
}

export function classNames(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function locatorLabel(value: Record<string, unknown> | string | null | undefined) {
  if (!value) return "";
  if (typeof value === "string") return value;
  return Object.entries(value).filter(([key]) => key !== "metadata").map(([key, item]) => `${key}: ${String(item)}`).join(" · ");
}
