import { createContext, useContext } from "react";
import type { StudioApi } from "./api";

export const ApiContext = createContext<StudioApi | null>(null);

export function useApi() {
  const value = useContext(ApiContext);
  if (!value) throw new Error("Studio API is unavailable");
  return value;
}
