import { ReactNode, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import { Database, Inbox, Network, Settings2, Sparkles, Wifi } from "lucide-react";
import logo from "@trisynapse-logo";
import { initialConnection, StudioApi } from "./api";
import { ApiContext } from "./context";
import { classNames } from "./util";
import { MemoryPage } from "./pages/MemoryPage";
import { SourcesPage } from "./pages/SourcesPage";
import { QueriesPage } from "./pages/QueriesPage";
import { ConfigurationPage } from "./pages/ConfigurationPage";
import { ConnectionPage } from "./pages/ConnectionPage";
import type { Connection } from "./types";

export function App() {
  const [connection, setConnection] = useState<Connection>(initialConnection);
  const api = useMemo(() => new StudioApi(connection), [connection]);
  const health = useQuery({ queryKey: ["health", connection.baseUrl], queryFn: () => api.health(), refetchInterval: 5_000 });
  return (
    <ApiContext.Provider value={api}>
      <div className="app-shell">
        <aside className="sidebar" aria-label="Studio navigation">
          <div className="brand-block">
            <img src={logo} alt="Trisynapse" />
            <div><strong>Trisynapse</strong><span>Memory Studio</span></div>
          </div>
          <nav aria-label="Studio sections">
            <SideLink to="/memory" icon={Network}>Memory Viewer</SideLink>
            <SideLink to="/sources" icon={Inbox}>Sources</SideLink>
            <SideLink to="/queries" icon={Sparkles}>Queries</SideLink>
            <SideLink to="/configuration" icon={Settings2}>Configuration</SideLink>
            <SideLink to="/connection" icon={Wifi}>Connection</SideLink>
          </nav>
          <div className="sidebar-footer">
            <div className="namespace-dot"><span />{connection.namespace.project_id || "default"}</div>
            <small>{health.data?.storage_ready ? "Store ready" : health.isError ? "Server unavailable" : "Checking store…"}</small>
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
              <Route path="/memory" element={<MemoryPage />} />
              <Route path="/sources" element={<SourcesPage />} />
              <Route path="/queries" element={<QueriesPage />} />
              <Route path="/configuration" element={<ConfigurationPage />} />
              <Route path="/connection" element={<ConnectionPage value={connection} onSave={setConnection} />} />
              <Route path="*" element={<Navigate to="/memory" replace />} />
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
