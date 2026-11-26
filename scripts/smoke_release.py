"""Exercise the installed wheel and npm tarball together without provider calls."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen

from trisynapse_memory import MemoryNamespace
from trisynapse_memory.terminal import MemoryTerminal


async def open_terminal(store: Path) -> None:
    app = MemoryTerminal(store, MemoryNamespace())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt")


def check_server(server: subprocess.Popen, log_path: Path, base_url: str, version: str) -> None:
    deadline = monotonic() + 40
    while server.poll() is None and monotonic() < deadline:
        try:
            with urlopen(f"{base_url}/api/v1/health", timeout=2) as response:
                health = json.load(response)
        except URLError:
            sleep(0.2)
            continue
        if health["status"] != "ready" or health["version"] != version:
            raise RuntimeError(f"Unexpected packaged server health response: {health}")
        return
    raise RuntimeError(f"Packaged server did not start:\n{log_path.read_text(encoding='utf-8', errors='replace')}")


def check_studio(base_url: str) -> None:
    with urlopen(f"{base_url}/studio/", timeout=5) as response:
        html = response.read().decode()
    entry = re.search(r'src="(/studio-assets/assets/[^"]+\.js)"', html)
    if entry is None:
        raise RuntimeError("The Python wheel did not serve the built Studio entry point.")
    with urlopen(base_url + entry.group(1), timeout=5) as response:
        if not response.read(1):
            raise RuntimeError("Studio's JavaScript bundle is empty.")


def check_sdk(sdk_directory: Path, base_url: str, key: str) -> None:
    script = """
import { TrisynapseMemory } from "@trisynapse/trisynapse-memory";
const memory = new TrisynapseMemory({ baseUrl: process.argv[1], apiKey: process.env.TRISYNAPSE_MEMORY_API_KEY });
const config = await memory.getRetrievalConfiguration();
await memory.setRetrievalConfiguration({
  ...config, enabled_routes: ["bm25"], raw_candidates_only: true, graph_hops: 0,
  reranker_enabled: false, max_refinement_rounds: 0, readout: "ranked",
});
await memory.add("The release reference is trimer-030.");
const found = await memory.search("trimer-030");
if (!found.hits.some(hit => hit.text.includes("trimer-030"))) {
  throw new Error("The installed SDK could not recall the added source.");
}
"""
    subprocess.run(
        ["node", "--input-type=module", "-e", script, base_url],
        cwd=sdk_directory, env={**os.environ, "TRISYNAPSE_MEMORY_API_KEY": key},
        check=True, timeout=60,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("sdk_directory", type=Path)
    args = parser.parse_args()
    cli = Path(sys.executable).with_name("trisynapse-memory" + (".exe" if sys.platform == "win32" else ""))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        store = root / "store"
        if subprocess.check_output([cli, "--version"], text=True, cwd=root).strip() != args.version:
            raise RuntimeError("The installed CLI version does not match the release.")
        command = [cli, "--path", str(store)]
        subprocess.run([*command, "--json", "init"], cwd=root, check=True, timeout=45)
        subprocess.run([*command, "bench", "omni", "--help"], cwd=root, check=True, timeout=45)
        asyncio.run(open_terminal(store))
        key = (store / ".api-key").read_text(encoding="utf-8").strip()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"
        log_path = root / "server.log"
        with log_path.open("w", encoding="utf-8") as log:
            with subprocess.Popen(
                [*command, "serve", "--studio", "--port", str(port)],
                cwd=root, env={**os.environ, "TRISYNAPSE_MEMORY_API_KEY": key},
                stdout=log, stderr=subprocess.STDOUT,
            ) as server:
                try:
                    check_server(server, log_path, base_url, args.version)
                    check_studio(base_url)
                    check_sdk(args.sdk_directory.resolve(), base_url, key)
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server.kill()
    print("Installed Python, CLI, terminal, Studio, and npm SDK smoke checks passed.")


if __name__ == "__main__":
    main()
