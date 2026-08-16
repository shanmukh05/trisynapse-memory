import { defineConfig, devices } from "@playwright/test";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const localCli = resolve(here, "../../.venv/bin/trisynapse-memory");
const cli = existsSync(localCli) ? localCli : "trisynapse-memory";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://127.0.0.1:8877/studio/",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "narrow", use: { ...devices["iPhone 13"], browserName: "chromium" } },
  ],
  webServer: {
    command: `${cli} --path /tmp/trisynapse-studio-e2e serve --studio --no-auth --port 8877`,
    url: "http://127.0.0.1:8877/studio/",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
