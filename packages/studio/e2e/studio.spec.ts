import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("opens Memory Viewer as home and keeps the five-view shell", async ({ page }, testInfo) => {
  await page.goto("./");
  await expect(page.getByRole("heading", { name: "Memory Viewer" })).toBeVisible();
  for (const name of ["Memory Viewer", "Sources", "Queries", "Configuration", "Connection"]) {
    await expect(page.getByRole("link", { name })).toBeVisible();
  }
  const logo = page.locator('img[alt="Trisynapse"]:visible');
  await expect(logo).toBeVisible();
  expect(await logo.evaluate(image => ({ width: (image as HTMLImageElement).naturalWidth, height: (image as HTMLImageElement).naturalHeight }))).toEqual({ width: 1536, height: 1024 });

  await page.getByRole("link", { name: "Sources" }).click();
  await expect(page.getByRole("heading", { name: "Sources" })).toBeVisible();
  await page.getByRole("button", { name: "Add source" }).click();
  await expect(page.getByRole("dialog", { name: "Add sources" })).toBeVisible();
  await page.getByRole("button", { name: "Single source" }).click();
  for (const name of ["Text", "Document", "Code / Notebook", "Image", "Web page", "Public Git", "Archive"]) {
    await expect(page.getByRole("button", { name })).toBeVisible();
  }

  const results = await new AxeBuilder({ page }).exclude(".react-flow").exclude(".vector-canvas").analyze();
  expect(results.violations, `${testInfo.project.name} accessibility violations`).toEqual([]);
});

test("switches Recall helper tabs on Memory Viewer", async ({ page }, testInfo) => {
  await page.goto("./");
  await expect(page.getByRole("heading", { name: "Memory Viewer" })).toBeVisible();
  const tabs = page.getByRole("tablist", { name: "Recall helpers" });
  await expect(tabs.getByRole("tab", { name: "BM25" })).toBeVisible();
  await tabs.getByRole("tab", { name: "BM25" }).click();
  await expect(page).toHaveURL(/helper=bm25/);
  await expect(page.getByPlaceholder("Look up a term")).toBeVisible();
  await tabs.getByRole("tab", { name: "Vectors" }).click();
  await expect(page.getByText(/embedded \/|No embeddings yet/)).toBeVisible();
  await tabs.getByRole("tab", { name: "Claims" }).click();
  await expect(page).toHaveURL(/helper=claims/);
  await expect(page.getByText(/No compiled claims|Claims unavailable/)).toBeVisible();
  const results = await new AxeBuilder({ page }).exclude(".react-flow").exclude(".vector-canvas").analyze();
  expect(results.violations, `${testInfo.project.name} accessibility violations`).toEqual([]);
});
