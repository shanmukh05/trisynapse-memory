import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("renders the five-view shell with the canonical logo", async ({ page }, testInfo) => {
  await page.goto("./");
  await expect(page.getByRole("heading", { name: "Sources" })).toBeVisible();
  for (const name of ["Sources", "Queries", "Memory Viewer", "Configuration", "Connection"]) {
    await expect(page.getByRole("link", { name })).toBeVisible();
  }
  const logo = page.locator('img[alt="Trisynapse"]:visible');
  await expect(logo).toBeVisible();
  expect(await logo.evaluate(image => ({ width: (image as HTMLImageElement).naturalWidth, height: (image as HTMLImageElement).naturalHeight }))).toEqual({ width: 1536, height: 1024 });

  await page.getByRole("button", { name: "Add source" }).click();
  await expect(page.getByRole("dialog", { name: "Add sources" })).toBeVisible();
  await page.getByRole("button", { name: "Single source" }).click();
  for (const name of ["Text", "Document", "Code / Notebook", "Image", "Web page", "Public Git", "Archive"]) {
    await expect(page.getByRole("button", { name })).toBeVisible();
  }

  const results = await new AxeBuilder({ page }).exclude(".react-flow").exclude(".cytoscape").analyze();
  expect(results.violations, `${testInfo.project.name} accessibility violations`).toEqual([]);
});

test("keeps navigation usable on a narrow screen", async ({ page }) => {
  await page.goto("./queries");
  await expect(page.getByRole("heading", { name: "Queries" })).toBeVisible();
  await page.getByRole("link", { name: "Memory Viewer" }).click();
  await expect(page.getByRole("heading", { name: "Memory Viewer" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Accessible list" })).toBeVisible();
});
