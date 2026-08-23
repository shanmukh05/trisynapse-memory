I’ve prepared the npm release automation locally. It is not pushed yet because npm requires one manual bootstrap publication before trusted GitHub publishing can be configured.

## One-time setup

1. Create/sign in to npm and ensure you own the `trisynapse` scope, usually by creating the `trisynapse` npm organization.

2. Enable two-factor authentication and log in:

```bash
npm login
npm whoami
```

3. Publish the existing SDK once:

```bash
pnpm install --frozen-lockfile
pnpm --filter @trisynapse/memory check
pnpm --filter @trisynapse/memory build

npm publish ./packages/js-sdk --access public
npm view @trisynapse/memory@0.1.1 version
```

Scoped packages require `--access public`. [npm scoped-package documentation](https://docs.npmjs.com/creating-and-publishing-scoped-public-packages/)

4. In GitHub, create an Actions environment named exactly:

```text
npm
```

No npm token or secret is required.

5. In the npm settings for `@trisynapse/memory`, add a trusted publisher:

| Setting | Value |
|---|---|
| Provider | GitHub Actions |
| Owner | `shanmukh05` |
| Repository | `trisynapse-memory` |
| Workflow | `release.yml` |
| Environment | `npm` |
| Allowed action | `npm publish` |

Alternatively, using npm 11.15+:

```bash
npm trust github @trisynapse/memory \
  --file release.yml \
  --repo shanmukh05/trisynapse-memory \
  --env npm \
  --allow-publish
```

npm requires the package to exist before this trusted relationship can be created. [npm trusted-publishing documentation](https://docs.npmjs.com/trusted-publishers/)

## Every future release

After the one-time setup, the normal release process is enough:

```bash
uv run python scripts/version.py set 0.1.2

git add .
git commit -m "Prepare 0.1.2"
git push origin main

git tag -a v0.1.2 -m "Trisynapse Memory 0.1.2"
git push origin v0.1.2
```

The tag will automatically:

1. Run Python, TypeScript, and Studio tests.
2. Build `@trisynapse/memory`.
3. Publish it to npm using tokenless OIDC.
4. Install the exact npm version in a clean project and import it.
5. Publish the matching Python package to PyPI.
6. Create the GitHub release.
7. Test Python installers on Linux, macOS, and Windows.

Prereleases are handled correctly:

- Python/tag: `0.2.0rc1` / `v0.2.0rc1`
- npm: `0.2.0-rc.1`
- npm dist-tag: `next`
- Stable releases use npm dist-tag: `latest`