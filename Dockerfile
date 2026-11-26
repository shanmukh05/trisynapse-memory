FROM node:22-bookworm-slim AS studio

WORKDIR /app
COPY pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages/js-sdk/package.json packages/js-sdk/
COPY packages/studio/package.json packages/studio/
RUN npm install --global pnpm@10.24.0 \
    && pnpm install --filter @trisynapse/studio --frozen-lockfile
COPY packages/studio/ packages/studio/
COPY public/assets/logo.png public/assets/
RUN pnpm --filter @trisynapse/studio build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/.cache/huggingface \
    TRISYNAPSE_MEMORY_PATH=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system trisynapse \
    && adduser --system --ingroup trisynapse trisynapse
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY --from=studio /app/src/trisynapse_memory/studio/dist/ src/trisynapse_memory/studio/dist/
RUN pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.13.0+cpu' \
    && pip install --no-cache-dir '.[all]' \
    && mkdir -p /data \
    && chown -R trisynapse:trisynapse /data

USER trisynapse
VOLUME ["/data"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/v1/health'))['status']=='ready'"
CMD ["trisynapse-memory", "--path", "/data", "serve", "--host", "0.0.0.0", "--port", "8765", "--studio"]
