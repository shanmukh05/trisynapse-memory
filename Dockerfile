FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRISYNAPSE_MEMORY_PATH=/data

RUN addgroup --system trisynapse && adduser --system --ingroup trisynapse trisynapse
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir '.[all]' && mkdir -p /data && chown -R trisynapse:trisynapse /data

USER trisynapse
VOLUME ["/data"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/v1/health'))['status']=='ready'"
CMD ["trisynapse-memory", "--path", "/data", "serve", "--host", "0.0.0.0", "--port", "8765", "--studio"]
