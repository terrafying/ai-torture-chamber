FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=false BROWSER_USE_CLOUD_SYNC=false \
    OBSERVATORY_DB=/data/state.sqlite3 PLAYWRIGHT_BROWSERS_PATH=/opt/browsers
WORKDIR /app
COPY observatory/requirements*.txt ./observatory/
RUN pip install -r observatory/requirements-research.txt \
    && python -m playwright install --with-deps chromium \
    && useradd --create-home --uid 10001 researcher \
    && mkdir /data && chown researcher:researcher /data
COPY observatory ./observatory
COPY site ./site
COPY live/server.py live/press.json /app/live/
USER researcher
EXPOSE 8060
CMD ["python", "-m", "uvicorn", "observatory.app:app", "--host", "0.0.0.0", "--port", "8060", "--workers", "1"]
