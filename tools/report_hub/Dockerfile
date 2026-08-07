# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim

RUN groupadd --system report-hub \
    && useradd --system --gid report-hub --home-dir /app report-hub \
    && mkdir -p /app/tools /var/lib/report-hub \
    && chown -R report-hub:report-hub /app /var/lib/report-hub

WORKDIR /app
COPY --chown=report-hub:report-hub tools/__init__.py /app/tools/__init__.py
COPY --chown=report-hub:report-hub tools/report_hub /app/tools/report_hub

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REPORT_HUB_DATABASE=/var/lib/report-hub/report-hub.sqlite3 \
    REPORT_HUB_HOST=0.0.0.0 \
    REPORT_HUB_PORT=4180

USER report-hub
EXPOSE 4180
VOLUME ["/var/lib/report-hub"]
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4180/api/v1/health', timeout=3)"]

CMD ["python", "-m", "tools.report_hub"]
