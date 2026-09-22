# Sandbox runtime for agent-written code (CPU). Build: docker build -t exp-rt:cpu env/
# requirements.lock is the resolved set from the first build (pip freeze inside the image).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 procps && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
