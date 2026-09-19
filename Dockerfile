# Continual-Counsel Edge Inference Service Dockerfile
# Physical Separation: Inference ONLY. No training libraries or egress capabilities.
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only online serving, audit, and configs
COPY configs/base.yaml /app/configs/base.yaml
COPY configs/training.yaml /app/configs/training.yaml
COPY src/online/ /app/src/online/
COPY src/audit/ /app/src/audit/
COPY src/__init__.py /app/src/__init__.py

# Copy pre-bundled local artifacts (FAISS index, quantized GGUF weights, registry DB)
COPY artifacts/ /app/artifacts/

# Security & Compliance: Run as non-root user, no network egress permissions
RUN useradd -u 8888 edgeuser && chown -R edgeuser:edgeuser /app
USER edgeuser

ENV PYTHONUNBUFFERED=1
ENV CONTINUAL_COUNSEL_OFFLINE_MODE=true

EXPOSE 8000

# Health check and entrypoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; exit(0)" || exit 1

CMD ["uvicorn", "src.online.api:app", "--host", "0.0.0.0", "--port", "8000"]
