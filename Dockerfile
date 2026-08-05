# Scientific Reproducibility Container for Mem4ristor
# (l'en-tete annoncait « v2.9.3 » jusqu'au 2026-08-05 ; la version fait foi dans
#  pyproject.toml et VERSION, pas dans un commentaire.)
# Based on slim Python image for minimal footprint

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set python path to include src
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Default command: Run the Resilience Benchmark
CMD ["python", "experiments/attack_resilience.py"]
