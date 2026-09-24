FROM python:3.11-slim

# Set root working directory
WORKDIR /app

# Install system dependencies (helps with psycopg2, FAISS)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies first (its own layer, cached independently
# of source-code changes below).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source code, including the docuqueryai package under src/,
# pyproject.toml, README.md and LICENSE (both read by pyproject.toml).
# .dockerignore keeps .env, .git, caches, and the test suite out of this
# and every layer above it.
COPY . .

# Properly install the package itself (--no-deps: dependencies are already
# installed above; this just registers "docuqueryai" as an importable,
# installed package rather than relying on a PYTHONPATH/cwd convention).
RUN pip install --no-cache-dir --no-deps .

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Set environment port and expose it
ENV PORT=10000
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:10000/health')"

# Run app with optimized settings. docuqueryai is now a properly installed
# package (see above), not dependent on the container's working directory.
CMD ["uvicorn", "docuqueryai.api.main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "2"]