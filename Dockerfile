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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source code including `api/` folder
COPY . .

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Set working dir to where main.py is located
WORKDIR /app/api

# Set environment port and expose it
ENV PORT=10000
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:10000/health')"

# Run app with optimized settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "2"]