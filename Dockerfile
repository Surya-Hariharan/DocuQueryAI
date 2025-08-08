FROM python:3.11-slim

# Set root working directory
WORKDIR /app

# Install system dependencies (helps with psycopg2)
RUN apt-get update && apt-get install -y gcc libpq-dev

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code including `api/` folder
COPY . .

# Set working dir to where main.py is located
WORKDIR /app/api

# Set environment port and expose it
ENV PORT=10000
EXPOSE 10000

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]