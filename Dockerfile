FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements_db.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_db.txt
RUN pip install --no-cache-dir mysqlclient

# Copy all source code
COPY . .

# Create output directory
RUN mkdir -p /app/output_train

# Expose API port
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "run_daily_pipeline.py"]