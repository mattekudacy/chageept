# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# The knowledge base now lives in Qdrant Cloud, not baked into the image, so
# no build-time crawl is needed here. chat_ui.py runs the initial crawl on
# first startup (only when the Qdrant collection is empty) and reuses it on
# every subsequent deploy since the data persists externally.

# Expose port
EXPOSE 8000

# Run via start script (properly handles PORT variable)
CMD ["./start.sh"]
