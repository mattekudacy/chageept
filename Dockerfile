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

# Pre-build the knowledge base at image build time. Embedding now goes through
# Gemini's hosted API, so GEMINI_API_KEY must be available during the build -
# set it as a build-time env var on the host (e.g. Render dashboard). This
# bakes a ready chroma_db/ into the image so cold starts on free-tier hosts
# (no persistent disk) don't have to re-crawl the CHAGEE site before serving
# the first request.
ARG GEMINI_API_KEY
ENV GEMINI_API_KEY=$GEMINI_API_KEY
RUN python -m scripts.seed_crawler

# Expose port
EXPOSE 8000

# Run via start script (properly handles PORT variable)
CMD ["./start.sh"]
