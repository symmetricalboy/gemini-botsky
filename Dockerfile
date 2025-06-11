# Use Python 3.11 slim image for smaller footprint
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY bot.py .
COPY .env.example .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash botuser && \
    chown -R botuser:botuser /app
USER botuser

# Set environment variable for Python
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import psutil; exit(0 if psutil.Process().is_running() else 1)"

# Run the bot
CMD ["python", "bot.py"] 