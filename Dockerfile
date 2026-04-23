FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY data/ ./data/

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
