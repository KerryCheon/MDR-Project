
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Work from /app — this file is intended to be built from the Temporal/ directory with:
# docker build -f Pipeline/Dockerfile -t temporal-pipeline .
WORKDIR /app

# Install only the Pipeline requirements initially for caching
COPY Pipeline/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the whole Temporal directory into /app/Temporal so absolute imports work
COPY . /app/Temporal

# Set PYTHONPATH so imports like `Temporal.Pipeline` work
ENV PYTHONPATH=/app

# Start in the pipeline folder
WORKDIR /app/Temporal/Pipeline

CMD ["python", "main.py"]
