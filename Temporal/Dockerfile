FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Use Pipeline as the working directory (main.py lives in Pipeline/)
WORKDIR /app/Temporal/Pipeline

# Copy and install only Pipeline requirements to speed up builds
COPY Pipeline/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the repository into /app so imports like `from Temporal.Pipeline...` work
COPY . /app/Temporal

# Make sure Python can import the top-level `Temporal` package
ENV PYTHONPATH=/app

CMD ["python", "main.py"]
