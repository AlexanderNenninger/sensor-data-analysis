# Dockerfile for sensor-data-analysis with Ray
FROM python:3.13-slim

WORKDIR /app

COPY . /app

# Install uv and dependencies
RUN pip install --upgrade pip && \
    pip install uv && \
    uv pip install .

# Expose Ray dashboard port
EXPOSE 8265

CMD ["python", "dependency_inversion.py"]
