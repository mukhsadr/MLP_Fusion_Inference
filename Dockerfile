# Use slim Python base
FROM python:3.10-slim

# Update system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget && \
    rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /code

# Copy requirements
COPY requirements.txt /code/

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /code

# Default command (runs main.py)
CMD ["python", "/code/main.py"]
