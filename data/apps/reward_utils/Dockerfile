FROM python:3.9.20-bullseye

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir pyext>=0.7 numpy psutil==5.7.0
RUN pip install --no-cache-dir datasets

# Set working directory
WORKDIR /sandbox

# Copy the necessary files
COPY reward_utils/test_one_solution.py /sandbox/test_one_solution.py
COPY reward_utils/testing_util.py /sandbox/testing_util.py
COPY reward_utils/check_correctness.py /sandbox/check_correctness.py

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1 

# Ensure proper SSL certificates are available
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
