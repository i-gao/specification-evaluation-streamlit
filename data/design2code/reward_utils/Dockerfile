# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /sandbox

# Copy requirements files first for better caching
COPY reward_utils/requirements-minimal.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Install Playwright and browsers
RUN playwright install-deps
RUN playwright install

# Copy the entire project
COPY reward_utils/ .
COPY assets/testset_final ./testset_final
COPY assets/testset_final/rick.jpg ./rick.jpg

# Set default command
CMD ["bash"]
