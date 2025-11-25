FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY train.py .
COPY inference.py .
COPY view_predict_slice.py .
COPY models.py .
COPY preprocess.py .
COPY view_hdr.py .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/models /app/outputs /app/artifacts

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command
CMD ["python", "--version"]