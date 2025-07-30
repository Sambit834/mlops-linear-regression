# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /application

# Copy requirements first for better caching
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./source/
COPY src/models/ ./artifacts/

# Set Python path
ENV PYTHONPATH=/application/source

# Default command to run predict.py
CMD ["python", "source/predict.py"]
