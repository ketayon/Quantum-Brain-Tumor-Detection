# Use Python 3.12 slim image for lightweight build
FROM python:3.12-slim

# Prevents Python from writing .pyc files
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt

# Ensure correct package versions are installed
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt

# Ensure all scripts use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Create necessary directories for dataset and models
RUN mkdir -p /app/datasets /app/models

# Copy dataset and models if available
COPY datasets/ /app/datasets/
COPY models/ /app/models/

# Copy entire project into container
COPY . .

# Set the Python path to the app directory
ENV PYTHONPATH="/app"

# Expose the web application port
EXPOSE 5000

# Start Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "interfaces.web_app.app:app", "--timeout", "300", "--workers", "2"]
