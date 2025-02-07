# Use Python base image
FROM python:latest

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Expose Flask API port
EXPOSE 3000

# Run Flask app
CMD ["python", "app.py"]
