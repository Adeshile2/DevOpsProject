# Use Python base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model, encoder, and app files
COPY ml_model.pkl .
COPY encoder.pkl .
COPY app.py .

# Expose API port
EXPOSE 3000

# Run Flask app
CMD ["python", "app.py"]
