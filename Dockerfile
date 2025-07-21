FROM python:3.11-slim

WORKDIR /app

# Install only minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy only what's necessary
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r app_requirements.txt

COPY app/ app/
COPY helpers/FeatureExtractor.py helpers/FeatureExtractor.py
COPY models/combined_model_scripted.pth models/combined_model_scripted.pth

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]