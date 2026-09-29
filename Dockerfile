FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENV STREAMLIT_TELEMETRY_ENABLED=false \
    STREAMLIT_SERVER_HEADLESS=true

CMD ["bash", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
