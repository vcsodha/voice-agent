FROM python:3.11-slim

# HOME is inside /app so the models we download below are found again at
# runtime, whichever user the host runs the container as
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/app/home \
    PORT=8000 \
    CHROMA_PATH=/tmp/chroma \
    LOG_TRANSCRIPTS=0

WORKDIR /app
RUN mkdir -p /app/home

# install python packages first so docker can cache this step
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# download every model into the image, so starting the container never has
# to download anything (this is what keeps cold starts short on a host that
# stops the app when nobody is using it)
RUN python -m piper.download_voices --download-dir /app/models en_US-lessac-medium
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up'])"
RUN chmod -R a+rX /app
ENV HF_HUB_OFFLINE=1

COPY backend /app/backend
COPY frontend /app/frontend
COPY data /app/data

WORKDIR /app/backend
EXPOSE 8000

# the host can choose the port by setting PORT (docker compose uses 8000)
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
