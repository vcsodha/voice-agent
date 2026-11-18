FROM python:3.11-slim

# Where the models live is pinned here, not left to whatever the host decides.
# HF_HOME is where the speech model (faster-whisper) is kept, and HOME is where
# chroma keeps its small embedding model. The start command at the bottom sets
# HOME again, because some hosts start the container with a different one.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/app/home \
    HF_HOME=/app/hf-cache \
    PORT=8000 \
    CHROMA_PATH=/tmp/chroma \
    ANONYMIZED_TELEMETRY=False \
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

# build check: the speech model must be found even when HOME is something else
# (this is what broke the first Vercel deploy). If it is not, the build stops
# here instead of the site failing later.
RUN HOME=/nonexistent python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8'); print('speech model found')"

COPY backend /app/backend
COPY frontend /app/frontend
COPY data /app/data

WORKDIR /app/backend
EXPOSE 8000

# the host can choose the port by setting PORT (docker compose uses 8000);
# HOME is set again here so chroma finds its model whoever runs the container
CMD ["sh", "-c", "export HOME=/app/home; exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
