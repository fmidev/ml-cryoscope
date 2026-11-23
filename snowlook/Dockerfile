# SnowLook, CryoSCOPE SWE model dashboard (Streamlit), image and container name snowlook
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY snowlook.py *.png ./
# results and optional logos in the mounted /data/cryoscope, Optuna db in /data/ML; SNOWLOOK_PUBLIC=1 hides the data paths box
ENV SNOWMODEL_DIR=/data/cryoscope/snowmodel/ OPTUNA_DB=/data/ML/MLexperiments.sqlite3 LOGO_DIR=/data/cryoscope SNOWLOOK_PUBLIC=1
# same uid as ubuntu on the host, so the parquet cache (validationresults_*/app_cache/) is not owned by root
RUN useradd -m -u 1000 snowlook
USER snowlook
EXPOSE 8503
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8503/_stcore/health',timeout=4)"
CMD ["streamlit","run","snowlook.py","--server.port=8503","--server.address=0.0.0.0","--server.headless=true","--server.fileWatcherType=none","--browser.gatherUsageStats=false","--client.toolbarMode=minimal","--client.showErrorDetails=false"]