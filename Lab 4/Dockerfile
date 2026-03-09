FROM apache/airflow:2.7.3

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY --chown=airflow:0 requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

RUN mkdir -p /opt/airflow/data /opt/airflow/mlflow-artifacts/artifacts

WORKDIR /opt/airflow
