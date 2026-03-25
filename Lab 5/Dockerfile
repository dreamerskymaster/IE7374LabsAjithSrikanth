FROM apache/airflow:2.7.3-python3.10

USER root
# Install build essentials for heavy packages like TFDV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/requirements.txt
# To avoid large image sizes or pip conflict with TFDV, we might install it with no-deps if necessary, 
# but per instructions standard install is fine.
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN mkdir -p /opt/airflow/data /opt/airflow/mlflow-artifacts/artifacts /opt/airflow/mlmd
