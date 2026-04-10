#!/bin/bash
set -e
PROJECT="ajithmlopsie7374"
DATASET="cnc_warehouse"
echo "=== BigQuery Warehouse Setup ==="
bq --project_id=${PROJECT} mk --dataset --location=US --description="CNC Manufacturing Data Warehouse (Lab 6)" ${PROJECT}:${DATASET} 2>/dev/null || echo "Dataset already exists"
echo "Dataset ${DATASET} ready. Tables created by warehouse_loader.py"
