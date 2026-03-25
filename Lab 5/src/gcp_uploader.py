import os
import json
from google.cloud import storage as gcs
from utils import get_logger, log_stage, GCS_BUCKET, RAW_DATA_PATH, VALIDATED_DATA_PATH, FEATURES_PATH, RESULTS_DIR, MODELS_DIR, TFDV_STATS_HTML

logger = get_logger(__name__)

@log_stage("GCS Upload")
def upload_pipeline_artifacts(tracker=None, context_id=None):
    if not GCS_BUCKET:
        logger.warning("GCS_BUCKET is not set. Skipping upload.")
        return

    logger.info(f"Uploading artifacts to gs://{GCS_BUCKET}/lab5/run_latest/")
    try:
        client = gcs.Client()
        bucket = client.bucket(GCS_BUCKET)
        
        files_to_upload = [
            RAW_DATA_PATH,
            VALIDATED_DATA_PATH,
            FEATURES_PATH,
            os.path.join(RESULTS_DIR, "model_comparison_summary.json")
        ]
        
        # Add all *_result.json
        if os.path.exists(RESULTS_DIR):
            for f in os.listdir(RESULTS_DIR):
                if f.endswith("_result.json") or f.endswith("_confusion_matrix.png") or f.endswith("_feature_importance.png"):
                    files_to_upload.append(os.path.join(RESULTS_DIR, f))
                    
        # Add TFDV stats
        if os.path.exists(TFDV_STATS_HTML):
            files_to_upload.append(TFDV_STATS_HTML)
            
        # Add Champion Model
        champion_path = os.path.join(MODELS_DIR, "champion_model.joblib")
        if os.path.exists(champion_path):
            files_to_upload.append(champion_path)

        uploaded_count = 0
        for file_path in files_to_upload:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                blob_path = f"lab5/run_latest/{file_name}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(file_path)
                size_kb = os.path.getsize(file_path) / 1024
                logger.info(f"Uploaded {file_name} ({size_kb:.1f} KB) to gs://{GCS_BUCKET}/{blob_path}")
                uploaded_count += 1
                
        logger.info(f"Successfully uploaded {uploaded_count} files to GCS.")

        if tracker and context_id is not None:
            tracker.record_execution(
                exec_type=tracker.gcs_upload_exec_type,
                properties={"n_files": (uploaded_count, "int"), "bucket": (GCS_BUCKET, "string")},
                input_ids=[],
                output_ids=[],
                context_id=context_id
            )
            
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}")
