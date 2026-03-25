import os
import ml_metadata
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2
from utils import get_logger, MLMD_DB_PATH

logger = get_logger(__name__)

class PipelineTracker:
    def __init__(self, db_path=MLMD_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        connection_config = metadata_store_pb2.ConnectionConfig()
        connection_config.sqlite.filename_uri = self.db_path
        connection_config.sqlite.connection_mode = 3 # READWRITE_OPENCREATE
        
        logger.info(f"Initializing MLMD store at {self.db_path}")
        self.store = metadata_store.MetadataStore(connection_config)
        self._register_types()

    def _register_types(self):
        # Artifact Types
        self.dataset_type = self._put_artifact_type("CNCDataset", {"path": metadata_store_pb2.STRING, "n_samples": metadata_store_pb2.INT, "version": metadata_store_pb2.STRING})
        self.validated_dataset_type = self._put_artifact_type("ValidatedDataset", {"path": metadata_store_pb2.STRING, "n_samples": metadata_store_pb2.INT, "anomalies_found": metadata_store_pb2.INT})
        self.feature_set_type = self._put_artifact_type("FeatureSet", {"path": metadata_store_pb2.STRING, "n_features": metadata_store_pb2.INT, "n_samples": metadata_store_pb2.INT})
        self.model_type = self._put_artifact_type("TrainedModel", {"model_name": metadata_store_pb2.STRING, "accuracy": metadata_store_pb2.DOUBLE, "f1_weighted": metadata_store_pb2.DOUBLE, "framework": metadata_store_pb2.STRING})
        self.metrics_type = self._put_artifact_type("ModelMetrics", {"model_name": metadata_store_pb2.STRING, "report_path": metadata_store_pb2.STRING})
        self.tfdv_stats_type = self._put_artifact_type("TFDVStats", {"path": metadata_store_pb2.STRING, "n_features": metadata_store_pb2.INT})
        self.tfdv_schema_type = self._put_artifact_type("TFDVSchema", {"path": metadata_store_pb2.STRING})
        
        # Execution Types
        self.generation_exec_type = self._put_execution_type("DataGeneration", {"n_samples": metadata_store_pb2.INT, "seed": metadata_store_pb2.INT})
        self.validation_exec_type = self._put_execution_type("DataValidation", {"checks_passed": metadata_store_pb2.INT, "rows_dropped": metadata_store_pb2.INT, "tfdv_used": metadata_store_pb2.INT})
        self.feature_exec_type = self._put_execution_type("FeatureEngineering", {"n_features": metadata_store_pb2.INT})
        self.training_exec_type = self._put_execution_type("ModelTraining", {"model_type": metadata_store_pb2.STRING, "training_time_s": metadata_store_pb2.DOUBLE})
        self.evaluation_exec_type = self._put_execution_type("ModelEvaluation", {"champion": metadata_store_pb2.STRING})
        self.gcs_upload_exec_type = self._put_execution_type("GCSUpload", {"n_files": metadata_store_pb2.INT, "bucket": metadata_store_pb2.STRING})

        # Context Type
        self.run_context_type = self._put_context_type("PipelineRun", {"pipeline_version": metadata_store_pb2.STRING, "run_date": metadata_store_pb2.STRING})
        logger.info("MLMD types registered successfully.")

    def _put_artifact_type(self, type_name, properties):
        artifact_type = metadata_store_pb2.ArtifactType()
        artifact_type.name = type_name
        for key, val_type in properties.items():
            artifact_type.properties[key] = val_type
        return self.store.put_artifact_type(artifact_type)

    def _put_execution_type(self, type_name, properties):
        execution_type = metadata_store_pb2.ExecutionType()
        execution_type.name = type_name
        for key, val_type in properties.items():
            execution_type.properties[key] = val_type
        return self.store.put_execution_type(execution_type)

    def _put_context_type(self, type_name, properties):
        context_type = metadata_store_pb2.ContextType()
        context_type.name = type_name
        for key, val_type in properties.items():
            context_type.properties[key] = val_type
        return self.store.put_context_type(context_type)

    def create_pipeline_run(self, pipeline_version, run_date_str):
        context = metadata_store_pb2.Context()
        context.type_id = self.run_context_type
        context.name = f"run_{run_date_str}"
        context.custom_properties["pipeline_version"].string_value = pipeline_version
        context.custom_properties["run_date"].string_value = run_date_str
        context_id = self.store.put_contexts([context])[0]
        logger.info(f"Created PipelineRun context: {context_id}")
        return context_id

    def get_or_create_pipeline_run(self, run_id: str, pipeline_version="1.0"):
        from datetime import datetime
        contexts = self.store.get_contexts_by_type("PipelineRun")
        for c in contexts:
            if c.name == run_id:
                return c.id
        
        context = metadata_store_pb2.Context()
        context.type_id = self.run_context_type
        context.name = run_id
        context.custom_properties["pipeline_version"].string_value = pipeline_version
        context.custom_properties["run_date"].string_value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context_id = self.store.put_contexts([context])[0]
        logger.info(f"Created new PipelineRun context: {run_id} ({context_id})")
        return context_id

    def record_artifact(self, type_id, uri, properties, context_id=None):
        artifact = metadata_store_pb2.Artifact()
        artifact.type_id = type_id
        artifact.uri = uri
        for key, (val, val_type) in properties.items():
            if val_type == "int":
                artifact.custom_properties[key].int_value = int(val)
            elif val_type == "string":
                artifact.custom_properties[key].string_value = str(val)
            elif val_type == "double":
                artifact.custom_properties[key].double_value = float(val)
        
        artifact_id = self.store.put_artifacts([artifact])[0]
        
        if context_id is not None:
            attribution = metadata_store_pb2.Attribution()
            attribution.artifact_id = artifact_id
            attribution.context_id = context_id
            self.store.put_attributions_and_associations([attribution], [])
            
        return artifact_id

    def record_dataset(self, path, n_samples, version, context_id=None):
        logger.info(f"Recording CNCDataset artifact: {path}")
        return self.record_artifact(
            self.dataset_type, path,
            {"path": (path, "string"), "n_samples": (n_samples, "int"), "version": (version, "string")},
            context_id
        )

    def record_validation(self, path, n_samples, anomalies_found, context_id=None):
        logger.info(f"Recording ValidatedDataset artifact: {path}")
        return self.record_artifact(
            self.validated_dataset_type, path,
            {"path": (path, "string"), "n_samples": (n_samples, "int"), "anomalies_found": (anomalies_found, "int")},
            context_id
        )

    def record_model(self, model_name, accuracy, f1, framework, path, context_id=None):
        logger.info(f"Recording TrainedModel artifact: {model_name}")
        return self.record_artifact(
            self.model_type, path,
            {"model_name": (model_name, "string"), "accuracy": (accuracy, "double"), 
             "f1_weighted": (f1, "double"), "framework": (framework, "string")},
            context_id
        )

    def record_execution(self, exec_type, properties, input_ids, output_ids, context_id=None):
        execution = metadata_store_pb2.Execution()
        execution.type_id = exec_type
        
        for key, (val, val_type) in properties.items():
            if val_type == "int":
                execution.custom_properties[key].int_value = int(val)
            elif val_type == "string":
                execution.custom_properties[key].string_value = str(val)
            elif val_type == "double":
                execution.custom_properties[key].double_value = float(val)
                
        execution_id = self.store.put_executions([execution])[0]
        
        events = []
        for a_id in input_ids:
            event = metadata_store_pb2.Event()
            event.artifact_id = a_id
            event.execution_id = execution_id
            event.type = metadata_store_pb2.Event.INPUT
            events.append(event)
            
        for a_id in output_ids:
            event = metadata_store_pb2.Event()
            event.artifact_id = a_id
            event.execution_id = execution_id
            event.type = metadata_store_pb2.Event.OUTPUT
            events.append(event)
            
        if events:
            self.store.put_events(events)
            
        if context_id is not None:
            association = metadata_store_pb2.Association()
            association.execution_id = execution_id
            association.context_id = context_id
            self.store.put_attributions_and_associations([], [association])
            
        logger.info(f"Recorded Execution {execution_id} with {len(input_ids)} inputs and {len(output_ids)} outputs.")
        return execution_id

    def get_all_pipeline_runs(self):
        contexts = self.store.get_contexts_by_type("PipelineRun")
        runs = []
        for c in contexts:
            runs.append({
                "id": c.id,
                "name": c.name,
                "version": c.custom_properties["pipeline_version"].string_value,
                "date": c.custom_properties["run_date"].string_value
            })
        return runs

    def get_model_lineage(self, model_artifact_id):
        # Simplify: go backwards from output model to its input training features, 
        # then find its execution, then input validated dataset, etc.
        lineage = {}
        events = self.store.get_events_by_artifact_ids([model_artifact_id])
        output_event = [e for e in events if e.type == metadata_store_pb2.Event.OUTPUT]
        if not output_event:
            return lineage
            
        exec_id = output_event[0].execution_id
        input_events = [e for e in self.store.get_events_by_execution_ids([exec_id]) if e.type == metadata_store_pb2.Event.INPUT]
        
        lineage["training_execution"] = exec_id
        lineage["inputs"] = [e.artifact_id for e in input_events]
        return lineage

    def get_full_lineage_graph(self):
        # A simple export of nodes and edges
        nodes = []
        edges = []
        
        try:
            artifacts = self.store.get_artifacts()
            executions = self.store.get_executions()
            events = self.store.get_events_by_artifact_ids([a.id for a in artifacts])
            
            # Helper to ge type name
            art_types = {t.id: t.name for t in self.store.get_artifact_types()}
            exec_types = {t.id: t.name for t in self.store.get_execution_types()}
            
            for a in artifacts:
                nodes.append({
                    "id": f"a_{a.id}",
                    "type": "artifact",
                    "subtype": art_types.get(a.type_id, "Unknown"),
                    "uri": a.uri
                })
                
            for e in executions:
                nodes.append({
                    "id": f"e_{e.id}",
                    "type": "execution",
                    "subtype": exec_types.get(e.type_id, "Unknown")
                })
                
            for ev in events:
                if ev.type == metadata_store_pb2.Event.INPUT: # Execution consumed Artifact
                    edges.append({"source": f"a_{ev.artifact_id}", "target": f"e_{ev.execution_id}", "type": "INPUT"})
                elif ev.type == metadata_store_pb2.Event.OUTPUT: # Execution produced Artifact
                    edges.append({"source": f"e_{ev.execution_id}", "target": f"a_{ev.artifact_id}", "type": "OUTPUT"})
                    
        except Exception as e:
            logger.error(f"Error building lineage graph: {e}")
            
        return {"nodes": nodes, "edges": edges}
