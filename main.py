import hydra                    # for configuration management
from omegaconf import OmegaConf
from src.schema import AppConfig    # Pydantic for stric data validation and can catch multiple error at once
from dotenv import load_dotenv      # to load environment variable
import mlflow
from src.utils.logging import setup_primary_logging
from datetime import datetime
from eval.eval_run import *
import asyncio
import pandas as pd

import logging
logger = logging.getLogger(__name__)          # logger for this file

def get_validated_config(cfg) -> AppConfig:
    OmegaConf.resolve(cfg)              # (Turn ${} into real values)
    OmegaConf.set_readonly(cfg, True)   # Freeze the variable to prevent it modify in code
    # raw_config = OmegaConf.to_container(cfg, resolve=True)  # if resolve=False; resulting dictionary contain the string "${oc.env:TOKEN}", it won't actually look up the value.
    raw_dict = {
        "paths": cfg.paths,
        "api":{
            "hugging_face_access_token": cfg.api.hugging_face_access_token
        },
        "models":{
            "embedding_model_id": cfg.embedding_provider + '/' + cfg.embedding_model,
            "llm_model_id": cfg.llm_provider + '/' + cfg.llm_model
        },
        "mlflow_config":{
            "tracking_uri": cfg.mlflowConfig.tacking_uri,
            "experiment_name": cfg.mlflowConfig.experiment_name,
            "run_name" : cfg.mlflowConfig.run_name,
            "enabled": cfg.mlflowConfig.enabled
        },
        "data_pipeline": cfg.ingestion,
        "VectorDB": cfg.VectorDB,
        "prompt_detail": cfg.selected_prompt_detail
    }
    return AppConfig(**raw_dict)

def mlflow_setup(config):
    mlflow.llama_index.autolog()
    mlflow.set_tracking_uri(config.mlflow_config.tracking_uri)
    mlflow.set_experiment(config.mlflow_config.experiment_name)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg) -> None:
    log_file_path = 'logs/' + str(datetime.now().strftime("%Y-%m-%d")) + 'debug_full.log'
    log_listener = setup_primary_logging(log_file_path)       # central log writer
    
    logger.info("Welcome to Building Code Reviewer!")
    config = get_validated_config(cfg)
    path_dict = config.paths.as_strs()
    logger.debug("Config file has beed read and validated: %s", config)

    # from src.ingestion.registry import IngestionProcess
    # ingestion_process_object = IngestionProcess(config)
    # ingestion_process_object.run_data_ingestion_pipeline(path_dict['raw_data_dir'])
   
    # mlflow setup
    # mlflow server --host 127.0.0.1 --port 5000
    # mlflow_setup(config)

    # with mlflow.start_run(run_name = cfg.VectorDB.collection_name):
    #     mlflow.log_params(config.model_dump(mode='json'))
    #     # df_results = run_experiment_evaluation(config)
    #     df_results = run_experiment_evaluation_async_try_2(config)

    #     mlflow.log_metrics({
    #         "mean_hit_rate": df_results['retrieval_hit'].mean(),
    #         "mean_mrr": df_results['mrr'].mean(),
    #         "mean_faithfulness": df_results['faithfulness'].mean(),
    #         "mean_relevancy": df_results['answer_relevancy'].mean(),
    #         "mean_context_precision": df_results.get('context_precision', pd.Series([0])).mean(),
    #         "mean_context_recall": df_results['context_recall'].mean()
    #     })
        
    #     mlflow.log_table(data=df_results, artifact_file="evaluation_details.json")
    #     df_results.to_csv("eval_results.csv", index=False)
    #     mlflow.log_artifact("eval_results.csv")

    #     logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    # Log metrics
    # latency, retrieval_latency,  faithfulness, ... check databricks    

if __name__ == "__main__":
    load_dotenv()           # load variables into os.environ first; so that whatever variables are inside .env can be accessible by hydra
    try:
        main()
    except Exception as e:
        print(f"Configuartion Error: {e}")
        logger.exception(f"Error in the main function; log from the main.py")