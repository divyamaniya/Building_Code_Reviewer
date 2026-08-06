import os
import requests
from pathlib import Path
from pydantic import BaseModel, SecretStr, Field, field_validator, model_validator
from src.utils.downloader import download_hf_model

class PathChecker(BaseModel):
    # Paths Fields
    raw_data_dir: Path
    processed_data_dir: Path
    vector_db_dir: Path
    embedding_model_dir: Path
    llm_model_dir: Path
    # reranker_dir: Path
    intermediate_data: Path
    llm_judge: Path
    embedding_judge: Path
    rerank_model_dir: Path

    @field_validator("*", mode="before") # The "*" takes EVERY field
    @classmethod
    def ensure_dir_exists(cls, v: str) -> Path:
        """Checks if directory exists, create if it doesn't"""

        path = Path(v).resolve()            # resolve(); if the path is relative make it absolute and if not exist return false
        print("v: ",v)
        if not path.exists():
            print(f"Creating missing directory: {path}")
            path.mkdir(parents=True, exist_ok=True)         # parents=True (if parent folder is not exist, it will create those, otherwise throw an error), exist_ok=True (if you try to create a folder that already exists, Python throw an error and stop program, True will ignore the command and keep going)
        return path

    def as_strs(self):
        # returns all paths as strings
        return {k: str(v) for k, v in self.model_dump().items()}


class APIConfig(BaseModel):
    hugging_face_access_token: SecretStr

    # @model_validator is designed to inspect or mutate the entire object; it receives fully formed object(self) and you must return object so pydantic can hand it back to rest of the program. if you don't your config object will become None.
    @model_validator(mode="after")          # this runs after pydandic has validated types
    def validate_api_access(self) -> "AppConfig":
        """Checks if Hugging face token is valid before starting"""
        token = self.hugging_face_access_token.get_secret_value()
        headers = {"Authorization": f"Bearer {token}"}
        print("validating hf API...")
        # trying simple ping to Hugging face API to verify token
        try:
            response = requests.get(
                "https://huggingface.co/api/whoami-v2",
                headers=headers,
                timeout=5
            )
            if response.status_code != 200:
                raise ValueError(f"Invalid HF token (Status: {response.status_code})")
        except requests.exceptions.RequestException:
            print("Warning: Could not verify HF token (Connection Error)")

        return self
    


class ModelConfig(BaseModel):

    # @model_validator(mode="after")
    # def download_model_if_not_exist(self) -> "AppConfig":
    # we call this function using self. method inside AppConfig, so python automatically passes ModelConfig instance as the first argument, if function defination does not have that then python gets confused and thinks you are passing too many things.
    def setup_models(self, paths: "PathChecker", token: SecretStr):
        """If model is not exist in local then it will download from internet"""
        download_hf_model(
            local_dir=paths.embedding_model_dir, 
            token=token.get_secret_value()
        )
        download_hf_model(
            local_dir=paths.llm_model_dir, 
            token=token.get_secret_value()
        )
        download_hf_model(
            local_dir=paths.rerank_model_dir, 
            token=token.get_secret_value()
        )
        # need to add reranker here



class MLflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "bulding_code_rag"
    run_name: str
    enabled: bool = True



class DataPipeline(BaseModel):
    parser_method: str
    loader_method: str
    chunk_strategy: dict

class VectorDBConfig(BaseModel):
    name: str
    collection_name: str
    persist_directory: str
    initial_top_k : int
    use_reranker: bool
    reranker_type: str
    final_top_k: int
    force_reindex: bool = False


class PromptConfig(BaseModel):
    date: str
    author: str
    description: str
    text: str


class AppConfig(BaseModel):

    paths: PathChecker
    api: APIConfig
    models: ModelConfig
    mlflow_config: MLflowConfig
    data_pipeline: DataPipeline
    VectorDB: VectorDBConfig
    prompt_detail: PromptConfig

    @model_validator(mode="after")
    def run_setup(self) -> "AppConfig":
        """Once pydentic automatically validate path and API we can now setup the models"""
        self.models.setup_models(
            paths = self.paths,
            token = self.api.hugging_face_access_token
        )
        return self

    



# Runtime checks on in/out messages with user