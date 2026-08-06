import logging
import chromadb
from pathlib import Path

logger = logging.getLogger(__name__)

class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.db = chromadb.PersistentClient(path= config.paths.vector_db_dir)

    def get_or_init_index(self):
        collection_names = self.config.VectorDB.collection_name
        existing_collections = [c.name for c in self.db.list_collections()]

        force_reindex = getattr(self.config.VectorDB, "force_reindex", False)
        
        if collection_names in existing_collections and force_reindex:
            logger.info(f"force_reindex is True. Deleting existing collection {collection_names} ...")
            self.db.delete_collection(collection_names)
            existing_collections.remove(collection_names)

        if collection_names in existing_collections:
            logger.info(f"Collection '{collection_names}' exists. Loading index...")
        else:
            logger.info(f"Collection '{collection_names}' not found. Starting light ingestion from JSONs...")
            self._build_from_json()
    
    def _build_from_json(self):
        from src.ingestion.registry import IngestionProcess
        ingestion_ = IngestionProcess(self.config)
        ingestion_.run_data_ingestion_pipeline(self.config.paths.raw_data_dir)
        logger.info("Read JSON, Chunk, and embed into vectorDB...")
        # return self._load_index()

    # def _load_index(self):
    #     from llama_index.core import Document, VectorStoreIndex, StorageContext
    #     from llama_index.vector_stores.chroma import ChromaVectorStore
    #     chroma_collection = self.db.get_collection(self.config.VectorDB.collection_name)
    #     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    #     return VectorStoreIndex.from_vector_store(vector_store)