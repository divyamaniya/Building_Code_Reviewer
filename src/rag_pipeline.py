from llama_index.core import StorageContext
from src.retrieval.retriever import RetrievalPipeline, HybridRetrievalPipeline
from src.generation.generation import GenerationPipeline
from src.utils.loader import LocalModelLoader, vLLMLocalModelLoader
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import logging
from llama_index.core import Settings
from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)

def init_settings(config):
    vLLMLocalModelLoader().load_llm(config.paths.llm_model_dir)
    # vLLMLocalModelLoader().load_embedding(config.paths.embedding_model_dir)
    vLLMLocalModelLoader().local_TEI(
        model_path=config.paths.embedding_model_dir, 
        tei_url="http://localhost:8095"
    )

    # Settings.llm = llm
    # Settings.embed_model = embed_model

    # 1. Connect to the existing ChromaDB on disk
    db = chromadb.PersistentClient(path=config.VectorDB.persist_directory)
    # db = chromadb.HttpClient(host="localhost", port=8045)
    chroma_collection = db.get_or_create_collection(config.VectorDB.collection_name)

    # 3. FIX: Fetch existing data from Chroma to populate the in-memory docstore
    logger.info("Extracting raw text nodes from ChromaDB...")
    existing_data = chroma_collection.get(include=["documents", "metadatas"])
    
    nodes = []
    if existing_data and existing_data.get("ids"):
        for idx in range(len(existing_data["ids"])):
            node = TextNode(
                id_=existing_data["ids"][idx],
                text=existing_data["documents"][idx],
                metadata=existing_data["metadatas"][idx] if existing_data["metadatas"] else {},
            )
            nodes.append(node)
        logger.info(f"Extracted {len(nodes)} nodes for hybrid search processing.")
    else:
        logger.error("ChromaDB returned no data! Make sure your data ingestion ran successfully.")

    # 2. Create the Vector Store object
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 3. Build the index from the existing vector store
    # This avoids looking for docstore.json and pulls directly from Chroma
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
 
    # Fallback to empty initialization if no data exists yet
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context
    )

    return index, nodes


def run_rag_pipeline(query: str, config):
    index, nodes = init_settings(config)

    logger.debug(f"Index: {index}")
    retrieval_module = RetrievalPipeline(index, config)
    # retrieval_module = HybridRetrievalPipeline(index, nodes, config)
    

    generative_module = GenerationPipeline(retrieval_module, config)

    # response = generative_module.answer(query)
    # return response
    return retrieval_module, generative_module