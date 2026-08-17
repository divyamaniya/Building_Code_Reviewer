# uvicorn app:app --reload --port 8000
# npm run dev

import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import List, Optional

from src.utils.config import load_config
from src.ingestion.registry import IngestionRegistry
from src.retrieval.retriever import RetrievalPipeline, HybridRetrievalPipeline
from src.generation.generation import GenerationPipeline
from llama_index.core import Settings
from src.utils.loader import vLLMLocalModelLoader
from fastapi.staticfiles import StaticFiles
from src.guardrails.guardrail import GuardrailPipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced RAG PDF Chat & Citation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount your local split PDF directory
app.mount(
    "/api/documents",
    StaticFiles(directory="data/temp_data/splitted"), 
    name="documents"
)

# Global Application State Holders
config = None
vector_store_manager = None
generation_pipeline = None
guardrail_pipeline = None

class QueryRequest(BaseModel):
    query: str
    use_hybrid: Optional[bool] = False

@app.on_event("startup")
def startup_event():
    global config, generation_pipeline, guardrail_pipeline
    config = load_config("config/config.yaml")
    guardrail_pipeline = GuardrailPipeline(config)
    
    model_loader = vLLMLocalModelLoader()
    
    embed_model_path = Path(config.paths.embedding_model_dir)
    llm_model_path = Path(config.paths.llm_model_dir)
    
    # This injects them into LlamaIndex's global Settings object
    model_loader.load_embedding(embed_model_path)
    model_loader.load_llm(llm_model_path)
    logger.info("Local LLM and Embedding models successfully bound to global Settings.")

    # 2. Proceed with initializing your vector store and pipelines safely
    vector_store_manager = IngestionRegistry.get_vector_store(
        name=config.VectorDB.name,
        collection_name=config.VectorDB.collection_name,
        persist_dir=config.VectorDB.persist_directory,
        embed_model_path=embed_model_path
    )
    
    retriever_module = RetrievalPipeline(vector_store_manager.index, config)
    generation_pipeline = GenerationPipeline(retriever_module, config)
    logger.info("Application startup sequence completed completely local.")

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    global generation_pipeline, guardrail_pipeline
    if not generation_pipeline:
        raise HTTPException(status_code=500, detail="Generation pipeline not initialized.")
    if not guardrail_pipeline:
        raise HTTPException(status_code=500, detail="Guardrail pipeline not initialized.")
    
    try:
        # 1. Check query intent using the guardrail pipeline
        intent = guardrail_pipeline.classify_query_intent(request.query)
        
        # Unified handling for greetings, casual chat, and off-topic queries
        # Reuse the generation pipeline's handler for general/off-topic inputs
        if intent == "GENERAL_OR_OFF_TOPIC":
            answer_text = generation_pipeline.handle_conversation(request.query)
            return {
                "answer": answer_text,
                "citations": []
            }

        # Retrieve context nodes explicitly to construct structured citations
        retriever_mod = generation_pipeline.query_engine.retriever
        nodes_with_scores = retriever_mod.retrieve(request.query)
        
        citations = []
        for item in nodes_with_scores:
            node = item.node
            meta = node.metadata

            citations.append({
                "source_doc": meta.get("source_doc", "unknown.pdf"),
                "page_number": meta.get("page_number"),
                "has_precise_highlight": meta.get("has_precise_highlight", False),
                "coord_x1": meta.get("coord_x1"),
                "coord_y1": meta.get("coord_y1"),
                "coord_x2": meta.get("coord_x2"),
                "coord_y2": meta.get("coord_y2"),
                "snippet_text": node.get_content()[:300]
            })

        # Execute generation response
        response_obj = generation_pipeline.answer(request.query)
        answer_text = str(response_obj)

        return {
            "answer": answer_text,
            "citations": citations
        }
    except Exception as e:
        logger.error(f"Error handling chat query: {e}")
        raise HTTPException(status_code=500, detail=str(e))