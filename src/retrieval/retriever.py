'''
This file is implementation of retrieval and reranking
'''

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.core import QueryBundle
import logging
from functools import partial
import asyncio
from src.utils.loader import vLLMLocalModelLoader

logger = logging.getLogger(__name__)

# in llama reranking is handled by NodePostprocessors
# retrieval evaluation: Hit Rate/MRR

class RetrievalPipeline:
    def __init__(self, index, config):
        self.index = index     # LlamaIndex 'index' object (linked to your ChromaDB)
        self.config = config
        self.use_reranker = config.VectorDB.use_reranker

        # setup how we search the vector DB
        self.retriever = VectorIndexRetriever(
            index = self.index,
            similarity_top_k= config.VectorDB.initial_top_k
        )

        if self.use_reranker:
        # reranker on initial results
            if config.VectorDB.reranker_type == "llm":
                self.reranker = LLMRerank(top_n=config.VectorDB.final_top_k) # ,llm=Settings.llm)
            else:
                self.reranker = SentenceTransformerRerank(
                    model = config.paths.rerank_model_dir,
                    top_n = config.VectorDB.final_top_k
                )

        logger.debug("Init of RetrievalPipeline completed!")

    def retrieve_context(self, query_str: str):
        # initial retrieval
        # with llamaindex, we don't make call to chromadb.query() and retrive top k results; VectorIndexRetriever handles it
        # below function call performs; embeds the query_str > calls Chroma using internal .query() > returns a list of NodeWithScore objects
        nodes = self.retriever.retrieve(query_str)

        # Logic check: if no reranker, just return initial nodes
        if not self.use_reranker:
            return nodes

        # Wraps the string query into a LlamaIndex object for processing
        query_bundle = QueryBundle(query_str)

        # Executes the reranking logic: Query + Nodes -> Re-scored Nodes.
        reranked_nodes = self.reranker.postprocess_nodes(
            nodes, query_bundle=query_bundle
        )
        logger.debug(f"retrieve_context {reranked_nodes}")
        return reranked_nodes
    
    async def aretrieve_try1(self, query_str: str):
        """
        Asynchronously retrieves and optionally reranks nodes matching the query string.
        """
        nodes = await self.retriever.aretrieve(query_str)

        if not self.use_reranker:
            return nodes
        
        # 2. Package the query for the post-processor
        query_bundle = QueryBundle(query_str)

        # 3. Asynchronous Reranking
        # We call 'apostprocess_nodes' to ensure execution stays non-blocking
        reranked_nodes = await self.reranker.apostprocess_nodes(
            nodes, query_bundle=query_bundle
        )
        
        logger.debug(f"Async retrieve_context count: {len(reranked_nodes)}")
        return reranked_nodes
    

    async def aretrieve_try2(self, query_str: str):
        """
        Asynchronously retrieves nodes, offloading synchronous DB I/O.
        """
        # Force the synchronous database read to run safely in a worker thread
        loop = asyncio.get_running_loop()
        nodes = await loop.run_in_executor(
            None, 
            partial(self.retriever.retrieve, query_str)  # Note: calling .retrieve, NOT .aretrieve
        )

        return nodes
    

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.core import Settings

# pip install llama-index-postprocessor-tei-rerank
# pip install llama-index-retrievers-bm25

class HybridRetrievalPipeline:
    # chromadb doesnot support native hybrid search inside llamaindex via single standard retriever wrapper
    # so the cleanest way to use it using QueryFusionRetriever; this automatically runs both vector retrieval and BM25 (sparse keyword search)
    
    def __init__(self, index, nodes, config):
        self.index = index     # LlamaIndex 'index' object (linked to your ChromaDB)
        self.config = config
        self.use_reranker = config.VectorDB.use_reranker

        # Base Dense Retriever
        dense_retriever = VectorIndexRetriever(
            index = self.index,
            similarity_top_k= config.VectorDB.initial_top_k
        )

        # 2. FIX: Initialize BM25 directly using the extracted text nodes
        if not nodes:
            raise ValueError("Cannot initialize BM25Retriever because nodes list is empty!")

        # Sparse Retriever (Keyword search over same index document)
        sparse_retriever = BM25Retriever.from_defaults(
            nodes= nodes,
            similarity_top_k= config.VectorDB.initial_top_k
        )

        # Hybrid search setup using QueryFusionRetriever
        self.retriever = QueryFusionRetriever(
            retrievers= [dense_retriever, sparse_retriever],
            similarity_top_k= config.VectorDB.initial_top_k,
            num_queries= 4,      # to keep original query unchanged; # Generates 3 alternate phrasings + the original query
            # mode= "reciprocal_rerank",       # combines keyword and vector search
            mode= "relative_score",       # Preserves raw semantic/lexical features for your TEI reranker)
            llm = Settings.llm,
            use_async=True
        )

        # 4. Correctly setup Reranker Object
        if self.use_reranker:
            logger.info(f"Initializing Reranker: {config.VectorDB.reranker_type}")
            if config.VectorDB.reranker_type == "llm":
                self.reranker = LLMRerank(top_n=config.VectorDB.final_top_k)
            else:
                # This perfectly accepts BAAI/bge-reranker-large or bge-reranker-v2-m3
                self.reranker = SentenceTransformerRerank(
                    model=str(config.paths.rerank_model_dir),
                    top_n=config.VectorDB.final_top_k
                )

                # self.reranker = vLLMLocalModelLoader().local_TEI_reranker(
                #     tei_url="http://localhost:8096", # Your TEI port
                #     top_n=config.VectorDB.final_top_k,
                #     model_path=str(config.paths.rerank_model_dir)
                # )
        else:
            self.reranker = None

        logger.debug("Init of RetrievalPipeline completed!")