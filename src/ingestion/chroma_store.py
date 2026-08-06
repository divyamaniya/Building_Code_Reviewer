import chromadb
import logging
import hashlib
from src.ingestion.base import BaseVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from src.utils.loader import LocalModelLoader

logger = logging.getLogger(__name__)

'''
In VectorDB, Database/Instance are top-level container; chroma instance
Collection/Index; equivalent to table
Document; single entry in database; row
Metadata/Payload; Extra information - column
ID(UUID); Unique identifier for the point
Vector Index (HNSW, IVF); mathematical structure that allows nearest neighbor search
Vector store = Vector Database

https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
'''



class ChromaVectorManager(BaseVectorStore):
    def __init__(self, collection_name: str, persist_dir: str, embed_model_path):
        
        logger.debug(f"Collection name: {collection_name}, {persist_dir}, {embed_model_path}")

        self.embed_model = LocalModelLoader().load_embedding(embed_model_path)
        
        # Initialize Chroma Client
        self.client_db = chromadb.PersistentClient(path=persist_dir)
        self.chroma_collection = self.client_db.get_or_create_collection(collection_name)      # if we add self.embed_model, chroma will call model every time you add text or query the DB. 

        # Setup LlamaIndex Vector Store
        self.vector_store = ChromaVectorStore(chroma_collection = self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store = self.vector_store)

        # this class also wants to calculate embeddings first and then sends the vectors to chroma
        self.index = VectorStoreIndex.from_vector_store(
            vector_store= self.vector_store,
            storage_context = self.storage_context,
            embed_model= self.embed_model
        )

    def add_documents(self, chunks, batch_size=64):
        '''
        Converts dictionary chunks to LlamaIndex Nodes and indexes them
        '''
        nodes = []
        all_ids = set()
        already_in = []

        for chunk in chunks:
            # Create a unique hash for the chunk based on its content 
            # to prevent duplicates if the pipeline runs again.
            content_hash = hashlib.sha256(chunk["text"].encode()).hexdigest()
            # hashlib.sha256(f"{chunk['text']}_{chunk['metadata'].get('page_number')}".encode())

            node_id = f"node_{content_hash}"
            if node_id in all_ids:
                already_in.append(chunk['text'])
                continue
            all_ids.add(node_id)
            
            # llamaindex internaly uses UUIDs and automatically generate UUID4 for every single node
            # The problem is it is non-deterministic, even the text is identical, every time you run the ingestion script, a brand-new unique id created.
            node = TextNode(
                text = chunk["text"],
                id_ = node_id,          # chunk['metadata'].get('node_id'),       # wiht manual hash f"node_{content_hash}"
                metadata = chunk['metadata']
            )

            # for hierarchicalChunker
            if "relationships" in chunk:
                node.relationships = chunk['relationships']

            # in the llm response, it got spell out the node_id and other metadata; so we will exclude metadata for llm only
            # node.excluded_llm_metadata_keys = ['node_id', 'file_path', 'page_label', 'page_number', 'source_doc', 'contains_image', 'is_markdown']    # this stay in chromaDb and are returned by the retriever but right before the text is sent to llm it will be deleted
            # Define a base list of keys to ALWAYS hide from the LLM
            base_exclusions = ['node_id', 'file_path', 'source_doc', 'is_markdown', 'is_hybrid', 'is_hierarchical']
            
            # Add keys that might appear in Docling Hybrid or StructureAware
            layout_exclusions = ['bbox', 'doc_items', 'dl_path', 'charspan', 'coord_origin']
            
            node.excluded_llm_metadata_keys = list(set(base_exclusions + layout_exclusions + list(chunk['metadata'].keys())))
            # Keep only the important ones for the LLM context
            # We WANT the LLM to see page numbers and section headers
            for important_key in ['page_number', 'section_title', 'heading']:
                if important_key in node.excluded_llm_metadata_keys:
                    node.excluded_llm_metadata_keys.remove(important_key)
            
            
            node.excluded_embed_metadata_keys = ['node_id', 'file_path', 'page_label']      # to keep embedding pure only based on text, this will exclude them from the embedder too

            # force llm to see only content
            # node.text_template = "{content}"

            nodes.append(node)
            
        logger.debug(f"already in : {already_in}")
        total_nodes = len(nodes)
        logger.info(f"Beginning indexing of {total_nodes} unique nodes in batches of {batch_size}")        
        
        for i in range(0, total_nodes, batch_size):
            batch = nodes[i : i + batch_size]
            
            # We use insert_nodes because it handles the embedding call internally
            # Each call here triggers the GPU for exactly 'batch_size' nodes
            try:
                self.index.insert_nodes(batch)
                logger.info(f"Successfully indexed batch {i//batch_size + 1}: nodes {i} to {min(i + batch_size, total_nodes)}")
            except Exception as e:
                logger.error(f"Failed at batch starting at index {i}: {str(e)}")
                # Optional: break or continue depending on your error policy
    
    def delete_collection(self):
        self.client_db.delete_collection(self.chroma_collection.name)
