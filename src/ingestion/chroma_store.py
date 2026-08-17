import chromadb
import logging
import hashlib
from src.ingestion.base import BaseVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from src.utils.loader import LocalModelLoader
from src.schema import HighlightCoords
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def flatten_citation_metadata(metadata: dict) -> dict:
    """
    Build the minimal metadata stored in Chroma.

    Chroma receives only metadata needed for:
    - citation
    - PDF retrieval
    - page navigation
    - precise highlighting
    - node tracking
    """

    raw_coords = metadata.get("highlight_coords")

    result = {
        "source_doc": metadata["source_doc"],
        "source_path": metadata["source_path"],
        "page_number": int(metadata.get("page_number", 0)),
        "node_id": metadata["node_id"],
        "has_precise_highlight": False,
    }

    if raw_coords:
        try:
            coords = HighlightCoords(**raw_coords)

            result["coord_x1"] = coords.x1
            result["coord_y1"] = coords.y1
            result["coord_x2"] = coords.x2
            result["coord_y2"] = coords.y2

            result["has_precise_highlight"] = True

        except ValidationError as e:
            logger.warning(
                "Malformed highlight coordinates dropped: %s",
                e
            )

    return result

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
            # Create a unique hash for the chunk based on hash(its content + source_doc + page)
            # to prevent duplicates if the pipeline runs again.
            metadata = chunk["metadata"]

            identity_string = "|".join([
                metadata["source_doc"],
                str(metadata.get("page_number", 0)),
                chunk["text"],
            ])

            content_hash = hashlib.sha256(
                identity_string.encode("utf-8")
            ).hexdigest()

            node_id = f"node_{content_hash}"
            if node_id in all_ids:
                already_in.append(chunk['text'])
                continue
            all_ids.add(node_id)

            chunk["metadata"]["node_id"] = node_id
            flat_metadata = flatten_citation_metadata(chunk['metadata'])
            
            # llamaindex internaly uses UUIDs and automatically generate UUID4 for every single node
            # The problem is it is non-deterministic, even the text is identical, every time you run the ingestion script, a brand-new unique id created.
            node = TextNode(
                text = chunk["text"],
                id_ = node_id,          # chunk['metadata'].get('node_id'),       # wiht manual hash f"node_{content_hash}"
                metadata = flat_metadata
            )

            # for hierarchicalChunker
            if "relationships" in chunk:
                node.relationships = chunk['relationships']

            # in the llm response, it got spell out the node_id and other metadata; so we will exclude metadata for llm only            
            node.excluded_llm_metadata_keys = ["source_doc", "source_path", "page_number", "coord_x1", "coord_y1", "coord_x2", "coord_y2", "has_precise_highlight", "node_id"]
            # Keep only the important ones for the LLM context
            # We WANT the LLM to see page numbers and section headers
            for important_key in ['page_number', 'section_title', 'heading']:
                if important_key in node.excluded_llm_metadata_keys:
                    node.excluded_llm_metadata_keys.remove(important_key)
            
            node.excluded_embed_metadata_keys = ["source_doc", "source_path", "page_number", "coord_x1", "coord_y1", "coord_x2", "coord_y2", "has_precise_highlight", "node_id"]      # to keep embedding pure only based on text, this will exclude them from the embedder too

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
