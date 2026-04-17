import os
import time
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader
)

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from llama_index.readers.docling import DoclingReader
from llama_index.readers.file import MarkdownReader

from llama_index.core.node_parser import(
    SentenceSplitter,
    SemanticSplitterNodeParser,
    MarkdownNodeParser
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

class VectorizePDFs:
    def __init__(self, db_path = "./vectorDB"):
        self.db_path = db_path
        self.db_client = chromadb.PersistentClient(path=self.db_path)

    def get_gpu_docling_reader(self):
        pipeline_options = ThreadedPdfPipelineOptions(
           accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
            page_batch_size=16  # Batching happens here
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        print("Returned DoclingReader")
        return DoclingReader(converter=converter)
    
    def get_node_parser(self, technique="sentence"):
        if technique == "sentence":
            return SentenceSplitter(chunk_size=512, chunk_overlap=50)
        elif technique == "semantic":
            return SemanticSplitterNodeParser(
                buffer_size=5,
                breakpoint_percentile_threshold=90,
                embed_model=Settings.embed_model
            )
        elif technique == "markdown":
            return MarkdownNodeParser()
        else:
            raise ValueError(f"Unknown technique: {technique}")
        
    def get_embedding_model(self, model_name="BAAI/bge-small-en-v1.5"):
        base_model_path = Path("../../models/embeddings/").resolve()
        full_path = base_model_path / model_name
        if not full_path.exists():
            raise FileNotFoundError(f"Model not found at: {full_path}")
            
        return HuggingFaceEmbedding(model_name= str(full_path), device="cuda", embed_batch_size=64)
    
    
    def run_experiment(self, source_dir, chunk_tech="sentence", embed_mode_name = "BAAI/bge-small-en-v1.5"):
        
        embed_model = self.get_embedding_model(embed_mode_name)
        Settings.embed_model = embed_model
        node_parser = self.get_node_parser(chunk_tech)
        Settings.node_parser = node_parser
        print("Loaded Embedding model and Chuncking Technique(node_parser)")

        collection_name = f"{embed_mode_name.replace('/', '_')}_{chunk_tech}"
        chroma_collection = self.db_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        reader = self.get_gpu_docling_reader()
        dir_reader = SimpleDirectoryReader(
            input_dir=source_dir,
            file_extractor={
                ".pdf": reader,
                ".md": MarkdownReader()
                },
            required_exts=[".pdf", ".md"],
            recursive=True
        )
        documents = dir_reader.load_data()      # num_workers=4

        pipeline = IngestionPipeline(
            transformations=[node_parser, embed_model],
            # vector_store=vector_store         # because we need to manually write 5000 nodes
        )
        print(f"--- Running Ingestion for {collection_name} ---")
        nodes = pipeline.run(documents=documents, show_progress=True)       # num_workers=4
        
        BATCH_SIZE = 5000
        print(f"Inserting {len(nodes)} nodes in batches of {BATCH_SIZE}")
        for i in range(0, len(nodes), BATCH_SIZE):
            batch = nodes[i: i+BATCH_SIZE]
            vector_store.add(batch)
            print(f"Inserted batch {i//BATCH_SIZE + 1}")

        # Build Index from the now-filled Vector Store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context = storage_context,
            embed_model=embed_model
        )
        num_records = chroma_collection.count()
        if num_records == 0:
            print(f"Collection {collection_name} is empty! Starting auto-ingestion...")
            self.ingest_data(embed_model_name=embed_mode_name)
        else:
            print(f"Collection found with {num_records} nodes.")

        print(f"--- Index built and stored in {self.db_path} ---")
        return index
    

if __name__ == "__main__":
    pipeline = VectorizePDFs(db_path="../../vectorDB")
    index1 = pipeline.run_experiment(source_dir='../../data/processed_data/', chunk_tech="semantic", embed_mode_name="BAAI/bge-large-en-v1.5")

    # query_engine = index1.as_query_engine()
    # print(query_engine.query("What is Onatario Building Code?"))