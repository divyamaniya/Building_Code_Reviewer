'''
This file registered and selects which class to use based on file type and parser technique/method given in config

Loader > Parser > Chunker > embedd > store in VectorDB

'''

from src.ingestion.loader import LocalPDFLoader, VirtualPDFSplitter
from src.ingestion.parser import DoclingParser, UnstructuredParser
from src.ingestion.chunker import *
from src.ingestion.chroma_store import ChromaVectorManager

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
from typing import List

import logging
import json
from pathlib import Path
from src.utils.logging import worker_configurer

logger = logging.getLogger(__name__)

class IngestionRegistry:        # better name ComponentRegistry
    _loaders = {
        "localPdf": LocalPDFLoader,
        # "url": WebLoader
    }

    _parser = {
        "docling": DoclingParser,
        "unstructured": UnstructuredParser
    }

    _chunkers = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "sliding_window": SlidingWindowChunker,
        "hierarchical": HierarchicalChunker,
        "structure": StructureAwareChunker,
        "hybrid": DoclingHybridChunker,
    }

    _vector_stores = {
        "chroma": ChromaVectorManager
    }

    @classmethod
    def get_loader(cls, source_type: str):
        loader_class = cls._loaders.get(source_type)
        logger.debug("fetching loader: %s", loader_class)
        if not loader_class:
            raise ValueError(f"No loader found for: {source_type}")
        return loader_class()
    
    @classmethod
    def get_parser(cls, method_name: str):
        parser_class = cls._parser.get(method_name)
        logger.debug("fetching parser: %s", parser_class)
        if not parser_class:
            raise ValueError(f"Parser {method_name} not supported")
        return parser_class()
    
    @classmethod
    def get_chunker(cls, method_name: str, *args, **kwargs):
        chunker_class = cls._chunkers.get(method_name)
        logger.debug("fetching chunker: %s", chunker_class)
        if not chunker_class:
            raise ValueError(f"Chuner {method_name} not supported")
        
        if method_name == "semantic":
            return chunker_class(*args, **kwargs)
        else:
            return chunker_class()
    
    @classmethod
    def get_vector_store(cls, name: str, **kwargs):
        vs_class = cls._vector_stores.get(name)
        if not vs_class:
            raise ValueError(f'Vector Store {name} not supported')
        return vs_class(**kwargs)
    
    # # Get the constructor parameters     TO AVOID BELOW IF ELSE CODE
    #     sig = inspect.signature(chunker_class.__init__)
    #     params = sig.parameters

    #     # Automatically inject dependencies if the class asks for them
    #     kwargs = {}
    #     if "embed_model_path" in params:
    #         kwargs["embed_model_path"] = config.paths.embedding_model_dir
            
    #     if "llm_model_path" in params:
    #         kwargs["llm_model_path"] = config.paths.llm_model_dir

    #     logger.debug(f"Initializing {method_name} with params: {list(kwargs.keys())}")
    #     return chunker_class(**kwargs)
    

class IngestionProcess:
    '''
    This connects all 3 components
    '''
    def __init__(self, config):
        self.config = config
        self.registry = IngestionRegistry()

    def process_single_file(self, file_path: str, method: str, destination_path: str):
        logger.debug("process_single_file for: %s", file_path)
        parser = self.registry.get_parser(method)
        result = parser.parse(file_path)
        self.save_to_jsonl(result, file_path, destination_path)
        return str(destination_path)
    
    def save_to_jsonl(self, parse_data, source, destination):
        import json
        data = parse_data.document.export_to_dict()
        source_path = Path(source).resolve()
        data['_source_pdf'] = {
            "source_doc": source_path.name,
            "source_path": str(source_path)
        }
        dest_path = Path(destination)/ "parsed" / Path(source).with_suffix(".json").name
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4, default=str)

    def process_batch(self, file_paths:List[str], method:str, destination_path):
        
        cpu_count = multiprocessing.cpu_count()
        logger.debug("process_batch: %s", file_paths)
        print(f"Starting multiprocess ingestion on {cpu_count} cores...")

        results = []
        with ProcessPoolExecutor(max_workers=cpu_count, initializer=worker_configurer) as executor:
            # map the worker function to all file paths
            futures = {
                executor.submit(self.process_single_file, fp, method, destination_path): fp
                for fp in file_paths
            }

            for future in as_completed(futures):
                file_name = futures[future]
                try:
                    saved_at = future.result()
                    # logger.info("Successfully finished %s -> %s", file_name, saved_at)
                except Exception as e:
                    logger.error("Failed to process %s: %s", file_name, e)


    def load_parse(self, source_folder):
        self.loader_method = self.config.data_pipeline.loader_method
        self.parser_method = self.config.data_pipeline.parser_method   # ingestion> method is define in config
        
        pdf_files_paths = self.registry.get_loader(self.loader_method).load(source_folder)      # raw_dir

        if len(pdf_files_paths)>1:
            self.process_batch(pdf_files_paths, self.parser_method, self.intermediate_data)
        else:
            self.process_single_file(pdf_files_paths[0], self.parser_method, self.intermediate_data)

    def chunk_single_file(self, json_path, embed_model_path, chunks_dir, chunk_technique, save_func):
        logger.info(f"Chunk for file {json_path}; ")
        try:
            chunker = self.registry.get_chunker(chunk_technique, embed_model_path)
            chunks = chunker.run_on_file(str(json_path))
            chunks_file_path = chunks_dir/f"{chunk_technique}_{json_path.name}"
            source_pdf_dir = Path(self.config.paths.splitted_data)     # Citation must point to the corresponding PDF.
            source_pdf_path = (
                source_pdf_dir / json_path.with_suffix(".pdf").name
            )
            if not source_pdf_path.exists():
                raise FileNotFoundError(
                    f"Original PDF not found: {source_pdf_path}"
                )
            save_func(chunks, chunks_file_path, source_pdf_path)
            return f"Finished {json_path.name}"
        except Exception as e:
            logger.debug(f"Error occured in single file chunk: {e}")

    def chunk_and_store(self):
        self.chunk_technique = self.config.data_pipeline.chunk_strategy.get('chunk_technique')
        embed_model_path = self.config.paths.embedding_model_dir if self.chunk_technique == "semantic" else None

        chunks_destination_dir = Path(self.intermediate_data) / "chunks" / self.chunk_technique
        chunks_destination_dir.mkdir(parents=True, exist_ok=True)

        source_file_list = list((Path(self.intermediate_data)/"parsed").glob("*.json"))
        logger.debug(f"""Debugging chunk and store: \nchunk_technique: {self.chunk_technique},
                    \nembed_model_path: {embed_model_path},
                    \nchunks_destination_dir: {chunks_destination_dir}""")
        # Use ProcessPoolExecutor for CPU-bound chunking
        # max_workers=None defaults to number of processors
        with ProcessPoolExecutor(max_workers=4) as executor:            # We wrap the extra arguments using partial
            worker = functools.partial(
                self.chunk_single_file,
                embed_model_path=embed_model_path, 
                chunks_dir=chunks_destination_dir, 
                chunk_technique=self.chunk_technique, 
                save_func=self.save_chunks
            )
            executor.map(worker, source_file_list)
            


    def embed_and_vectorDB(self):
        self.chunk_technique = self.config.data_pipeline.chunk_strategy.get('chunk_technique')
        logger.info("Starting Vector Storage Process...")
        # Load embedding model using your existing utility
        embed_model_path = Path(self.config.paths.embedding_model_dir)

        vector_store = self.registry.get_vector_store(
            name = self.config.VectorDB.name,
            collection_name = self.config.VectorDB.collection_name,
            persist_dir= self.config.VectorDB.persist_directory,
            embed_model_path = embed_model_path
        )
        chunks_dir = Path(self.intermediate_data) / "chunks" / self.chunk_technique
        logger.info(f"Chunk file path: {chunks_dir}")
        all_chunks = []         # Collect ALL chunks first (Very fast CPU task)
        for chunk_file in chunks_dir.glob("*.json"):
            logger.info(f"Processing chunk file {chunk_file} for Vector DB")
            chunks = self.load_chunks_from_file(chunk_file)

            if chunks:
                all_chunks.extend(chunks)
        
        # 2. Add to Vector DB in one or two large optimized batches
        # Your add_documents should handle the hashing/deduplication logic we wrote
        if all_chunks:
            logger.info(f"Sending {len(all_chunks)} chunks to GPU for embedding...")
            vector_store.add_documents(all_chunks)
        
        logger.info("pipeline execution complete")


    def save_chunks(self, chunks: list, output_dir: str, source_pdf_path: str):
        """
        Save chunks as JSONL.

        The JSONL file is only an intermediate artifact.
        Citation metadata always points back to the original PDF.
        """

        source_pdf = Path(source_pdf_path).resolve()

        try:
            with open(output_dir, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    metadata = chunk.setdefault("metadata", {})
                    # Original PDF identity
                    metadata["source_doc"] = source_pdf.name
                    metadata["source_path"] = str(source_pdf)

                    json_line = json.dumps(chunk, ensure_ascii=False)
                    f.write(json_line + "\n")

            logger.info("Successfully saved %s chunks for %s", len(chunks), source_pdf.name)

        except Exception as e:
            logger.error("Error saving chunks for %s: %s", source_pdf, e)


    def load_chunks_from_file(self, file_path: Path):
        '''
        helper to read the json chunk files
        '''
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
        # If your file is JSONL (one JSON object per line)
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks


    def run_data_ingestion_pipeline(self, source_folder: str):

        self.intermediate_data = self.config.paths.intermediate_data
        

        # self.load_parse(source_folder)
        # Now we got bunch of JSON file
        self.chunk_and_store()
        self.embed_and_vectorDB()