import os
import re
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate
)
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import torch

class RAGTester:
    def __init__(self, db_path):
        self.db_client = chromadb.PersistentClient(path=db_path)

    def _get_chat_formatter(self, model_path):
        def messages_to_prompt(messages):
            prompt = ""
            for message in messages:
                if message.role == 'user':
                    prompt += f"<|user|>\n{message.content}<|end|>\n"
                elif message.role == 'assistant':
                    prompt += f"<|assistant|>Model\n{message.content}<|end|>\n"
            prompt += "<|assistant|>Model\n"
            return prompt
        return messages_to_prompt

    def load_session(self, llm_path, embed_model_path, chunk_tech="markdown"):
        # embed_model_path = '../../models/embeddings/' + embed_model_name         .... no need after config
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_path, device='cuda')
        Settings.llm = HuggingFaceLLM(
            model_name=llm_path,
            tokenizer_name=llm_path,
            context_window=4096,
            max_new_tokens=1024,
            generate_kwargs={"temperature": 0.7, "do_sample": True},
            messages_to_prompt=self._get_chat_formatter(llm_path),
            completion_to_prompt=lambda x: f"<|user|>user\n{x}<|end|>\n<|assistant|>model\n",
            model_kwargs={"torch_dtype": torch.float16},
            device_map='auto'
        )
        llm = Settings.llm
        if hasattr(llm, "_model"):
            print(f"LLM INFO: {llm.model_name}")
            print(f"Context Window: {llm.context_window}")
        else:
            print("LLM is not a HuggingFaceLLM or not loaded yet.")

        # clean_name = embed_model_name.split("/")[-1].replace(".","-")
        # collection_name = f"{clean_name}_{chunk_tech}"
        positions = [p.start() for p in re.finditer(r'/', embed_model_path)]
        embed_model_name = embed_model_path[positions[-2]+1:]
        collection_name = f"{embed_model_name.replace('/', '_')}_{chunk_tech}"

        print(f"Connecting to collection: {collection_name}")

        chroma_collection = self.db_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        num_records = chroma_collection.count()
        if num_records > 0:
            print(f"LOG: VectorDB loaded successfully. Collection '{collection_name}' contains {num_records} nodes.")
        else:
            print(f"LOG: VectorDB '{collection_name}' is currently empty. You may need to ingest data.")    # BAAI_bge-small-en-v1.5_markdown

        index = VectorStoreIndex.from_vector_store(vector_store)

        chat_engine = index.as_chat_engine(
            chat_mode="context",
            system_prompt="You are a helpful researcher answering questions about PDFs."
        )

        return chat_engine

    
    def start_terminal_chat(self, chat_engine):
        print("\n" + "="*50)
        print("SYSTEM: Chat started. Type 'exit' or 'quit' to stop.")
        print("="*50 + "\n")

        while True:
            user_input = input("USER: ")
            if user_input.lower() in ['exit', 'quit']:
                break

            response = chat_engine.chat(user_input)
            # DEBUG: If response is still null, check if chunks were even found
            if not response.response:
                print(f"DEBUG: No text generated. Chunks found: {len(response.source_nodes)}")
            else:
                print(f"\nASSISTANT: {response.response}\n")


if __name__ == "__main__":
    LLM_PATH = "../../models/llms/meta-llama/Llama-3.2-3B"
    EMBED_MODEL = "BAAI/bge-large-en-v1.5"
    DB_PATH = "../../vectorDB"
    chat_obj = RAGTester(DB_PATH)

    engine = chat_obj.load_session(LLM_PATH, EMBED_MODEL, "semantic")
    chat_obj.start_terminal_chat(engine)