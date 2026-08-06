''' This file loads any model embedding or llm given path and this file can be use in Embedding, Retreival as well as generation'''

from abc import ABC, abstractmethod
from llama_index.core import Settings
from transformers import AutoConfig
import logging
import subprocess
import time
import os
import signal
from pydantic import BaseModel, PrivateAttr

logger = logging.getLogger(__name__)

class BaseModelLoader(ABC):
    @abstractmethod
    def load_llm(self, model_name_or_path: str, **kwargs):
        pass

    @abstractmethod
    def load_embedding(self, model_name_or_path: str, **kwargs):
        pass


class LocalModelLoader(BaseModelLoader):
    def load_llm(self, model_path: str, **kwargs):
        from llama_index.llms.huggingface import HuggingFaceLLM

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")
        try:
            # if model is hallucinate the conversation between user and model on its own, it means that it lost it's boundry between user prompt/input and its response begins.
            # every model has their own special token (bos_token, eos_token, and pad_token)
            # if you send plain text without proper markers, then model will hallucinate the conversation
            # So every model have their own chat template (<|user|> and <|assistant|>)

            # Another option to tune temperature settings; if temperature is high then model become creative at the same time hallucinate also.
            # Repetation penalty around 1.1 or 1.2 discourages the model from saying same words over and over.
            
            # To automatically take out chat template we will use AutoTokenizer; so that we do not have to manually maintain chat template keywords and special tokens
            # from transformers import AutoTokenizer
            # tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            model_str = str(model_path)

            # fetch model config
            model_config = AutoConfig.from_pretrained(model_str, trust_remote_code=True)

            hf_context_window = getattr(model_config, "max_position_embeddings",       # standard name
                                getattr(model_config, "sliding_window", 4096))  # this is for mistral

            context_window = kwargs.get("context_window") or hf_context_window
            max_new_tokens = kwargs.get("max_new_tokens") or int(context_window * 0.1)
            max_new_tokens = min(max_new_tokens, 2048)
            logger.info(f"Detected Context Window for {model_str}: {context_window}")

            llm_instance = HuggingFaceLLM(
                model_name= model_str,
                tokenizer_name= model_str,    # llama internally calls AutoTokenizer.from_pretrained()
                context_window= context_window,
                max_new_tokens= max_new_tokens,
                # model_kwargs={
                #     "torch_dtype": torch.bfloat16, 
                #     "attn_implementation": "flash_attention_2", 
                # },
                generate_kwargs={
                    "temperature": kwargs.get("temperature", 0.6),
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                    "do_sample": True if kwargs.get("temperature", 0.6) > 0 else False
                },
                # is_chat_model = True,       # by this llamaindex uses Chat Engine (internally apply_chat_template)
                device_map= "auto"
            )
            Settings.llm = llm_instance
            # logger.info(f"Loading local LLM from: {llm_instance}")
        except Exception as e:
            logger.debug(f"Exception in llm loading: {e}")
        return llm_instance         # UnboundLocalError; try to return variable that was never successfully created!
    
    def load_embedding(self, model_path, **kwargs):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        model_str = str(model_path)
        
        # Nomic and Qwen both need this as they use custom architecture not the standard BERT
        kwargs.setdefault("trust_remote_code", True)
        
        # Instruction-based embedding models (like GTE-Qwen2); design to understand context. they perform better when we instruct and tell why they are looking at a piece of text.
        if "gte-Qwen2" in model_str:
            # GTE models perform significantly better with a query instruction; because there is a mathematical difference between "searching for an answer" and "storeing a fact"
            kwargs.setdefault("query_instruction", "Represent this query for retrieving relevant documents: ")
            kwargs.setdefault("embed_batch_size", 4) # 7B models are VRAM heavy!

        embed_model = HuggingFaceEmbedding(model_name= model_str, **kwargs, device="cuda")
        Settings.embed_model = embed_model
        logger.info(f"Loading Local Embedding from: {embed_model}")
        return embed_model
    

# This is the fix of https://github.com/run-llama/llama_index/issues/21371

from typing import Dict, Any
@property
def patched_model_kwargs(self) -> Dict[str, Any]:
    base_kwargs = {
        "temperature": self.temperature,
        "max_tokens": self.max_new_tokens,
        "n": self.n,
        "frequency_penalty": self.frequency_penalty,
        "presence_penalty": self.presence_penalty,
        # "best_of": self.best_of,  <-- Culprit removed
        "ignore_eos": self.ignore_eos,
        "stop": self.stop,
        "logprobs": self.logprobs,
        "top_k": self.top_k,
        "top_p": self.top_p,
    }
    return {k: v for k, v in base_kwargs.items() if v is not None}

class vLLMLocalModelLoader(BaseModelLoader):

    def load_llm(self, model_path: str, **kwargs):
        from llama_index.llms.vllm import Vllm
        Vllm._model_kwargs = patched_model_kwargs       # applied the patch to the class

        # To derive parameters just from model, like HF AutoConfig is not direct in vLLM
        # because vLLM has its own logic for context
        model_str = str(model_path)
        
        # initializing a temporary config
        vllm_kwargs = {     # this vllm_kwargs passed to SamplingParams
            "gpu_memory_utilization": kwargs.get("gpu_util", 0.75), # Leave room for Embedding/Rerank
            # "device": "cuda",
            # "max_model_len": kwargs.get("context_window", 4096),
        }

        # Initialize the LlamaIndex Wrapper
        # The wrapper will initialize the vllm.LLM engine internally
        llm_instance = Vllm(
            model= model_str,
            vllm_kwargs= vllm_kwargs,
            temperature= kwargs.get("temperature", 0.1),
            max_new_tokens= kwargs.get("max_new_tokens", 512),
            trust_remote_code = False,
            # reuse_client= True,
            presence_penalty= kwargs.get("repetition_penalty", 1.1),
            stop=["user:", "system:", "assistant:"]
        )

        # Derive context window dynamically from the loaded vLLM engine
        # vLLM stores this in its logical engine config
        try:
            engine_config = llm_instance._client.llm_engine.model_config
            hf_config = engine_config.hf_config
            
            actual_context_len = engine_config.max_model_len
            logger.debug(f"The vLLM engine is running with context length: {actual_context_len}")

            # hf_context_window = getattr(hf_config, "max_position_embeddings", 
            #                  getattr(hf_config, "sliding_window", 4096))
            
            # resolved_context = kwargs.get("context_window") or hf_context_window
            # resolved_max_tokens = min(kwargs.get("max_new_tokens", 512), resolved_context // 4)

            # # Update the instance attributes
            # llm_instance.max_model_len = resolved_context    # or max_model_len
            # llm_instance.max_new_tokens = resolved_max_tokens

            # logger.debug(f"vLLM Detected Context Window: {resolved_context}")
        except Exception as e:
            logger.warning(f"Could not derive vLLM params: {e}. Using default 4096.")

        Settings.llm = llm_instance
        return llm_instance
    
    def load_embedding(self, model_path, **kwargs):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        model_str = str(model_path)
        
        # Nomic and Qwen both need this as they use custom architecture not the standard BERT
        kwargs.setdefault("trust_remote_code", True)
        
        # Instruction-based embedding models (like GTE-Qwen2); design to understand context. they perform better when we instruct and tell why they are looking at a piece of text.
        if "gte-Qwen2" in model_str:
            # GTE models perform significantly better with a query instruction; because there is a mathematical difference between "searching for an answer" and "storeing a fact"
            kwargs.setdefault("query_instruction", "Represent this query for retrieving relevant documents: ")
            kwargs.setdefault("embed_batch_size", 4) # 7B models are VRAM heavy!

        embed_model = HuggingFaceEmbedding(model_name= model_str, **kwargs, device="cuda", embed_batch_size=32)
        Settings.embed_model = embed_model
        logger.info(f"Loading Local Embedding from: {embed_model}")
        return embed_model
    
    def local_TEI(self, model_path: str, tei_url: str = "http://localhost:8095", **kwargs):
        """
        Launches the TEI CLI server locally via a Python subprocess, 
        waits for it to be ready, and connects the LlamaIndex client.
        """
        from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
        import requests
        model_str = str(model_path)
        logger.info(f"starting TEI CLI Server for model: {model_str}")

        cmd = [
            "text-embeddings-router",
            "--model-id", model_str,
            "--port", tei_url.split(":")[-1].replace("/", ""), # Extracts port (e.g., '8080')
            "--dtype", "float16",
            "--max-concurrent-requests", "32", # Increase queue handling capacity
            "--max-batch-tokens", "16384",       # Maximizes your GPU's parallel capabilities
            "--max-client-batch-size", "32"      # Allows bigger batch handling per client request
        ]

        tei_log_file = open("logs/tei_server.log", "w")

        try: # launch CLI process in background
            self._tei_embed_process = subprocess.Popen(
                cmd,
                stdout= tei_log_file,
                stderr= subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid
            )
        except FileNotFoundError:
            raise RuntimeError(
                "The 'text-embeddings-router' binary was not found in your PATH. "
                "Make sure you ran 'source ~/.bashrc' or configured your Cargo environment."
            )
        
        # 3. Wait for the server to spin up and become healthy
        health_url = f"{tei_url.rstrip('/')}/health"
        logger.info("Waiting for TEI server to initialize kernels and load weights...")
        
        timeout = 180 # Give a 7B model up to 3 minutes to load entirely into VRAM
        start_time = time.time()

        while True:
            # Check if the process crashed immediately on startup (e.g., OOM)
            if self._tei_embed_process.poll() is not None:
                raise RuntimeError("TEI Server failed to start. Check your VRAM allocation or model path.")
            
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200 or response.ok:
                    logger.info("TEI Server is up, healthy, and ready!")
                    break
            except requests.exceptions.RequestException:
                pass # Server isn't ready yet, keep polling
            
            if time.time() - start_time > timeout:
                self.close_embedding()
                raise TimeoutError("Timed out waiting for TEI server to become healthy.")
            
            time.sleep(2)

        # 4. Instantiate the LlamaIndex Wrapper Client
        embed_model = TextEmbeddingsInference(
            model_name=model_str,
            base_url=tei_url,
            timeout=kwargs.get("timeout", 180.0),
            embed_batch_size=kwargs.get("embed_batch_size", 32)
        )
        
        Settings.embed_model = embed_model
        return embed_model
    

    def local_TEI_reranker(self, model_path: str, tei_url: str = "http://localhost:8096", **kwargs):
        """
        Launches the TEI CLI server locally via a Python subprocess, 
        waits for it to be ready, and connects the LlamaIndex client.
        """
        from llama_index.postprocessor.tei_rerank import TextEmbeddingInference
        import requests
        model_str = str(model_path)
        logger.info(f"starting TEI CLI Server for model: {model_str}")

        cmd = [
            "text-embeddings-router",
            "--model-id", model_str,
            "--port", tei_url.split(":")[-1].replace("/", ""), # Extracts port (e.g., '8080')
            "--dtype", "float16",
            "--max-concurrent-requests", "32", # Increase queue handling capacity
            "--max-batch-tokens", "16384",       # Maximizes your GPU's parallel capabilities
            "--max-client-batch-size", "32"      # Allows bigger batch handling per client request
        ]

        tei_log_file = open("logs/tei_server_reranker.log", "w")

        try: # launch CLI process in background
            self._tei_rerank_process = subprocess.Popen(
                cmd,
                stdout= tei_log_file,
                stderr= subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid
            )
        except FileNotFoundError:
            raise RuntimeError(
                "The 'text-embeddings-router' binary was not found in your PATH. "
                "Make sure you ran 'source ~/.bashrc' or configured your Cargo environment."
            )
        
        # 3. Wait for the server to spin up and become healthy
        health_url = f"{tei_url.rstrip('/')}/health"
        logger.info("Waiting for TEI server to initialize kernels and load weights...")
        
        timeout = 180 # Give a 7B model up to 3 minutes to load entirely into VRAM
        start_time = time.time()

        while True:
            # Check if the process crashed immediately on startup (e.g., OOM)
            if self._tei_rerank_process.poll() is not None:
                raise RuntimeError("TEI Server failed to start. Check your VRAM allocation or model path.")
            
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200 or response.ok:
                    logger.info("TEI Server is up, healthy, and ready!")
                    break
            except requests.exceptions.RequestException:
                pass # Server isn't ready yet, keep polling
            
            if time.time() - start_time > timeout:
                self.close_embedding()
                raise TimeoutError("Timed out waiting for TEI server to become healthy.")
            
            time.sleep(2)

        # 4. Instantiate the LlamaIndex Wrapper Client
        rerank_model = TextEmbeddingInference(
            model_name=model_str,
            base_url=tei_url,
            timeout=kwargs.get("timeout", 180.0),
            top_n=kwargs.get("top_n", 7)
        )
        
        return rerank_model
    
    def close_embedding(self):
        """Safely kills both TEI background subprocess groups."""
        import signal
        
        # Kill Embedding Server
        if hasattr(self, '_tei_embed_process') and self._tei_embed_process:
            try:
                os.killpg(os.getpgid(self._tei_embed_process.pid), signal.SIGTERM)
                logger.info("Terminated TEI Embedding server process group.")
            except Exception as e:
                logger.error(f"Error killing TEI embedding process: {e}")
                
        # Kill Reranker Server
        if hasattr(self, '_tei_rerank_process') and self._tei_rerank_process:
            try:
                os.killpg(os.getpgid(self._tei_rerank_process.pid), signal.SIGTERM)
                logger.info("Terminated TEI Reranker server process group.")
            except Exception as e:
                logger.error(f"Error killing TEI reranker process: {e}")

    
    
    def kill_process_by_name(self, process_name="text-embeddings-router"):
        """Scans system processes and forcefully terminates any matching the name."""
        import psutil
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Searching for active system processes matching: '{process_name}'")
        
        killed_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Avoid crashing if the process doesn't expose a name or cmdline attribute
                p_name = proc.info.get('name') or ""
                p_cmdline = proc.info.get('cmdline') or []
                
                name_match = process_name.lower() in p_name.lower()
                cmd_match = any(process_name.lower() in str(arg).lower() for arg in p_cmdline)
                
                if name_match or cmd_match:
                    logger.warning(f"Found rogue TEI process [PID: {proc.info['pid']}]. Terminating...")
                    
                    p = psutil.Process(proc.info['pid'])
                    for child in p.children(recursive=True):
                        child.kill()
                    p.kill() 
                    killed_count += 1
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        if killed_count > 0:
            logger.info(f"Successfully cleaned up {killed_count} '{process_name}' process(es).")
        else:
            logger.debug(f"No active processes found matching '{process_name}'.")


# class VLLMModelLoader:
#     def load_llm(self, model_path, **kwargs):
#         from llama_index.llms.vllm import Vllm
#         model_str = str(model_path)
        
#         # 1. Derive parameters from HF Config (Just like your previous HF Loader)
#         hf_config = AutoConfig.from_pretrained(model_str, trust_remote_code=False)
#         hf_context_window = getattr(hf_config, "max_position_embeddings", 
#                             getattr(hf_config, "sliding_window", 4096))
        
#         context_window = kwargs.get("context_window") or hf_context_window
#         # Ensure max_new_tokens is safe
#         max_new_tokens = kwargs.get("max_new_tokens") or int(context_window * 0.1)
        
#         logger.info(f"vLLM Loading: {model_str} | Derived Context: {context_window}")

#         # 2. Map HF settings to vLLM Engine Args
#         # Note: vLLM uses 'dtype' instead of 'torch_dtype'
#         vllm_engine_kwargs = {
#             "model": model_str,
#             "trust_remote_code": False, # Security guard for recent vulnerabilities
#             "gpu_memory_utilization": kwargs.get("gpu_util", 0.75), # LEAVE ROOM FOR EMBEDDINGS
#             "max_model_len": context_window,
#             "dtype": "auto", 
#             "enforce_eager": True, # Helps with memory stability on single GPUs
#         }

#         # 3. Initialize the Wrapper
#         llm_instance = Vllm(
#             model=model_str,
#             vllm_kwargs=vllm_engine_kwargs,
#             temperature=kwargs.get("temperature", 0.1),
#             max_new_tokens=max_new_tokens,
#             additional_kwargs={
#                 "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
#             }
#         )
        
#         Settings.llm = llm_instance
#         return llm_instance


class APIModelLoader(BaseModelLoader):
    pass