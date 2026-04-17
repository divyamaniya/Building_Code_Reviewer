import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from huggingface_hub import login
from huggingface_hub import snapshot_download

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

hugging_face_access_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
HUGGING_FACE_MODEL_ID = 'BAAI/bge-large-en-v1.5'
LOCAL_MODEL_DIR = './models/embeddings/' + HUGGING_FACE_MODEL_ID

if not hugging_face_access_token:
    raise ValueError("No API token found. Check your .env file.")

def download_model_if_not_exists(model_id, local_dir):

    if Path(local_dir).exists() and any(Path(local_dir).iterdir()):
        print(f"Model directory '{local_dir}' already exists and contains files. Skipping download.")
        return
    
    print(f"Model not found locally. Starting download of {model_id} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    login(token=hugging_face_access_token)

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

    print("Download complete.")

def load_model(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
        print(f"Model and tokenizer loaded successfully from {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit()

def try_model(model_path):
    try:
        model, tokenizer = load_model(model_path)
        # Ensure tokenizer has a pad token to avoid the warning
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Fallback template if missing
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<start_of_turn>model\n' }}"
                "{% endif %}"
            )

        print("Start chatting (type 'exit' or 'quit' to end): ")
        chat_history = []
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", 'quit']:
                break

            chat_history.append({"role":"user", "content":user_input})
            stop_tokens = [tokenizer.eos_token, "<|im_end|>"]
            
            text = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9,
                                            repetition_penalty=1.2,
                                            # pad_token_id=tokenizer.pad_token_id,
                                            # eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id)
                                            eos_token_id=tokenizer.eos_token_id)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()

            print(f"Bot: {response}")
            chat_history.append({"role": "model", "content": response})

    except Exception as e:
        print(f"Error from try_model fun: {e}")
        sys.exit()

if __name__ == "__main__":
    download_model_if_not_exists(HUGGING_FACE_MODEL_ID, LOCAL_MODEL_DIR)
    # try_model(LOCAL_MODEL_DIR)