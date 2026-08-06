import pandas as pd
from src.rag_pipeline import run_rag_pipeline
from .experiment_manager import ExperimentManager
import logging
import re

from llama_index.core.evaluation import(
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    BatchEvalRunner
)

# from ragas import evaluate
# from datasets import Dataset
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     context_precision,
#     context_recall,
# )
# from ragas.llms import LlamaIndexLLMWrapper
# from ragas.embeddings import LlamaIndexEmbeddingsWrapper

# from sentence_transformers import SentenceTransformer, util
from src.utils.loader import LocalModelLoader, vLLMLocalModelLoader
import asyncio

logger = logging.getLogger(__name__)


# chunk_strategy = semantic, recursive, 
# import os
# os.environ["RAGAS_DO_NOT_TRACK"] = "True"
# similarity_model = SentenceTransformer('all-MiniLM-L6-v2')


import asyncio
import pandas as pd
import logging
from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)

async def process_stage1_item(row, generative_module):
    """Asynchronously run the RAG pipeline for a single row."""
    question = row.get('Question with Option')
    
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    query_bundle = QueryBundle(
        query_str=question,
       custom_embedding_strs=[f"{BGE_QUERY_PREFIX}{question}"]
    )
    # Use the async variant if available (e.g., aquery or aanswer)
    # If your generative_module doesn't support async, remove asyncio.to_thread
    # response_obj = await asyncio.to_thread(generative_module.answer, question)
    response_obj = await generative_module.query_engine.aquery(query_bundle)
    
    logger.info(f"Number of nodes sent to LLM: {len(response_obj.source_nodes)}")
    return {
        "question": question,
        "answer": response_obj.response,
        "contexts": [node.node.get_content() for node in response_obj.source_nodes],
        "ground_truth": str(row.get('Answer')),
        "ref_text": str(row.get('Extracted Text'))
    }

async def run_stage1_async(eval_rows, generative_module):
    """Manager: Throttles and batches all workers concurrently."""
    # Allows vLLM to batch up to 8 queries at the exact same time
    semaphore = asyncio.Semaphore(4) 
    
    async def sem_worker(row):
        async with semaphore:
            return await process_stage1_item(row, generative_module)
            
    # Gather and run everything
    return await asyncio.gather(*(sem_worker(row) for row in eval_rows))

async def evaluate_single_item(item, evaluators):
    """Asynchronously evaluate a single item across all metrics concurrently."""
    # Run all 4 evaluation metrics for this item in parallel
    f_task = evaluators['faithfulness'].aevaluate(
        query=item['question'], response=item['answer'], contexts=item['contexts']
    )
    r_task = evaluators['relevancy'].aevaluate(
        query=item['question'], response=item['answer'], contexts=item['contexts']
    )
    c_task = evaluators['correctness'].aevaluate(
        query=item['question'], response=item['answer'], reference=item['ground_truth']
    )
    s_task = evaluators['similarity'].aevaluate(
        query=item['question'], response=item['answer'], reference=item['ref_text']
    )

    f_result, r_result, c_result, s_result = await asyncio.gather(f_task, r_task, c_task, s_task)

    return {
        "question": item['question'],
        "answer": item['answer'],
        "contexts": item['contexts'],
        "ground_truth": item['ground_truth'],
        "faithfulness": float(f_result.passing),
        "answer_relevancy": float(r_result.passing),
        "correctness": c_result.score / 5.0,
        "context_recall": s_result.score,
        "context_precision": float(r_result.passing),
        "retrieval_hit": 1 if s_result.score > 0.85 else 0,
        "mrr": 1.0 if s_result.score > 0.85 else 0.0
    }


def robust_correctness_parser(eval_response: str):
    """Safely extracts score and reasoning using regex instead of strict split."""
    # Clean up response spacing
    response_text = eval_response.strip()
    
    # Regex to locate 'Score: X.X'
    score_match = re.search(r"Score:\s*([0-9.]+)", response_text, re.IGNORECASE)
    # Regex to locate 'Reasoning: ...'
    reasoning_match = re.search(r"Reasoning:\s*(.*)", response_text, re.IGNORECASE)
    
    # Fallbacks if parsing behaves unexpectedly 
    score = float(score_match.group(1)) if score_match else 1.0
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text
    
    return score, reasoning

def run_experiment_evaluation_async_try_2(cfg):
    manager = ExperimentManager(cfg)
    manager.get_or_init_index()
    
    retrieval_module, generative_module = run_rag_pipeline('na', cfg)
    eval_df = pd.read_csv("eval/Eval_with_Ref_text.csv")
    
    # Drop pandas row tracking overhead
    eval_rows = eval_df.to_dict('records')

    # --- STAGE 1: BATCH GENERATION ---
    logger.info("Starting Stage 1: Async RAG Generation")
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        intermediate_data = loop.run_until_complete(run_stage1_async(eval_rows, generative_module))
    else:
        intermediate_data = asyncio.run(run_stage1_async(eval_rows, generative_module))
        # Re-get the loop if we need to run Stage 2 with it
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    # Cleanup Stage 1
    del generative_module, retrieval_module
    flush_memory()
    vLLMLocalModelLoader().kill_process_by_name()
    logger.info("RAG models cleared. VRAM flushed.")

    # --- STAGE 2: EVALUATION ---
    logger.info("Starting Stage 2: Async LLM Evaluation")
    
    judge_llm = vLLMLocalModelLoader().load_llm(cfg.paths.llm_judge, temperature=0.0)
    judge_embedding = vLLMLocalModelLoader().load_embedding(cfg.paths.embedding_judge)
    
    evaluators = {
        "faithfulness": FaithfulnessEvaluator(llm=judge_llm),
        "relevancy": RelevancyEvaluator(llm=judge_llm),
        "correctness": CorrectnessEvaluator(llm=judge_llm),
        "similarity": SemanticSimilarityEvaluator(embed_model=judge_embedding)
    }

    from llama_index.core.prompts import PromptTemplate
    CUSTOM_CORRECTNESS_TEMPLATE = PromptTemplate(
        "You are a strict grading assistant.\n"
        "Compare the Generated Answer against the Reference Answer for the given question.\n\n"
        "Question: {query}\n"
        "Reference Answer: {reference_answer}\n"
        "Generated Answer: {generated_answer}\n\n"
        "Give a score between 1.0 and 5.0, where 5.0 means perfectly correct and 1.0 means completely wrong or irrelevant.\n\n"
        "Return your output EXACTLY in this format, with no extra text:\n"
        "Score: [Your numerical score here]\n"
        "Reasoning: [One sentence explaining why]\n"
    )
    

    evaluators["correctness"].update_prompts({"parser_prompt": CUSTOM_CORRECTNESS_TEMPLATE})
    evaluators["correctness"].parser_function = robust_correctness_parser

    # Run evaluations concurrently
    async def run_stage2():
        semaphore = asyncio.Semaphore(4)
        async def sem_task(item):
            async with semaphore:
                return await evaluate_single_item(item, evaluators)
        return await asyncio.gather(*(sem_task(item) for item in intermediate_data))

    if loop and loop.is_running():
        results_data = loop.run_until_complete(run_stage2())
    else:
        results_data = asyncio.run(run_stage2())

    # Final Cleanup
    del evaluators, judge_llm, judge_embedding
    flush_memory()
    
    return pd.DataFrame(results_data)


def flush_memory():
    import gc
    import torch

    from llama_index.core import Settings
    if hasattr(Settings.llm, "_client") and Settings.llm._client:
        try:
            logger.debug("shuttingdown.....")
            Settings.llm._client.llm_engine.shutdown()
        except Exception:
            pass
    
    logger.debug("Setting llm and embed_model: None")
    Settings.llm = None
    Settings.embed_model = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


async def run_experiment_evaluation_async(cfg):
    manager = ExperimentManager(cfg)
    manager.get_or_init_index()

    retrieval_module, generative_module = run_rag_pipeline('na',cfg)
    eval_df = pd.read_csv("eval/Eval_with_Ref_text.csv")#.head(20)

    logger.info(f"Starting Stage 1: Async Generation for {len(eval_df)} rows")

    gen_sem = asyncio.Semaphore(32)

    async def sem_generate(row):
        async with gen_sem:
            return await generative_module.aanswer(row['Question with Option'])

    # create a list of tasks (coroutines)
    tasks = [sem_generate(row) for _, row in eval_df.iterrows()]

    # run all generations concurrently; multiple requests hit the model at once
    responses = await asyncio.gather(*tasks)

    intermediate_data = []
    for i, response_obj in enumerate(responses):
        row = eval_df.iloc[i]
        intermediate_data.append({
            "question": row['Question with Option'],
            "answer": response_obj.response,
            "contexts": [node.node.get_content() for node in response_obj.source_nodes],
            "ground_truth": str(row['Answer']),
            "ref_text": str(row['Extracted Text'])
        })

    # CLEANUP: Free VRAM for the Judge Model
    del generative_module, retrieval_module
    flush_memory()
    logger.info("Generation complete. Models cleared for Evaluation Stage.")

    import json
    try:
        with open("eval/intermediate_data.json", "w") as f:
            json.dump(intermediate_data, f, indent=4)
        logger.info("Saved Stage 1 results to eval/intermediate_data.json")
    except: 
        logger.debug("Error in saving intermediate_data")

    # # --- STAGE 2: BATCH EVALUATION ---
    logger.info("Starting Stage 2: LLM Evaluation")
    judge_llm = vLLMLocalModelLoader().load_llm(cfg.paths.llm_judge, temperature=0.0)
    judge_embedding = vLLMLocalModelLoader().load_embedding(cfg.paths.embedding_judge)

    evaluators = {
        "faithfulness": FaithfulnessEvaluator(llm= judge_llm),
        "relevancy": RelevancyEvaluator(llm= judge_llm),
        "correctness": CorrectnessEvaluator(llm= judge_llm)
    }
    similarity_eval = SemanticSimilarityEvaluator(embed_model=judge_embedding)

    queries = [item['question'] for item in intermediate_data]
    responses = [item['answer'] for item in intermediate_data]
    contexts_list = [item['contexts'] for item in intermediate_data]
    ground_truths = [item['ground_truth'] for item in intermediate_data]

    logger.info(f"Running Batch Evaluation for {len(intermediate_data)} items...")

    eval_sem = asyncio.Semaphore(30)

    async def sem_eval(i):
        async with eval_sem:
            q_preview = intermediate_data[i]['question'][:30].replace('\n', ' ')
            logger.debug(f"[Eval Start] Row {i}: {q_preview}..., {queries[i][:30]}")

            f_task = evaluators["faithfulness"].aevaluate(
                query=queries[i], response=responses[i], contexts=contexts_list[i]
            )
            r_task = evaluators["relevancy"].aevaluate(
                query=queries[i], response=responses[i], contexts=contexts_list[i]
            )
            c_task = evaluators["correctness"].aevaluate(
                query=queries[i], response=responses[i], reference=ground_truths[i]
            )
            results = await asyncio.gather(f_task, r_task, c_task)
            logger.info(
                f"[Eval Complete] Row {i} | Faith: {results[0].passing} | "
                f"Rel: {results[1].passing} | Corr Score: {results[2].score}"
            )
            return results
        
    # Execute all evaluation tasks
    eval_tasks = [sem_eval(i) for i in range(len(intermediate_data))]
    all_eval_results = await asyncio.gather(*eval_tasks)

    results_data = []
    for i, item in enumerate(intermediate_data):
        # 4. Context Recall / Similarity 
        # This uses 'ref_text' (the extracted text from your CSV)
        s_result = similarity_eval.evaluate(
            query=item['question'],
            response=item['answer'],
            reference=item['ref_text']
        )

        f_res, r_res, c_res = all_eval_results[i]

        raw_correctness = 0.0
        if c_res is not None and c_res.score is not None:
            raw_correctness = c_res.score
        else:
            logger.warning(f"Score is still None for row {i} (Question: {item['question'][:30]}...)")

        results_data.append({
            "question": item['question'],
            "answer": item['answer'],
            "contexts": item['contexts'],
            "ground_truth": item['ground_truth'],
            "ref_text": item['ref_text'],
            "faithfulness": float(f_res.passing),
            "answer_relevancy": float(r_res.passing),
            "correctness": float(raw_correctness) / 5.0, # Normalizing LlamaIndex 1-5 scale
            "context_recall": s_result.score,
            "retrieval_hit": 1 if s_result.score > 0.85 else 0,
            "mrr": 1.0 if s_result.score > 0.85 else 0.0
        })


    return pd.DataFrame(intermediate_data)



def run_experiment_evaluation(cfg):
    manager = ExperimentManager(cfg)
    manager.get_or_init_index()
    
    retrieval_module, generative_module = run_rag_pipeline('na',cfg)
    eval_df = pd.read_csv("eval/Eval_with_Ref_text.csv")

    intermediate_data = []
    for i, row in eval_df.iterrows():
        question = row['Question with Option']
        ground_truth_ref = str(row['OBC reference'])
        ground_truth_text = str(row['Extracted Text'])
        ground_truth_ans = str(row['Answer'])
        

        response_obj = generative_module.answer(question)
        actual_response_text = response_obj.response
        retrieved_contexts = [node.node.get_content() for node in response_obj.source_nodes]

        intermediate_data.append({
            "question": question,
            "answer": actual_response_text,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truth_ans,
            "ref_text": ground_truth_text
        })

        del response_obj
        if i % 2 == 0: # Flush every 2 rows to keep VRAM stable
            flush_memory()


    generative_module = None
    retrieval_module = None
    
    del generative_module
    del retrieval_module
    flush_memory()
    logger.info("RAG models cleared. VRAM flushed.")

    # --- STAGE 2: EVALUATION ---
    logger.info("Starting Stage 2: LLM Evaluation")
    
    judge_llm = LocalModelLoader().load_llm(cfg.paths.llm_judge, temperature=0.0)
    judge_embedding = LocalModelLoader().load_embedding(cfg.paths.embedding_judge)
    
    evaluators = {
        "faithfulness": FaithfulnessEvaluator(llm= judge_llm),
        "relevancy": RelevancyEvaluator(llm= judge_llm),
        "correctness": CorrectnessEvaluator(llm= judge_llm),
        "similarity": SemanticSimilarityEvaluator(embed_model=judge_embedding)
    }

    from llama_index.core.prompts import PromptTemplate
    CUSTOM_CORRECTNESS_TEMPLATE = PromptTemplate(
        "You are a strict grading assistant.\n"
        "Compare the Generated Answer against the Reference Answer for the given question.\n\n"
        "Question: {query}\n"
        "Reference Answer: {reference_answer}\n"
        "Generated Answer: {generated_answer}\n\n"
        "Give a score between 1.0 and 5.0, where 5.0 means perfectly correct and 1.0 means completely wrong or irrelevant.\n\n"
        "Return your output EXACTLY in this format, with no extra text:\n"
        "Score: [Your numerical score here]\n"
        "Reasoning: [One sentence explaining why]\n"
    )

    # 3. Force the correctness evaluator to use the lightweight template
    evaluators["correctness"].update_prompts(
        {"parser_prompt": CUSTOM_CORRECTNESS_TEMPLATE}
    )

    results_data = []

    for item in intermediate_data:
        # Faithfulness: answer vs context
        f_result = evaluators['faithfulness'].evaluate(
            query=item['question'],
            response=item['answer'],
            contexts=item['contexts']
        )

        # Relevancy: Answer vs Query
        r_result = evaluators['relevancy'].evaluate(
            query=item['question'],
            response=item['answer'],
            contexts=item['contexts']
        )

        # Correctness: Answer vs Ground Truth
        c_result = evaluators['correctness'].evaluate(
            query=item['question'],
            response=item['answer'],
            reference=item['ground_truth']
        )

        # 4. Context Recall (Check if ground_truth_text exists in retrieved chunks)
        # We simulate Context Recall by checking similarity between GT Text and retrieved nodes
        s_result = evaluators["similarity"].evaluate(
            query=item['question'],
            response=item['answer'],
            reference=item['ref_text'],
        )

        results_data.append({
            "question": item['question'],
            "answer": item['answer'],
            "contexts": item['contexts'],
            "ground_truth": item['ground_truth'],
            "faithfulness": float(f_result.passing),
            "answer_relevancy": float(r_result.passing),
            "correctness": c_result.score / 5.0,
            "context_recall": s_result.score,
            "context_precision": float(r_result.passing),
            "retrieval_hit": 1 if s_result.score > 0.85 else 0,
            "mrr": 1.0 if s_result.score > 0.85 else 0.0
        })

        # --- MEMORY MANAGEMENT ---
        # Explicitly delete the result objects to free up their tensors/references
        del f_result, r_result, c_result, s_result

        # Flush every 2 items just like in Stage 1
        if i % 2 == 0:
            flush_memory()
            logger.debug(f"Evaluated {i+1}/{len(intermediate_data)} items. VRAM Flushed.")

    # Final Cleanup
    evaluators = None
    del judge_llm
    del judge_embedding
    flush_memory()
    
    return pd.DataFrame(results_data)
    # to Log everything into MLflow we will return results



#  - torch==2.6.0+cu124
#  + torch==2.11.0
#  + torch-c-dlpack-ext==0.1.5
#  - torchaudio==2.6.0+cu124
#  + torchaudio==2.11.0
#  - torchvision==0.21.0+cu124
#  + torchvision==0.26.0