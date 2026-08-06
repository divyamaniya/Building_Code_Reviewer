import logging
import logging.handlers
import multiprocessing

# Because ProcessPoolExecutor runs multiple processes and try to write in same debug file at the same time millisecond
# so it is better to use QueueHandler, gloabl queue that coordinate between processes

_log_queue = multiprocessing.Queue()

def setup_primary_logging(file_path):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    logging.getLogger("docling").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("llama_index").setLevel(logging.WARNING)
    logging.getLogger("docling_ibm_models").setLevel(logging.WARNING)
    # logging.getLogger("vllm").setLevel(logging.INFO)
    logging.getLogger("git").setLevel(logging.WARNING)

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.propagate = True

    file_handler = logging.FileHandler(file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s")
    file_handler.setFormatter(formatter)

    listener = logging.handlers.QueueListener(_log_queue, file_handler)
    listener.start()

    queue_handler = logging.handlers.QueueHandler(_log_queue)
    root.addHandler(queue_handler)

    return listener

def worker_configurer():
    h = logging.handlers.QueueHandler(_log_queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)
