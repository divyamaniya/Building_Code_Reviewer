from llama_index.core import PromptTemplate, ChatPromptTemplate
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
import logging
from llama_index.core import Settings

logger = logging.getLogger(__name__)

# evaluation: (Faithfulness/Correctness)

class GenerationPipeline:
    def __init__(self, retriever_module, config):
        self.config = config

        # Converts your raw string from config into a LlamaIndex Prompt object.
        # qa_prompt_tmpl_str = (
        #     "Context infomration is below. \n"
        #     "-------------------\n"
        #     "{context_str}\n"
        #     "-------------------\n"
        #     "Given the context information and not prior knowledge, "
        #     "answer the query: {query_str}\n"
        #     "Answer: "
        # )
        # self.qa_prompt = PromptTemplate(qa_prompt_tmpl_str)
        # logger.debug(f"Given prompt is {config.prompt_detail.text}, qa_prompt: {self.qa_prompt}")

        chat_text_qa_msgs = self.get_formatted_chat_messages(config)
        self.chat_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
        logger.debug(f"ChatPromptTemplate: {self.chat_qa_template}")


        # this tells how to bundle text if it's too long for the LLM, compact mmode stuffs as many as node possible into one prompt
        self.response_synthesizer= get_response_synthesizer(
            response_mode= "compact",   # e.g compact or refine: if retrieved context is larger than the context window, it will try to refine and make multiple llm calls
                                        # if want to stick with single call tree_summarize, use tree_summarize to ensure context_window
            text_qa_template=self.chat_qa_template
            # Can also pass refine_template
        )

        # building list of postprocessors dynamically based on the toggle
        node_postprocessors = []
        if hasattr(retriever_module, 'reranker'):
            logger.debug("post processing using reranker.....")
            node_postprocessors.append(retriever_module.reranker)

        # This connects the retriever to the answer/synthesizer
        # for chatboat we need to add memory so we can use ChatEngine or CondenseQuestionChatEngine instead of RetrieverQueryEngine
        # CondenseQuestionChatEngine; takes chat history + new message and create single standalone query
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever= retriever_module.retriever,
            response_synthesizer= self.response_synthesizer,
            node_postprocessors=node_postprocessors        # actual object here
        )

    def get_formatted_chat_messages(self, config):
        # some llm have system role that uses System prompt using System tag or assistant tag; chat ml
        # some uses Instruction tags that contains system prompt + user message
        # all model have their own special tags INST, system, assistant... and this is define in tokenizer_config.json
        # So we need to prepare template according to model

        llm = Settings.llm   # get tokenizer
        tokenizer = None
        supports_system = None
        try:
            if hasattr(llm, "_client"):     # vLLM Case
                tokenizer = llm._client.llm_engine.tokenizer.tokenizer
            elif hasattr(llm, 'tokenizer'):       # HF Case
                tokenizer = llm.tokenizer
            elif hasattr(llm, '_model') and hasattr(llm._model, 'tokenizer'):
                tokenizer = llm._model.tokenizer
            elif hasattr(llm, '_tokenizer'):        # Try the protected _tokenizer attribute
                tokenizer = llm._tokenizer
            logger.debug(f"found tokenizer: {tokenizer}")
            
            # detect chat template from tokenizer and/or llm
            chat_template = ""          # Extract the template string
            if tokenizer and hasattr(tokenizer, "chat_template"):
                chat_template = tokenizer.chat_template or ""
            logger.debug(f"chat_template from tokenizer: {chat_template}")

            # If still empty, check if the LLM object itself has a template attribute
            if not chat_template and hasattr(llm, "chat_template"):
                chat_template = llm.chat_template or ""
            logger.debug(f"chat_template from llm: {chat_template}")

            # 2. THE FIX: Advanced Detection
            # We check for the word 'system' AND ensure it's not followed by 'not supported' or 'raise_exception'
            has_system_word = "system" in chat_template.lower()
            
            # Specific patterns used by Google/Mistral to block system roles
            is_explicitly_unsupported = (
                "system role not supported" in chat_template.lower() or 
                "system role is unsupported" in chat_template.lower() or
                "raise_exception('system" in chat_template.lower().replace(" ", "")
            )

            # Actual support means the word is there and it's NOT explicitly blocked
            supports_system = has_system_word and not is_explicitly_unsupported

            logger.debug(f"Template Analysis -> Word present: {has_system_word}, Explicitly blocked: {is_explicitly_unsupported}")
            logger.info(f"Final Decision -> supports_system: {supports_system}")

        except Exception as e:
            logger.debug(f"Tokenizer or Template extraction failed: {e}.")
            supports_system = False

        # Get system prompt from config using getattr (OmegaConf override access config... in getattr so commenting out)
        prompt_detail = config.selected_prompt_detail
        system_instruction = prompt_detail.get("system", "You are a helpful assistant.")
        user_tmpl = prompt_detail.get("user_tmpl", "context: {context_str} \nQuestion: {query_str}")

        if supports_system:
            # Standard: System -> User
            return [
                ChatMessage(role=MessageRole.SYSTEM, content=system_instruction),
                ChatMessage(role=MessageRole.USER, content=user_tmpl)
            ]
        else:
            # Merged: User (Instruction + Context)
            # We prefix the user message with the instructions to satisfy the 
            # 'must start with user' and 'must alternate' rules.
            combined_content = f"INSTRUCTIONS:\n{system_instruction}\n\nINPUT:\n{user_tmpl}"
            return [
                ChatMessage(role=MessageRole.USER, content=combined_content)
            ]

    def handle_conversation(self, query_str: str) -> str:
        """
        Handles general conversation and off-topic queries using the 
        native LLM instance while keeping the formatting clean.
        """
        from llama_index.core.base.llms.types import MessageRole, ChatMessage
        
        chat_messages = [
            ChatMessage(
                role=MessageRole.SYSTEM, 
                content="You are a helpful, polite AI assistant dedicated strictly to helping users with the Ontario Building Code. Answer greetings naturally and briefly guide users to building code inquiries. For off-topic questions, politely remind them of your specialization."
            ),
            ChatMessage(
                role=MessageRole.USER, 
                content=query_str
            )
        ]
        
        response = Settings.llm.chat(chat_messages)
        return str(response.message.content).strip()


    def answer(self, query_str: str):
        logger.debug(f"Question asked: {query_str}")
        # this is blocking it stops the whole program until GPU finishes
        return self.query_engine.query(query_str)
    
    async def aanswer(self, query_str: str):
        logger.debug(f"Question asked (async): {query_str}")
        return await self.query_engine.aquery(query_str)