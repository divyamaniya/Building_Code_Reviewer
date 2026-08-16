from llama_index.core import Settings
import logging

logger = logging.getLogger(__name__)

class GuardrailPipeline:
    def __init__(self, config=None):
        self.config = config

    def classify_query_intent(self, query_str: str) -> str:
        """
        Classifies the user query into:
        - 'CONVERSATION': Greetings, general pleasantries, how are you, etc.
        - 'OFF_TOPIC': Questions about movies, sports, coding, recipes, etc.
        - 'BUILDING_CODE': Anything related to buildings, construction, permits, OBC, etc.
        """
        prompt = f"""
        Classify the following user query into one of these three categories:
        1. CONVERSATION (greetings, pleasantries, thanking, personal chat)
        2. GENERAL_OR_OFF_TOPIC (greetings, pleasantries, small talk, or topics unrelated to building codes)
        
        User Query: "{query_str}"

        Return ONLY the category name (BUILDING_CODE or GENERAL_OR_OFF_TOPIC). Do not include any other text.        
        """
        
        try:
            response = Settings.llm.complete(prompt)
            intent = response.text.strip().upper()
            logger.debug(f"Query intent classified as: {intent}")
            
            if "BUILDING_CODE" in intent:
                return "BUILDING_CODE"
            else:
                return "GENERAL_OR_OFF_TOPIC"
        except Exception as e:
            logger.error(f"Intent classification failed: {e}. Defaulting to BUILDING_CODE.")
            return "BUILDING_CODE"
        