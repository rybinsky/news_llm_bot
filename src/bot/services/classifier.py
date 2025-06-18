import logging
from typing import Any, Dict, List, Optional, Set

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama.llms import OllamaLLM

from .logging import CustomLogger


class NewsClassifier:
    def __init__(
        self,
        topics: Set[str],
        example_articles: List[Dict[str, str]],
        model_name: str = "llama2",
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.topics = topics
        self.example_articles = example_articles
        self.model_name = model_name
        self.temperature = temperature
        self.logger = logger or CustomLogger(__name__).get_logger()
        self.chain = self._initialize_chain()

    def _initialize_chain(self) -> RunnableSerializable:
        """Initialize the LLM chain for topic classification."""
        example_prompt = PromptTemplate(
            input_variables=["text", "category"],
            template="Текст: {text}\nТема: {category}",
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=self.example_articles,
            example_prompt=example_prompt,
            prefix="Определи самую подходящую тематику текста из списка и выведи только ее без других символов: "
            + ", ".join(self.topics),
            suffix="Текст: {input}\nТема:",
            input_variables=["input"],
        )

        llm = OllamaLLM(model=self.model_name, temperature=self.temperature)
        return few_shot_prompt | llm

    def classify(self, text: str, max_attempts: int = 3) -> str:
        """Classify news article text into one of the predefined topics."""
        attempt = 0
        while attempt < max_attempts:
            try:
                answer = self.chain.invoke({"input": text}).strip()
                if answer in self.topics:
                    return answer
                self.logger.warning("Invalid topic received: %s", answer)
            except Exception as e:
                self.logger.error("Classification attempt %d failed: %s", attempt + 1, str(e))
            attempt += 1

        self.logger.warning("Failed to classify after %d attempts", max_attempts)
        return "Другое"
