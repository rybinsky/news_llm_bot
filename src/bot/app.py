import os
import textwrap
from typing import Sequence

import streamlit as st
import torch
from dotenv import load_dotenv
from langchain.schema import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings as HFEmbedder
from langchain_ollama.llms import OllamaLLM

from bot.models import NewsArticle
from bot.services import EXAMPLES_CLS_TOPIC, DatabaseManager, TopicClassifier, load_config, setup_logging

torch.classes.__path__ = []

PROMPT = ChatPromptTemplate.from_template(
    textwrap.dedent(
        """
        Представь что ты стендап-комик и ты очень смешно комментируешь последние новости. За хорошие шутки тебе платят очень хорошие деньги.
        Твоя задача придумать хорошую шутку, опираясь на контекст по теме, которую я тебе дам. Твоя шутка должна быть тонкой и остроумной.
        Шутка должна быть в 1-2 предложения, должна быть смешной и тонкой и опираться на данный контекст!

        Контекст:
        {context}

        Тема: {question}

        Шутка:
    """
    )
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join([d.page_content for d in docs])


def generate_response(
    llm: OllamaLLM, news: Sequence[NewsArticle], embedder: HFEmbedder, splitter: CharacterTextSplitter, query: str
) -> str:
    split_documents = splitter.create_documents([new.text for new in news])
    db = FAISS.from_documents(split_documents, embedder)
    retriever = db.as_retriever()
    chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | PROMPT | llm | StrOutputParser()
    return chain.invoke(query)


def main():
    """Main application entry point."""
    load_dotenv()
    st.title("🤖 AI-агент: Генератор мемов по последним новостям")
    st.markdown("Введите запрос на любом языке")

    config = load_config()

    logger = setup_logging(config.logging)

    text_embedder = HFEmbedder(
        model_name=config.text_embedder,
        model_kwargs={"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")},
    )
    db_manager = DatabaseManager(logger)

    classifier = TopicClassifier(
        topics=set(config.classifier.topics),
        example_articles=EXAMPLES_CLS_TOPIC,
        model_name=config.ollama.model,
        temperature=config.classifier.temperature,
        max_attempts=config.classifier.max_attempts,
        logger=logger,
    )

    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    llm = OllamaLLM(model=config.generator.model, temperature=config.generator.temperature)

    try:
        db_manager.initialize(
            {
                "user": os.getenv("POSTGRES_USER"),
                "password": os.getenv("POSTGRES_PASS"),
                "host": os.getenv("POSTGRES_HOST"),
                "port": os.getenv("POSTGRES_PORT"),
                "database": os.getenv("POSTGRES_DB"),
            }
        )

        user_query = st.text_input("О чем вы хотите мем/комментарий?", "")

        if user_query:
            with st.spinner("Анализируем запрос и ищем похожие новости..."):
                topic = classifier.classify(user_query)
                news = db_manager.get_last_news_by_topic(topic)
                response = generate_response(llm, news, text_embedder, splitter, user_query)

                st.subheader(f"Читаем новости по теме '{topic}'...")
                st.markdown("💬 **Мем:**")
                st.write(response)

    except Exception as e:
        logger.error("Application error: %s", str(e), exc_info=True)

    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
