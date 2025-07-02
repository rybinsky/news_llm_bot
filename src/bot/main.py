import os

import streamlit as st
import torch
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings as HFEmbedder
from langchain_ollama.llms import OllamaLLM

from bot.app import generate_response, run_news_updater
from bot.services import EXAMPLES_CLS_TOPIC, DatabaseManager, TopicClassifier, load_config, setup_logging

torch.classes.__path__ = []


FIRST_RUN_FLAG = ".first_run"


def check_first_run() -> bool:
    """Проверяет, первый ли это запуск приложения"""
    if not os.path.exists(FIRST_RUN_FLAG):
        with open(FIRST_RUN_FLAG, "w") as f:
            f.write("1")
        return True
    return False


def main() -> None:
    load_dotenv()

    if check_first_run():
        run_news_updater()

    st.title("🤖 AI-агент: Генератор мемов по последним новостям")

    config = load_config()
    logger = setup_logging(config.logging)

    llm = OllamaLLM(model=config.generator.model, temperature=config.generator.temperature)

    text_embedder = HFEmbedder(
        model_name=config.text_embedder,
        model_kwargs={"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")},
    )

    splitter = CharacterTextSplitter(
        separator="\n\n", chunk_size=500, chunk_overlap=100, length_function=len, is_separator_regex=False
    )

    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)

    db_manager = DatabaseManager(logger)
    classifier = TopicClassifier(
        topics=set(config.classifier.topics),
        example_articles=EXAMPLES_CLS_TOPIC,
        model_name=config.ollama.model,
        temperature=config.classifier.temperature,
        max_attempts=config.classifier.max_attempts,
        logger=logger,
    )

    try:
        db_manager.initialize(
            {
                "user": st.secrets["POSTGRES_USER"],
                "password": st.secrets["POSTGRES_PASS"],
                "host": st.secrets["POSTGRES_HOST"],
                "port": st.secrets["POSTGRES_PORT"],
                "database": st.secrets["POSTGRES_DB"],
            }
        )

        user_query = st.text_input("О чем вы хотите мем/комментарий?", "")

        if user_query:
            with st.spinner("Анализируем запрос и ищем похожие новости..."):
                topic = classifier.classify(user_query)
                news = db_manager.get_last_news_by_topic(topic)
                joke, summaries = generate_response(llm, news, text_embedder, summarize_chain, splitter, user_query)

                st.subheader(f"📰 Новости по теме '{topic}':")
                for summary in summaries:
                    st.write(summary)

                st.markdown("💬 **Мем:**")
                st.success(joke)

    except Exception as e:
        logger.error("❌ Ошибка: %s", str(e), exc_info=True)
        st.error("Произошла ошибка")

    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
