import os

import streamlit as st
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM

from bot.services import EXAMPLES_CLS_TOPIC, DatabaseManager, TopicClassifier, load_config, setup_logging


def geenrate_response() -> None:
    pass


def main():
    """Main application entry point."""
    load_dotenv()
    st.title("🤖 AI-агент: Генератор мемов по последним новостям")
    st.markdown("Введите запрос на любом языке")

    config = load_config()

    logger = setup_logging(config.logging)

    db_manager = DatabaseManager(logger)

    classifier = TopicClassifier(
        topics=set(config.classifier.topics),
        example_articles=EXAMPLES_CLS_TOPIC,
        model_name=config.ollama.model,
        temperature=config.classifier.temperature,
        max_attempts=config.classifies.max_attempts,
        logger=logger,
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
                response = geenrate_response()

                st.subheader(f"Читаем новости по теме '{topic}'...")
                st.markdown("💬 **Мем:**")
                st.write(response)

    except Exception as e:
        logger.error("Application error: %s", str(e), exc_info=True)

    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
