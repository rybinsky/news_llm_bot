import os
import textwrap
from typing import Dict

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image, ImageDraw, ImageFont

# Загрузка переменных окружения
load_dotenv()


class NewsAIAgent:
    def __init__(self):
        self.available_topics = ["technology", "politics", "entertainment"]
        self.llm_model = "llama3.1:8b"
        self.llm = self._init_llm()

    def _init_llm(self, temperature: float = 0.7) -> Ollama:
        """Инициализация языковой модели"""
        return Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=self.llm_model,
            temperature=temperature,
        )

    def _classify_query_topic(self, user_query: str) -> str:
        """Определение тематики пользовательского запроса (на английском)"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
             Identify the main topic of the user's query from these options:
             {', '.join(self.available_topics)}.
             Return only the topic name without additional explanations.
             """,
                ),
                ("user", "Query: {query}"),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        topic = chain.invoke({"query": user_query}).strip()
        return topic if topic in self.available_topics else "Other"

    def _get_english_news(self, topic: str, num_news: int = 3) -> list[dict]:
        """Получение англоязычных новостей по заданной теме"""
        news_items = []
        try:
            url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "lxml-xml")

            for item in soup.find_all("item")[:num_news]:
                news_items.append(
                    {
                        "title": item.title.text,
                        "link": item.link.text,
                        "source": item.source.text if item.source else "Unknown source",
                        "topic": topic,
                    }
                )
        except Exception as e:
            st.error(f"Ошибка при получении новостей: {e}")
        return news_items

    def _generate_russian_response(self, news_item: Dict) -> Dict:
        """Генерация ответа на русском на основе английской новости"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    textwrap.dedent(
                        """
                Ты — креативный генератор контента для русскоязычной аудитории.
                На основе английской новости создай на русском языке:
                1. Мем (верхняя и нижняя фраза)
                2. Острый комментарий
                3. Короткий саркастичный вывод

                Формат:
                Мем: [верхняя фраза] | [нижняя фраза]
                Комментарий: [текст]
                Вывод: [текст]
            """
                    ),
                ),
                ("user", "Английская новость: {title}\nИсточник: {source}"),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"title": news_item["title"], "source": news_item["source"]})

        # Парсинг результата
        response = {"Мем": "", "Комментарий": "", "Вывод": ""}
        current_key = None

        for line in result.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                if key in response:
                    current_key = key
                    response[current_key] = value.strip()
            elif current_key:
                response[current_key] += " " + line.strip()

        return response

    def process_user_request(self, user_query: str) -> Dict:
        """Основной метод обработки пользовательского запроса"""
        # Определяем тему на английском
        topic = self._classify_query_topic(user_query)
        st.info(f"Определена тема: {topic}")

        # Получаем англоязычные новости
        news_items = self._get_english_news(topic)
        if not news_items:
            return {"error": "Не удалось найти новости по данной теме"}

        # Выбираем новость
        selected_news = news_items[0]

        # Генерируем русскоязычный контент
        content = self._generate_russian_response(selected_news)

        return {
            "topic": topic,
            "news": selected_news,
            "content": content,
        }


def main():
    st.title("🤖 Англоязычный новостной AI-агент с русскими ответами")
    st.markdown("Введите запрос на любом языке и получите мем/комментарий на русском на основе англоязычных новостей")

    # Инициализация агента
    agent = NewsAIAgent()

    # Пользовательский ввод
    user_query = st.text_input("О чем вы хотите мем/комментарий?", "")

    if user_query:
        with st.spinner("Анализируем запрос и ищем англоязычные новости..."):
            result = agent.process_user_request(user_query)

            if "error" in result:
                st.error(result["error"])
                return

            # Отображение результатов
            st.subheader(f"Английская новость по теме '{result['topic']}':")
            st.markdown(f"**{result['news']['title']}**")
            st.caption(f"Источник: {result['news']['source']}")

            # Разделение на две колонки
            col2 = st.columns(1)

            with col2:
                st.subheader("Анализ на русском")
                if result["content"]["Комментарий"]:
                    st.markdown("💬 **Комментарий:**")
                    st.write(result["content"]["Комментарий"])

                if result["content"]["Вывод"]:
                    st.markdown("🔍 **Вывод:**")
                    st.write(result["content"]["Вывод"])

            # Кнопка для генерации нового контента
            if st.button("Сгенерировать другой вариант"):
                with st.spinner("Создаем новый вариант..."):
                    new_content = agent._generate_russian_response(result["news"])
                    if "|" in new_content["Мем"]:
                        st.image(agent._create_meme_image(*new_content["Мем"].split("|")[:2]))
                    st.write("💬 **Новый комментарий:**", new_content["Комментарий"])
                    st.write("🔍 **Новый вывод:**", new_content["Вывод"])


if __name__ == "__main__":
    main()
