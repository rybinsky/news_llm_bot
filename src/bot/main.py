import os
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import textwrap
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Загрузка переменных окружения
load_dotenv()

# Инициализация Ollama
def init_llm(model_name="gemma3:1b"):
    return Ollama(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=model_name,
        temperature=0.7,
    )

# Получение последних новостей
def get_news(topics=["technology", "politics", "entertainment"], num_news=5):
    news_items = []
    try:
        for topic in topics:
            url = f"https://news.google.com/topics/CAAqIQgKIhtDQkFTRGdvSUwyMHZNRFppYm5vU0FuSjFLQUFQAQ?hl=ru&gl=RU&ceid=RU%3Aru"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "lxml-xml")  # Используем lxml для XML
            items = soup.find_all("item")[:num_news]
            for item in items:
                news_items.append({
                    "title": item.title.text,
                    "link": item.link.text,
                    "source": item.source.text,
                    "topic": topic
                })
    except Exception as e:
        st.error(f"Error fetching news: {e}")
    return news_items

# Генерация идеи для мема на основе новости
def generate_meme_idea(news_item, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты — генератор вирусных мемов на русском языке. У тебя отличное чувство юмора, ты видишь места для него.
        Я дам тебе новость, а ты на основе новости придумай:
        1. Острую первую фразу (верх мема)
        2. Неожиданную вторую фразу (низ мема)
        3. Короткое объяснение юмора

        Формат ответа:
        Верх: [фраза 1]
        Низ: [фраза 2]
        Соль: [объяснение юмора]"""),
        ("user", "Новость: {news_title}\nИсточник: {news_source}\nТема: {news_topic}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "news_title": news_item["title"],
        "news_source": news_item["source"],
        "news_topic": news_item["topic"]
    })
    return result

# Создание мема (заглушка - в реальном приложении нужно добавить генерацию изображений)
def create_meme_image(top_text, bottom_text, format_suggestion):
    # В реальном приложении здесь будет код для генерации мема
    # Это примерная реализация с созданием простого изображения
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Разбиваем текст на строки
    top_lines = textwrap.wrap(top_text, width=20)
    bottom_lines = textwrap.wrap(bottom_text, width=20)
    
    # Рисуем верхний текст
    y_text = 10
    for line in top_lines:
        w, h = d.textsize(line, font=font)
        d.text(((width - w) / 2, y_text), line, font=font, fill=(0, 0, 0))
        y_text += h + 10
    
    # Рисуем нижний текст
    y_text = height - 100
    for line in bottom_lines:
        w, h = d.textsize(line, font=font)
        d.text(((width - w) / 2, y_text), line, font=font, fill=(0, 0, 0))
        y_text += h + 10
    
    # Добавляем информацию о формате
    d.text((10, height - 30), f"Suggested format: {format_suggestion}", font=ImageFont.load_default(), fill=(150, 150, 150))
    
    return img

# Основное приложение Streamlit
def main():
    st.title("🎭 AI-агент для генерации трендовых мемов")
    st.markdown("Анализируем последние новости и создаем на их основе вирусные мемы")
    
    # Выбор модели
    model_name = st.sidebar.selectbox(
        "Выберите модель",
        ["gemma3:1b"],
        index=0
    )
    
    # Получение новостей
    if st.sidebar.button("Получить свежие новости"):
        st.session_state.news = get_news()
    
    if "news" not in st.session_state:
        st.session_state.news = []
    
    if st.session_state.news:
        st.subheader("Последние новости")
        news_index = st.selectbox(
            "Выберите новость для мема",
            range(len(st.session_state.news)),
            format_func=lambda x: st.session_state.news[x]["title"]
        )
        selected_news = st.session_state.news[news_index]
        
        if st.button("Сгенерировать мем на основе этой новости"):
            with st.spinner("Генерируем идею для мема..."):
                llm = init_llm(model_name)
                meme_idea = generate_meme_idea(selected_news, llm)
                st.session_state.meme_idea = meme_idea
            
        if "meme_idea" in st.session_state:
            st.subheader("Идея для мема")
            st.text(st.session_state.meme_idea)
            
            # Парсинг ответа
            try:
                parts = {
                    "Caption": "",
                    "Punchline": "",
                    "Format": "",
                    "Context": ""
                }
                current_part = None
                
                for line in st.session_state.meme_idea.split("\n"):
                    if ":" in line:
                        part_name, content = line.split(":", 1)
                        part_name = part_name.strip()
                        if part_name in parts:
                            current_part = part_name
                            parts[current_part] = content.strip()
                        else:
                            if current_part:
                                parts[current_part] += "\n" + line
                    elif current_part:
                        parts[current_part] += "\n" + line
                print(parts)
                # Создание мема
                if parts["Caption"] and parts["Punchline"]:
                    with st.spinner("Создаем мем..."):
                        meme_img = create_meme_image(
                            parts["Caption"],
                            parts["Punchline"],
                            parts.get("Format", "Unknown format")
                        )
                        st.image(meme_img, caption=f"Мем на тему: {selected_news['title']}")
                        
                        st.markdown("**Почему это смешно:**")
                        st.write(parts.get("Context", "Объяснение отсутствует"))
            except Exception as e:
                st.error(f"Ошибка при создании мема: {e}")
    else:
        st.info("Нажмите 'Получить свежие новости' в боковой панели, чтобы начать")

if __name__ == "__main__":
    main()