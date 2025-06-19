import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional

from newspaper import Article
from newspaper import build as get_last_news
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.orm import sessionmaker

from bot.models import NewsArticle

from .classifier import TopicClassifier
from .logging import CustomLogger


class NewsScraper:
    def __init__(self, logger: Optional[logging.Logger] = None, max_workers: Optional[int] = None) -> None:
        self.logger = logger or CustomLogger(__name__).get_logger()
        self.max_workers = max_workers

    def scrape_article(self, url: str) -> Optional[Article]:
        """Scrape article content from given URL."""
        article = Article(url)
        try:
            article.download()
            article.parse()
            self.logger.debug("Successfully scraped article: %s", url)
            return article
        except Exception as e:
            self.logger.error("Error scraping article %s: %s", url, str(e))
            return None

    def extract_article_data(
        self, article_dict: dict[str, Any], text_field: str, classifier: Optional[TopicClassifier] = None
    ) -> dict[str, Any]:
        """Extract and process article data from raw dictionary."""
        data = {
            "url": article_dict.get("url"),
            "source": article_dict.get("source_url"),
            "title": article_dict.get("title"),
            "text": article_dict.get(text_field),
            "summary": article_dict.get("summary"),
            "publish_date": article_dict.get("publish_date"),
            "language": article_dict.get("meta_lang"),
            "keywords": article_dict.get("keywords", []) + article_dict.get("meta_keywords", []),
        }

        if classifier and data["text"]:
            data["topic"] = classifier.classify(data["text"])
        else:
            data["topic"] = "Другое"

        if data["publish_date"] and isinstance(data["publish_date"], datetime):
            data["publish_date"] = data["publish_date"].isoformat()

        return data

    def store_article(
        self,
        db_session: DBSession,
        article: Article,
        text_field: str,
        classifier: Optional[TopicClassifier] = None,
    ) -> bool:
        """Store article in database if it doesn't exist."""
        if db_session.query(NewsArticle).filter(NewsArticle.url == article.url).first():
            self.logger.info("Article already exists: %s", article.url)
            return False

        try:
            parsed_article = self.extract_article_data(article.__dict__, text_field, classifier)
            new_article = NewsArticle(**parsed_article)
            db_session.add(new_article)
            db_session.commit()
            self.logger.info("Article stored successfully: %s", article.url)
            return True
        except Exception as e:
            db_session.rollback()
            self.logger.error("Error storing article %s: %s", article.url, str(e))
            return False

    def process_single_article(
        self,
        article_url: str,
        text_field: str,
        db_session_factory: sessionmaker,
        classifier: Optional[TopicClassifier] = None,
    ) -> bool:
        """Process a single article (scrape + store) with its own db session."""
        db_session = db_session_factory()
        try:
            scraped_article = self.scrape_article(article_url)
            if scraped_article:
                return self.store_article(db_session, scraped_article, text_field, classifier)
            return False
        finally:
            db_session.close()

    def scrape_from_source(
        self,
        source_url: str,
        text_field: str,
        db_session_factory: sessionmaker,
        classifier: Optional[TopicClassifier] = None,
    ) -> int:
        """Scrape articles from a single source and store them using parallel processing."""
        count = 0
        try:
            news_source = get_last_news(source_url)
            self.logger.info("Parsed last %d news", len(news_source.articles))
            article_urls = [article.url for article in news_source.articles]

            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(self.process_single_article, url, text_field, db_session_factory, classifier)
                    for url in article_urls
                ]

                # Обрабатываем результаты по мере завершения
                for future in as_completed(futures):
                    if future.result():  # Если статья успешно сохранена
                        count += 1

        except Exception as e:
            self.logger.error("Error processing source %s: %s", source_url, str(e))

        return count
