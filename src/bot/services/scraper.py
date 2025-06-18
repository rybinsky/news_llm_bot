import logging

from typing import Dict, Any, Optional
from datetime import datetime
from newspaper import Article, build as get_last_news

from models import NewsArticle
from .logging import CustomLogger
from .classifier import NewsClassifier
from .database import DBSession

class NewsScraper:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or CustomLogger(__name__).get_logger()
    
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
        self, 
        article_dict: Dict[str, Any], 
        text_field: str,
        classifier: Optional[NewsClassifier] = None
    ) -> Dict[str, Any]:
        """Extract and process article data from raw dictionary."""
        data = {
            'url': article_dict.get('url'),
            'source': article_dict.get('source_url'),
            'title': article_dict.get('title'),
            'text': article_dict.get(text_field),
            'summary': article_dict.get('summary'),
            'publish_date': article_dict.get('publish_date'),
            'language': article_dict.get('meta_lang'),
            'keywords': article_dict.get('keywords', []) + article_dict.get('meta_keywords', []),
        }

        if classifier and data["text"]:
            data['topic'] = classifier.classify(data["text"])
        else:
            data['topic'] = "Другое"

        if data['publish_date'] and isinstance(data['publish_date'], datetime):
            data['publish_date'] = data['publish_date'].isoformat()

        return data
    
    def store_article(
        self, 
        db_session: DBSession, 
        article: Article, 
        text_field: str,
        classifier: Optional[NewsClassifier] = None
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
    
    def scrape_from_source(
        self,
        source_url: str,
        text_field: str,
        db_session: DBSession,
        classifier: Optional[NewsClassifier] = None,
        max_articles: int = 5
    ) -> int:
        """Scrape articles from a single source and store them."""
        count = 0
        try:
            news_source = get_last_news(source_url)
            self.logger.info("Parsed last %d news", len(news_source.articles))
            for article in news_source.articles[:max_articles]:
                scraped_article = self.scrape_article(article.url)
                if scraped_article and self.store_article(
                    db_session, scraped_article, text_field, classifier
                ):
                    count += 1
        except Exception as e:
            self.logger.error("Error processing source %s: %s", source_url, str(e))
        
        return count
