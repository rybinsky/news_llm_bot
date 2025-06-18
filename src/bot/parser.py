import os
from typing import Dict, Any
from dotenv import load_dotenv
import yaml
from omegaconf import OmegaConf, DictConfig

from bot.services import DatabaseManager, NewsScraper, NewsClassifier, setup_logging


def load_config() -> DictConfig:
    """Load application configuration from YAML file."""
    with open("src/config/config.yaml", 'r', encoding='utf-8') as file:
        return OmegaConf.create(yaml.safe_load(file))


def main():
    """Main application entry point."""
    load_dotenv()
    config = load_config()
    
    logger = setup_logging(config.logging)
    
    db_manager = DatabaseManager(logger)
    scraper = NewsScraper(logger)
    
    # Example articles for few-shot learning
    example_articles = [
        {"text": "Ученые открыли новый вид динозавров", "category": "Наука"},
        {"text": "Выборы президента пройдут в следующем месяце", "category": "Политика"},
        {"text": "Футбольный клуб нашего города выиграл чемпионат!", "category": "Спорт"},
    ]
    
    classifier = NewsClassifier(
        topics=set(config.classifier.topics),
        example_articles=example_articles,
        model_name=config.ollama.model,
        temperature=config.classifier.temperature,
        logger=logger
    )
    
    try:
        db_manager.initialize({
            'user': os.getenv("POSTGRES_USER"),
            'password': os.getenv("POSTGRES_PASS"),
            'host': os.getenv("POSTGRES_HOST"),
            'port': os.getenv("POSTGRES_PORT"),
            'database': os.getenv("POSTGRES_DB")
        })
        
        db_session = db_manager.get_session()
        
        total_articles = 0
        for source_name, source_cfg in config.news_sources.items():
            logger.info("Processing news source: %s", source_name)
            count = scraper.scrape_from_source(
                source_url=source_cfg.url,
                text_field=source_cfg.text_field,
                db_session=db_session,
                classifier=classifier,
                max_articles=config.scraper.max_articles
            )
            total_articles += count
            logger.info("Stored %d articles from %s", count, source_name)
        
        logger.info("Finished. Total articles stored: %d", total_articles)
    
    except Exception as e:
        logger.error("Application error: %s", str(e), exc_info=True)
    
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()