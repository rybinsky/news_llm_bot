import threading
import time

from dotenv import load_dotenv

from bot.services import (
    EXAMPLES_CLS_TOPIC,
    DatabaseManager,
    NewsScraper,
    TopicClassifier,
    get_db_config,
    load_config,
    setup_logging,
)


def run_news_updater() -> None:
    load_dotenv()
    config = load_config()
    logger = setup_logging(config.logging)

    db_manager = DatabaseManager(logger)
    db_manager.initialize(
        get_db_config(),
        clear_tables=True,
    )
    scraper = NewsScraper(logger)

    classifier = TopicClassifier(
        topics=set(config.classifier.topics),
        example_articles=EXAMPLES_CLS_TOPIC,
        model_name=config.ollama.model,
        temperature=config.classifier.temperature,
        max_attempts=config.classifier.max_attempts,
        logger=logger,
    )

    def update_loop() -> None:
        while True:
            try:
                db_session = db_manager.get_session()

                for source_name, source_cfg in config.news_sources.items():
                    logger.info("🗞️ Updating news from: %s", source_name)
                    scraper.scrape_from_source(
                        source_url=source_cfg.url,
                        text_field=source_cfg.text_field,
                        db_session_factory=db_session,
                        classifier=classifier,
                    )

                logger.info("✅ News updated successfully.")
            except Exception as e:
                logger.error("❌ Error in news update: %s", str(e), exc_info=True)
            finally:
                db_manager.close()

            time.sleep(3600)  # wait 1 hour

    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()
