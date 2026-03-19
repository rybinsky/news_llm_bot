import logging
from typing import Optional, Sequence

from sqlalchemy import create_engine, desc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from bot.models import Base, NewsArticle

from .logging import CustomLogger


class DatabaseManager:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or CustomLogger(__name__).get_logger()
        self.engine = None

    def get_database_url(self, config: dict) -> str:
        """Construct database connection URL from configuration."""
        return (
            f"postgresql://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['database']}"
        )

    def initialize(self, db_config: dict, clear_tables: bool = False) -> None:
        """Initialize database connection."""
        try:
            db_url = self.get_database_url(db_config)
            self.engine = create_engine(db_url)
            self.session: sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            if clear_tables:
                Base.metadata.drop_all(bind=self.engine)
                self.logger.info("All tables dropped")
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database initialized successfully")
        except SQLAlchemyError as e:
            self.logger.error("Database initialization failed: %s", str(e))
            raise

    def get_last_news_by_topic(self, topic: str, limit: int = 15) -> Sequence[NewsArticle]:
        """Get latest news of given topic."""
        stmt = (
            select(NewsArticle).where(NewsArticle.topic == topic).order_by(desc(NewsArticle.publish_date)).limit(limit)
        )
        result = self.session().execute(stmt)
        return result.scalars().all()

    def get_session(self) -> sessionmaker:
        """Get a new database session."""
        if not self.session:
            raise RuntimeError("Database not initialized")
        return self.session

    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection closed")
