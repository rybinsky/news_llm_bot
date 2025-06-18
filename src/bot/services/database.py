import os
import logging
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as DBSession
from sqlalchemy.exc import SQLAlchemyError

from models import Base
from .logging import CustomLogger


class DatabaseManager:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or CustomLogger(__name__).get_logger()
        self.engine = None
        self.SessionLocal = None
    
    def get_database_url(self, config: dict) -> str:
        """Construct database connection URL from configuration."""
        return (
            f"postgresql://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['database']}"
        )
    
    def initialize(self, db_config: dict) -> None:
        """Initialize database connection."""
        try:
            db_url = self.get_database_url(db_config)
            self.engine = create_engine(db_url)
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine
            )
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database initialized successfully")
        except SQLAlchemyError as e:
            self.logger.error("Database initialization failed: %s", str(e))
            raise
    
    def get_session(self) -> DBSession:
        """Get a new database session."""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()
    
    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection closed")
