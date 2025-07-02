from sqlalchemy import Any, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.ext.declarative import declarative_base

Base: Any = declarative_base()


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    text = Column(Text)
    topic = Column(String(50), index=True)
    publish_date = Column(DateTime)
    url = Column(String(1000), unique=True)
    source = Column(String(200), index=True)
    keywords = Column(PG_ARRAY(String(50)))
    summary = Column(Text)
    language = Column(String(10))

    def __repr__(self) -> str:
        return f"<NewsArticle(title='{self.title}', source='{self.source}')>"
