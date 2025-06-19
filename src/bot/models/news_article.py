from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


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
    embedding = Column(JSONB)

    def __repr__(self):
        return f"<NewsArticle(title='{self.title}', source='{self.source}')>"
