from typing import Sequence

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.schema import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings as HFEmbedder
from langchain_ollama.llms import OllamaLLM

from bot.models import NewsArticle

from .prompt import PROMPT


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join([d.page_content for d in docs])


def generate_response(
    llm: OllamaLLM,
    news: Sequence[NewsArticle],
    embedder: HFEmbedder,
    summarize_chain: BaseCombineDocumentsChain,
    splitter: CharacterTextSplitter,
    query: str,
) -> tuple[str, list[str]]:
    split_documents = splitter.create_documents([new.text for new in news])
    db = FAISS.from_documents(split_documents, embedder)
    retriever = db.as_retriever()
    relevant_docs = retriever.invoke(query)

    summaries = []
    for doc in relevant_docs:
        summary = summarize_chain.invoke([doc])
        summaries.append(f"📌 {summary['output_text']}")

    chain = (
        {"context": lambda _: format_docs(relevant_docs), "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    joke = chain.invoke(query)
    return joke, summaries
