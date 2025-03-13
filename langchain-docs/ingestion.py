import os
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    loader = ReadTheDocsLoader(path="langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} documents to Pinecone")
    PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name="langchain-docs",
    )
    print("****Loading to vectorstore done ****")

def ingest_docs2() -> None:
    from langchain_community.document_loaders import FireCrawlLoader

    langchain_documents_base_urls = [
        "https://python.langchain.com/docs/integrations/chat/",
        "https://python.langchain.com/docs/integrations/retrievers/",
        "https://python.langchain.com/docs/integrations/tools/",
        "https://python.langchain.com/docs/integrations/document_loaders/",
        "https://python.langchain.com/docs/integrations/vectorstores/",
        "https://python.langchain.com/docs/integrations/text_embedding/",
    ]

    for url in langchain_documents_base_urls:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params={
                "crawlerOptions": {"limit": 5},
                "pageOptions": {"onlyMainContent": True},
                "wait_until_done": True,
            }
        )
        docs = loader.load()

        print(f"Adding {len(docs)} documents from {url} to Pinecone")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name="firecrawl-index"
        )
        print(f"Added {len(docs)} documents from {url} to Pinecone")


if __name__ == "__main__":
    ingest_docs2()