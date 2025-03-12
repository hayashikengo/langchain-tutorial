import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_loaders.slack import SlackChatLoader
from langchain_community.chat_loaders.utils import (
    map_ai_messages,
    merge_chat_runs,
)
from langchain_core.chat_sessions import ChatSession
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    loader = SlackChatLoader(path="slack_dump.zip")
    raw_messages = loader.load()
    print(f"Loaded {len(raw_messages)} raw_messages")

    # スレッドごとにメッセージをマージ
    # merged_messages = list(merge_chat_runs(raw_messages))
    # print(f"Merged into {len(merged_messages)} threads")

    # 各スレッドのメッセージを文書として保存するための準備
    documents = []
    for thread in raw_messages:
        print(thread)
        # メッセージが空の場合はスキップ
        if not thread["messages"] or len(thread["messages"]) == 0:
            continue

        try:
            # スレッド内の全メッセージを結合
            thread_text = "\n\n".join([
                f"{msg.additional_kwargs.get('sender', 'Unknown')}: {msg.content}"
                for msg in thread["messages"]
            ])

            # メタデータの作成
            metadata = {
                "thread_id": thread["messages"][0].additional_kwargs.get("thread_ts", ""),
                "channel": thread["messages"][0].additional_kwargs.get("channel", ""),
                "timestamp": thread["messages"][0].additional_kwargs.get("ts", "")
            }

            # Documentオブジェクトを作成
            documents.append(Document(
                page_content=thread_text,
                metadata=metadata
            ))
        except Exception as e:
            print(f"Error processing thread: {e}")
            continue

    print(f"Going to add {len(documents)} threads to Pinecone")

    if documents:  # ドキュメントが存在する場合のみVectorStoreに保存
        PineconeVectorStore.from_documents(
            documents,
            embeddings,
            index_name="rag-slack",
        )
        print("****Loading to vectorstore done ****")
    else:
        print("No valid documents to process")


if __name__ == "__main__":
    ingest_docs()