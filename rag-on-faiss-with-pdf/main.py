import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,OpenAI
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":
    # loader
    pdf_path = "/Users/kengo/ghq/github.com/kengo/langchain-tutorial/pdf-loader/2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    # テキスト分割
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    # vector store
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embeddings)
    # 永続化の為にローカルに保存する
    vectorStore.save_local("faiss_index_react")
    # ローカルに保存したベクトルストアを読み込む
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    # chain
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in "})
    print(res["answer"])
