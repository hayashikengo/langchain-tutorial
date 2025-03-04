import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv(override=True)

# ドキュメントをフォーマット
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    print("Hello, World!")
    print(os.getenv("LANGCHAIN_PROJECT"))

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI()

    query = "what is Pinecone is machine learning?"
    chain = PromptTemplate.from_template(query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    # VectorStore
    vectorStore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )

    # LangHubでデータ取得
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrieval_chain = create_retrieval_chain(
    #     retriever=vectorStore.as_retriever(),
    #     combine_docs_chain=combine_docs_chain,
    # )

    # result = retrieval_chain.invoke(input={"input": query})
    # print(result)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer: """

    # 自前でカスタムプロンプト定義
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": vectorStore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    result = rag_chain.invoke(query)
    print(result.content)
