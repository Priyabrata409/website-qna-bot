import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def get_rag_chain():
    """
    Creates and returns a RAG chain for question answering.
    """
    # Model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Vector Store
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME must be set in .env")

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    # Prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Combine documents using a simple join (or use a more advanced combiner if needed)
    def combine_context(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def rag_chain(inputs):
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(inputs["input"])
        # Combine context
        context = combine_context(docs)
        # Format prompt
        formatted_prompt = prompt.format(context=context, input=inputs["input"])
        # Get LLM response
        response = llm.invoke(formatted_prompt)
        return {"answer": response.content}

    return rag_chain
