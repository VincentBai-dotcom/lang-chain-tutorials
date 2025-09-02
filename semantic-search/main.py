"""
Main module for the semantic search.
"""

import getpass
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from typing import List


def load_documents():
    """
    Load the documents from the file.
    """
    return [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]


def display_some_data_from_documents(docs):
    """
    Display some data from the documents.
    """
    print(len(docs))
    print(f"{docs[0].page_content[:200]}\n")
    print(docs[0].metadata)
    print("=" * 100)


def load_env():
    """
    Load the environment variables.
    """
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


def embed_documents(embeddings, all_splits):
    """
    Embed the documents.
    """
    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)
    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    print(vector_1[:10])


def query_vector_store(vector_store, query):
    """
    Query the vector store.
    """
    print(f"Query: {query}")
    results = vector_store.similarity_search_with_score(query)
    doc, score = results[0]
    print(f"Score: {score}\n")
    print(doc)
    print("=" * 100)


def main():
    """
    Main function to run the semantic search.
    """
    print("Hello from semantic-search!")

    file_path = "./example_data/nke-10k-2023.pdf"

    loader = PyPDFLoader(file_path)

    docs = loader.load()

    display_some_data_from_documents(docs)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    display_some_data_from_documents(all_splits)
    load_env()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embed_documents(embeddings, all_splits)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)
    print("=" * 100)
    query_vector_store(vector_store, "What was Nike's revenue in 2023?")
    query_vector_store(
        vector_store, "How many distribution centers does Nike have in the US?"
    )
    query_vector_store(vector_store, "How were Nike's margins impacted in 2023?")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    print(
        retriever.batch(
            [
                "How many distribution centers does Nike have in the US?",
                "When was Nike incorporated?",
            ]
        )
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
