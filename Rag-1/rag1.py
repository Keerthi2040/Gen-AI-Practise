from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

def load_documents(directory):
    """
    Load all text files from a given directory
    """
    """documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            loader = TextLoader(filepath, encoding='utf-8')
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents from {directory}")
    return documents"""
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.txt'):
            try:
                # Try UTF-8 first
                loader = TextLoader(filepath, encoding='utf-8')
                documents.extend(loader.load())
            except UnicodeDecodeError:
                try:
                    # Fallback to latin-1 or other encodings
                    loader = TextLoader(filepath, encoding='latin-1')
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        elif filename.endswith('.pdf'):
            try:
                # Load PDF files
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    print(f"Loaded {len(documents)} documents from {directory}")
    return documents


def create_rag_chain(documents, model_name='llama3.2:1b'):
    """
    Create a RAG chain with Ollama
    
    Args:
        documents: List of documents to index
        model_name: Ollama model to use (default: llama3)
    """
    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)

    # Embedding and Vector Store
    embeddings = OllamaEmbeddings(model=model_name)
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # RAG Prompt
    template = """You are an expert assistant. Use the following context to answer the question.
    If the context doesn't contain enough information, say so.

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    llm = ChatOllama(model=model_name, temperature=0.7)

    # RAG Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def main():
    # Directory containing your text files
    document_directory = r'C:\Users\Lenovo\Desktop\GEN-AI-Rag\Rag-1\Pdfs'

    # Load documents
    documents = load_documents(document_directory)
    
    if not documents:
        print("No documents found. Please add text files to the documents directory.")
        return

    # Create RAG chain
    rag_chain, retriever = create_rag_chain(documents)

    # Interactive query loop
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        
        if query.lower() == 'quit':
            break

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(query)
        print("\n--- Retrieved Documents ---")
        for doc in retrieved_docs:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content Preview: {doc.page_content[:200]}...\n")

        # Generate response
        response = rag_chain.invoke(query)
        print("\n--- AI Response ---")
        print(response)

if __name__ == "__main__":
    main()