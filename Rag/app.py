from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load and split PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

# Create vector store
def create_vector_store(docs):
    embeddings = OllamaEmbeddings(model="llama3.2:1b")
    return Chroma.from_documents(docs, embeddings)

# RAG chain setup
def setup_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOllama(model="llama3.2:1b")
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Main chat function
def chat_with_pdf(pdf_path):
    # Load and process PDF
    docs = load_pdf(pdf_path)
    vectorstore = create_vector_store(docs)
    
    # Create RAG chain
    rag_chain = setup_rag_chain(vectorstore)
    
    # Interactive chat loop
    while True:
        query = input("Ask a question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        response = rag_chain.invoke(query)
        print("Response:", response)

# Usage
if __name__ == "__main__":
    pdf_path = r"C:\Users\Lenovo\Desktop\GEN-AI-Rag\Rag\PDE-Google.pdf"
    chat_with_pdf(pdf_path)