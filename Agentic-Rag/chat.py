from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Setup SQLite Database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Initialize Ollama Chat Model
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.1,
)

# Create a prompt template that includes database context and chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful database assistant. 
    Use the following database context to answer questions precisely:
    Available Tables: {tables}
    
    Guidelines:
    - Only use SQL queries to retrieve information
    - Provide clear and concise answers
    - If you cannot find the information, explain why"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# Create a chain that combines the prompt, LLM, and database context
def get_sql_chain(db):
    def _get_tables():
        return ", ".join(db.get_usable_table_names())
    
    return (
        prompt.partial(tables=_get_tables())
        | llm
    )

# Function to execute SQL and get results
def execute_sql_query(query):
    try:
        return db.run(query)
    except Exception as e:
        return f"Error executing query: {str(e)}"

# Main chat function
def chat_with_database(question, session_id="default_session"):
    # Setup chat history using SQLite
    chat_history = SQLChatMessageHistory(
        session_id=session_id, 
        connection_string="sqlite:///chat_history.db"
    )
    
    # Create a chain with message history
    chain_with_history = RunnableWithMessageHistory(
        get_sql_chain(db),
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    
    # Invoke the chain
    config = {"configurable": {"session_id": session_id}}
    response = chain_with_history.invoke(
        {"question": question}, 
        config=config
    )
    
    return response.content

# Example usage
def main():
    # Interactive chat loop
    print("Database Chat Assistant (type 'exit' to quit)")
    session_id = "user_session_1"
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            break
        
        response = chat_with_database(user_input, session_id)
        print("Assistant:", response)

if __name__ == "__main__":
    main()