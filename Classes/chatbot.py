from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import Runnable

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])
 
class ChatBot():
    def __init__(self, memory: bool, model="https://7af2-195-230-200-203.ngrok-free.app/v1", api_key="EMPTY"):
        self.chatbot = ChatOpenAI(openai_api_base=model, api_key=api_key)
        self.memory = memory
        self.retriever = None
        self.template = None
        self.prompt = None
        self.context = None
        self.relevant_documents = []  # Store relevant documents for context

    def ingest(self, vector, k):
        """Ingests the vector and prepares the retriever and prompt template."""
        self.retriever = vector.retriever(k)

        # Define the prompt template based on whether memory is enabled
        if self.memory:
            self.template = """
            Answer the question based ONLY on the context and the conversation you and me have had up to now.

            This is our conversation up to now (If I ask you something about our previous conversation, you can draw from here):
            
            {chat_history}
            
            This is the context:
            
            {context}
            
            IT IS FORBIDDEN to use external knowledge you have in your memory to answer the question!!!!
            
            Question: {question}
            """
        else:
            self.template = """
            Answer the question based only on the following context:
            
            {context}
            
            Question: {question}
            """
        
        self.prompt = ChatPromptTemplate.from_template(self.template)
        return self.prompt

    def ask(self, query, stream=False):
        """Asks a query and returns the response, handling both memory and non-memory scenarios."""
        response = ''
        self.relevant_documents = []  # Reset for each new query

        if not self.memory:
            # Non-memory case: Format the context and use the chatbot
            chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.chatbot
                | StrOutputParser()
            )
            if not stream:
                result = chain.invoke(query)
                response = result
                # Update relevant documents used in context
                self.relevant_documents = self.retriever.retrieve(query)
            else:
                for s in chain.stream(query):
                    print(s, end="", flush=True)
                    response += s
                self.relevant_documents = self.retriever.retrieve(query)
        else:
            # Memory case: Use ConversationBufferMemory and conversational retrieval chain
            message_history = BaseChatMessageHistory()
            chat_memory = ConversationBufferMemory(memory_key="chat_history",
                                                   output_key="answer",
                                                   chat_memory=message_history,
                                                   return_messages=True)

            chain = ConversationalRetrievalChain.from_llm(
                llm=self.chatbot,
                retriever=self.retriever,
                condense_question_prompt=self.prompt,
                chain_type="stuff",
                memory=chat_memory,
                return_source_documents=True,
                get_chat_history=lambda h: h
            )
            if not stream:
                result = chain.invoke(input=query)
                response = result["answer"]
                print(response)
                # Save relevant documents for context
                self.relevant_documents = result["source_documents"]
            else:
                for s in chain.stream(query):
                    print(s["answer"], end="", flush=True)
                    response += s["answer"]
                self.relevant_documents = chain.retrieve(query)

        return response

    def get_relevant_documents(self):
        """Returns the relevant documents used for context."""
        return self.relevant_documents
