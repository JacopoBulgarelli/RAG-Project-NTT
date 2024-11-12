from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

class LangchainManager:
    def __init__(self, openai_api_base, api_key):
        self.llm = ChatOpenAI(openai_api_base=openai_api_base, api_key=api_key)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Definisci il template di prompt
        self.prompt_template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
        
        # Inizializza self.prompt usando self.prompt_template
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
    
    def invoke(self, question, context):
        """Esegue una query al modello di linguaggio."""
        formatted_context = "\n\n".join(context)
        input_data = {"context": formatted_context, "question": question}
        
        output = self.prompt | self.llm | StrOutputParser()
        response = output.invoke(input_data)
        return response
    
    def create_chain(self, retriever):
        """Crea una catena di QA utilizzando il retriever fornito."""
        
        # Usa RetrievalQA.from_chain_type per configurare la QA chain
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",   # "stuff" combina tutti i documenti in un contesto unico
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def get_conversation_history(self):
        """Restituisce lo storico della conversazione."""
        return self.memory.messages

    def clear_memory(self):
        """Pulisce lo storico della memoria."""
        self.memory.clear()
