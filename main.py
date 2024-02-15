import os
import ast
from dotenv import load_dotenv
from utils import load_data, embedding, format_docs

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def main():
    print('환경설정중...')
    load_dotenv()
    KEY = os.getenv("KEY")
    FILE_PATH = os.getenv("FILE_PATH")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    MODEL_KWARGS = ast.literal_eval(os.getenv("MODEL_KWARGS"))
    ENCODE_KWARGS = ast.literal_eval(os.getenv("ENCODE_KWARGS"))

    print('데이터로드중...')
    docs = load_data(FILE_PATH)

    print('모델설정중...')
    embedding_model = embedding(EMBEDDING_MODEL_NAME, MODEL_KWARGS, ENCODE_KWARGS)
    db = Chroma.from_documents(documents=docs, embedding=embedding_model)
    retriever = db.as_retriever()
    llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0, openai_api_key=KEY)

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise. \

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt
        | llm
    )

    chat_history = []

    question = "가장 많이 일어난 범죄의 유형을 알려줘"
    ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
    print()
    print('Q1 :', question)
    print(ai_msg)
    
    chat_history.extend([HumanMessage(content=question), ai_msg])
    print()
    
    second_question = "그 범죄의 판례 번호를 3개만 뽑아줘"
    ai_msg = rag_chain.invoke({"question": second_question, "chat_history": chat_history})
    print('Q2 :', second_question)
    print(ai_msg)

    chat_history.extend([HumanMessage(content=question), ai_msg])
    print()

    third_question = "그럼 이것들의 평균형량을 알려줘"
    ai_msg = rag_chain.invoke({"question": third_question, "chat_history": chat_history})
    print('Q3 :', third_question)
    print(ai_msg)


if __name__ == '__main__':
    main()
