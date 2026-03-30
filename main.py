from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import *

from vector import retriever

model = OllamaLLM(model="llama3.2")
template = """
You are an expert in answering questions
Here is your reference {reference}
Here is your question {question}
"""
chat = ChatPromptTemplate.from_template(template)
chatActive = True
while chatActive:
    question = input("Question (done to quit): ")
    if question == "done":
        chatActive = False
        break
    elif question == "":
        chatActive = False
        break
    #chain = chat | model
    reference = retriever.invoke(question)
    print(reference)

    formatted_prompt = chat.format_prompt(reference= reference, question=question)

    result = model.invoke(formatted_prompt)
    #result = chain.invoke({"reference": "My age is 99", "question": "Capital of India?"})
    print(result)
