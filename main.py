from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import *

model = OllamaLLM(model="llama3.2")
template = """
You are an expert in answering questions
Here is your reference {reference}
Here is your question {question}
"""
chat = ChatPromptTemplate.from_template(template)
#chain = chat | model
formatted_prompt = chat.format_prompt(reference="My age is 99", question="Capital of India?")

result = model.invoke(formatted_prompt)
#result = chain.invoke({"reference": "My age is 99", "question": "Capital of India?"})
print(result)
