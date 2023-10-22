import os
import PyPDF2
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language
)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def init_chat(prompt0):
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # 1. Model
    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        streaming=True,
        callbacks=[ StreamingStdOutCallbackHandler() ],
        temperature=0.8
    )

    # 2. Load Document
    file_path = './greenguides.pdf'
    with open(file_path, 'rb') as pdf_file:
        pdf = PyPDF2.PdfReader(pdf_file)

        data = ''
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            data += page.extract_text()
    # print(data)

    # 3. Split Document
    md_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=128, chunk_overlap=32
    )
    split_md_docs = md_splitter.create_documents([data])
    # print(split_md_docs)

    # 4. Embeddings, Vector Store, Retriever
    db = Chroma.from_documents(split_md_docs, OpenAIEmbeddings())
    query = 'What language is contained in this file?'
    res = db.similarity_search(query)
    # print(res[0].page_content)

    retriever = db.as_retriever()
    # docs = retriever.get_relevant_documents(query)
    # ans = qa(query)

    class ViolationCheck(BaseModel):
        user_violation: str = Field(description='marketing violation made by user')
        section_number: str = Field(description='FTC section 2xx explaining the violation')
        recommendation: str = Field(description='better way to rephrase marketing sentence')
    parser = PydanticOutputParser(pydantic_object=ViolationCheck)

    # 6. Prompt + Retrieval + Chat
    prompt = PromptTemplate(
        template="""
            You are an AI assistant dedicated to watching out for FTC violations given a marketing sentence from a company.
            Use the following pieces of context to answer the question at the end.
            If you know the answer, explain in detail why it is a violation of the FTC, and most importantly reference the section number 2xx in the Green Guide.
            Then recommend a fix to better phrase the marketing sentence.
            Otherwise if you don't know the answer say you don't know.
            {context}

            Question: {question}
            Answer: {format_instructions}
        """,
        input_variables=['context', 'question'],
        partial_variables={'format_instructions': parser.get_format_instructions()},
    )

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            'prompt': prompt
        }
    )

    output = qa.run(prompt0)
    parser.parse(output)

    # return qa

if __name__ == '__main__':
    prompt = 'Spirit airlines: we are the lowest carbon emissions of any major airline'
    model = init_chat(prompt)
    # model.run('Scotts facial cream is certified organic, USDA organic and vegan')
    print()

# question = 'Hello, who are you?'
# print(question)
# qa.run(question)
# print()

# question = '你好 我叫苏学良'
# print('\n' + question)
# qa.run(question)
# print()

# question = 'Please convert the latin words to English'
# print('\n' + question)
# qa.run(question)
# print()

# question = 'Who are my best friends?'
# print('\n' + question)
# qa.run(question)
# print()

# question = 'How was your day today?'
# print('\n' + question)
# qa.run(question)
# print()

# question = 'Goodbye 👋'
# print('\n' + question)
# qa.run(question)
# print()

