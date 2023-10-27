import os
import PyPDF2
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def init_chat():
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # 1. Model
    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        model='gpt-4',
        streaming=True,
        callbacks=[ StreamingStdOutCallbackHandler() ],
        temperature=0.4
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    split_md_docs = text_splitter.create_documents([data])

    # 4. Embeddings, Vector Store, Retriever
    db = Chroma.from_documents(split_md_docs, OpenAIEmbeddings())
    retriever = db.as_retriever()

    # 5. Define Output Parser
    class ViolationCheck(BaseModel):
        user_violation: str = Field(description='exact marketing violation made by user')
        reason: str = Field(description='reason why marketing sentence violates FTC guidelines')
        section_number: str = Field(description='FTC section number explaining the violation')
        recommendation: str = Field(description='better way to rephrase marketing sentence')
    parser = PydanticOutputParser(pydantic_object=ViolationCheck)

    # 6. Prompt + Retrieval + Chat + Output Parser
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

    # 7. Model
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            'prompt': prompt
        }
    )

    return qa, parser

def analyze_statement(model, parser, statement):
    analyzed = model.run(statement)
    return parser.parse(analyzed)
    