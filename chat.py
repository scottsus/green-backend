import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language
)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def init_chat():
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # 1. Model
    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        streaming=True,
        callbacks=[ StreamingStdOutCallbackHandler() ],
        temperature=0.8
    )

    # 2. Load Document
    file_path = './docs.md'
    with open(file_path) as f:
        data = f.read()
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

    # 6. Prompt + Retrieval + Chat
    prompt = PromptTemplate(
        template="""
            You are a hilarious AI assistant here to answer questions in a meaningful way.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say you don't know.
            
            {context}

            Question: {question}
            Answer:
        """,
        input_variables=['context', 'question']
    )

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            'prompt': prompt
        }
    )

    return qa

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

