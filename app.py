import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def main():
    load_dotenv(os.getenv('OPENAI_API_KEY'))
    st.set_page_config(page_title='Chat with your PDF', page_icon=':books:')
    st.title('Chat with your PDF')
    st.header('Upload your PDFS')
    st.text_input("Ask your question here")

    with st.sidebar:
        st.subheader('Your documents')
        pdf=st.file_uploader('Upload your PDFS', type=['pdf'])
        st.button('Process')

    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        # st.write(text)
        splitter=CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=splitter.split_text(text)
        
        embeddings=OpenAIEmbeddings()
        knowledge_base=FAISS.from_texts(chunks, embeddings)
        user_question=st.text_input("Ask your question about your PDF here")
        if user_question is not None:
            docs=knowledge_base.similarity_search(user_question)
            llm=OpenAI()
            chain=load_qa_chain(llm,chain_type="stuff")
            response=chain.run(input_documents=docs,question=user_question)
            st.write(response)
            
if __name__ =='__main__':
    main()