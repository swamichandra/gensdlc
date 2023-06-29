# Author: Swami Chandrasekaran
# Last Updated: 06/23/2023
#
# This code is meant to serve as a template / boilerplate for building LLM based apps.
# Feel free to expand, extent and enhance.

import os
import vertexai
import streamlit as st
from langchain import OpenAI, PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from google.cloud import aiplatform
from vertexai.preview.language_models import TextGenerationModel

def generate_response(txt):
    # Instantiate the LLM model
    PRIMARY_MODEL = 'text-bison@001'
    try:
        llm = VertexAI()
    except:
        print ("Error during LLM model initialization ...")

    # Prompt Template
    prompt_template = """You are a master software engineer. Based on the requirements provided below, write the code following solid Python programming practices. Add relevant code comments. Don't explain the code, just generate the code.
    {text}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Text summarization
    res = None
    try:
        chain = LLMChain(prompt=PROMPT, llm=llm)
        res = chain.run(txt)
    except:
        print ("Error during LLM model chaining and invocation ...")

    return res

# Page title
st.set_page_config(page_title="Generative AI Text Summarization App",
                   page_icon=":random:", layout="centered")
st.title('📚 Generative AI Text Summarization App')

# aiplatform.init(project="project=learning-351419", location="us-central1")

# Create a file upload widget for the credentials JSON file
creds_file = st.file_uploader(
    "Upload Google Cloud credentials file", type="json")

if creds_file is not None:
    creds_contents = creds_file.read().decode("utf-8")
    with open("temp_credentials.json", "w") as f:
        f.write(creds_contents)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_credentials.json"

    # Text input
    txt_input = st.text_area('Enter your text to summarize', '', height=200)

    result = []
    if st.button("Submit"):
        with st.spinner('Performing magic ...'):
            st.info(txt_input)
            response = generate_response(txt_input.strip())
            result.append(response)

    # Display result
    if len(result):
        st.write(response)
