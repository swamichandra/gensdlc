# Author: Swami Chandrasekaran
# Last Updated: 06/23/2023
#
# This code is meant to serve as a template / boilerplate for building LLM based apps.
# Feel free to expand, extent and enhance.
import css
import os
import random
from datetime import datetime
import vertexai
import streamlit as st
from langchain import OpenAI, PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.preview.language_models import TextGenerationModel

# Page title
st.set_page_config(page_title="SDLC powered by GCP Vertex Generative A.I", page_icon=":random:", layout="wide")
st.write(f'<style>{css.v1}</style>', unsafe_allow_html=True)
st.title('👩‍💻 SDLC powered by Generative A.I')

# Global Settings and Config
project_id = "learning-351419"
loc = "us-central1"
primary_model_name = "text-bison@001"
temperature = 0.3
max_output_tokens = 1024
top_p = 0.8
top_k = 40
location = "us-central1"
model_name = "text-bison@001"

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

vertexai.init(project=project_id, location=loc, credentials=credentials)


@st.cache_resource
def generate_random_input():
    INPUT_TEXT_TMP = ""
    sample_code_gen_qns = [
        "Generate a semantic HTML and Tailwind CSS Contact Support form consisting of the user name, email, issue type, and message. The form elements should be stacked vertically and placed inside a card", 
        "Write a JavaScript function. It accepts a full name as input and returns avatar letters.",
        "Social food delivery app: Satisfy your cravings while supporting local businesses with a social food delivery app. Discover a wide range of culinary delights from nearby restaurants and food vendors. Engage with a community of food enthusiasts, share recommendations, and enjoy exclusive deals. Experience the joy of delicious food delivered right to your doorstep, all while fostering connections within your local food scene.",
        "An app to manage everyday tasks and items. This could include features such as adding tasks, deleting tasks, marking tasks as completed, and setting due dates for tasks.",
        "An app that can display the current weather conditions and forecast for a specific location. This could include features such as showing the temperature, humidity, wind speed, and precipitation.",
        "You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.",
        "Write an Express.js API to fetch the current user's profile information. It should make use of MongoDB", 
        "The database has students and course tables. Write a PostgreSQL query to fetch a list of students who are enrolled in at least 3 courses.", 
        "The ETR (electronic tool rental) will be a E-commerce online rental platform that	provides rental services of various home improvement tools like carpet cleaner rentals, woodchipper rentals, lawn rollers, saws for the wide range of vendors (plumbing technicians, Pipe fitters, Steam fitters, Gas service technician, Business owners and general consumers). Should provide rental services of tools with wide range of rental plans by eliminating the huge capital investment and maintenance efforts. Provides rental services of home improvement tools across the country which benefits the technicians/small scale business owners by eradicating the need of transporting the tools to different locations where they do repairs/services.",
        "The purpose of the online flight management system is to ease flight management and to create a convenient and easy-to-use application for passengers, trying to buy airline tickets. The system is based on a relational database with its flight management and reservation functions. We will have a database server supporting hundreds of major cities around the world as well as thousands of flights by various airline companies. Above all, we hope to provide a comfortable user experience along with the best pricing available.",
        "Simple Library Management System using which a librarian can add book details like ISBN number, book title, author name, edition, and publication details through a web page. In addition to this, the librarian or any user can search for the available books in the library by the book name. If book details are present in the database, the search details are displayed.",
        "Container tracking app: Streamline your logistics operations with a container tracking application. Track the movement of your shipments in real time, ensuring transparency and efficiency throughout the supply chain. Get instant updates on container status, location, and estimated arrival times. Simplify the management of your cargo and enhance collaboration with shipping partners, empowering you to stay ahead in the global marketplace.",
        
    
    ]
    INPUT_TEXT_TMP = random.choice(sample_code_gen_qns)
    return INPUT_TEXT_TMP

@st.cache_resource
def generate_code(txt, lang_option):
    PROJECT_ID = "learning-351419"  # @param {type:"string"}
    vertexai.init(project=PROJECT_ID, location="us-central1")

    # Prompt Template
    code_gen_prompt_template = """You are a master software engineer. Based on the requirements provided below, write the code following solid {lang_option} programming practices. Add relevant code comments. Don't explain the code, just generate the code.
    {text}
    """
    PROMPT = PromptTemplate(template=code_gen_prompt_template, input_variables=["text", "lang_option"])

    res = None

    # Instantiate the LLM model
    try:
        llm = VertexAI(model_name=primary_model_name, max_output_tokens=506,
                       temperature=0.1, top_p=0.8, top_k=40, verbose=True,)

        # Text summarization
        try:
            chain = LLMChain(prompt=PROMPT, llm=llm)
            res = chain.run({'text':txt, 'lang_option': lang_option})
        except Exception as e:
            st.error("Error during LLM model chaining and invocation")
            st.error(e)
    except Exception as e:
        st.error("Error during LLM model initialization")
        st.error(e)

    return res

@st.cache_resource
def generate_product_backlog(txt):
    # Prompt Template
    prod_backlog_gen_prompt_template = """You are a master product manager. Based on the requirements provided below, generate a product backlog. Organize the backlog into epics and features with associated priority. List the epics as well formatted bullet points and the features as sub-bullets undeneath the respective epics. For each of the bullet points include the folowing: Epic Name, Features, and Priority.
    {text}
    """
    PROMPT3 = PromptTemplate(template=prod_backlog_gen_prompt_template, input_variables=["text"])

    res = None

    # Instantiate the LLM model
    try:
        llm3 = VertexAI(model_name=primary_model_name, max_output_tokens=506,
                       temperature=0.1, top_p=0.8, top_k=40, verbose=True,)

        # Text summarization
        try:
            chain = LLMChain(prompt=PROMPT3, llm=llm3)
            res = chain.run({'text':txt})
        except Exception as e:
            st.error("Error during LLM model chaining and invocation")
            st.error(e)
    except Exception as e:
        st.error("Error during LLM model initialization")
        st.error(e)

    return res

@st.cache_resource
def generate_api(txt):
    # Prompt Template
    api_gen_prompt_template = """You are a master API designer and develop. Based on the requirements provided below and using leading practices for domain driven API design, come up with a list of API's including the input and output parameters. List the API's ONLY as bullet points.
    {text}
    """
    
    PROMPTapi = PromptTemplate(template=api_gen_prompt_template, input_variables=["text"])

    res = None
    # Instantiate the LLM model
    try:
        llmapi = VertexAI(model_name=primary_model_name, max_output_tokens=506,
                       temperature=0.1, top_p=0.8, top_k=40, verbose=True,)

        # Text summarization
        try:
            chain = LLMChain(prompt=PROMPTapi, llm=llmapi)
            res = chain.run({'text':txt})
        except Exception as e:
            st.error("Error during LLM model chaining and invocation")
            st.error(e)
    except Exception as e:
        st.error("Error during LLM model initialization")
        st.error(e)

    return res


@st.cache_resource
def generate_test_cases(txt, code):
    # Prompt Template
    test_case_gen_prompt_template = """You are a master software quality engineer. Based on the requirements and code provided below, generate test cases to validate features and functions. List the test cases ONLY asa table with the following Test Case ID, Test Scenario, Test Steps and Expected Results.
    {text}
    {code}
    """
    
    PROMPT2 = PromptTemplate(template=test_case_gen_prompt_template, input_variables=["text", "code"])

    res = None

    # Instantiate the LLM model
    try:
        llm2 = VertexAI(model_name=primary_model_name, max_output_tokens=506,
                       temperature=0.1, top_p=0.8, top_k=40, verbose=True,)

        # Text summarization
        try:
            chain = LLMChain(prompt=PROMPT2, llm=llm2)
            res = chain.run({'text':txt, 'code':code})
        except Exception as e:
            st.error("Error during LLM model chaining and invocation")
            st.error(e)
    except Exception as e:
        st.error("Error during LLM model initialization")
        st.error(e)

    return res

@st.cache_resource
def generate_documentation(txt, backlog, code):
    # Prompt Template
    doc_gen_prompt_template = """You are a master technical writer. Based on the requirements, backlog and code provided below, generate readthedocs style product documentation. Include the following: README, Detailed Description, Installation Instructions, API Documentation, Getting Started. Generate text version only.
    {text}
    {backlog}
    {code}
    """
    
    PROMPTdoc = PromptTemplate(template=doc_gen_prompt_template, input_variables=["text", "backlog", "code"])

    res = None

    # Instantiate the LLM model
    try:
        llmcode = VertexAI(model_name=primary_model_name, max_output_tokens=506,
                       temperature=0.1, top_p=0.8, top_k=40, verbose=True,)

        # Text summarization
        try:
            chain = LLMChain(prompt=PROMPTdoc, llm=llmcode)
            res = chain.run({'text':txt, 'backlog': backlog, 'code':code})
        except Exception as e:
            st.error("Error during LLM model chaining and invocation")
            st.error(e)
    except Exception as e:
        st.error("Error during LLM model initialization")
        st.error(e)

    return res

result_code = []
# Using the "with" syntax
#with st.form(key='sdlc_form', clear_on_submit = False):
text_input = st.text_area('Tell me about your app', generate_random_input(), height=200, key='fav1')

lang_option = st.radio("Target Programming Language:", ('Python', 'Java', 'JavaScript', 'Go', 'Rust'), horizontal=True)

submit_button = st.button('Submit')


if submit_button:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Product Backlog", "Generated Code", "API's", "Test Cases", "Deployment Script", "Documentation"])
    
    #st.write(lang_option)
    result_code = []
    response_code = None
    result_test_case = []
    response_test_case = None
    result_prod_backlog = []
    response_prod_backlog = None
    result_doc = []
    response_doc = None
    result_api = []
    response_api = None
    with st.spinner('Wait for the magic ...'):
        with tab1:
            response_prod_backlog = generate_product_backlog(text_input.strip())
            result_prod_backlog.append(response_prod_backlog)
            
            # Display backlog
            if len(result_prod_backlog):
                st.subheader("Feature Backlog")
                st.write(response_prod_backlog)
        
        with tab2:
            response_code = generate_code(text_input.strip(), lang_option)
            result_code.append(response_code)

            # Display code
            if len(result_code):
                st.subheader("The Code")
                st.write(response_code)
            
        with tab3:
            response_api = generate_api(text_input.strip())
            result_api.append(response_api)

            # Display api
            if len(result_api):
                st.subheader("API's")
                st.write(response_api)
            
        with tab4:
            response_test_case = generate_test_cases(text_input.strip(), response_code.strip())
            result_test_case.append(response_test_case)
            
            # Display test case
            if len(result_test_case):
                st.subheader("Test Cases")
                st.write(response_test_case)
        
        with tab5:
            st.subheader("Deployment Script")
            st.write("DevSecOps Coming soon...")
        
        with tab6:
            response_doc = generate_documentation(text_input.strip(), response_prod_backlog.strip(), response_code.strip())
            result_doc.append(response_doc)
            
            # Display documentation
            if len(result_doc):
                st.subheader("Documentation")
                st.write(response_doc)
