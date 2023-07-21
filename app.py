# Author: Swami Chandrasekaran
# Last Updated: 06/23/2023
#
# This code is meant to serve as a template / boilerplate for building LLM based apps.
# Feel free to expand, extent and enhance.
import css
import os
import time
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
st.title('💫 SDLC augmented by Gen A.I')

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1.5rem;
                    padding-bottom: 0rem;
                    padding-left: 1.5rem;
                    padding-right: 1.5rem;
                }
</style>
""", unsafe_allow_html=True)
        
# Global Settings and Config
project_id = "learning-351419"
loc = "us-central1"

# Confuguration Section Starts
primary_model_name = st.sidebar.selectbox("Model", ["text-bison@001"])
temperature = st.sidebar.number_input("Temperature _(Higher value will result in more **random** responses)_", 0.0, 1.0, 0.3)
max_output_tokens = st.sidebar.number_input("Max Output Tokens _(Number of tokens that the model will **generate**)_", 200, 2048, 1024)
top_p = st.sidebar.number_input("Top_p _(Higher value will result in more **creative** responses)_", 0.0, 1.0, 0.5)
top_k = st.sidebar.slider("Top_k _(Higher value will result in more **diverse** responses)_", 1, 100, 40)
# Confuguration Section Ends

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

vertexai.init(project=project_id, location=loc, credentials=credentials)

sample_code_gen_qns = [
    "Generate a semantic HTML and Tailwind CSS Contact Support form consisting of the user name, email, issue type, and message. The form elements should be stacked vertically and placed inside a card", 
    "Write a that accepts a full name as input and returns avatar letters.",
    "Social food delivery app: Satisfy your cravings while supporting local businesses with a social food delivery app. Discover a wide range of culinary delights from nearby restaurants and food vendors. Engage with a community of food enthusiasts, share recommendations, and enjoy exclusive deals. Experience the joy of delicious food delivered right to your doorstep, all while fostering connections within your local food scene.",
    "An app to manage everyday tasks and items. This could include features such as adding tasks, deleting tasks, marking tasks as completed, and setting due dates for tasks.",
    "An app that can display the current weather conditions and forecast for a specific location. This could include features such as showing the temperature, humidity, wind speed, and precipitation.",
    "You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.",
    "Write an Express.js API to fetch the current user's profile information. It should make use of MongoDB",
    "Railway tracking app: For citizens living in cities bustling with activity, public transportation is an integral part of their daily lives. Unfortunately, with trains often running late or leaving us unsure if they will even show up at all, an unforeseen disruption to our schedule can be the difference between being on time… or not. Wouldn’t it be nice to have a reliable resource that could provide you with the exact info about when your train will arrive? With a railway tracking app, commuters would have access to useful information about where their train’s location is in real-time – potentially saving us from awkward misunderstandings or added stress as we try to coordinate other transport methods when a train is delayed. And who knows? If utilized correctly, a railway tracking app could even make life on the rails more enjoyable for everyday commuters.",
    "The ETR (electronic tool rental) will be a E-commerce online rental platform that	provides rental services of various home improvement tools like carpet cleaner rentals, woodchipper rentals, lawn rollers, saws for the wide range of vendors (plumbing technicians, Pipe fitters, Steam fitters, Gas service technician, Business owners and general consumers). Should provide rental services of tools with wide range of rental plans by eliminating the huge capital investment and maintenance efforts. Provides rental services of home improvement tools across the country which benefits the technicians/small scale business owners by eradicating the need of transporting the tools to different locations where they do repairs/services.",
    "The purpose of the online flight management system is to ease flight management and to create a convenient and easy-to-use application for passengers, trying to buy airline tickets. The system is based on a relational database with its flight management and reservation functions. We will have a database server supporting hundreds of major cities around the world as well as thousands of flights by various airline companies. Above all, we hope to provide a comfortable user experience along with the best pricing available.",
    "Simple Library Management System using which a librarian can add book details like ISBN number, book title, author name, edition, and publication details through a web page. In addition to this, the librarian or any user can search for the available books in the library by the book name. If book details are present in the database, the search details are displayed.",
    "Container tracking app: Streamline your logistics operations with a container tracking application. Track the movement of your shipments in real time, ensuring transparency and efficiency throughout the supply chain. Get instant updates on container status, location, and estimated arrival times. Simplify the management of your cargo and enhance collaboration with shipping partners, empowering you to stay ahead in the global marketplace.",
    "Bike servicing app: A door-step bike servicing platform and application which will use technology for the convenience of two-wheeler owners by providing them a transparent connection with high-quality vehicle maintenance providers. The platform can provide assisted door-step pick-up and drop, an in-built inventory management system that enables reduction of waiting-time, smarter stock allocation, an order management system etc.",
    "Fitness App: A healthy lifestyle web application targeting health conscious people to track their habits assisted by registered nutritionists, pathologists and health coaches in order to ultimately lower the risk of lifestyle disorders. The application would be equipped with several charts that help the user manage their overall health- like weight, sugar, heart rate, blood pressure etc. User is also equipped with individual meal charts, lifestyle plans, nutrition plans as per their condition. It will also be integrated with chat facility that allows users to talk with the community as well as health professionals.",
    ]

@st.cache_resource
def generate_code(txt, lang_option):
    PROJECT_ID = "learning-351419"  # @param {type:"string"}
    vertexai.init(project=PROJECT_ID, location="us-central1")

    # Prompt Template
    code_gen_prompt_template = """You are a master software engineer. Based on the requirements provided below, write the code following solid {lang_option} programming practices. Add relevant code comments. Don't explain the code, just generate the code.
    {text}
    """
    
    code_gen_prompt_template = """
    Generate clean, well-structured code based on the requirements described below. Follow best practices for {lang_option} coding style, naming conventions, modularity, and documentation.

    Use proper indentation, spacing, and formatting to maximize readability. Apply precise descriptive names for variables, functions, classes, and files. Modularize functionality into reusable components with clear interfaces. Validate all inputs and handle errors gracefully. Include comprehensive comments and docstrings explaining logic and complex sections. Follow recommended coding guidelines and style guides for the language. Implement logging, error handling, and debuggable code.

    Structure the classes and functions logically to minimize complexity. Break code into small, single-responsibility functions. Use optimal data structures and algorithms to ensure efficiency. Write secure code free from vulnerabilities and risks. Test the code thoroughly covering edge cases. Focus on maintainability, scalability, and extensibility. Apply principles of good API design for interfaces. Generate clean, well-organized code that is compliant, functional, readable and production-ready.
    {text}
    """
    
    PROMPT = PromptTemplate(template=code_gen_prompt_template, input_variables=["text", "lang_option"])

    res = None

    # Instantiate the LLM model
    try:
        llm = VertexAI(model_name=primary_model_name, max_output_tokens=506, temperature=temperature, top_p=top_p, top_k=top_k, verbose=True,)

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
    prod_backlog_gen_prompt_template = """You are a master product manager. Based on the requirements provided below, generate a product backlog. Organize the backlog into epics and a list of asscoiated features. Assign a level of priority (High, Medium, Low) for each of the features. Include non-functional epics and features as well. List the backlog as a table with the following: Epic Name, Features, and Priority.
    {text}
    """
    
    prod_backlog_gen_prompt_template = """Based on the requirements below generate a prioritized product backlog for as a table with the following columns: Epic, Feature, Description, Priority

    Come up with 5-7 logical epics that group related features. Under each epic, list out 3-5 concrete features that are valuable to users. Write clear and concise descriptions for each epic and feature. Assign priority numbers from 1 to 10, with 1 being the highest priority feature. Ensure the features align to the product vision and strategy. Include a mix of must-have and nice-to-have capabilities. Structure the epics and features to maximize business value, mitigate risk, and allow for incremental delivery. Order the rows by priority with the most important items on top.
    {text}
    """    
    
    PROMPTbacklog = PromptTemplate(template=prod_backlog_gen_prompt_template, input_variables=["text"])

    res = None

    # Instantiate the LLM model
    try:
        llmbacklog = VertexAI(model_name=primary_model_name, max_output_tokens=506, temperature=temperature, top_p=top_p, top_k=top_k, verbose=True,)

        # Text summarization
        try:
            chain = LLMChain(prompt=PROMPTbacklog, llm=llmbacklog)
            res = chain.run({'text':txt})
        except Exception as e:
            st.error("Error during LLM model chaining and invocation")
            st.error(e)
    except Exception as e:
        st.error("Error during LLM model initialization")
        st.error(e)

    return res

@st.cache_resource
def generate_api(txt, backlog):
    # Prompt Template
    api_gen_prompt_template = """Based on the requirements below generate a lit of RESTful API's. Create definitions that follows domain drive API best practices. Use descriptive names and consistent conventions. Include HTTP methods, endpoint paths, request and response examples in JSON format. Document all endpoints thoroughly explaining the functionality, required parameters, sample requests/responses and error conditions. Implement proper authentication, input validation, error handling, rate limiting, and idempotent endpoints. Provide sensible defaults and optional parameters where applicable. Make the API intuitive and easy to use. Focus on simplicity without unnecessary complexity in the design. Use proper versioning and pagination. Follow REST principles and HTTP standards.
    {text}
    {backlog}
    """
    
    PROMPTapi = PromptTemplate(template=api_gen_prompt_template, input_variables=["text", "backlog"])

    res = None
    # Instantiate the LLM model
    try:
        llmapi = VertexAI(model_name=primary_model_name, max_output_tokens=506, temperature=temperature, top_p=top_p, top_k=top_k, verbose=True,)

        # Text summarization
        try:
            chain = LLMChain(prompt=PROMPTapi, llm=llmapi)
            res = chain.run({'text':txt, 'backlog': backlog})
        except Exception as e:
            st.error("Error during LLM model chaining and invocation")
            st.error(e)
    except Exception as e:
        st.error("Error during LLM model initialization")
        st.error(e)

    return res


@st.cache_resource
def generate_test_cases(txt, backlog):
    # Prompt Template    
    test_case_gen_prompt_template = """Using the details below generate a comprehensive quality assurance test plan and test cases for an application in a well-formatted table with columns: Test Case ID, Test Type, Description, Input Data, Expected Result

    Read through the provided application description and code. Create test cases to evaluate functionality, usability, performance, security, compatibility, reliability and other quality attributes. Outline superb test data covering valid, invalid, edge case scenarios. Specify detailed test steps and expected results. Include positive, negative, destructive, exploratory, regression, user acceptance testing. Evaluate against quality standards and requirements. Recommend optimal test environments, tools and techniques. Develop a formal test plan covering scope, schedule, time estimation, environment needs, metrics, team structure and responsibilities. Apply best practices for requirement traceability and risk based testing. Focus on maximizing test coverage and defect detection. Document all test cases clearly in an easy to read tabular format with proper alignment, spacing and headings. Produce a high quality, reusable test suite following industry standards and guidelines.
        
    {text}
    {backlog}
    """
    
    PROMPTtestcase = PromptTemplate(template=test_case_gen_prompt_template, input_variables=["text", "backlog"])

    res = None

    # Instantiate the LLM model
    try:
        llmtestcase = VertexAI(model_name=primary_model_name, max_output_tokens=506, temperature=temperature, top_p=top_p, top_k=top_k, verbose=True,)

        # Text summarization
        try:
            chain = LLMChain(prompt=PROMPTtestcase, llm=llmtestcase)
            res = chain.run({'text':txt, 'backlog':backlog})
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
    
    doc_gen_prompt_template = """Based on the requirements, backlog and code provided below, generate complete documentation for the application in the ReadTheDocs format following best practices. Start with a project overview explaining the purpose and key capabilities. Provide installation and usage instructions with example code snippets. Use a hierarchical structure with logical grouping and labeling. Include README, Getting Started, and Installation Instructions. Include detailed API documentation with explanations, parameters, input/output formats, exceptions etc. Document all classes and functions thoroughly with docstrings and type annotations. Create tutorials and how-to guides for common tasks and integrations. Explain key concepts, architecture and design decisions. Use diagrams and visuals to illustrate complex workflows. Link related topics for discoverability. Follow style principles to maximize readability - consistent formatting, succinct language, appropriate headings and highlighting. Use Markdown formatting for portability. Check for accuracy, completeness, and clarity throughout. Generate comprehensive, well-structured documentation that enables easy understanding and usage of the codebase.
    {text}
    {backlog}
    {code}
    """
    
    PROMPTdoc = PromptTemplate(template=doc_gen_prompt_template, input_variables=["text", "backlog", "code"])

    res = None

    # Instantiate the LLM model
    try:
        llmcode = VertexAI(model_name=primary_model_name, max_output_tokens=506, temperature=temperature, top_p=top_p, top_k=top_k, verbose=True,)

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

#button_random = st.button("Randomize App Description")
#if button_random:
app_val = "Fitness App: A healthy lifestyle web application targeting health conscious people to track their habits assisted by registered nutritionists, pathologists and health coaches in order to ultimately lower the risk of lifestyle disorders. The application would be equipped with several charts that help the user manage their overall health- like weight, sugar, heart rate, blood pressure etc. User is also equipped with individual meal charts, lifestyle plans, nutrition plans as per their condition. It will also be integrated with chat facility that allows users to talk with the community as well as health professionals."

#random_index = random.randint(1, len(sample_code_gen_qns))
#app_val = sample_code_gen_qns[random_index]

text_input = st.text_area('Describe in a few sentences the app you want to develop. I will then use AI generative capabilities to create some key artifacts that can help jumpstart your app development process.', value=app_val, height=200, key='fav1')

#randomize_button = st.button('Randomize', on_click=None)
#if randomize_button == True:
#    text_input = sample_code_gen_qns[random.randint(1, len(sample_code_gen_qns))]

col1, col2 = st.columns([1, 1])
with col1:
    lang_option = st.radio("Target Programming Language:", ('Python', 'Java', 'JavaScript', 'Go', 'Rust'), horizontal=True)
with col2:
    cloud_provider = st.radio("Target Cloud Platform:", ('GCP', 'Azure', 'AWS', 'Hybrid'), horizontal=True, disabled=True)
    
submit_button = st.button('**GENERATE**')


if submit_button:
    # states
    step1 = "Product Backlog"
    step2 = "API's" if "two" not in st.session_state else "API's"
    step3 = "Generated Code" if "three" not in st.session_state else "Generated Code"
    step4 = "Test Cases" if "four" not in st.session_state else "Test Cases"
    step5 = "Deployment Script"
    step6 = "Documentation" if "six" not in st.session_state else "Documentation"
    steps = [step1, step2, step3, step4, step5, step6]

    BACKLOG_GEN = False
    API_GEN = False
    CODE_GEN = False
    TESTCASE_GEN = False
    DOC_GEN = False
    
    #tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Product Backlog", "API's", "Generated Code", "Test Cases", "Deployment Script", "Documentation"])
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(steps)
    
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
    
    factoid_list = ["The average person's left hand does 56% of the typing.", "Sudan has more pyramids than any country in the world", "In Alaska, it is legal to shoot bears. However, waking a sleeping bear for the purpose of taking a photograph is prohibited.", "A 'jiffy' is an actual unit of time for 1/100th of a second.", "Floccinaucinihilipilification, the declaration of an item being useless, is the longest non-medical term in the English language.", "Before Google launched Gmail, “G-Mail” was the name of a free e-mail service offered by Garfield’s website.", "Les Misérables has a three-page, 823-word sentence that is divided by ninety-three commas, fifty-one semicolons, and four dashes. Why? According to rumors someone suffocated from lack of oxygen in the 1940's just short of the seventy-third comma while giving a dramatic reading from the work.", "The Main Library at Indiana University sinks over an inch every year because engineers failed to account for the weight of all the books that it would eventually hold.", "Ping Pong balls can travel off the paddle at speeds up to 160 km/hr. That's just under 100 mph.", "No word in the English language rhymes with 'MONTH.'", "Spiral staircases in medieval castles run clockwise. This is because all knights used to be right-handed. When the intruding army would climb the stairs, they would not be able to use their right hand, which was holding the sword, because of the difficulties of climbing the stairs. Left-handed knights would have had no trouble, except left-handed people could never become knights because they were assumed to be descendants of the devil.", "Hydrogen gas is the least dense substance in the world, at 0.08988g/cc."]
    
    with st.spinner("Hang tight for a few .... Wait for the magic ... "):
        
        progress_text = "I'm generating a set of things to bootstrap your app build." + '\n\n\n' + "💡 Did you know: " + random.choice(factoid_list)
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(.48)
            my_bar.progress(percent_complete + 1, text=progress_text)
        my_bar.empty()
        
        #time.sleep(10)
        with tab1:
            response_prod_backlog = generate_product_backlog(text_input.strip())
            result_prod_backlog.append(response_prod_backlog)
            
            # Display backlog
            if len(result_prod_backlog):
                st.subheader("Feature Backlog")
                st.write(response_prod_backlog)
                BACKLOG_GEN = True
        
        with tab2:
            if BACKLOG_GEN:
                response_api = generate_api(text_input.strip(), response_prod_backlog)
                result_api.append(response_api)
                st.session_state["two"] = True

            # Display api
            if len(result_api):
                st.subheader("API's")
                st.write(response_api)
                API_GEN = True
                
        with tab3:
            response_code = generate_code(text_input.strip(), lang_option)
            result_code.append(response_code)
            st.session_state["three"] = True

            # Display code
            if len(result_code):
                st.subheader("The Code")
                st.write(response_code)
                CODE_GEN = True
            
        with tab4:
            if BACKLOG_GEN:
                response_test_case = generate_test_cases(text_input.strip(), response_prod_backlog)
                result_test_case.append(response_test_case)
                st.session_state["four"] = True
            
            # Display test case
            if len(result_test_case):
                st.subheader("Test Cases")
                st.write(response_test_case)
                TESTCASE_GEN = True
                
        
        with tab5:
            st.subheader("Deployment Script")
            st.write("DevSecOps Coming soon...")
        
        with tab6:
            if CODE_GEN and BACKLOG_GEN:
                response_doc = generate_documentation(text_input.strip(), response_prod_backlog.strip(), response_code.strip())
                result_doc.append(response_doc)
                st.session_state["six"] = True
            
            # Display documentation
            if len(result_doc):
                st.subheader("Documentation")
                st.write(response_doc)
                DOC_GEN = True
