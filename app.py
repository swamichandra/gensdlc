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
from vertexai.preview.language_models import TextGenerationModel

# Global Settings and Config
project_id = "learning-351419"
loc = "us-central1"
primary_model_name = "text-bison@001"
temperature = 0.2
max_output_tokens = 1024
top_p = 0.8
top_k = 40
location = "us-central1"
model_name = "text-bison@001"
vertexai.init(project=project_id, location=loc)


@st.cache_resource
def generate_random_input():
    INPUT_TEXT_TMP = ""
    sample_code_gen_qns = [
        "Generate a semantic HTML and Tailwind CSS Contact Support form consisting of the user name, email, issue type, and message. The form elements should be stacked vertically and placed inside a card", 
        "Write a JavaScript function. It accepts a full name as input and returns avatar letters.",
        "Given a string s, return the longest palindromic substring in s.",
        "Define a class with a generator which can iterate the numbers, which are divisible by 7, between a given range 0 and n",
        "An app to manage everyday tasks and items. This could include features such as adding tasks, deleting tasks, marking tasks as completed, and setting due dates for tasks.",
        "An app that can display the current weather conditions and forecast for a specific location. This could include features such as showing the temperature, humidity, wind speed, and precipitation.",
        "You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.",
        "Write an Express.js API to fetch the current user's profile information. It should make use of MongoDB", 
        "The database has students and course tables. Write a PostgreSQL query to fetch a list of students who are enrolled in at least 3 courses.", "Write a function that checks if a year is a leap year.",
        "You are given a string num, which represents a large integer. You are also given a 0-indexed integer array change of length 10 that maps each digit 0-9 to another digit. More formally, digit d maps to digit change[d]. You may choose to mutate a single substring of num. To mutate a substring, replace each digit num[i] with the digit it maps to in change (i.e. replace num[i] with change[num[i]]).Return a string representing the largest possible integer after mutating (or choosing not to) a single substring of num.",
        "A sentence is a list of words that are separated by a single space with no leading or trailing spaces. Each word consists of lowercase and uppercase English letters. A sentence can be shuffled by appending the 1-indexed word position to each word then rearranging the words in the sentence. For example, the sentence 'This is a sentence' can be shuffled as 'sentence4 a3 is2 This1' or 'is2 sentence4 This1 a3'. Given a shuffled sentence s containing no more than 9 words, reconstruct and return the original sentence.",
        "A company is planning to interview 2n people. Given the array costs where costs[i] = [aCosti, bCosti], the cost of flying the ith person to city a is aCosti, and the cost of flying the ith person to city b is bCosti. Return the minimum cost to fly every person to a city such that exactly n people arrive in each city.",
        "You are given a license key represented as a string s that consists of only alphanumeric characters and dashes. The string is separated into n + 1 groups by n dashes. You are also given an integer k. We want to reformat the string s such that each group contains exactly k characters, except for the first group, which could be shorter than k but still must contain at least one character. Furthermore, there must be a dash inserted between two groups, and you should convert all lowercase letters to uppercase. Return the reformatted license key.",
        "By using list comprehension, please write a program to print the list after removing the 0th,4th,5th numbers in [12,24,35,70,88,120,155].",
        "Define a function which can print a dictionary where the keys are numbers between 1 and 20 (both included) and the values are square of keys.",
    
    ]
    INPUT_TEXT_TMP = random.choice(sample_code_gen_qns)
    return INPUT_TEXT_TMP

@st.cache_resource
def generate_code(txt):
    PROJECT_ID = "learning-351419"  # @param {type:"string"}
    vertexai.init(project=PROJECT_ID, location="us-central1")

    # Prompt Template
    code_gen_prompt_template = """You are a master software engineer. Based on the requirements provided below, write the code following solid Python programming practices. Add relevant code comments. Don't explain the code, just generate the code.
    {text}
    """
    PROMPT = PromptTemplate(template=code_gen_prompt_template, input_variables=["text"])

    res = None

    # Instantiate the LLM model
    try:
        llm = VertexAI(model_name=primary_model_name, max_output_tokens=506,
                       temperature=0.1, top_p=0.8, top_k=40, verbose=True,)

        # Text summarization
        try:
            chain = LLMChain(prompt=PROMPT, llm=llm)
            res = chain.run(txt)
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
    test_case_gen_prompt_template = """You are a master software quality engineer. Based on the requirements and code provided below, generate test cases to validate features and functions. List the test cases ONLY as bullet points. For each of the bullet points include the folowing as nicely formatted sub-bullets: Test Case ID, Test Scenario, Test Steps and Expected Results.
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

# Page title
st.set_page_config(page_title="Generative AI Text Summarization App", page_icon=":random:", layout="centered")
st.write(f'<style>{css.v1}</style>', unsafe_allow_html=True)
st.title('📚 Generative AI Text Summarization App')

# Create a file upload widget for the credentials JSON file
creds_file = st.file_uploader(
    "Upload Google Cloud credentials file", type="json")

result = []
if creds_file is not None:
    now = datetime.now()
    randomfilename = "temp_credentials_" + now.strftime("%m%d%Y_%H%M%S")

    creds_contents = creds_file.read().decode("utf-8")
    with open(randomfilename, "w") as f:
        f.write(creds_contents)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = randomfilename
    
    # Using the "with" syntax
    #with st.form(key='sdlc_form', clear_on_submit = False):
    text_input = st.text_area('Tell me about your app', generate_random_input(), height=200, key='fav1')
    submit_button = st.button('Submit', key='rand1')
    #random_button = st.button('Randomize', key='randrand', on_click=text_input.value = generate_random_input())
    #st.write(random_button)
    
    col1, buff, col2 = st.columns([2, 1, 2])
    col3, buff, col4, col5 = st.columns([2, 1, 1, 1])
    
    if submit_button:
        result = []
        response = None
        result2 = []
        response2 = None
        with st.spinner('Wait for the magic ...'):
            with col1:
                st.subheader("Feature Backlog")
                st.write("Coming soon...")
            
            with col2:
                response = generate_code(text_input.strip())
                result.append(response)

                # Display code
                if len(result):
                    st.subheader("The Code")
                    st.write(response)
                
            with col3:
                #st.write("place holder for test cases")
                #st.write(response)
                response2 = generate_test_cases(text_input.strip(), response)
                result2.append(response2)
                
                # Display test case
                if len(result2):
                    st.subheader("Test Cases")
                    st.write(response2)
            
            with col4:
                st.subheader("Deployment Script")
                st.write("DevSecOps Coming soon...")
            
            with col5:
                st.subheader("Documentation")
                st.write("Coming soon...")
