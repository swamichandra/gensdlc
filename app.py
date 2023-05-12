import streamlit as st
import streamlit.components.v1 as components

import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
import os
from google.cloud import aiplatform


st.set_page_config(page_title="Reimagine SLDC with Google Vertex, Bard, PaLM-2", page_icon=":random:", layout="wide")
st.title('Reimagine SLDC with Google Vertex, Bard, PaLM-2')


# Global Settings and Config
project_id = "swamigenaihive"
model_name = "chat-bison@001"
temperature = 0.2
max_output_tokens = 256
top_p = 0.8
top_k = 40
location = "us-central1"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="swamigenaihive-f663031ef703.json"

with st.form(key='my_form_to_submit'):
    user_request = st.text_area("The program requirements or comments from an existing program header")
    submit_button = st.form_submit_button(label='Convert')
    
if submit_button:
    # check if the user has entered a request
    if not user_request:
        st.error('Please enter a request')
        st.stop()
    
    # role-prompting where we instuct GPT
    intro = 'Your task is to act as a senior software engineer who has good knowledge of both COBOL and Python. You will complete a task and write the results. I have provided the DESCRIPTION of what the COBOL program does in the context.'
    
    # the user entered requirements in natural language 
    ctx = f' {user_request}.\n'
    
    prompt = f'Generate Python code based on the program description provided in the context'
    
    prompt += f'Generate a well written Python code following the top-10 Python coding practices. GUIDELINES: Use descriptive variable names. Use comments to explain your code. Use white space to format your code. Use functions to break down your code into smaller, more manageable chunks. Use descriptive names for all variables, function names, constants, and other identifiers. Follow the PEP 8 style guide. Break your code into functions. Use modules and packages. Document the generated code. I only need the code.'
    
    with st.expander("See Prompt"):
        st.code(prompt)

    with st.spinner(text="Converting your text query to KSQL and executing ..."):
        # Create a VertexAI client object.
        """Predict using a Large Language Model."""
        vertexai.init(project=project_id, location=location, credentials="AIzaSyCtqJUjWvVB96LE6YkO7m1BLujha204oMk",)#credentials=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],)
        
        # Create a VertexAI client object.
        #client = vertexai.Client()

        # Use the client object to initialize the vertexai.init() function.

        
        chat_model = ChatModel.from_pretrained(model_name)
        parameters = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "top_p": top_p,
        "top_k": top_k,
        #credentials: Optional[google.auth.credentials.Credentials] = None,
        }
        
        chat = chat_model.start_chat(context=ctx, examples=[])
        response=chat.send_message(prompt,**parameters)
        st.markdown(response.text)