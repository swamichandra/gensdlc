import streamlit as st
import streamlit.components.v1 as components

from google.cloud import aiplatform
import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
from vertexai.preview.language_models import TextGenerationModel
from google.oauth2 import service_account
import os

st.set_page_config(page_title="Reimagine SLDC with Google Vertex, Bard, PaLM-2", page_icon=":random:", layout="wide")
st.title('Reimagine SLDC with Google Vertex, Bard, PaLM-2')


# Global Settings and Config
project_id = "learning-351419"
model_name = "chat-bison@001"
temperature = 0.2
max_output_tokens = 1024
top_p = 0.8
top_k = 40
location = "us-central1"
model_name = "text-bison@001"

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="learning-351419-2f2aca25aadf.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account.Credentials.from_service_account_file('learning-351419-2f2aca25aadf.json')

#bootstrap text for text area
ta_val = '''* CBDEM1 IS A SIMPLE EXAMPLE PROGRAM WHICH ADDS NEW EMPLOYEE
* ROWS TO THE PERSONNEL DATA BASE. CHECKING IS DONE TO
* INSURE THE INTEGRITY OF THE DATA BASE. EMPLOYEE NUMBERS
* ARE AUTOMATICALLY SELECTED USING THE CURRENT MAXIMUM
* EMPLOYEE NUMBER AS THE START. DUPLICATE NUMBERS ARE SKIPPED.
* THE PROGRAM QUERIES THE USER FOR DATA AS FOLLOWS:
*
*		 Enter employee name  :
*		 Enter employee job   :
*		 Enter employee salary:
*		 Enter employee dept  :
*
* TO EXIT THE PROGRAM, ENTER A CARRIAGE RETURN AT THE
* PROMPT FOR EMPLOYEE NAME. IF THE ROW IS SUCCESSFULLY 
* INSERTED, THE FOLLOWING IS PRINTED:
*
* ENAME added to DNAME department as employee # NNNNN
*
* THE MAXIMUM LENGTHS OF THE 'ENAME', 'JOB', AND 'DNAME'
* COLUMNS WILL BE DETERMINED BY THE ODESCR CALL.'''

with st.form(key='my_form_to_submit'):
    user_request = st.text_area("The program requirements or comments from an existing program header", height=50, value=ta_val)
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

    with st.spinner(text="Converting your software requirements into a bootstrappable code ..."):
        # Create a VertexAI client object.
        #Predict using a Large Language Model.
        vertexai.init(project=project_id, location=location, credentials=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],)
        model = TextGenerationModel.from_pretrained(model_name)
        
        parameters = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "top_p": top_p,
        "top_k": top_k,
        #credentials: Optional[google.auth.credentials.Credentials] = None,
        }
        
        #chat = chat_model.start_chat(context=ctx, examples=[])
        #response=chat.send_message(prompt,**parameters)
        #st.markdown(response.text)
        
        response = model.predict(intro+prompt+ctx, temperature=temperature, max_output_tokens=max_output_tokens, top_k=top_k, top_p=top_p,)
        st.markdown(response.text)
        #predict_large_language_model_sample("learning-351419", "text-bison@001", 0.2, 1024, 0.8, 40, '''Generate Python code based on the program description provided in the contextGenerate a well written Python code following the top-10 Python coding practices. GUIDELINES: Use descriptive variable names. Use comments to explain your code. Use white space to format your code. Use functions to break down your code into smaller, more manageable chunks. Use descriptive names for all variables, function names, constants, and other identifiers. Follow the PEP 8 style guide. Break your code into functions. Use modules and packages. Document the generated code. I only need the code.
