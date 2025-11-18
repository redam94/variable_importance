from typing import TypedDict, Union
import streamlit as st
from langchain_ollama import ChatOllama
from langchain.messages import AIMessage, HumanMessage, SystemMessage
import httpx
import tempfile
import os
import asyncio
from model import code_workflow, OutputCapturingExecutor, OutputManager

def fetch_models():
    url = "http://100.91.155.118:11434/v1/models"
    response = httpx.get(url)
    if response.status_code == 200:
        return [model['id'] for model in response.json()['data']]
    else:
        return {"error": f"Failed to fetch models: {response.status_code}"}
    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages: list[Union[HumanMessage, AIMessage, SystemMessage]] = []

executor = OutputCapturingExecutor()

models = fetch_models()
with st.sidebar:
    st.header("Available Models")
    model = st.selectbox("Select Model", options=models, index=0)
    clear_chat = st.button("Clear Chat")
    if clear_chat:
        st.session_state.messages = []
    work_id = st.text_input("Work ID", value="test_work_id")
    if work_id:
        output_mgr = OutputManager(workflow_id=work_id)
    else:
        output_mgr = OutputManager(workflow_id="default_work_id")
    stage_name = st.text_input("Stage Name", value="default_stage")

    st.write(stage_name)    
    uploaded_file = st.file_uploader("Upload a file", type=["csv"])
temp_file_path = None

if uploaded_file is not None:
    # Create a NamedTemporaryFile to store the uploaded file
    # The 'delete=False' argument ensures the file is not deleted immediately after closing,
    # allowing access by other processes if needed. Remember to manage its deletion later.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        # Write the content of the uploaded file to the temporary file
        temp_file.write(uploaded_file.getvalue())
        
        # Get the path of the temporary file
        temp_file_path = temp_file.name

    st.success(f"File saved temporarily at: {temp_file_path}")

    # You can now use temp_file_path with other libraries or functions
    # For example, to display the content of a text file:
    if uploaded_file.type == "text/plain":
        with open(temp_file_path, "r") as f:
            st.text_area("File Content", f.read())

llm = ChatOllama(
    model=model,
    temperature=0.0,
    base_url="http://100.91.155.118:11434"
    # other params...
)

st.title("Echo Bot")
st.write(f"This is a simple echo bot using Streamlit's chat interface and Ollama's LLMs. {model}")

class Message(TypedDict):
    role: str
    content: str

def response_generator(messages: list[Union[HumanMessage, AIMessage, SystemMessage]]) -> str:
    for message in llm.stream(messages):
        yield message.content

tabs = st.tabs(["Chat", 'Artifacts'])
with tabs[1]:
    st.header("Artifacts")
    if stage_name:
        stage_dir = output_mgr.get_stage_dir(stage_name) / 'plots'
        st.write(f"Artifacts for stage: {stage_name}")
        if stage_dir.exists():
            for artifact in stage_dir.iterdir():
                st.image(artifact)
        else:
            st.write("No artifacts found for this stage.")
    if st.button("Refresh Artifacts"):
        st.rerun()
with tabs[0]:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role = 'user' if isinstance(message, HumanMessage) else 'assistant'
        content = message.content
        if "```python" in content:
            with st.chat_message("code", avatar=":material/code:"):
                with st.expander("Code"):
                    st.markdown(content.replace("Code:\n\n", ""))
        elif "```" in content and not "```python" in content:
            role = "code_output"
            with st.chat_message(role, avatar=":material/terminal:"):
                with st.expander("Code Output"):
                    st.markdown(content)
        else:
            with st.chat_message(role):
                st.markdown(content)


    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        #response = llm.invoke(st.session_state.messages)
        # Display assistant response in chat message container
        # with st.chat_message("assistant"):
        #     response = st.write_stream(response_generator(st.session_state.messages))
        with st.spinner("Thinking..."):
            response = asyncio.run(code_workflow.ainvoke(
                {
                    "messages": st.session_state.messages,
                    "input_data_path": temp_file_path,
                    "stage_name": stage_name
                },
                context={
                    'executor': executor,
                    'output_manager': output_mgr
                }
            ))
        if 'code' in response:
            code = response['code']
            with st.chat_message("code", avatar=":material/code:"):
                with st.expander("Code"):
                    st.markdown(f"```python\n{code}\n```")
        
        if 'code_output' in response:
            code_output = response['code_output']
            with st.chat_message("code_output", avatar=":material/terminal:"):
                with st.expander("Code Output"):
                    st.markdown(f"**Stdout:**\n```\n{code_output.stdout}\n```\n\n**Errors:**\n```\n{code_output.error}\n```")

        with st.chat_message("assistant"):
            
            st.write(response.get('summary', 'No summary available'))

        # Add assistant response to chat history
        if 'code' in response:
            st.session_state.messages.append(AIMessage(content=f"Code:\n\n```python\n{response['code']}\n```"))
        if 'code_output' in response:
            st.session_state.messages.append(AIMessage(content=f"**Stdout:**\n\n```\n{response['code_output'].stdout}\n```\n\n**Errors:**\n```\n{response['code_output'].error}\n```"))
        st.session_state.messages.append(AIMessage(content=response['summary']))



