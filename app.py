# imports
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

# set page parameters
st.set_page_config(
    page_title="Chat with Streamlit docs powered by LlamaIndex",
    page_icon=":books:",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
    )
openai.api_key = st.secrets.openai_key

st.title("Chat with Streamlit docs, powered by LlamaIndex")
st.info("Copied from streamlit [blog here](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

# initialize chat message history with default start message if no messages in history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about streamlit."}
    ]
    
# load and index data in streamlit with llamaindex
@st.cache_resource(show_spinner=False)
def load_data():
    """
    Function to load data from a SimpleDirectoryReader, create a ServiceContext, and return a VectorStoreIndex.
    """
    with st.spinner(text="Loading and Indexing Streamlit docs. Should take a few minutes!"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.5,
                system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features.",
                verbose=True,
            ),
        )
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
    
index = load_data()

# initialize chat engine
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# prompt user for input and add the input to the message history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# pass query to chat engine and get response from chatbot 
# if last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assisant"):
        with st.spinner(text="Waiting for response..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)