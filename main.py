import streamlit as st
from langchain import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO

def load_LLM(deepseek_api_key):
    # Make sure your openai_api_key is set as an environment variable
    llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    api_key=deepseek_api_key
)
    return llm

#Page title and header
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")


#Intro: instructions
col1, col2 = st.columns(2)

with col1:
    st.markdown("DeepSeek non può riassumere testi lunghi. Ora lo puoi fare con questa App.")

with col2:
    st.write("Contatta [SU] per costruire i tuoi Progetti AI")

#Input DeepSeekAI API Key
st.markdown("## Enter Your DeepSeek API Key")

def get_openai_api_key():
    input_text = st.text_input(label="DeepSeek API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="deepseek_api_key_input", type="password")
    return input_text

deepseek_api_key = get_openai_api_key()

# Input
st.markdown("## Upload the text file you want to summarize")

uploaded_file = st.file_uploader("Choose a file", type="txt")

# Output
st.markdown("### Here is your Summary:")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)

    file_input = string_data

    if len(file_input.split(" ")) > 20000:
        st.write("Please enter a shorter file. The maximum length is 20000 words.")
        st.stop()

    if file_input:
        if not deepseek_api_key:
            st.warning('Please insert OpenAI API Key. \
            Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
            st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], 
        chunk_size=5000, 
        chunk_overlap=350
        )

    splitted_documents = text_splitter.create_documents([file_input])

    llm = load_LLM(deepseek_api_key=deepseek_api_key)

    summarize_chain = load_summarize_chain(
        llm=llm, 
        chain_type="map_reduce"
        )

    summary_output = summarize_chain.run(splitted_documents)

    st.write(summary_output)