import streamlit as st
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel
from typing import Annotated
import os
import time
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class Response(BaseModel):
    is_publishable: Annotated[bool, ..., "Whether the research paper is Publishable or not"]

def initialize_llm(api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=api_key
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def load_pdf(path: str):
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        content = []
        for doc in docs:
            content.append(doc.page_content)
        context = "".join(content)
        print(len(context.split(" ")))
        return context
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

def remove_stopwords_nltk(text):
    if text is None:
        return ""
    filtered_text = [word for word in text.split(" ") if word.lower() not in stop_words]
    return " ".join(filtered_text)

def invoke_with_retry(agent, prompt, max_retries=3, initial_delay=60, backoff_factor=2):
    retry_delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = agent.invoke(prompt)
            return response
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed: {e}")
            if "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    logging.info(f"Rate limit encountered. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= backoff_factor
                else:
                    st.error("Too many retries due to rate limit. Please try again later.")
                    return None
            else:
                st.error(f"Error during analysis: {e}")
                return None

def main():
    st.title("Research Paper Publishability Classifier")

    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("Enter your Google Gemini API Key:", type="password")

    st.sidebar.header("Upload Research Paper")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file", type=["pdf"]
    )

    if api_key:
        llm = initialize_llm(api_key)

        if llm is None:
            st.error("Please check your API Key")
            return

        if uploaded_file is not None:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            user_paper = load_pdf("temp.pdf")

            if user_paper:
                with st.spinner('Analyzing paper...'):
                    rp1 = load_pdf("R004.pdf")
                    rp3 = load_pdf("R010.pdf")

                    prompt = f"""
                        You are an expert Research Paper analyst. Based on the given Examples, classify the user's research paper into 'Publishable' and 'Non Publishable' Paper. Examples are without stopwords.
                        Non-publishable:
                        {remove_stopwords_nltk(rp1)}
                        Publishable:
                        {remove_stopwords_nltk(rp3)}
                        User's research paper:
                        {remove_stopwords_nltk(user_paper)}
                        """

                    agent = llm.with_structured_output(Response)
                    try:
                        response = invoke_with_retry(agent, prompt)
                        if response:
                            if response.is_publishable:
                                st.success("The research paper is classified as **Publishable**.")
                            else:
                                st.warning("The research paper is classified as **Non-Publishable**.")
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")

    else:
        st.warning("Please enter your Google Gemini API key in the sidebar to proceed.")

if __name__ == "__main__":
    main()
