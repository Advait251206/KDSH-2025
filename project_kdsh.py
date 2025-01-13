import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel
from typing import Annotated
load_dotenv()

class Response(BaseModel):
  is_publishable:Annotated[bool,..., "Whether the research paper is Publishable or not"]


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
    # other params...
)


def load_pdf(path:str):
  loader = PyPDFLoader(
      path
  )
  docs=loader.load()
  content = []
  for doc in docs:
    content.append(doc.page_content)
  context="".join(content)
  print(len(context.split(" ")))
  return context



nltk.download('stopwords')  # Download stopwords data

def remove_stopwords_nltk(text):
    stop_words = set(stopwords.words('english'))  # You can change the language if needed

    filtered_text = [word for word in text.split(" ") if word.lower() not in stop_words]
    print(len(filtered_text))
    return " ".join(filtered_text)

#Non-publishable
rp1=load_pdf("R004.pdf")
rp2=load_pdf("R002.pdf")
#Publishable
rp3=load_pdf("R010.pdf")
rp4=load_pdf("R014.pdf")
#Test
test1=load_pdf("P001.pdf")
test2=load_pdf("P002.pdf")
test3=load_pdf("P003.pdf")
test4=load_pdf("P004.pdf")
test5=load_pdf("P005.pdf")

prompt=f"""
You are an expert Reasearch Paper analyst. Based on the given Examples, classify the user's research paper into 'Publishable' and 'Non Publishable' Paper.Examples are without stopwords.
Non-publishable:
{remove_stopwords_nltk(rp1)}
Publishable:
{remove_stopwords_nltk(rp3)}
User's research paper:
{remove_stopwords_nltk(rp4)}
"""


agent=llm.with_structured_output(Response)
agent.invoke(prompt)

