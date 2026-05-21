# 📄 Research Paper Publishability Classifier

### *KDSH 2025 – AI-Driven Academic Evaluation Engine*

---

<div align="center">
  <img src="https://img.shields.io/badge/Competition-Kharagpur_Data_Science_Hackathon-blue?style=for-the-badge&logo=hackaday" alt="Hackathon Badge" />
  <img src="https://img.shields.io/badge/Model-Gemini_1.5_Pro-9C27B0?style=for-the-badge&logo=google-gemini" alt="Gemini Badge" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit Badge" />
</div>

---

## 🎯 Project Overview

This repository houses the **Research Paper Publishability Classifier** built for **KDSH 2025** (Kharagpur Data Science Hackathon). 

The platform is designed to automate the initial screening of scientific manuscripts. Leveraging the state-of-the-art **Gemini 1.5 Pro** LLM via LangChain, it performs few-shot semantic analysis to classify academic drafts into either **Publishable** or **Non-Publishable** categories.

---

## 🛠️ Advanced Architectural Features

* **🧠 Few-Shot Learning Context:** Loads reference benchmark papers (`R004.pdf` as a baseline for non-publishable drafts, and `R010.pdf` as a baseline for standard publishable work) to align the LLM's classification logic.
* **✂️ Context Footprint Reduction:** Leverages NLTK tokenizer pipelines to completely strip stopwords from the context blocks of the loaded PDFs, significantly saving API tokens and accelerating the classification cycle.
* **🛡️ Structured Output Verification:** Utilizes structured schemas powered by **Pydantic V2** to constrain LLM responses into strict, type-safe boolean objects (`is_publishable`).
* **⏳ Robust Rate-Limit Handlers:** Implements custom exponential backoff loops to gracefully handle high-frequency request rate limits during massive dataset evaluations.

---

## 🧬 Tech Stack & Dependencies

The system is fully written in Python and is highly modularized:
* **Core LLM Interface:** `langchain_google_genai` 🦜🔗
* **Document Parser:** `PyPDFLoader` (via `langchain_community` & `pypdf`) 📄
* **Structured Modeling:** `pydantic` 🛡️
* **Web Client Interface:** `streamlit` 🎨
* **Semantic Filters:** `nltk` (Natural Language Toolkit stopwords corpus) 🌾
* **Environment Controller:** `python-dotenv` 🔑

---

## 🚀 Local Installation & Quick Start

Get the classifier running locally on your environment by executing the following steps:

### **1. Clone & Set Up Directory**
Navigate to your KDSH workspace and establish a dedicated virtual environment:
```bash
# Set up a Python virtual environment
python -m venv venv
source venv/Scripts/activate     # On Windows
# source venv/bin/activate       # On Linux/macOS
```

### **2. Install Dependencies**
Install NLTK, Streamlit, Langchain, and Pypdf extensions:
```bash
pip install -r requirements.txt
```

### **3. Set Up Credentials**
Create a `.env` file in the root of the `KDSH` folder to configure your credentials:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### **4. Boot the Streamlit Web Application**
Start the local server. The interface will automatically load in your default browser:
```bash
streamlit run app.py
```

---

## 📂 Repository Layout

```
├── .gitignore         # Ignores pycache, .env, and uploaded temporary PDFs
├── README.md          # Visual project roadmap and architecture description
├── app.py             # Streamlit graphical user interface and runner
├── project_kdsh.py    # Modular Python scripting pipeline for custom datasets
├── requirements.txt   # Complete list of verified project packages
├── R004.pdf           # Benchmarked Non-Publishable sample research paper
├── R010.pdf           # Benchmarked Publishable sample research paper
└── temp.pdf           # Temporary cache for user-uploaded manuscripts
```

---

<br>
<p align="center">
  <i>Developed to streamline scholarly evaluation and bring AI alignment to academic research 🔬.</i>
</p>