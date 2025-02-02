# app.py
import os
import tempfile
import csv
import streamlit as st
from pydantic import BaseModel
from phi.assistant import Assistant
from phi.vectordb.chroma import ChromaDb
from phi.embedder.openai import OpenAIEmbedder
from phi.document.reader.pdf import PDFReader
from phi.tools.file import FileTools
import base64
from io import BytesIO
from pdf2image import convert_from_path
from phi.knowledge import AssistantKnowledge
from typing import Generator

# Fix for Colab (MUST BE FIRST)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Streamlit setup
st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon=":scroll:",
)

# Get OpenAI key securely
openai_key = st.text_input("OpenAI API Key", type="password")
if not openai_key:
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_key

# Data model
class ContractAnalysis(BaseModel):
    file_name: str
    summary: str
    key_terms: list[str]
    obligations: list[str]
    risks: list[str]

# Initialize ChromaDB with correct parameters
def get_vector_db():
    return ChromaDb(
        collection="legal_contracts",
        embedder=OpenAIEmbedder(model="text-embedding-3-small"),
        path="./chroma_db"
    )

# PDF processing function
def process_pdf(file_path: str, file_name: str):

    # Initialize PDF Reader with chunking
    pdf_reader = PDFReader(chunk=True)
    
    # 1. Extract text from PDF
    documents = pdf_reader.read(file_path)
    
    # Filter empty docs and create unique IDs
    unique_docs = []
    for idx, doc in enumerate(documents):
        if not doc.content.strip():
            continue
        doc.id = f"{file_name}_{idx}"
        unique_docs.append(doc)
    
    text_content = "\n".join([doc.content for doc in unique_docs])
    
    # 2. Analyze images
    images = convert_from_path(file_path)
    image_descriptions = []
    
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        vision_assistant = Assistant(model="gpt-4-vision-preview", streaming=False)
        response = vision_assistant.run(
            "Describe this legal document image in detail, focusing on text and terms",
            images=[{"url": f"data:image/jpeg;base64,{img_base64}"}],
        )
        # Handle generator response
        image_description = "".join(response) if hasattr(response, '__iter__') else str(response)
        image_descriptions.append(image_description)
        buffered.close()

    # 3. Combine content
    full_content = f"Text:\n{text_content}\n\nImages:\n{' '.join(image_descriptions)}"
    
    # 4. Store in vector DB
    vector_db = get_vector_db()
    
     # Create collection if not exists
    try:
        collection = vector_db.client.get_collection("legal_contracts")
    except Exception:
        collection = vector_db.client.create_collection("legal_contracts")
    
    # Delete existing entries for this file
    collection.delete(where={"file_name": file_name})
    
    # Add metadata and insert
    for doc in unique_docs:
        doc.metadata.update({"file_name": file_name})
    vector_db.insert(documents=unique_docs)
    
    return full_content

# Analysis Assistant
def get_analysis_assistant():
    return Assistant(
        model="gpt-4o",
        tools=[FileTools()],
        knowledge_base=AssistantKnowledge(
            vector_db=get_vector_db(),
            num_documents=3
        ),
        system_prompt="""Analyze legal contracts and respond ONLY with valid JSON containing:
        - summary: string
        - key_terms: list of strings
        - obligations: list of strings  
        - risks: list of strings""",
        output_model=ContractAnalysis,
        read_chat_history=False
    )

# Streamlit UI
st.title("Legal Contract Analyzer :scroll:")

uploaded_files = st.file_uploader(
    "Upload PDF Contracts", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    results = []
    with st.status("Processing...", expanded=True) as status:
        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_path = tmp_file.name

                content = process_pdf(file_path, uploaded_file.name)
                assistant = get_analysis_assistant()

                analysis = assistant.run(
                    messages=[
                        {"role": "user", "content": "Extract: summary, key terms, obligations, risks"},
                        {"role": "user", "content": content},
                    ],
                    output_model=ContractAnalysis,
                )
    
                # Ensure we get the model instance (handle streaming responses)
                if isinstance(analysis, ContractAnalysis):
                    valid_analysis = analysis
                else:
                    # Handle generator/text responses
                    if isinstance(analysis, (Generator, list, tuple)):
                        raw_response = " ".join(str(item) for item in analysis)
                        valid_analysis = ContractAnalysis.model_validate_json(raw_response)
                    else:
                        valid_analysis = ContractAnalysis.model_validate_json(str(analysis))

                results.append({
                    "File Name": uploaded_file.name,
                    "Summary": valid_analysis.summary,
                    "Key Terms": ", ".join(valid_analysis.key_terms),
                    "Obligations": ", ".join(valid_analysis.obligations),
                    "Risks": ", ".join(valid_analysis.risks),
                })

                st.success(f"Processed: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Analysis failed for {uploaded_file.name}: {str(e)}")
    if results:
        csv_path = "contract_analysis.csv"
        with open(csv_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        st.download_button(
            "Download CSV",
            open(csv_path).read(),
            "contract_analysis.csv",
            "text/csv"
        )
