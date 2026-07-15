import requests
import streamlit as st 
import os 

st.set_page_config(
    page_title="Enterprise AI System",
    page_icon="🤖",
    layout="wide"
)



API_URL=os.getenv("API_URL", "http://localhost:8000/ask")

st.title("Enterprise AI system")
st.caption("AI-Powered Enterprise Document Intelligence using Hybrid RAG")

question=st.text_input("Ask a Question")

if st.button("Ask"):
    if question.strip()=="":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer...."):
           try:
               response=requests.post(
                 API_URL,
                 json={"question":question},
                 timeout=120
               )
               
               response.raise_for_status()
               result=response.json()
               st.write(result)

           except requests.exceptions.RequestException as e:
               st.error(f"API Error: {e}")
               st.stop()

               st.subheader("Answer")
               st.markdown(result["answer"])

               st.subheader("Sources")

               for source in result["sources"]:
                   st.success(source)

