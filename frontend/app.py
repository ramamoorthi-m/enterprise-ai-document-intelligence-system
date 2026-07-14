import requests
import streamlit as st 
import os 

API_URL=os.getenv("API_URL", "http://localhost:8000/ask")

st.title("Enterprise AI system")
question=st.text_input("Ask a Question")
if st.button("Ask"):
    if question.strip()=="":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer...."):
            response=requests.post(
                API_URL,
                json={"question":question}
            )

            result=response.json()

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")

            for source in result["sources"]:
                st.write(f"- {source}")

