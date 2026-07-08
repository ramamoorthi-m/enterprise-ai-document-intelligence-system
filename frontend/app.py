import requests
import streamlit as st 
st.title("Enterprise AI system")
question=st.text_input("Ask a Question")
if st.button("Ask"):
    if question.strip()=="":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answe...."):
            response=requests.post(
                "http://127.0.0.1:8000/ask",
                json={"question":question}
            )

            result=response.json()

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")

            for source in result["sources"]:
                st.write(f"- {source}")

