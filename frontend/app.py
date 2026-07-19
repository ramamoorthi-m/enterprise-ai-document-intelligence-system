import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

st.set_page_config(
    page_title="Enterprise AI System",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📄 Enterprise AI System")
st.caption("AI-Powered Enterprise Document Intelligence")

question = st.text_input(
    "Ask a question",
    placeholder="Ask questions about your enterprise documents using Hybrid Retrieval-Augmented Generation(RAG).",
)

if st.button("Ask", use_container_width=True):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching documents and generating answer..."):

        try:

            response = requests.post(
                API_URL,
                json={
                    "question": question
                 },
                timeout=60
            )

            if response.status_code == 200:

                result = response.json()

                st.success("Answer generated successfully.")

                st.divider()

                st.subheader("Answer")

                st.write(result["answer"])

                st.divider()

                st.subheader("Sources")

                if result["sources"]:

                    for source in result["sources"]:
                        st.markdown(f"- 📄 {source}")

                else:
                    st.info("No sources found.")

            else:

                st.error(f"API Error ({response.status_code})")

                st.json(response.json())

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI.\n\n"
                "Make sure the backend is running."
            )

        except requests.exceptions.Timeout:

            st.error("Request timed out.")

        except Exception as e:

            st.error(str(e))