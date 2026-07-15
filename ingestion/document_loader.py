"""
Loads PDF documents  from the data folder 
and extracts page-level text
with metadata.
"""

from pathlib import Path
import fitz
from ingestion.preprocessing import preprocess_text

def load_documents(data_dir: str="data/pdfs"):
    """
    Load all PDF documents from the specified directory.

    Each page is extracted, preprocessed, and stored with metadata.

    Args:
        data_dir (str):
            Directory containing PDF files.

    Returns:
        list[dict]:
            List of dictionaries containing:
                - text
                - source
                - page
    """

    documents=[]

    pdf_files=sorted(Path(data_dir).glob("*.pdf"))


    for pdf_file in pdf_files:

        pdf=fitz.open(pdf_file)

        for page_number,page in enumerate(pdf):

            text=page.get_text()

            if text:
               text=preprocess_text(text)

            if not text.strip():
                continue

            documents.append({
                "text":text,
                "source":pdf_file.name,
                "page":page_number +1
       
       
       
            })
        
        pdf.close()
    return documents

if __name__=="__main__":
    documents=load_documents()
    print(f"Loaded {len(documents)} pages\n")
    for document in documents[:5]:
        print("="*80)
        print("Source:",document["source"])
        print("Page:",document["page"])
        print(document["text"][:400])
