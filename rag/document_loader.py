from pathlib import Path
from pypdf import PdfReader

def load_documents(data_dir="data"):
    documents=[]

    pdf_files=Path(data_dir).glob("*.pdf")
    
    for pdf_file in pdf_files:
        reader=PdfReader(pdf_file)

        for page_no,page in enumerate(reader.pages):
            text=page.extract_text()

            if text and text.strip():
                documents.append({
                    "text":text,
                    "source":pdf_file.name,
                    "page":page_no + 1
                })
    return documents


if __name__=="__main__":
    documents=load_documents()
    print("Total pages:", len(documents))

    for i,doc in enumerate(documents):
        print("="*80)
        print(i)
        print(doc["source"])
        print(doc["page"])
        print(doc["text"][:300])
        