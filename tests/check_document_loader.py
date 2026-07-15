from ingestion.document_loader import load_documents

if __name__=="__main__":
    documents=load_documents()
    print(f" Total Loaded {len(documents)} pages\n")
    for document in documents[:3]:
        print("="*80)
        print("Source:",document["source"])
        print("Page:",document["page"])
        print(document["text"][:400])