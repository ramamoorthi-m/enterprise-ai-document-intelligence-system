"""
    Text preprocessing utilities for document ingestion.
    This module cleans extracted PDF text while preserving 
    its semantic meaning for embedding and retrieval.

    """
import re
import unicodedata

def preprocess_text(text:str):
    
    #Normalize unicode characters
    text= unicodedata.normalize("NFKC", text)

    #Remove email addresses'
    text=re.sub(r"\S+@\S+", " ", text)

    #Remove URLs
    text=re.sub(r"http\S+|www\.\S+", " ", text)

    #Replace tabs with spaces
    text=text.replace("\t", " ")

    #Collapse multiple spaces
    text=re.sub(r"[ \t]{2,}", " ", text)

    #Collapse multiple blank lines
    text=re.sub(r"\n{3,}", "\n\n",  text)

    #Trim leading/trailing whitespace
    text=text.strip()

    return text


if __name__=="__main__":
    sample= """
       Hello      World!!

       Contact:
       abc@gmail.com

       Website:
       https://openai.com 

       This     is     a    test:
       """

    print(preprocess_text(sample))


        
