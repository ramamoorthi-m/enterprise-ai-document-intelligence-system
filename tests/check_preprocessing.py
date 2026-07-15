from ingestion.preprocessing import preprocess_text

if __name__=="__main__":
    sample= """
       Hello      World!!

       Contact:
       abc@gmail.com

       Website:
       https://openai.com 

       This     is     a    test:

       Transformer\t\tuses\tself-attention.
       """

    print(preprocess_text(sample))


        