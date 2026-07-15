from generation.llm_generator import generate_answer

if __name__=="__main__":
        prompt="""
        Context:
        LoRA is a parameter efficient fine tuning technique.

        Question:
        What is LoRA?

        Answer:
        """



        answer=generate_answer(prompt)

        print(answer)