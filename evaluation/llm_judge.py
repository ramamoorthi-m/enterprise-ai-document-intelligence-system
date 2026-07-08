import json
from evaluation.judge_model import judge_completion

def judge_answer(question,context,ground_truth,generated_answer):
    prompt=f"""
    You are an expert evaluator for Retrieval-Augmented Generation(RAG)
    Evaluate the generated answer.

    Question:
    {question}

    Retrieved Context:
    {context}

    Ground_Truth:
    {ground_truth}

    Generated Answer:
    {generated_answer}

    Score the answer between 0 and 1.

    Evaluate:

    1.Faithfulnes
    -Is every claim supported by the retrieved context?

    2.Answer Relevancy
    -Does the answer actually answer the question?

    3.Correctness
    -Does the answer agree with the ground truth?

    4.Completeness
    -Does the answer include the important information from ground truth?

    Return ONLY raw JSON.
    Do NOT wrap the json inside markdown.
    Do NOT use '''json or '''.
    your response must begin with {{ and end with }}.

    {{
        "faithfulness":0.0,
        "answer_relevancy":0.0,
        "correctness":0.0,
        "completeness":0.0,
        "reason":""
    }}
    """
    response=judge_completion(prompt)
    cleaned=response.strip()
    cleaned=cleaned.replace("'''JSON","")
    cleaned=cleaned.replace("'''","")
    cleaned=cleaned.strip()
    result=json.loads(cleaned)
    return result

if __name__=="__main__":
    result=judge_answer(
        question="What is LoRA?",

        context="""
        LoRA freezes pretrained weights and trains low-rank metrices.
        """,

        ground_truth="""
        LoRA is a parameter-efficient fine-tuning technique.
        """,

        generated_answer="""
        LoRA freezes pretrained weights and trains low-rank metrices.
        """
    )
    print(json.dumps(result,indent=4))