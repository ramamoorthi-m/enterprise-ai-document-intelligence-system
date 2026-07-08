from ingestion.document_loader import load_documents
from ingestion.text_chunker import chunk_documents
from retrieval.bm25_retriever import build_bm25_index
from rag_pipeline import ask
from evaluation.evaluation_dataset import EVALUATION_DATASET
from evaluation.llm_judge import judge_answer



print("Loading documents...")
documents=load_documents()
chunks=chunk_documents(documents)
build_bm25_index(chunks)
print("BM25 Ready")

total_questions=len(EVALUATION_DATASET)
faithfulness=0
answer_relevancy=0
correctness=0
completeness=0

for sample in EVALUATION_DATASET:
    question=sample["question"]
    ground_truth=sample["ground_truth"]

    generated_answer,contexts,sources=ask(question)

    context="\n\n".join(contexts)

    result=judge_answer(
        question=question,
        context=context,
        ground_truth=ground_truth,
        generated_answer=generated_answer
)

    print("="*80)
    print(f"Question: {question}")
    print()

    print(f"Faithfulness : {result['faithfulness']:.3f}")
    print(f"Answer Relevancy : {result['answer_relevancy']:.3f}")
    print(f"Correctness : {result['correctness']:.3f}")
    print(f"completeness : {result['completeness']:.3f}")

    print("\nReason:")
    print(result["reason"])

    faithfulness +=result["faithfulness"]
    answer_relevancy +=result["answer_relevancy"]
    correctness +=result["correctness"]
    completeness +=result["completeness"]

faithfulness /=total_questions
answer_relevancy /=total_questions
correctness /=total_questions
completeness /=total_questions

print("\n" + "=" *80)
print("FINAL GENERATION EVALUATION REPORT")
print("=" *80)
    
print(f"Total Questions : {total_questions}")

print(f"Average Faithfulness : {faithfulness:.3f}")
print(f"Average Answer Relevancy : {answer_relevancy:.3f}")
print(f"Average Correctnes : {correctness:.3f}")
print(f"Average Completeness : {completeness:.3f}")

print("="*80)