from ingestion.document_loader import load_documents
from ingestion.text_chunker import chunk_documents
from retrieval.bm25_retriever import build_bm25_index
from evaluation.evaluation_dataset import EVALUATION_DATASET
from retrieval.hybrid_retriever import hybrid_search
import math


def evaluate_retrieval():
    print("Loading documents...")
    documents=load_documents()
    chunks=chunk_documents(documents)
    build_bm25_index(chunks)
    print("BM25 Ready")

    total_questions=len(EVALUATION_DATASET)
    hits=0
    hit1=0
    hit3=0
    hit5=0
    mrr=0
    recall5=0
    precision5=0
    map_score=0
    ndcg=0

    for sample in EVALUATION_DATASET:

        question=sample["question"]
        expected_source=sample["expected_source"]

        retrieved_chunks=hybrid_search(question,top_k=5)

        retrieval_sources=[]

        for chunk in retrieved_chunks:
            source=chunk.get("metadata",{}).get("source")

            if source:
                retrieval_sources.append(source)

        retrieval_sources=list(dict.fromkeys(retrieval_sources))

        reciprocal_rank=0

        if isinstance(expected_source,list):

            ranks=[]
            for src in expected_source:
                if src in retrieval_sources:
                    ranks.append(retrieval_sources.index(src) + 1)

        
            if len(ranks)>0:
                reciprocal_rank=1/min(ranks)

        else:

            if expected_source in retrieval_sources:
                rank=retrieval_sources.index(expected_source)+1
                reciprocal_rank=1/rank
        mrr+=reciprocal_rank 

        dcg=0

        for i,src in enumerate(retrieval_sources,start=1):
            
            if isinstance(expected_source,list):
                if src in expected_source:
                    dcg +=1/math.log2(i+1)

            else:
                if src==expected_source:
                    dcg +=1/math.log2(i+1)
        
        if isinstance(expected_source,list):

            idcg=0
            for i in range(1,len(expected_source)+1):
                idcg +=1/math.log2(i+1)
        
        else:
            idcg=1

        ndcg_score=dcg/idcg if idcg>0 else 0
        ndcg +=ndcg_score

        
        if isinstance(expected_source,list):
            relevant=0
            for src in expected_source:
                if src in retrieval_sources[:5]:
                    relevant +=1
            recall=relevant/len(expected_source)
        else:
            recall=1 if expected_source in retrieval_sources[:5] else 0

        recall5 +=recall


        if isinstance(expected_source,list):
            relevant=sum(
                src in expected_source
                for src in retrieval_sources[:5]
            )
        else:
            relevant=sum(
                src == expected_source
                for src in retrieval_sources[:5]
            )
        precision=relevant/min(5,len(retrieval_sources))
        precision5 +=precision


        if isinstance(expected_source,list):
            relevant=0
            precision_sum=0

            for i,src in enumerate(retrieval_sources,start=1):
                if src in expected_source:
                    relevant +=1
                    precision_sum +=relevant/i
            average_precision=precision_sum/len(expected_source)
        else:
            average_precision=0
            if expected_source in retrieval_sources:
                rank=retrieval_sources.index(expected_source) +1
                average_precision=1/rank
        map_score +=average_precision

                
        if  isinstance(expected_source,list):
            hit_at_1=all(src in retrieval_sources[:1] for src in expected_source)
            hit_at_3=all(src in retrieval_sources[:3] for src in expected_source)
            hit_at_5=all(src in retrieval_sources[:5] for src in expected_source)           
        else:
            hit_at_1=expected_source in retrieval_sources[:1]
            hit_at_3=expected_source in retrieval_sources[:3]
            hit_at_5=expected_source in retrieval_sources[:5]
        if hit_at_1:
            hit1 +=1
        
        if hit_at_3:
            hit3 +=1

        if hit_at_5:
            hit5 +=1

        print("="*60)
        print("Question:")
        print(question)
        print()

        print("Expected:")
        print(expected_source)
        print()

        print("Retrieved:")
        print(retrieval_sources)
        print()

        print(f"Reciprocal Rank:{reciprocal_rank:.3f}")
        print(f"Recall@5:{recall:.3f}")
        print(f"Precision@5 :{precision:.3f}")
        print(f"Average Precision:{average_precision:.3f}")
        print(f"nDCG :{ndcg_score:.3f}")
        

        print(
            "Result:",
            "PASS" if hit_at_5 else "FAIL"
        )

    hit_rate=hits/total_questions
    recall5=recall5/total_questions
    precision5 /=total_questions
    map_score /=len(EVALUATION_DATASET)
    ndcg /=len(EVALUATION_DATASET)

    print("="*80)
    print(f"Total Questions:{total_questions}")
    print(f"Hit@1  :{hit1/total_questions:.2%}")
    print(f"Hit@3  :{hit3/total_questions:.2%}")
    print(f"Hit@5  :{hit5/total_questions:.2%}")
    print(f"MRR :{mrr/total_questions:.3f}")
    print(f"Recall@5 : {recall5:.3f}")
    print(f"Precision@5 :{precision:.3f}")
    print(f"MAP: {map_score:.3f}")
    print(f"nDCG : {ndcg:.3f}")


if __name__=="__main__":
    evaluate_retrieval()



