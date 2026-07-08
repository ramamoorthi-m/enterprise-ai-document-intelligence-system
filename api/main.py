from fastapi import FastAPI 
from pydantic import BaseModel 
from fastapi import HTTPException
from rag_pipeline import ask, initialize

app=FastAPI()

initialize()


class QuestionRequest(BaseModel):
    question: str 

class QuestionResponse(BaseModel):
    question: str 
    answer:str
    sources:list[str]    



@app.get("/")
def home():
    return {
        "message":"Welcome to Enterprise AI System"
        } 

@app.post("/ask",response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    try:

        answer,contexts,sources=ask(request.question)

    

        return {
             "question": request.question,
             "answer": answer,
             "sources":sources
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )