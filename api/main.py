from fastapi import FastAPI 
from pydantic import BaseModel 
from fastapi import HTTPException
from contextlib import asynccontextmanager
from rag_pipeline import ask, initialize

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Initializing Enterprise AI System...")
    initialize()
    print("Initialization Complete.")
    yield

app=FastAPI(
    title="Enterprise AI System API",
    version="1.0.0",
    lifespan=lifespan
)


class QuestionRequest(BaseModel):
    question: str 

class QuestionResponse(BaseModel):
    question: str 
    answer:str
    sources:list[str]    



@app.get("/")
def home():
    return {
        "message":"Enterprise AI System API is running"
        } 

@app.post("/ask",response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

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
            detail=f"Internal Server Error: {str(e)}"
        )