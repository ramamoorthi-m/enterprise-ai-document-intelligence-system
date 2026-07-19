from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import initialize
from rag_pipeline import ask

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 80)
    print("Initializing Enterprise AI System...")

    try:
        initialize()
        print("Initialization Complete.")
    except Exception as e:
        print(f"Initialization Failed: {e}")
    
    print("=" * 80)

    yield 

    print("Shutting down Enterprise AI System...")


app = FastAPI(
    title="Enterprise AI System API",
    description="Enterprise Document Intelligence using Hybrid RAG",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/")
def home():

    return {
        "message": "Enterprise AI System API Running"
    }

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    question= request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question Cannot be empty."
        )

    try:

        print("=" * 80)
        print(f"Question: {question}")

        answer, contexts, sources = ask(question)

        print("Request Completed")
        print("=" * 80)

        return QuestionResponse(
            question=question,
            answer=answer,
            sources=sources
        )
    except HTTPException:
        raise

    except Exception as e:

        print(f"API Error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        ) 