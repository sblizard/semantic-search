#external imports
from fastapi import FastAPI

#built-in imports
from contextlib import asynccontextmanager

#local imports
from src.io import Query, ChatGPTResponse
from src.hyphoteticalDocument import create_hyphotetical_document
@asynccontextmanager
async def lifespan(app: FastAPI):
    #load model
    yield
    #close model

app: FastAPI = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/query")
async def search(query: Query):
    #TODO: create hyphotetical document
    #TODO: embed hyphotetical document
    #TODO: create query
    #TODO: embed query
    #TODO: search
    #TODO: return results
    return {"query": query.query}

@app.get("/hdTest")
async def create_hyphotetical_document_test() -> ChatGPTResponse:
    queryStr: str = "What is the capital of the moon?"
    query: Query = Query(query=queryStr)
    return await create_hyphotetical_document(query)

