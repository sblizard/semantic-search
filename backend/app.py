#external imports
from fastapi import FastAPI

#built-in imports
from contextlib import asynccontextmanager

#local imports
from src.io import Query, ChatGPTResponse, GetEmbeddingParams, EmbeddingOutput, UpsertInput, SearchOutput, SearchQuery
from src.query.hyphoteticalDocument import create_hyphotetical_document
from src.embed import upsert_vectors, get_embedding, semantic_search


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
    #TODO: search
    #TODO: return results
    return {"query": query.query}


@app.get("/hdTest")
async def create_hyphotetical_document_test() -> ChatGPTResponse:
    queryStr: str = "What is the capital of the moon?"
    query: Query = Query(query=queryStr)
    return await create_hyphotetical_document(query)


@app.post("/embed")
async def embed(text: GetEmbeddingParams) -> EmbeddingOutput:
    embeddingOut: EmbeddingOutput = await get_embedding(text)
    upsert_vectors(UpsertInput(data=embeddingOut, metadata={"text": text.text}))
    return embeddingOut

@app.post("/search")
async def search_endpoint(query: SearchQuery) -> SearchOutput:
    return await semantic_search(query=query)