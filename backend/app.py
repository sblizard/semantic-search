#external imports
from fastapi import FastAPI

#built-in imports
from contextlib import asynccontextmanager

#local imports
from src.io import Query, ChatGPTResponse, GetEmbeddingParams, EmbeddingOutput, UpsertInput, SearchOutput, SearchQuery
from src.query.hyphoteticalDocument import create_hyphotetical_document
from src.embed import upsert_vectors, get_embedding, semantic_search, search_by_embedding


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
    print("**********************")
    print(f'query: {query}')
    chatGPTResponse: ChatGPTResponse = await create_hyphotetical_document(query=query)
    print(f'GPT Response: {chatGPTResponse}')
    embeddedParams: GetEmbeddingParams = GetEmbeddingParams(text=chatGPTResponse.response)
    print(f' embeddedParams: {embeddedParams}')
    gptEmbedding: EmbeddingOutput = await get_embedding(params=embeddedParams)
    print(f'gptEmbedding: {gptEmbedding}')
    searchResults: SearchOutput = await search_by_embedding(embedding=gptEmbedding)
    print(f'searchResult: {searchResults}')
    print("**********************")
    return searchResults


@app.get("/hdTest")
async def create_hyphotetical_document_test() -> ChatGPTResponse:
    queryStr: str = "What is the capital of the moon?"
    query: Query = Query(query=queryStr)
    return await create_hyphotetical_document(query)


@app.post("/embed")
async def embed(text: GetEmbeddingParams) -> EmbeddingOutput:
    embeddingOut: EmbeddingOutput = await get_embedding(params=text)
    upsert_vectors(UpsertInput(data=embeddingOut, metadata={"text": text.text}))
    return embeddingOut


@app.post("/search")
async def search_endpoint(query: SearchQuery) -> SearchOutput:
    return await semantic_search(query=query)