#external imports
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Dict
from pdfplumber import open as open_pdf
import io
import mammoth
from semantic_text_splitter import TextSplitter

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

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile) -> dict:
    try:
        file_contents = await file.read()

        if file.filename.endswith(".pdf"):
            with io.BytesIO(file_contents) as pdf_stream:
                with open_pdf(pdf_stream) as pdf:
                    text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

        elif file.filename.endswith(".docx"):
            with io.BytesIO(file_contents) as docx_stream:
                result = mammoth.extract_raw_text(docx_stream)
                text = result.value

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        splitter = TextSplitter(50)
        chunks = splitter.chunks(text)

        for chunk in chunks:
            embedding = await get_embedding(GetEmbeddingParams(text=chunk))
            upsert_vectors(UpsertInput(data=embedding, metadata={"text": chunk}))

        return {"filename": file.filename, "split-text": chunks}

    except Exception as e:
        print("Error processing file:", str(e))
        # raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
