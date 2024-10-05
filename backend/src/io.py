#external imports
from pydantic import BaseModel

class Query(BaseModel):
    query: str

class HyphoteticalDocument(BaseModel):
    document: str

class ChatGPTResponse(BaseModel):
    response: str

class Embeddings(BaseModel):
    values: list[float]

class PineconeUsage(BaseModel):
    total_tokens: int

class EmbeddingsList(BaseModel):
    model: str
    data: Embeddings
    usage: PineconeUsage