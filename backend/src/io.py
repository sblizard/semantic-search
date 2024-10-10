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

class GetEmbeddingParams(BaseModel):
    text: str

class EmbeddingOutput(BaseModel):
    embedding: list[float]

class UpsertInput(BaseModel):
    data: EmbeddingOutput
    metadata: dict[str, str]

class SearchQuery(BaseModel):
    text: str

class Match(BaseModel):
    id: str
    metadata: dict[str, str]

class SearchOutput(BaseModel):
    matches: list[Match]
