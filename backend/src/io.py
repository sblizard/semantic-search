#external imports
from pydantic import BaseModel

class Query(BaseModel):
    query: str

class HyphoteticalDocument(BaseModel):
    document: str

class ChatGPTResponse(BaseModel):
    response: str