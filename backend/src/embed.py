#external imports
from pinecone import Pinecone

#built-in imports
from dotenv import load_dotenv
from os import getenv

#local imports
from src.io import EmbeddingsList

load_dotenv()

client: Pinecone = Pinecone(api_key=getenv("PINECONE_API_KEY"))

index = client.Index("semantic-search-incubator")

async def embed(text: str) -> EmbeddingsList:
    response = await client.inference.embed(
        model="multilingual-e5-large",
        input=text,
        parameters={"input_type": "passage", "truncate": "END"}
    )
    return EmbeddingsList(model="text-embedding-3-small", data=response.data, usage=response.usage)

async def upsert(embeddings: EmbeddingsList):
    index.upsert(
        values=embeddings.data,
        namespace="default"
    )