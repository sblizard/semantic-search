#external imports
from pinecone import Pinecone
from openai import AsyncOpenAI
from pinecone.grpc import GRPCIndex as Index

#built-in imports
from dotenv import load_dotenv
from os import getenv
from uuid import uuid4 

#local imports
from src.io import EmbeddingsList, UpsertInput, GetEmbeddingParams, EmbeddingOutput, SearchOutput, SearchQuery, Match


load_dotenv()


pinecode_client: Pinecone = Pinecone(api_key=getenv("PINECONE_API_KEY"))
openai_client: AsyncOpenAI = AsyncOpenAI(api_key=getenv("OPENAI_API_KEY"))
index: Index = pinecode_client.Index("oct7boil")

index = pinecode_client.Index("semantic-search-incubator")


async def embed(text: str) -> EmbeddingsList:
    response = await pinecode_client.inference.embed(
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


def upsert_vectors(params: UpsertInput) -> None:
    vectors: list[list[float]] = []

    vectors.append({
        "id": str(uuid4()),
        "values": params.data.embedding,
        "metadata": params.metadata
    })
    index.upsert(vectors=vectors)


async def get_embedding(params: GetEmbeddingParams) -> EmbeddingOutput:
    embedding: list[float] = await openai_client.embeddings.create(input=params.text, model="text-embedding-ada-002")
    return EmbeddingOutput(embedding=embedding.data[0].embedding)


async def semantic_search(query: SearchQuery) -> SearchOutput:
    embedding: list[float] = await openai_client.embeddings.create(
    input=query.text, 
    model="text-embedding-ada-002")

    embedding_output: EmbeddingOutput = EmbeddingOutput(embedding=embedding.data[0].embedding)

    results = index.query(
        vector=embedding_output.embedding,
        top_k=3,
        include_metadata=True
    )

    search_output: SearchOutput = SearchOutput(matches=[])

    for i in range(len(results['matches'])):
        match_data = results['matches'][i]
        match: Match = Match(
            id=match_data['id'], 
            metadata=match_data['metadata'] 
        )
        search_output.matches.append(match)
    
    return search_output

async def search_by_embedding(embedding: EmbeddingOutput):
    results = index.query(
        vector=embedding.embedding,
        top_k=3,
        include_metadata=True
    )

    search_output: SearchOutput = SearchOutput(matches=[])

    for i in range(len(results['matches'])):
        match_data = results['matches'][i]
        match: Match = Match(
            id=match_data['id'], 
            metadata=match_data['metadata'] 
        )
        search_output.matches.append(match)
    
    return search_output
