#external imports
from openai import AsyncOpenAI

#built-in imports
from dotenv import load_dotenv
from os import getenv
from src.embed import embed

#local imports
from src.io import Query, ChatGPTResponse, EmbeddingsList

load_dotenv()

client: AsyncOpenAI = AsyncOpenAI(api_key=getenv("OPENAI_API_KEY"))

async def create_hyphotetical_document(query: Query) -> ChatGPTResponse:
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant capable of generating hypothetical documents. Create a detailed, well-structured document based on the user's query."},
            {"role": "user", "content": query.query}
        ]
    )
    return ChatGPTResponse(response=response.choices[0].message.content)

async def create_hyphotetical_document_embedding(query: Query) -> EmbeddingsList:
   embedding: EmbeddingsList = await embed(query.query)
   return embedding