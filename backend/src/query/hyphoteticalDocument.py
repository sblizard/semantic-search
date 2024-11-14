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
    query_text: str = query.query
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Generate a document that addresses '{query_text}'. The document should include realistic but fictional data, examples, and content that mimics what an authentic response would entail.Ensure the information is plausible and contextually appropriate, but clearly marked as hypothetical."},
            {"role": "user", "content": query_text}
        ]
    )
    print("****************" + response.choices[0].message.content + "****************")
    return ChatGPTResponse(response=response.choices[0].message.content)

async def create_hyphotetical_document_embedding(query: Query) -> EmbeddingsList:
   embedding: EmbeddingsList = await embed(query.query)
   return embedding