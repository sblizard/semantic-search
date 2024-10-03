#external imports
from openai import AsyncOpenAI

#built-in imports
from dotenv import load_dotenv
from os import getenv

#local imports
from src.io import Query, ChatGPTResponse

load_dotenv()

client: AsyncOpenAI = AsyncOpenAI(api_key=getenv("OPENAI_API_KEY"))

async def create_hyphotetical_document(query: Query) -> ChatGPTResponse:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                   {"role": "user", "content": query.query}]
    )
    return ChatGPTResponse(response=response.choices[0].message.content)

