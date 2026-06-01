from openai import AsyncOpenAI

class researcher:
    def __init__(self, api_key: str, model: str ) -> None: 
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def response(self, user_message: str, system_prompt: str):

        response = await self.client.responses.create(
            model= self.model,
            tools=[{"type": "web_search"}],
            input=user_message,
            instructions=system_prompt,
        )

        if response:
            return response.output_text

        return "I received your message, but I could not extract a text response."

