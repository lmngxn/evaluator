from openai import AsyncOpenAI

class openAIAgent:
    def __init__(self, api_key: str, model: str ) -> None: 
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.messages = []

    async def response(self, user_message: str):

        self.messages.append({"role": "user","content": user_message})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )

        if response:
            self.messages.append({"role": "assistant",  "content": response.choices[0].message.content})
            return response.choices[0].message.content

    def clear_history(self):
        self.messages = []