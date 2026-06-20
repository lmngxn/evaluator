from anthropic import AsyncAnthropic

class claudeAgent:
    def __init__(self, api_key: str, model: str, max_tokens: int = 4096) -> None:
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.messages = []

    async def response(self, user_message: str):

        self.messages.append({"role": "user",  "content": user_message})

        response = await self.client.messages.create(
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
        )
        
        if response:
            self.messages.append({"role": "assistant",  "content": response.content[0].text})
            return response.content[0].text
    
    def clear_history(self):
        self.messages = []