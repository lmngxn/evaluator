from google import genai

class geminiAgent:
    def __init__(self, api_key: str, model: str ) -> None: 
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.messages = []

    async def response(self, user_message: str):
        
        self.messages.append({"role": "user",  "parts": [{"text":user_message}]})
                             
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=self.messages,
        )
        
        if response:
            self.messages.append({"role": "assistant",  "parts": [{"text":response.text}]})
            return response.text

    def clear_history(self):
        self.messages = []