from openai import OpenAI
import json
from pydantic import BaseModel, ValidationError

SYSTEM_PROMPT = """
Your job is to search the web based on the research statement provided.

Available next_agent values:
- research_scope
Send back to the main agent when the research is done. So that it can add on more scope if the user has any

Rules:
1. Use web search for current and factual information.
2. Extract only facts supported by sources.
3. Prioritise official and reputable sources.
4. Capture the source title, URL, publisher, publication date if available, and accessed date if available.
5. Do not invent missing facts.
6. Do not overstate confidence.
7. If sources conflict, preserve both versions and mark the conflict.
10. Do not write the final polished report. Produce structured research notes.

Output format:
Always respond with valid JSON only.
Do not include markdown, comments, or any text outside the JSON object.

The JSON object must follow this schema:

JSON schema:
{
  "next_agent": "research_scope",
  "response_to_user": "Present the results of your research here",
  "context_next_agent": "Research is done. Restart conversation with user",
}
""".strip()

class result(BaseModel):
    next_agent: str
    response_to_user: str
    context_next_agent: str

class generic_search:
    def __init__(self, api_key: str, model: str ) -> None: 
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.response_id = ''

    def response(self, user_message: str):

        if self.response_id:
            response = self.client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                input=user_message,
                previous_response_id=self.response_id,
                instructions=SYSTEM_PROMPT,
            )
        else:
            response = self.client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                input=user_message,
                instructions=SYSTEM_PROMPT,
            )
            self.response_id = response.id
        
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            try:
                data = json.loads((response.output_text).strip())
                results = result(**data)
                break
            
            except json.JSONDecodeError as e:
                response = self.client.responses.create(
                    model=self.model,
                    input="""
                    Output was not in JSON format. Please strictly follow the output format and only respond with the JSON object as specified in the instructions. 
                    'Do not include any text outside the JSON object.
                    """,
                    previous_response_id=self.response_id,
                    instructions=SYSTEM_PROMPT,
                )
                print(f"Agent output is not valid JSON: {e}. Retrying...")
                continue
            
            except ValidationError as e:
                response = self.client.responses.create(
                    model=self.model,
                    input="""
                    Output was not correct schema. Please strictly follow the output format and ensure the JSON object matches the specified schema in the instructions.
                    'Do not include any text outside the JSON object.'
                    """,
                    previous_response_id=self.response_id,
                    instructions=SYSTEM_PROMPT,
                )
                print(f"Agent output does not match schema: {e}. Retrying...")
                continue

        else:
            raise ValueError(
                f"Agent failed to return valid AgentDecision after {max_attempts} attempts. "
                f"Last output was:\n{response.output_text}"
            )

        if results:
            return data

        return "I received your message, but I could not extract a text response."

    def clear_history(self):
        self.response_id = ''
