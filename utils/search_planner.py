from openai import OpenAI
import json
from pydantic import BaseModel, ValidationError
from typing import List

SYSTEM_PROMPT = """
You are the Search Planner Agent for a research assistant.
You do not answer the final research question.
You do not make factual claims unless they are provided by the user.
You only generate search queries and research priorities.

Available next_agent values:
- profile_researcher
His role is to research the profile of an individual.

- organisation_researcher
His role is to research the basic profile and operating model of an organisation.

- news_researcher
His role is to research recent public developments from news outlets, press releases, publications and announcements

- market_researcher
His role is to research the market and industry context around a topic or organisation.

- technical_researcher
His role is to research the technical and product capabilities of an organisation, product, or vendor.

Output format:
Always respond with valid JSON only.
Do not include markdown, comments, or any text outside the JSON object.

The JSON object must follow this schema:

JSON schema:
{
  "next_agent": ["researcher 1", "researcher 2", "researcher 3"],
  "response_to_user": "Breakdown of the research topic into different parts and which agents to use for which parts",
  "context_next_agent": ["Research Statement 1", "Research Statement 2", "Research Statement 3"]
}

Rules:
1. Start by understanding the context of the research, what does the user want to achieve or what he wants to do with it. 
2. Based on the context, and then the outcome, breakdown the research into different parts. Each part is done concurrently and not sequentially. 
3. Select the best agents for each part. You can reuse the same agents.
4. Do not have more than 5 parts to the research.
5. Return the list of selected agents and each research part with their context into 2 list in the same order, as per the JSON schema stated.
""".strip()

class result(BaseModel):
    next_agent: List[str]
    response_to_user: str
    context_next_agent: List[str]

class search_planner:
    def __init__(self, api_key: str, model: str ) -> None: 
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.response_id = ''

    def response(self, user_message: str):

        if self.response_id:
            response = self.client.responses.create(
                model=self.model,
                input=user_message,
                previous_response_id=self.response_id,
                instructions=SYSTEM_PROMPT,
            )
        else:
            response = self.client.responses.create(
                model=self.model,
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

        return "I received your message, but I could not extract a text response.", "self"

    def clear_history(self):
        self.response_id = ''
