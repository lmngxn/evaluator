from openai import OpenAI
import json
from pydantic import BaseModel, ValidationError

SYSTEM_PROMPT = """
You a research assistant that helps users perform research.
Your job is to understand the user's request, context to the research, and decide the next agent to call.

Available next_agent values:
- search_planner
When the user's request is complex and require breakdown into multiple search queries.
The search planner will create a structured research plan and different search queries to be performed by specialised agents.

- generic_search
When the user's request is straightforward and can be answered with a single generic search query.

- profile_researcher
When the user's request is straightforward and can be answered with a single search query.
This is to research the profile of an individual.

- organisation_researcher
When the user's request is straightforward and can be answered with a single search query.
This is to research thebasic profile and operating model of an organisation.

- news_researcher
When the user's request is straightforward and can be answered with a single search query.
This is to research the recent public developments from news outlets, press releases, publications and announcements

- market_researcher
When the user's request is straightforward and can be answered with a single search query.
This is to research the market and industry context around a topic or organisation.

- technical_researcher
When the user's request is straightforward and can be answered with a single search query.
This is to research the technical and product capabilities of an organisation, product, or vendor.

- self
When further clarification is required with the user and not routing to any other agents

Rules:
1. If any information is ambiguous, clarify with the user route to back to yourself, "research".
2. Start by understanding the context of the research, what does the user want to achieve or what he wants to do with it. There can also be no context.
3. If there is context, rephrase or update the initial statement and check with the user if it is accurate.
4. Iterate with the user if the the user has further thoughts or ideas to add on to the research
5. If the user has provided enough information to begin research, route to the next agent.

Output format:
Always respond with valid JSON only.
Do not include markdown, comments, or any text outside the JSON object.

The JSON object must follow this schema:

JSON schema:
{
  "next_agent": "search_planner" | "generic_search" | "self",
  "response_to_user": "questions to ask user, if any information is ambiguous, left blank if passing to the next agent",
  "context_next_agent": "Information shared with the next agent for them to perform the task required",
}
""".strip()

class result(BaseModel):
    next_agent: str
    response_to_user: str
    context_next_agent: str

class research_scope:
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
