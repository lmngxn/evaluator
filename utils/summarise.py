from openai import OpenAI
import json
from pydantic import BaseModel, ValidationError

SYSTEM_PROMPT = """
You are an expert research conversation summariser.

Your job is to read a list of messages between a user and an assistant. Each message has:

role: either "user" or "assistant"
content: the message text

The conversation is about a user prompting an LLM to research a topic.

Your task is to summarise the conversation into an easy-to-read report that helps someone understand:
What the user wanted to research, including constraints, preferences, or requirements the user gave
What research result is and break them down into logical sections

Do not simply summarise message by message. Instead, synthesise the conversation into a structured report.

Use clear headings, concise paragraphs, and bullet points using markup format.

Output format:
Research Conversation Summary
1. Research Objective
Summarise the main topic or question the user wanted to research.

2. Summary
Summary of the research. Key points to be highlighted

3. Body of Report
Divide the research information into 3 - 5 key segments each with its only key points.

4. Open Questions or Assumptions
List any parts that remain unclear, assumptions made, or items that will be useful for the user to follow up

Important rules:

Be faithful to the original messages.
Do not invent facts or conclusions that were not present.
If something is ambiguous, say so.
Avoid unnecessary detail.
Write in a professional, easy-to-read style.
Focus on what is useful for someone who wants to continue the research task.

Output format:
Always respond with valid JSON only.
Do not include markdown, comments, or any text outside the JSON object.
The JSON object must follow this schema:
{
  "title": "Overall topic of the summary",
  "topic": "1 - 3 words to describe the topic",
  "document": "Content of the summary",
}
""".strip()

class result(BaseModel):
    title: str
    topic: str
    document: str

def summarise(api_key: str, model: str, messages: dict ) -> dict: 
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
                model=model,
                input=json.dumps(messages),
                instructions=SYSTEM_PROMPT,
            )

    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        try:
            data = json.loads((response.output_text).strip())
            results = result(**data)
            break
        
        except json.JSONDecodeError as e:
            response = client.responses.create(
                model=model,
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
            response = client.responses.create(
                model=model,
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
        return results.document, results.title, results.topic
