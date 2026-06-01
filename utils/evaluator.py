from utils.openai import openAIAgent
from utils.anthropic import claudeAgent
from utils.gemini import geminiAgent
import json
from pydantic import BaseModel, ValidationError
from typing import List

base_prompt = """
You are an impartial evaluator assessing the quality of responses from 2 to 3 different AI models.

Your job is to evaluate each response independently using the rubric below.

Important rules:
- Do not assume that longer responses are better.
- Do not favor any response based on style alone.
- Do not infer which model produced the response.
- Penalize unsupported claims, factual errors, vague reasoning, and failure to follow the user’s instructions.
- If the user request is ambiguous, reward responses that make reasonable assumptions or ask useful clarifying questions.
- If the response contains unsafe, misleading, or fabricated information, penalize it heavily.
- Evaluate only the quality of the answer, not the identity of the model.

Rubric:

1. Instruction Following, 1–5
How well does the response satisfy the user’s actual request?
- 1 = Does not answer the request
- 3 = Partially answers but misses important requirements
- 5 = Fully follows the request and constraints

2. Correctness / Factual Accuracy, 1–5
Is the response factually correct and free from hallucinations?
- 1 = Mostly incorrect or fabricated
- 3 = Some correct information but notable errors or unsupported claims
- 5 = Accurate, grounded, and reliable

3. Completeness, 1–5
Does the response cover the necessary points?
- 1 = Very incomplete
- 3 = Covers the basics but misses useful details
- 5 = Comprehensive without unnecessary filler

4. Reasoning Quality, 1–5
Is the answer logically sound and well-reasoned?
- 1 = Confusing, contradictory, or poorly reasoned
- 3 = Mostly understandable but with weak logic
- 5 = Clear, coherent, and well-justified

5. Practical Usefulness, 1–5
Would the user be able to act on this response?
- 1 = Not useful
- 3 = Somewhat useful but requires substantial work
- 5 = Directly useful and actionable

6. Clarity and Communication, 1–5
Is the response easy to understand?
- 1 = Hard to follow
- 3 = Understandable but could be clearer
- 5 = Clear, concise, and well-structured

After scoring each response, provide:
- A short explanation for each score
- The biggest strength of each response
- The biggest weakness of each response

Return your evaluation in the following JSON format only.
Each item in the array should represent one candidate response.
Use this exact schema:

[
    {
        "response_id": "Response A",
        "instruction_following": 1-5,
        "correctness": 1-5,
        "completeness": 1-5,
        "reasoning_quality": 1-5,
        "practical_usefulness": 1-5,
        "clarity": 1-5,
        "average_score": average,
        "strength": "<2–3 sentence justification>",
        "weakness": "<2–3 sentence justification>",
        "explanation": "<justification in short phrases>"
    }
]
""".strip()

class evaluation(BaseModel):
    response_id: str
    instruction_following: int
    correctness: int
    completeness: int
    reasoning_quality: int
    practical_usefulness: int
    clarity: int
    average_score: float
    strength: str
    weakness: str
    explanation: str

class evaluatorAgent:
    def __init__(self, api_key: str, provider:str, model: str ) -> None:
        if provider == "openai":
            self.agent = openAIAgent(api_key=api_key, model=model)
        if provider == "anthropic":
            self.agent = claudeAgent(api_key=api_key, model=model)
        if provider == "gemini":
            self.agent = geminiAgent(api_key=api_key, model=model)

    async def response(self, user_message: str, responses: List[str]):      
        input = base_prompt + f"Prompt given to the model:\n{user_message}\n\n"
        
        for i, response in enumerate(responses):
            input += f"Response {i+1}:\n{response}\n\n"

        response = await self.agent.response(input)

        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            try:
                data = json.loads((response).strip())
                validated_data = [evaluation(**d) for d in data]
                break
            
            except json.JSONDecodeError as e:
                response = self.agent.response("""
                    Output was not in JSON format. Please strictly follow the output format and only respond with the JSON object as specified in the instructions.
                    Do not include any text outside the JSON object.
                    """
                )
                print(f"Agent output is not valid JSON: {e}. Retrying...")
                continue
            
            except ValidationError as e:
                response = self.agent.response("""
                    Output was not correct schema. Please strictly follow the output format and ensure the JSON object matches the specified schema in the instructions.
                    Do not include any text outside the JSON object.
                    """
                )
                print(f"Agent output does not match schema: {e}. Retrying...")
                continue
        
        else:
            raise ValueError(
                f"Agent failed to return valid AgentDecision after {max_attempts} attempts. "
                f"Last output was:\n{response}"
            )

        return data

    def format_response(self, agent_response):
        score_fields = [
            "instruction_following",
            "correctness",
            "completeness",
            "reasoning_quality",
            "practical_usefulness",
            "clarity",
            "average_score",
        ]

        text_fields = [
            "explanation",
            "strength",
            "weakness",
        ]

        formatted_response = "### Scores\n\n"

        formatted_response += "  \n".join(
            f"**{field.replace('_', ' ').title()}**: {agent_response.get(field, '')}"
            for field in score_fields
        )

        formatted_response += "\n\n### Comments\n\n"

        formatted_response += "\n\n".join(
            f"**{field.replace('_', ' ').title()}**: {agent_response.get(field, '')}"
            for field in text_fields
        )
        return formatted_response

