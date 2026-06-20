from utils.openai import openAIAgent
from utils.anthropic import claudeAgent
from utils.gemini import geminiAgent
import asyncio
import json
import re
from pydantic import BaseModel, ValidationError
from typing import List, cast

base_prompt = """
You are an impartial evaluator assessing the quality of responses from 2 to 3 different AI models.

Your job is to evaluate each response independently using the rubric below.

Important rules:
- Do not assume that longer responses are better.
- Do not favor any response based on style alone.
- Do not infer which model produced the response.
- Penalize unsupported claims, factual errors, vague reasoning, and failure to follow the user's instructions.
- If the user request is ambiguous, reward responses that make reasonable assumptions or ask useful clarifying questions.
- If the response contains unsafe, misleading, or fabricated information, penalize it heavily.
- Evaluate only the quality of the answer, not the identity of the model.

Rubric:

1. Instruction Following, 1–5
How well does the response satisfy the user's actual request?
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
        "response_id": "Response 1",
        "instruction_following": 1-5,
        "correctness": 1-5,
        "completeness": 1-5,
        "reasoning_quality": 1-5,
        "practical_usefulness": 1-5,
        "clarity": 1-5,
        "strength": "<2–3 sentence justification>",
        "weakness": "<2–3 sentence justification>",
        "explanation": "<justification in short phrases>"
    }
]
""".strip()

_SCORE_FIELDS = [
    "instruction_following",
    "correctness",
    "completeness",
    "reasoning_quality",
    "practical_usefulness",
    "clarity",
]

# Fixed judge models — one per provider for cross-provider balance.
JUDGE_MODELS = {
    "openai": "gpt-5.4-nano",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.5-pro",
}


class evaluation(BaseModel):
    response_id: str
    instruction_following: float
    correctness: float
    completeness: float
    reasoning_quality: float
    practical_usefulness: float
    clarity: float
    average_score: float = 0.0
    strength: str
    weakness: str
    explanation: str


class evaluatorAgent:
    def __init__(
        self,
        openai_api_key: str = "",
        anthropic_api_key: str = "",
        gemini_api_key: str = "",
    ) -> None:
        self.judges: list[tuple[str, object]] = []

        if openai_api_key:
            self.judges.append(("GPT-5.4-Nano", openAIAgent(api_key=openai_api_key, model=JUDGE_MODELS["openai"])))
        if anthropic_api_key:
            self.judges.append(("Claude Sonnet", claudeAgent(api_key=anthropic_api_key, model=JUDGE_MODELS["anthropic"])))
        if gemini_api_key:
            self.judges.append(("Gemini 2.5 Pro", geminiAgent(api_key=gemini_api_key, model=JUDGE_MODELS["gemini"])))

        if not self.judges:
            raise ValueError("At least one API key must be provided for evaluation.")

    async def _evaluate_with_judge(self, agent, user_message: str, responses: List[str]) -> List[evaluation]:
        eval_prompt = base_prompt + f"\n\nPrompt given to the model:\n{user_message}\n\n"
        for i, r in enumerate(responses):
            eval_prompt += f"Response {i+1}:\n{r}\n\n"

        raw_response = await agent.response(eval_prompt)

        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            try:
                cleaned = re.sub(r"^```(?:json)?\s*", "", raw_response.strip())
                cleaned = re.sub(r"\s*```$", "", cleaned)
                data = json.loads(cleaned.strip())
                validated_data = [evaluation(**d) for d in data]
                for item in validated_data:
                    scores = [getattr(item, f) for f in _SCORE_FIELDS]
                    item.average_score = round(sum(scores) / len(scores), 2)
                break

            except json.JSONDecodeError as e:
                print(f"Judge output is not valid JSON: {e}. Retrying...")
                raw_response = await agent.response(
                    "Output was not in JSON format. Please strictly follow the output format and only respond with the JSON object as specified in the instructions. "
                    "Do not include any text outside the JSON object."
                )
                continue

            except ValidationError as e:
                print(f"Judge output does not match schema: {e}. Retrying...")
                raw_response = await agent.response(
                    "Output was not correct schema. Please strictly follow the output format and ensure the JSON object matches the specified schema in the instructions. "
                    "Do not include any text outside the JSON object."
                )
                continue

        else:
            raise ValueError(
                f"Judge failed to return valid evaluation after {max_attempts} attempts. "
                f"Last output was:\n{raw_response}"
            )

        return validated_data

    def _aggregate(self, evals: List[evaluation]) -> evaluation:
        averaged = {
            f: round(sum(getattr(e, f) for e in evals) / len(evals), 1)
            for f in _SCORE_FIELDS
        }
        avg_score = round(sum(averaged.values()) / len(_SCORE_FIELDS), 2)
        return evaluation(
            response_id=evals[0].response_id,
            **averaged,
            average_score=avg_score,
            strength="\n".join(f"• {e.strength}" for e in evals),
            weakness="\n".join(f"• {e.weakness}" for e in evals),
            explanation="\n".join(f"• {e.explanation}" for e in evals),
        )

    async def response(self, user_message: str, responses: List[str]):
        tasks = [
            self._evaluate_with_judge(agent, user_message, responses)
            for _, agent in self.judges
        ]
        judge_results = await asyncio.gather(*tasks, return_exceptions=True)

        valid: list[tuple[str, list[evaluation]]] = []
        for (name, _), result in zip(self.judges, judge_results):
            if isinstance(result, Exception):
                print(f"Judge {name!r} failed: {result}")
            else:
                valid.append((name, cast(list[evaluation], result)))

        if not valid:
            raise ValueError("All judges failed to produce valid evaluations.")

        n = len(responses)
        aggregated = []
        for i in range(n):
            judge_evals = [evals[i] for _, evals in valid if i < len(evals)]
            aggregated.append({
                "consensus": self._aggregate(judge_evals),
                "judges": [
                    {"name": name, "evaluation": evals[i]}
                    for name, evals in valid
                    if i < len(evals)
                ],
            })

        return aggregated

    def format_response(self, agent_response: dict) -> str:
        consensus = agent_response["consensus"]
        judges = agent_response["judges"]
        judge_names = ", ".join(j["name"] for j in judges)

        display_score_fields = _SCORE_FIELDS + ["average_score"]

        out = f"### Scores *(consensus across {len(judges)} judges: {judge_names})*\n\n"
        out += "  \n".join(
            f"**{f.replace('_', ' ').title()}**: {getattr(consensus, f)}"
            for f in display_score_fields
        )

        out += "\n\n### Comments\n\n"
        out += "\n\n".join(
            f"**{f.replace('_', ' ').title()}**:\n{getattr(consensus, f)}"
            for f in ["explanation", "strength", "weakness"]
        )

        out += "\n\n---\n\n#### Per-Judge Breakdown\n\n"
        for j in judges:
            name = j["name"]
            eva = j["evaluation"]
            out += f"**{name}** *(avg {eva.average_score})*  \n"
            out += "  \n".join(
                f"{f.replace('_', ' ').title()}: {getattr(eva, f)}"
                for f in _SCORE_FIELDS
            )
            out += f"  \n*Strength*: {eva.strength}  \n*Weakness*: {eva.weakness}\n\n"

        return out
