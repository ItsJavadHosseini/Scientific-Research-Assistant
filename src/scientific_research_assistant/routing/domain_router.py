from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from scientific_research_assistant.generation.llm import get_llm


Domain = Literal[
    "llm",
    "machine_learning",
    "computer_vision",
]


class RoutingDecision(BaseModel):
    domains: list[Domain] = Field(
        description="All research domains relevant to the user's question."
    )

    reasoning: str = Field(
        description="A brief explanation of why these domains were selected."
    )


class DomainRouter:

    def __init__(self):
        self.llm = get_llm(temperature=0.0)

        # Initialize a Pydantic parser for routing decisions.
        self.parser = PydanticOutputParser(
            pydantic_object=RoutingDecision
        )

    def route(self, question: str) -> RoutingDecision:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        format_instructions = self.parser.get_format_instructions()

        prompt = f"""
You are a domain classification system for a scientific
research assistant.

Classify the user's question into one or more of these domains:

1. llm:
   Large Language Models, Transformers, NLP, attention,
   prompting, fine-tuning, RAG, BERT, and related topics.

2. machine_learning:
   General machine learning, optimization, neural network
   training, regularization, model interpretability,
   classification, and related topics.

3. computer_vision:
   Image processing, image classification, object detection,
   image segmentation, CNNs, ResNet, ViT, GANs, and related topics.

Rules:
- Select every domain that is genuinely relevant.
- A question can belong to multiple domains.
- Do not select a domain merely because it is indirectly related.
- If the question is unrelated to all three domains,
  return an empty domains list.
- Provide a short explanation.
- Return only valid JSON.
- Do not include Markdown code fences.

{format_instructions}

User question:
{question}
"""

        # Generate the routing decision.
        response = self.llm.invoke(prompt)

        # Extract text from the model response.
        content = response.content

        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
            )

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Domain router received an empty response from the LLM."
            )

        # Parse and validate the JSON response.
        try:
            decision = self.parser.parse(content)

        except OutputParserException as exc:
            raise ValueError(
                f"Invalid routing response from LLM: {content}"
            ) from exc

        return decision