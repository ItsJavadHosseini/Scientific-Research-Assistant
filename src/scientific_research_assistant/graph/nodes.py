from scientific_research_assistant.graph.state import RAGState
from scientific_research_assistant.routing.domain_router import DomainRouter
from scientific_research_assistant.retrieval.retriever import Retriever
import time
import re

from scientific_research_assistant.generation.prompts import RAG_PROMPT


def create_router_node(domain_router: DomainRouter):

    def router_node(state: RAGState) -> dict:

        question = state["question"]

        # Use manually selected domains when provided.
        category = state.get("category")

        if category is not None:

            domains = (
                [category]
                if isinstance(category, str)
                else category
            )

            reasoning = "Domains were manually selected."

        else:

            # Automatically detect relevant research domains.
            decision = domain_router.route(question)

            domains = decision.domains
            reasoning = decision.reasoning

        return {
            "domains": domains,
            "routing_reasoning": reasoning,
        }

    return router_node



def create_retriever_node(
    retriever: Retriever,
    top_k: int = 5,
):

    def retriever_node(state: RAGState) -> dict:

        question = state["question"]
        domains = state.get("domains", [])
        paper_id = state.get("paper_id")

        # Skip retrieval when no relevant domain was identified.
        if not domains:
            return {"documents": []}

        # Retrieve relevant documents from Qdrant.
        documents = retriever.retrieve(
            question=question,
            top_k=top_k,
            paper_id=paper_id,
            category=domains,
        )

        return {
            "documents": documents,
        }

    return retriever_node

def create_context_builder_node(rag_chain):

    def context_builder_node(state: RAGState) -> dict:

        documents = state.get("documents", [])

        # Handle empty retrieval results.
        if not documents:
            return {
                "context": "",
                "sources": [],
            }

        # Reuse the existing document formatting logic.
        context, sources = rag_chain._format_documents(documents)

        return {
            "context": context,
            "sources": sources,
        }

    return context_builder_node

def create_generation_node(rag_chain):

    def generation_node(state: RAGState) -> dict:

        question = state["question"]
        context = state.get("context", "")

        if not context.strip():
            return {
                "answer": (
                    "The available research papers do not provide "
                    "sufficient evidence to answer this question."
                )
            }

        # Prepare the prompt messages.
        messages = RAG_PROMPT.format_messages(
            context=context,
            question=question,
        )

        answer_parts = []

        generation_start = time.perf_counter()

        # Generate the answer using the existing LLM instance.
        for chunk in rag_chain.llm.stream(messages):

            content = chunk.content

            if isinstance(content, str):
                answer_parts.append(content)

            elif isinstance(content, list):

                for block in content:

                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                    ):
                        answer_parts.append(
                            block.get("text", "")
                        )

        answer = "".join(answer_parts)

        # Fall back to non-streaming generation if needed.
        if not answer.strip():

            response = rag_chain.llm.invoke(messages)

            content = response.content

            if isinstance(content, str):
                answer = content

            elif isinstance(content, list):
                answer = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                )

        if not answer.strip():
            raise RuntimeError("LLM returned an empty response.")

        generation_time = (
            time.perf_counter() - generation_start
        )

        return {
            "answer": answer,
            "generation_time": generation_time,
        }

    return generation_node

def create_citation_validator_node():

    def citation_validator_node(state: RAGState) -> dict:

        answer = state.get("answer", "")
        sources = state.get("sources", [])

        # Extract citation IDs from the generated answer.
        used_citations = list(
            dict.fromkeys(
                re.findall(r"\[(S\d+)\]", answer)
            )
        )

        # Collect valid source IDs.
        valid_source_ids = {
            source["source_id"]
            for source in sources
        }

        # Find citations that do not exist in the sources.
        invalid_citations = [
            citation
            for citation in used_citations
            if citation not in valid_source_ids
        ]

        errors = []

        if invalid_citations:
            errors.append(
                f"Invalid citations: {invalid_citations}"
            )

        if sources and not used_citations:
            errors.append(
                "No citations were found in the generated answer."
            )

        return {
            "used_citations": used_citations,
            "citation_valid": not errors,
            "citation_errors": errors,
        }

    return citation_validator_node

def create_out_of_domain_node():

    def out_of_domain_node(state: RAGState) -> dict:

        return {
            "answer": (
                "This question is outside the scope of the "
                "available research domains: LLM, Machine Learning, "
                "and Computer Vision."
            ),
            "documents": [],
            "context": "",
            "sources": [],
            "generation_time": 0.0,
            "used_citations": [],
            "citation_valid": True,
            "citation_errors": [],
        }

    return out_of_domain_node


def create_insufficient_evidence_node():

    def insufficient_evidence_node(state: RAGState) -> dict:

        return {
            "answer": (
                "The available research papers do not provide "
                "sufficient evidence to answer this question."
            ),
            "context": "",
            "sources": [],
            "generation_time": 0.0,
            "used_citations": [],
            "citation_valid": True,
            "citation_errors": [],
        }

    return insufficient_evidence_node