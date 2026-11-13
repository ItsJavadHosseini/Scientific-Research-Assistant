from langgraph.graph import END, START, StateGraph

from scientific_research_assistant.graph.state import RAGState

from scientific_research_assistant.graph.nodes import (
    create_router_node,
    create_retriever_node,
    create_context_builder_node,
    create_generation_node,
    create_citation_validator_node,
    create_out_of_domain_node,
    create_insufficient_evidence_node,
)


def build_rag_graph(rag_chain):

    # Initialize the graph builder.
    builder = StateGraph(RAGState)

    # Create graph nodes using existing RAG components.
    builder.add_node(
        "router",
        create_router_node(rag_chain.domain_router),
    )

    builder.add_node(
        "retriever",
        create_retriever_node(
            rag_chain.retriever,
            top_k=rag_chain.top_k,
        ),
    )

    builder.add_node(
        "context_builder",
        create_context_builder_node(rag_chain),
    )

    builder.add_node(
        "generation",
        create_generation_node(rag_chain),
    )

    builder.add_node(
        "citation_validator",
        create_citation_validator_node(),
    )

    builder.add_node(
        "out_of_domain",
        create_out_of_domain_node(),
    )

    builder.add_node(
        "insufficient_evidence",
        create_insufficient_evidence_node(),
    )
        # Define the entry point.
    builder.add_edge(START, "router")

    # Route questions based on detected domains.
    builder.add_conditional_edges(
        "router",
        lambda state: (
            "retriever" if state.get("domains") else "out_of_domain"
        ),
        {
            "retriever": "retriever",
            "out_of_domain": "out_of_domain",
        },
    )

    # Route based on retrieval results.
    builder.add_conditional_edges(
        "retriever",
        lambda state: (
            "context_builder"
            if state.get("documents")
            else "insufficient_evidence"
        ),
        {
            "context_builder": "context_builder",
            "insufficient_evidence": "insufficient_evidence",
        },
    )

    # Define the successful RAG execution path.
    builder.add_edge("context_builder", "generation")
    builder.add_edge("generation", "citation_validator")

    # Define terminal nodes.
    builder.add_edge("citation_validator", END)
    builder.add_edge("out_of_domain", END)
    builder.add_edge("insufficient_evidence", END)

    return builder.compile()