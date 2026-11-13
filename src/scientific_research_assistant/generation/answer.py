import time
from collections import defaultdict

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from scientific_research_assistant.generation.llm import get_llm
from scientific_research_assistant.generation.prompts import RAG_PROMPT
from scientific_research_assistant.retrieval.retriever import Retriever
from scientific_research_assistant.routing.domain_router import DomainRouter


console = Console()


class RAGChain:

    def __init__(
        self,
        strategy: str = "recursive",
        top_k: int = 5,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.top_k = top_k

        self._log("Initializing RAG components...")

        with console.status(
            "[bold cyan]Loading embedding model and initializing components...",
            spinner="dots",
        ):
            self.retriever = Retriever(strategy=strategy)
            self.domain_router = DomainRouter()
            self.llm = get_llm()

        # Build the LangGraph workflow.
        from scientific_research_assistant.graph.workflow import (
            build_rag_graph,
        )

        self.graph = build_rag_graph(self)

        self._log(
            "[bold green]RAG components initialized successfully."
        )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    def _log(self, message: str):

        if self.verbose:
            console.print(message)

    # --------------------------------------------------
    # Document Formatting
    # --------------------------------------------------

    def _format_documents(self, documents):

        grouped_documents = defaultdict(list)

        for document in documents:

            metadata = document.metadata

            paper_id = metadata.get("paper_id", "unknown")
            page = metadata.get("page", "unknown")
            title = metadata.get("title", "Unknown Paper")

            key = (paper_id, page)

            grouped_documents[key].append({
                "title": title,
                "content": document.page_content,
                "score": metadata.get("retrieval_score"),
            })

        context_parts = []
        sources = []

        for index, ((paper_id, page), chunks) in enumerate(
            grouped_documents.items(),
            start=1,
        ):

            source_id = f"S{index}"
            title = chunks[0]["title"]

            combined_content = "\n\n".join(
                chunk["content"] for chunk in chunks
            )

            context_parts.append(
                f"""
[Source {source_id}]
Paper: {title}
Paper ID: {paper_id}
Page: {page}

Content:
{combined_content}
"""
            )

            scores = [
                chunk["score"]
                for chunk in chunks
                if chunk["score"] is not None
            ]

            sources.append({
                "source_id": source_id,
                "paper_id": paper_id,
                "title": title,
                "page": page,
                "retrieval_score": max(scores) if scores else None,
            })

        context = "\n\n".join(context_parts)

        return context, sources

    # --------------------------------------------------
    # Streaming Generation
    # --------------------------------------------------

    def _generate_streaming_answer(
        self,
        context: str,
        question: str,
    ):

        messages = RAG_PROMPT.format_messages(
            context=context,
            question=question,
        )

        answer_parts = []

        generation_start = time.perf_counter()

        console.print(
            "\n[bold cyan]Generating Answer:[/bold cyan]\n"
        )

        with Live(
            Panel(
                Markdown(""),
                title="Generated Answer",
                border_style="green",
            ),
            console=console,
            refresh_per_second=12,
        ) as live:

            for chunk in self.llm.stream(messages):

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

                current_answer = "".join(answer_parts)

                live.update(
                    Panel(
                        Markdown(current_answer),
                        title="Generated Answer",
                        border_style="green",
                    )
                )

        answer = "".join(answer_parts)

        # Retry with non-streaming generation if streaming fails.
        if not answer.strip():

            console.print(
                "[yellow]Streaming returned an empty response. "
                "Retrying with non-streaming generation...[/yellow]"
            )

            response = self.llm.invoke(messages)

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

                raise RuntimeError(
                    "LLM returned an empty response in both "
                    "streaming and non-streaming modes."
                )

            console.print(
                Panel(
                    Markdown(answer),
                    title="Generated Answer (Fallback)",
                    border_style="yellow",
                )
            )

        generation_time = (
            time.perf_counter() - generation_start
        )

        return answer, generation_time

    # --------------------------------------------------
    # Source Display
    # --------------------------------------------------

    def _display_sources(self, sources):

        table = Table(title="Retrieved Sources")

        table.add_column("ID", style="cyan")
        table.add_column("Paper")
        table.add_column("Page", justify="center")
        table.add_column("Score", justify="right")

        for source in sources:

            score = source["retrieval_score"]

            table.add_row(
                source["source_id"],
                source["title"],
                str(source["page"]),
                f"{score:.4f}" if score is not None else "N/A",
            )

        console.print(table)

    # --------------------------------------------------
    # Main RAG Pipeline (LangGraph)
    # --------------------------------------------------

    def invoke(
        self,
        question: str,
        category: str | list[str] | None = None,
        paper_id: str | None = None,
    ):

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        total_start = time.perf_counter()

        console.print(
            Panel(
                question,
                title="[bold cyan]New RAG Query",
                border_style="cyan",
            )
        )

        self._log(
            "\n[bold yellow]Executing LangGraph RAG workflow..."
        )

        # Initialize the graph state.
        initial_state = {
            "question": question,
            "category": category,
            "paper_id": paper_id,
        }

        # Execute the complete RAG workflow.
        # result = self.graph.invoke(initial_state)
                # Stream graph execution and display completed nodes.
        for event in self.graph.stream(
            initial_state,
            stream_mode="updates",
        ):

            for node_name, node_output in event.items():

                console.print(
                    f"\n[bold cyan]Completed Node:[/bold cyan] {node_name}"
                )

            result = self.graph.invoke(initial_state)
        total_time = time.perf_counter() - total_start

        # Extract the final results.
        answer = result.get("answer", "")
        domains = result.get("domains", [])
        routing_reasoning = result.get("routing_reasoning", "")
        sources = result.get("sources", [])
        generation_time = result.get("generation_time", 0.0)

        # Display the generated answer.
        console.print(
            Panel(
                Markdown(answer),
                title="Generated Answer",
                border_style="green",
            )
        )

        # Display retrieved sources.
        if sources and self.verbose:
            self._display_sources(sources)

        self._log(
            f"\n[cyan]Selected domains: {domains}"
        )

        self._log(
            f"[dim]{routing_reasoning}"
        )

        self._log(
            f"\n[green]Generation time: {generation_time:.2f}s"
        )

        self._log(
            f"[bold cyan]Total execution time: {total_time:.2f}s"
        )

        # Return results using the previous interface.
        return {
            "question": question,
            "domains": domains,
            "routing_reasoning": routing_reasoning,
            "answer": answer,
            "sources": sources,
            "execution_time": {
                "generation": generation_time,
                "total": total_time,
            },
            "citation_valid": result.get(
                "citation_valid",
                True,
            ),
            "citation_errors": result.get(
                "citation_errors",
                [],
            ),
        }

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    def close(self):

        self.retriever.close()

