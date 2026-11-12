
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

        self._log("[bold green]RAG components initialized successfully.")

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

    def _generate_streaming_answer(self, context: str, question: str):

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
    # Main RAG Pipeline
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

        # Step 1: Domain Routing

        routing_start = time.perf_counter()

        if category is None:

            with console.status(
                "[bold yellow]Step 1/4: Analyzing question and detecting domains...",
                spinner="dots",
            ):
                routing_result = self.domain_router.route(question)

            domains = routing_result.domains
            routing_reasoning = routing_result.reasoning

        else:

            domains = (
                [category]
                if isinstance(category, str)
                else category
            )

            routing_reasoning = "Domains were manually selected."

        routing_time = (
            time.perf_counter() - routing_start
        )

        self._log(
            f"\n[green]Routing completed in {routing_time:.2f}s"
        )

        self._log(
            f"[cyan]Selected domains: {domains}"
        )

        self._log(
            f"[dim]{routing_reasoning}"
        )

        # Step 2: Domain Validation

        if not domains:

            self._log(
                "[red]Question is outside supported research domains."
            )

            return {
                "question": question,
                "domains": [],
                "routing_reasoning": routing_reasoning,
                "answer": (
                    "This question is outside the research domains "
                    "supported by this assistant."
                ),
                "sources": [],
            }

        # Step 3: Retrieval

        retrieval_start = time.perf_counter()

        with console.status(
            f"[bold yellow]Step 2/4: Searching Qdrant in {len(domains)} domain(s)...",
            spinner="dots",
        ):

            documents = self.retriever.retrieve(
                question=question,
                top_k=self.top_k,
                category=domains,
                paper_id=paper_id,
            )

        retrieval_time = (
            time.perf_counter() - retrieval_start
        )

        self._log(
            f"\n[green]Retrieval completed in {retrieval_time:.2f}s"
        )

        self._log(
            f"[cyan]Retrieved chunks: {len(documents)}"
        )

        if not documents:

            self._log(
                "[red]No relevant documents were retrieved."
            )

            return {
                "question": question,
                "domains": domains,
                "routing_reasoning": routing_reasoning,
                "answer": (
                    "The available research papers do not provide "
                    "sufficient evidence to answer this question."
                ),
                "sources": [],
            }

        # Step 4: Context Preparation

        with console.status(
            "[bold yellow]Step 3/4: Preparing context and citations...",
            spinner="dots",
        ):

            context, sources = self._format_documents(documents)

        self._log(
            f"[green]Context prepared: {len(sources)} unique sources"
        )

        # Step 5: Streaming Generation

        answer, generation_time = self._generate_streaming_answer(
            context=context,
            question=question,
        )

        self._log(
            f"\n[green]Answer generated in {generation_time:.2f}s"
        )

        # Step 6: Display Results

        total_time = (
            time.perf_counter() - total_start
        )

        if self.verbose:

            self._display_sources(sources)

            self._log(
                f"\n[bold cyan]Total execution time: {total_time:.2f}s"
            )

        # Return Results

        return {
            "question": question,
            "domains": domains,
            "routing_reasoning": routing_reasoning,
            "answer": answer,
            "sources": sources,
            "execution_time": {
                "routing": routing_time,
                "retrieval": retrieval_time,
                "generation": generation_time,
                "total": total_time,
            },
        }

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    def close(self):

        self.retriever.close()
