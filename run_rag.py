from scientific_research_assistant.generation.answer import RAGChain


def main():
    rag_chain = RAGChain(
        strategy="recursive",
        top_k=5,
        verbose=True,
    )

    print("\nScientific Research Assistant")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            question = input("\nYour question: ").strip()

            if question.lower() in ["exit", "quit"]:
                break

            if not question:
                continue

            result = rag_chain.invoke(question)

            print("\nAnswer:")
            print(result["answer"])

    finally:
        rag_chain.close()


if __name__ == "__main__":
    main()