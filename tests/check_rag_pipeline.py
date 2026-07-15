from rag_pipeline import ask
from rag_pipeline import initialize

if __name__ == "__main__":

    initialize()

    while True:

        query = input("\nAsk Question (type 'exit' to quit): ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            break

        answer, contexts, sources = ask(query)

        print("\n" + "=" * 80)
        print("Answer\n")
        print(answer)

        print("\nSources")

        for source in sources:
            print(f"- {source}")

        print("=" * 80)