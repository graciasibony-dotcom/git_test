import os
import anthropic

SYSTEM_PROMPT = """You are an expert Python tutor with years of teaching experience. Your goal is to help learners of all levels understand Python clearly and confidently.

When answering questions:
- Provide clear, concise explanations tailored to the question's complexity
- Always include practical code examples with comments
- Explain *why* things work, not just *how*
- Point out common mistakes and how to avoid them
- Encourage best practices (PEP 8, idiomatic Python)
- If a concept builds on another, briefly mention the prerequisite

Format code examples in fenced code blocks with the `python` language tag.
"""

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def chat(conversation_history: list[dict]) -> str:
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                # Cache the system prompt — it never changes across turns
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=conversation_history,
    )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    reply = "\n".join(text_blocks)

    cache_stats = response.usage
    print(
        f"\n[tokens — input: {cache_stats.input_tokens}, "
        f"cache_write: {getattr(cache_stats, 'cache_creation_input_tokens', 0)}, "
        f"cache_read: {getattr(cache_stats, 'cache_read_input_tokens', 0)}, "
        f"output: {cache_stats.output_tokens}]"
    )

    return reply


def main() -> None:
    print("Python Learning Agent")
    print("Type your Python questions below. Type 'quit' or 'exit' to stop.\n")

    conversation: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        conversation.append({"role": "user", "content": user_input})

        print("\nTutor: ", end="", flush=True)
        reply = chat(conversation)
        print(reply)
        print()

        conversation.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
