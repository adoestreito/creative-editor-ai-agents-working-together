import re

import ollama


def get_word_count(text):
    return len(re.findall(r"\w+", text))


def agentic_creative_engine(genre, target_length, topic):
    WRITER_MODEL = "gemma2:9b"
    EDITOR_MODEL = "llama3.1:8b"
    max_iterations = 3

    print(f"🚀 Goal: {genre} story about '{topic}' (~{target_length} words)")

    writer_sys = (
        f"You are a master of {genre} fiction. Write with high style and focus on "
        f"{genre} tropes."
    )
    writer_task = (
        f"Write a {genre} piece about: {topic}. Target length: {target_length} words."
    )

    resp = ollama.chat(
        model=WRITER_MODEL,
        messages=[
            {"role": "system", "content": writer_sys},
            {"role": "user", "content": writer_task},
        ],
    )
    draft = resp["message"]["content"]

    sep = "=" * 72

    for i in range(max_iterations):
        actual_words = get_word_count(draft)
        print(f"\n{sep}")
        print(f"🧐 Iteration {i + 1}  ·  {actual_words} words")
        print(sep)
        print("\n📄 Draft under review\n")
        print(draft)
        print()

        editor_sys = f"""You are a senior {genre} editor. You evaluate text based on a strict scorecard.
        Score each category 1-10.

        CRITERIA:
        1. GENRE ADHERENCE: Does this feel like high-quality {genre}?
        2. LENGTH: Is it close to {target_length} words? (Current: {actual_words})
        3. ENGAGEMENT: Is the prose sharp and free of clichés?
        """

        editor_task = f"""Evaluate this draft. If the average score is below 8.5, give specific revision orders.
        If it is 8.5 or higher, start your response with 'APPROVED'.

        DRAFT:
        {draft}
        """

        feedback_resp = ollama.chat(
            model=EDITOR_MODEL,
            messages=[
                {"role": "system", "content": editor_sys},
                {"role": "user", "content": editor_task},
            ],
        )
        feedback = feedback_resp["message"]["content"]

        print("📝 Editor feedback\n")
        print(feedback)
        print()

        if "APPROVED" in feedback.upper():
            print("✅ Editor approved — stopping revisions.")
            break

        print("✏️  Revising draft from editor notes…\n")
        rewrite_resp = ollama.chat(
            model=WRITER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a {genre} writer revising your work based on editor notes.",
                },
                {
                    "role": "user",
                    "content": f"EDITOR NOTES:\n{feedback}\n\nORIGINAL DRAFT:\n{draft}",
                },
            ],
        )
        draft = rewrite_resp["message"]["content"]
        print("📄 Revised draft\n")
        print(draft)
        print()

    return draft


if __name__ == "__main__":
    final_piece = agentic_creative_engine(
        genre="Horror, Thriller, Mystery",
        target_length=200,
        topic="A family that is haunted by a ghost of a man that came from a distant future and is trying to avoid the apocalypse",
    )

    print("\n--- FINAL OUTPUT ---\n")
    print(final_piece)
