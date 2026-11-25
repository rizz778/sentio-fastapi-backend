from typing import Optional

SUMMARY_PROMPT_TEMPLATE = """You are an expert clinical summarizer for Sentio AI Therapy.
Your task is to produce an updated, highly condensed rolling summary of the ongoing therapy session incorporating the latest exchange between the client and therapist.

Current Running Summary:
{existing_summary}

Latest Turn:
Client: {user_text}
Therapist: {model_text}

Instructions:
1. Keep the summary under 180 words.
2. Focus on:
   - Core emotional states and trajectory (e.g., anxiety shifting to relief).
   - Key distress triggers and stressors mentioned.
   - Coping mechanisms or strategies explored/recommended.
   - Goals or unresolved issues discussed.
3. Discard polite chit-chat, greetings, small talk, and repetitive phrasing.
4. Use objective, concise third-person present tense (e.g., "Client shares feelings of exhaustion due to work deadlines...").
5. Return ONLY the plain text summary without any markdown headers, quotes, bullet titles, or meta-commentary.
"""


def build_summary_prompt(
    existing_summary: Optional[str],
    user_text: str,
    model_text: str,
) -> str:
    """Formats the clinical summarization prompt for the side LLM."""
    prior_context = (existing_summary or "").strip()
    if not prior_context:
        prior_context = "No prior summary. This is the start of the session."

    return SUMMARY_PROMPT_TEMPLATE.format(
        existing_summary=prior_context,
        user_text=(user_text or "").strip(),
        model_text=(model_text or "").strip(),
    )
