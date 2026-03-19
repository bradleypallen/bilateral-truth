"""
Situation variation generator for modal evaluation (Definition 25, BBL paper).

Generates meaning-preserving syntactic variants of situation descriptions
to support Monte Carlo estimation of the necessity operator □p:

    [[□p]]^M_s ≈ <min u_i, max v_i>   for VM(s'_i, p), i = 1..n

Variation types (Def. 25):
  - Lexical paraphrasing: synonym substitution
  - Syntactic reordering: clause/phrase reordering preserving meaning
  - Neutral elaboration: adding semantically neutral qualifiers
  - Formatting changes: punctuation, capitalisation, whitespace
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_VARIATION_PROMPT = """You are generating meaning-preserving paraphrases of a situation description for a formal logic experiment.

A "situation description" is a natural language context that frames an atomic factual claim. Your task is to produce {n} distinct paraphrases of the situation below. Each paraphrase must:

1. Preserve the meaning exactly — the same facts must be communicated.
2. Use different surface form — vary wording, sentence structure, or phrasing.
3. Remain natural and grammatically correct English.
4. Not add new information or omit existing information.
5. Be suitable as a neutral factual context (no rhetorical framing).

Apply a mix of these variation strategies:
- Lexical: substitute synonyms or equivalent phrases
- Syntactic: reorder clauses, change active/passive voice, restructure sentences
- Elaboration: add semantically neutral qualifiers ("It is known that...", "According to established records,...")
- Formatting: vary punctuation and capitalisation conventions

Original situation:
{situation}

Return ONLY a JSON array of {n} strings — the paraphrases — with no other text.
Example format: ["paraphrase 1", "paraphrase 2", ...]"""


class SituationVariationGenerator:
    """Generates meaning-preserving variants of situation descriptions.

    Uses Claude Opus 4.6 to produce n lexically and syntactically diverse
    paraphrases of a given situation string, for use in Monte Carlo
    estimation of the BBL necessity operator □p.
    """

    DEFAULT_MODEL = "claude-opus-4-6"

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """
        Args:
            api_key: Anthropic API key. If None, reads ANTHROPIC_API_KEY from environment.
            model: Anthropic model to use for generation.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.model = model
        import anthropic as _anthropic
        self.client = _anthropic.Anthropic(api_key=self.api_key)

    def generate(self, situation: str, n: int = 5) -> list[str]:
        """Generate n meaning-preserving paraphrases of a situation description.

        Args:
            situation: The situation description to paraphrase (δS(s) in the paper).
            n: Number of variants to generate (default 5).

        Returns:
            List of n paraphrase strings. If generation fails, returns fewer
            variants (possibly just the original) rather than raising.

        Raises:
            ValueError: If situation is empty or n < 1.
        """
        if not situation or not situation.strip():
            raise ValueError("situation must be a non-empty string")
        if n < 1:
            raise ValueError("n must be at least 1")

        prompt = _VARIATION_PROMPT.format(situation=situation.strip(), n=n)

        import time
        max_retries = 4
        base_delay = 2.0

        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=1.0,    # High temperature for lexical diversity
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()
                variants = self._parse_variants(raw, n)
                logger.info(
                    "Generated %d/%d variants for situation (len=%d)",
                    len(variants), n, len(situation)
                )
                return variants

            except Exception as e:
                error_str = (str(e) + type(e).__name__).lower()
                is_transient = any(m in error_str for m in (
                    "rate limit", "429", "503", "502", "overloaded",
                    "timeout", "timed out", "connection error",
                ))
                if not is_transient or attempt == max_retries:
                    logger.error(
                        "Variation generation failed after %d attempts: %s [API_ERROR]",
                        attempt, type(e).__name__
                    )
                    return [situation]  # fall back to original situation
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Transient error (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1, max_retries + 1, delay, type(e).__name__
                )
                time.sleep(delay)

        return [situation]

    def _parse_variants(self, raw: str, n: int) -> list[str]:
        """Parse the JSON array returned by the model.

        Tries strict JSON parse first; falls back to line-by-line extraction
        if the model wraps the array in markdown fences or adds preamble text.
        """
        # Strip markdown code fences if present
        text = raw
        if "```" in text:
            lines = text.splitlines()
            inner = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(inner).strip()

        # Attempt strict JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                variants = [str(v).strip() for v in parsed if str(v).strip()]
                if variants:
                    return variants[:n]
        except json.JSONDecodeError:
            pass

        # Fall back: find a JSON array substring
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                if isinstance(parsed, list):
                    variants = [str(v).strip() for v in parsed if str(v).strip()]
                    if variants:
                        return variants[:n]
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse variant JSON; returning raw lines [PARSE_FAILURE]")
        # Last resort: treat non-empty lines as variants
        lines = [l.strip().strip('"').strip("'").strip(",") for l in raw.splitlines()]
        variants = [l for l in lines if len(l) > 10]
        return variants[:n] if variants else [raw[:500]]
