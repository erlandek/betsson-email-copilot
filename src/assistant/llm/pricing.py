"""List prices per million tokens (input, output), used for the cost column in traces.

Checked 2026-10-22 against the providers' model pages; the GPT-6 prices were
added 2026-10-27 from published price lists. An unknown model costs 0.0 rather
than failing a run: cost is reporting, not control flow.
"""

PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    # OpenAI
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6.1-sol": (2.00, 10.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_PER_MILLION.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
