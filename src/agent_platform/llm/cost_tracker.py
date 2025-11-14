"""
Token usage and cost tracking for LLM operations.

This module provides detailed tracking of token usage and associated costs
across different LLM models and providers. It supports exporting data to
various formats for analysis and reporting.
"""

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


# Pricing information per 1K tokens (as of 2024)
# Prices are in USD per 1,000 tokens
PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI Models
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Anthropic Claude Models
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    # Google Gemini Models
    "gemini-pro": {"input": 0.00025, "output": 0.0005},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    # Ollama Models (free/local)
    "ollama/llama2": {"input": 0.0, "output": 0.0},
    "ollama/mistral": {"input": 0.0, "output": 0.0},
    "ollama/codellama": {"input": 0.0, "output": 0.0},
    "ollama/llama3": {"input": 0.0, "output": 0.0},
    "ollama/mixtral": {"input": 0.0, "output": 0.0},
}


@dataclass
class TokenUsage:
    """
    Record of token usage for a single LLM request.

    Attributes:
        prompt_tokens: Number of tokens in the prompt/input.
        completion_tokens: Number of tokens in the completion/output.
        total_tokens: Total tokens (prompt + completion).
        cost: Total cost in USD for this request.
        model: Model identifier used for this request.
        timestamp: When this request was made.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    model: str
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary with ISO format timestamp."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class CostTracker:
    """
    Track token usage and costs across LLM requests.

    This class maintains a history of all LLM requests with detailed
    token usage and cost information. It provides methods for querying
    statistics and exporting data for analysis.

    Example:
        >>> tracker = CostTracker()
        >>>
        >>> # Track a request
        >>> tracker.track_usage(
        ...     model="gpt-4",
        ...     prompt_tokens=100,
        ...     completion_tokens=50,
        ...     cost=0.009
        ... )
        >>>
        >>> # Get summary
        >>> summary = tracker.get_usage_summary()
        >>> print(f"Total cost: ${summary['total_cost']:.4f}")
        >>>
        >>> # Export to file
        >>> tracker.export_to_json("usage_report.json")
    """

    def __init__(self):
        """Initialize the cost tracker."""
        self.usage_history: List[TokenUsage] = []
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        self.usage_by_model: Dict[str, Dict] = {}

        logger.info("Initialized CostTracker")

    def track_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: Optional[float] = None,
    ) -> None:
        """
        Track usage for a single LLM request.

        Args:
            model: Model identifier.
            prompt_tokens: Number of tokens in the prompt.
            completion_tokens: Number of tokens in the completion.
            cost: Optional pre-calculated cost. If not provided, will be
                 calculated using the PRICING table.

        Example:
            >>> tracker.track_usage("gpt-4", 100, 50)
            >>> tracker.track_usage("claude-3-sonnet", 200, 100, cost=0.045)
        """
        total_tokens = prompt_tokens + completion_tokens

        # Calculate cost if not provided
        if cost is None:
            cost = calculate_cost(model, prompt_tokens, completion_tokens)

        # Create usage record
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            model=model,
            timestamp=datetime.utcnow(),
        )

        # Add to history
        self.usage_history.append(usage)

        # Update totals
        self.total_cost += cost
        self.total_tokens += total_tokens

        # Update per-model statistics
        if model not in self.usage_by_model:
            self.usage_by_model[model] = {
                "requests": 0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost": 0.0,
            }

        self.usage_by_model[model]["requests"] += 1
        self.usage_by_model[model]["total_tokens"] += total_tokens
        self.usage_by_model[model]["prompt_tokens"] += prompt_tokens
        self.usage_by_model[model]["completion_tokens"] += completion_tokens
        self.usage_by_model[model]["total_cost"] += cost

        logger.debug(
            f"Tracked usage for {model}",
            extra={
                "model": model,
                "tokens": total_tokens,
                "cost": cost,
            },
        )

    def get_total_cost(self) -> float:
        """
        Get the total cost across all requests.

        Returns:
            Total cost in USD.

        Example:
            >>> cost = tracker.get_total_cost()
            >>> print(f"Total spent: ${cost:.2f}")
        """
        return self.total_cost

    def get_usage_by_model(self) -> Dict[str, Dict]:
        """
        Get aggregated usage statistics per model.

        Returns:
            Dictionary mapping model names to their usage statistics.

        Example:
            >>> usage = tracker.get_usage_by_model()
            >>> for model, stats in usage.items():
            ...     print(f"{model}: {stats['requests']} requests, ${stats['total_cost']:.4f}")
        """
        return self.usage_by_model.copy()

    def get_usage_summary(self) -> Dict:
        """
        Get a comprehensive summary of all usage.

        Returns:
            Dictionary with total cost, tokens, averages, and per-model breakdown.

        Example:
            >>> summary = tracker.get_usage_summary()
            >>> print(f"Total requests: {summary['total_requests']}")
            >>> print(f"Average cost: ${summary['average_cost_per_request']:.4f}")
        """
        total_requests = len(self.usage_history)
        avg_cost = self.total_cost / total_requests if total_requests > 0 else 0.0
        avg_tokens = self.total_tokens / total_requests if total_requests > 0 else 0.0

        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_requests": total_requests,
            "average_cost_per_request": avg_cost,
            "average_tokens_per_request": avg_tokens,
            "usage_by_model": self.usage_by_model,
        }

    def export_to_json(self, filepath: str) -> None:
        """
        Export usage history to a JSON file.

        Args:
            filepath: Path to the output JSON file.

        Example:
            >>> tracker.export_to_json("reports/usage_2024.json")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "summary": self.get_usage_summary(),
            "usage_history": [usage.to_dict() for usage in self.usage_history],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Exported usage data to JSON",
            extra={"filepath": str(filepath), "records": len(self.usage_history)},
        )

    def export_to_csv(self, filepath: str) -> None:
        """
        Export usage history to a CSV file.

        Args:
            filepath: Path to the output CSV file.

        Example:
            >>> tracker.export_to_csv("reports/usage_2024.csv")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            if not self.usage_history:
                logger.warning("No usage history to export")
                return

            # Define CSV columns
            fieldnames = [
                "timestamp",
                "model",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for usage in self.usage_history:
                writer.writerow(usage.to_dict())

        logger.info(
            f"Exported usage data to CSV",
            extra={"filepath": str(filepath), "records": len(self.usage_history)},
        )

    def reset(self) -> None:
        """
        Clear all tracking data.

        This resets the tracker to its initial state, removing all
        usage history and statistics.

        Example:
            >>> tracker.reset()
            >>> assert tracker.get_total_cost() == 0.0
        """
        records_cleared = len(self.usage_history)

        self.usage_history.clear()
        self.total_cost = 0.0
        self.total_tokens = 0
        self.usage_by_model.clear()

        logger.info(f"Reset cost tracker", extra={"records_cleared": records_cleared})

    def get_cost_breakdown(self) -> Dict[str, float]:
        """
        Get cost breakdown by model.

        Returns:
            Dictionary mapping model names to their total costs.

        Example:
            >>> breakdown = tracker.get_cost_breakdown()
            >>> for model, cost in breakdown.items():
            ...     print(f"{model}: ${cost:.4f}")
        """
        return {
            model: stats["total_cost"]
            for model, stats in self.usage_by_model.items()
        }

    def get_token_breakdown(self) -> Dict[str, int]:
        """
        Get token usage breakdown by model.

        Returns:
            Dictionary mapping model names to their total token counts.

        Example:
            >>> breakdown = tracker.get_token_breakdown()
            >>> for model, tokens in breakdown.items():
            ...     print(f"{model}: {tokens:,} tokens")
        """
        return {
            model: stats["total_tokens"]
            for model, stats in self.usage_by_model.items()
        }


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the cost for a given model and token counts.

    Uses the PRICING table to determine costs. If the model is not found
    in the pricing table, returns 0.0 and logs a warning.

    Args:
        model: Model identifier.
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.

    Returns:
        Total cost in USD.

    Example:
        >>> cost = calculate_cost("gpt-4", 1000, 500)
        >>> print(f"Cost: ${cost:.4f}")
        Cost: $0.0600
    """
    # Try exact match first
    if model in PRICING:
        pricing = PRICING[model]
    else:
        # Try to find a partial match (for versioned models)
        matched_pricing = None
        for key in PRICING:
            if key in model or model in key:
                matched_pricing = PRICING[key]
                break

        if matched_pricing is None:
            logger.warning(
                f"No pricing data for model '{model}', using $0",
                extra={"model": model},
            )
            return 0.0

        pricing = matched_pricing

    # Calculate cost per 1K tokens
    input_cost = (prompt_tokens / 1000) * pricing["input"]
    output_cost = (completion_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost

    return total_cost


def get_model_pricing(model: str) -> Optional[Dict[str, float]]:
    """
    Get pricing information for a specific model.

    Args:
        model: Model identifier.

    Returns:
        Dictionary with 'input' and 'output' pricing per 1K tokens,
        or None if model not found.

    Example:
        >>> pricing = get_model_pricing("gpt-4")
        >>> print(f"Input: ${pricing['input']}, Output: ${pricing['output']}")
    """
    return PRICING.get(model)


def estimate_cost(
    model: str, prompt: str, estimated_completion_tokens: int
) -> Dict[str, float]:
    """
    Estimate the cost for a prompt before making the request.

    This provides a rough estimate based on character count for the prompt
    and an estimated completion length.

    Args:
        model: Model identifier.
        prompt: The prompt text.
        estimated_completion_tokens: Expected number of completion tokens.

    Returns:
        Dictionary with estimated prompt_cost, completion_cost, and total_cost.

    Example:
        >>> prompt = "Write a short story about a robot."
        >>> estimate = estimate_cost("gpt-4", prompt, 500)
        >>> print(f"Estimated total: ${estimate['total_cost']:.4f}")
    """
    # Rough estimate: 1 token ≈ 4 characters
    estimated_prompt_tokens = len(prompt) // 4

    # Calculate costs
    cost = calculate_cost(model, estimated_prompt_tokens, estimated_completion_tokens)

    pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
    prompt_cost = (estimated_prompt_tokens / 1000) * pricing["input"]
    completion_cost = (estimated_completion_tokens / 1000) * pricing["output"]

    return {
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_completion_tokens": estimated_completion_tokens,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": cost,
    }
