#!/usr/bin/env python3
"""
PIF Hallucination Threshold Experiment

Tests the hypothesis that LLM hallucinations are phase transitions occurring
when Information Load exceeds System Capacity.

Implements a "Needle in a Haystack" stress test across increasing fact counts.
"""

import json
import os
import random
import string
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from root .env
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import openai
from google import genai
from google.genai import types

# Configuration
LOAD_RANGE = range(50, 501, 50)  # 50 to 500, step 50 (higher load to find threshold)
TRIALS_PER_LOAD = 5  # Reduced trials for faster iteration
OUTPUT_DIR = Path(__file__).parent

# Word banks for generating random facts
NOUNS = [
    "flimflam", "zorbnak", "quibble", "snorkel", "blinkton", "grozzle",
    "plonkus", "whimwham", "dingleberry", "frobscottle", "splunge", "mugwump",
    "wobblekin", "snickerdoodle", "bumfuzzle", "lollygag", "snickersnee",
    "collywobbles", "cattywampus", "kerfuffle", "brouhaha", "hullabaloo"
]

ATTRIBUTES = [
    "colored", "shaped like", "originating from", "discovered in", "named after",
    "associated with", "composed of", "powered by", "connected to", "derived from"
]

VALUES = [
    "Blue", "Red", "Green", "Yellow", "Purple", "Orange", "Silver", "Gold",
    "a triangle", "a hexagon", "a spiral", "ancient Egypt", "the moon",
    "Dr. Wigglesworth", "quantum foam", "crystallized starlight", "compressed time",
    "the Northern Lights", "underground rivers", "volcanic ash", "frozen lightning"
]


def generate_random_word(length: int = 8) -> str:
    """Generate a random nonsense word."""
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def generate_facts(n: int) -> list[dict]:
    """
    Generate N random key-value facts.

    Returns list of dicts with 'noun', 'attribute', 'value' keys.
    """
    facts = []
    used_nouns = set()

    for _ in range(n):
        # Generate unique noun (use random word if we exhaust the bank)
        if len(used_nouns) < len(NOUNS):
            available = [noun for noun in NOUNS if noun not in used_nouns]
            noun = random.choice(available)
        else:
            noun = generate_random_word()
        used_nouns.add(noun)

        attribute = random.choice(ATTRIBUTES)
        value = random.choice(VALUES)

        facts.append({
            "noun": noun,
            "attribute": attribute,
            "value": value
        })

    return facts


def format_facts_as_context(facts: list[dict]) -> str:
    """Format facts into a system prompt context block."""
    lines = ["Here are some facts you must remember:\n"]
    for i, fact in enumerate(facts, 1):
        lines.append(f"{i}. The {fact['noun']} is {fact['attribute']}: {fact['value']}")
    return "\n".join(lines)


def create_question(fact: dict) -> str:
    """Create a question about a specific fact."""
    return f"What is the {fact['noun']} {fact['attribute']}?"


def get_system_prompt(facts_context: str) -> str:
    """Create the full system prompt with facts and response format instructions."""
    return f"""{facts_context}

You are a fact retrieval assistant. When asked about any of the facts above,
you must respond with the correct answer.

IMPORTANT: You MUST respond with valid JSON in this exact format:
{{"answer": "your answer here", "confidence_score": 0.0 to 1.0}}

The confidence_score should reflect how certain you are about your answer (0.0 = no confidence, 1.0 = completely certain).
Only output the JSON, nothing else."""


def query_openai(system_prompt: str, question: str) -> dict:
    """Query OpenAI gpt-4o-mini with JSON mode."""
    client = openai.OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=100
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except json.JSONDecodeError:
        return {"answer": content, "confidence_score": 0.5, "parse_error": True}
    except Exception as e:
        return {"answer": str(e), "confidence_score": 0.0, "error": True}


def query_gemini(system_prompt: str, question: str) -> dict:
    """Query Gemini 2.0 Flash with JSON mode using new SDK."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    try:
        # Combine system and user prompt
        full_prompt = f"{system_prompt}\n\nQuestion: {question}"

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=100,
                response_mime_type="application/json"
            )
        )

        content = response.text
        return json.loads(content)

    except json.JSONDecodeError:
        return {"answer": content, "confidence_score": 0.5, "parse_error": True}
    except Exception as e:
        return {"answer": str(e), "confidence_score": 0.0, "error": True}


def check_answer(response: dict, expected_value: str) -> bool:
    """Check if the model's answer matches the expected value."""
    answer = response.get("answer", "").lower().strip()
    expected = expected_value.lower().strip()

    # Check for exact match or containment
    return expected in answer or answer in expected


def run_experiment() -> pd.DataFrame:
    """Run the full experiment across all load sizes and trials."""
    results = []

    print("=" * 60)
    print("PIF HALLUCINATION THRESHOLD EXPERIMENT")
    print("=" * 60)
    print(f"Load range: {LOAD_RANGE.start} to {LOAD_RANGE.stop - 1} (step {LOAD_RANGE.step})")
    print(f"Trials per load: {TRIALS_PER_LOAD}")
    print(f"Models: gpt-4o-mini, gemini-2.0-flash")
    print("=" * 60)

    total_iterations = len(LOAD_RANGE) * TRIALS_PER_LOAD * 2  # 2 models

    with tqdm(total=total_iterations, desc="Running trials") as pbar:
        for load_size in LOAD_RANGE:
            for trial in range(TRIALS_PER_LOAD):
                # Generate facts for this trial
                facts = generate_facts(load_size)

                # Select random fact to query
                target_fact = random.choice(facts)

                # Create prompts
                facts_context = format_facts_as_context(facts)
                system_prompt = get_system_prompt(facts_context)
                question = create_question(target_fact)
                expected_value = target_fact["value"]

                # Test OpenAI
                openai_response = query_openai(system_prompt, question)
                openai_correct = check_answer(openai_response, expected_value)
                openai_confidence = openai_response.get("confidence_score", 0.5)

                results.append({
                    "load_size": load_size,
                    "trial": trial + 1,
                    "model": "gpt-4o-mini",
                    "is_correct": openai_correct,
                    "confidence_score": float(openai_confidence),
                    "expected": expected_value,
                    "answer": openai_response.get("answer", ""),
                    "had_error": openai_response.get("error", False)
                })
                pbar.update(1)

                # Small delay to avoid rate limits
                time.sleep(0.1)

                # Test Gemini
                gemini_response = query_gemini(system_prompt, question)
                gemini_correct = check_answer(gemini_response, expected_value)
                gemini_confidence = gemini_response.get("confidence_score", 0.5)

                results.append({
                    "load_size": load_size,
                    "trial": trial + 1,
                    "model": "gemini-2.0-flash",
                    "is_correct": gemini_correct,
                    "confidence_score": float(gemini_confidence),
                    "expected": expected_value,
                    "answer": gemini_response.get("answer", ""),
                    "had_error": gemini_response.get("error", False)
                })
                pbar.update(1)

                # Small delay to avoid rate limits
                time.sleep(0.1)

    return pd.DataFrame(results)


def generate_visualizations(df: pd.DataFrame):
    """Generate the phase transition visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color scheme
    colors = {
        "gpt-4o-mini": "#10a37f",      # OpenAI green
        "gemini-2.0-flash": "#4285f4"  # Google blue
    }

    # === SUBPLOT 1: Threshold Curve (Accuracy vs Load) ===
    ax1 = axes[0]

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        accuracy_by_load = model_df.groupby("load_size")["is_correct"].mean()

        ax1.plot(
            accuracy_by_load.index,
            accuracy_by_load.values,
            marker='o',
            linewidth=2,
            markersize=6,
            label=model,
            color=colors.get(model, "gray")
        )

    ax1.set_xlabel("Information Load (Number of Facts)", fontsize=12)
    ax1.set_ylabel("Average Accuracy", fontsize=12)
    ax1.set_title("The Threshold Curve\n(PIF Prediction: Accuracy drops when Load > Capacity)", fontsize=12)
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')

    # === SUBPLOT 2: Delusion Map (Confidence vs Load) ===
    ax2 = axes[1]

    # Separate correct and incorrect
    correct_df = df[df["is_correct"] == True]
    incorrect_df = df[df["is_correct"] == False]

    # Plot incorrect first (red) then correct (green) so correct points are on top
    ax2.scatter(
        incorrect_df["load_size"],
        incorrect_df["confidence_score"],
        c='red',
        alpha=0.5,
        s=40,
        label=f'Incorrect (n={len(incorrect_df)})'
    )

    ax2.scatter(
        correct_df["load_size"],
        correct_df["confidence_score"],
        c='green',
        alpha=0.5,
        s=40,
        label=f'Correct (n={len(correct_df)})'
    )

    ax2.set_xlabel("Information Load (Number of Facts)", fontsize=12)
    ax2.set_ylabel("Confidence Score", fontsize=12)
    ax2.set_title("The Delusion Map\n(Red points at top = High-confidence hallucinations)", fontsize=12)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # Highlight the "danger zone" (high confidence + incorrect)
    ax2.axhspan(0.8, 1.0, alpha=0.1, color='red', label='Danger Zone')
    ax2.text(145, 0.9, "Danger\nZone", fontsize=9, ha='right', color='darkred', alpha=0.7)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "phase_transition.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {output_path}")


def print_summary(df: pd.DataFrame):
    """Print experiment summary statistics."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        print(f"\n{model}:")
        print(f"  Overall Accuracy: {model_df['is_correct'].mean():.2%}")
        print(f"  Average Confidence: {model_df['confidence_score'].mean():.2f}")

        # Accuracy at low vs high load
        low_load = model_df[model_df["load_size"] <= 50]
        high_load = model_df[model_df["load_size"] >= 100]

        print(f"  Accuracy (Load ≤ 50): {low_load['is_correct'].mean():.2%}")
        print(f"  Accuracy (Load ≥ 100): {high_load['is_correct'].mean():.2%}")

        # High-confidence errors (hallucinations)
        hallucinations = model_df[(model_df["is_correct"] == False) & (model_df["confidence_score"] >= 0.8)]
        print(f"  High-confidence errors (conf ≥ 0.8): {len(hallucinations)}")

    # PIF validation
    print("\n" + "-" * 60)
    print("PIF HYPOTHESIS CHECK:")

    overall_low = df[df["load_size"] <= 50]["is_correct"].mean()
    overall_high = df[df["load_size"] >= 100]["is_correct"].mean()

    if overall_high < overall_low:
        print(f"  ✓ Accuracy dropped from {overall_low:.2%} (low load) to {overall_high:.2%} (high load)")
        print("  → SUPPORTS PIF: System shows phase transition behavior")
    else:
        print(f"  ✗ Accuracy did not significantly drop ({overall_low:.2%} → {overall_high:.2%})")
        print("  → DOES NOT SUPPORT PIF threshold hypothesis")

    high_conf_errors = len(df[(df["is_correct"] == False) & (df["confidence_score"] >= 0.8)])
    if high_conf_errors > 0:
        print(f"  ✓ Found {high_conf_errors} high-confidence errors (hallucinations)")
        print("  → SUPPORTS PIF: Confidence persists during failure (delusion)")

    print("=" * 60)


def main():
    """Main entry point."""
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment.")
        print("Please add it to .env file in the project root.")
        return

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found in environment.")
        print("Please add it to .env file in the project root.")
        return

    # Run experiment
    df = run_experiment()

    # Save raw data
    csv_path = OUTPUT_DIR / "hallucination_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nRaw data saved to: {csv_path}")

    # Generate visualizations
    generate_visualizations(df)

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
