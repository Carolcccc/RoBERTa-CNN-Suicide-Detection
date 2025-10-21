#!/usr/bin/env python3
"""
Data_api_public.py

Sanitized public version of the Data API example.

This file does NOT contain any API keys. Set your OpenAI API key in an
environment variable before running (zsh / macOS example shown below).

Usage:
  export OPENAI_API_KEY="sk-..."
  python "Data Api/Data_api_public.py" --text "I'm feeling hopeless."

This script performs a single example call and prints the model output.
"""

import os
import sys
import argparse
try:
    import openai
except Exception as e:
    print("Please install the OpenAI package: pip install openai")
    raise


def assess_suicidal_intent(text: str) -> str:
    """Call the OpenAI completion API to assess suicidal intent.

    Note: this function assumes OPENAI_API_KEY is set in the environment.
    """
    # You can change the model/engine as needed (eg 'text-davinci-003')
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Assess the following sentence for any indication of suicidal intent: \"{text}\"",
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.0,
    )
    return response.choices[0].text.strip()


def main():
    parser = argparse.ArgumentParser(description="Sanitized Data API demo (no API key in file)")
    parser.add_argument("--text", type=str, required=False, help="Text to assess")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set.\nSet it with (zsh):\n  export OPENAI_API_KEY=\"sk-...\"")
        sys.exit(1)

    openai.api_key = api_key

    text = args.text or "I'm feeling hopeless and don't want to go on."
    print("Input text:", text)
    try:
        result = assess_suicidal_intent(text)
    except Exception as e:
        print("API call failed:", e)
        sys.exit(1)

    print("\nModel output:\n", result)


if __name__ == "__main__":
    main()
