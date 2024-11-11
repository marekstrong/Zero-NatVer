
import json
import os
import sys


def build_template_chunk(claim):
    # System
    system_text = f"You are a helpful assistant."

    # Task definition
    user_text = ""
    user_text += f"Split the following text into smaller chunks that can be individually fact-checked:\n"
    user_text += f"\"{claim}\"\n\n"

    return system_text, user_text


def build_template_align(claim, evidence, expressions, separator):
    # System
    system_text = f"You are a helpful assistant."

    # Context
    user_text = ""
    user_text += f"CLAIM:\n\"{claim}\""
    user_text += f"\n\n"
    user_text += f"EVIDENCE:\n\"{evidence}\""
    user_text += f"\n\n-----\n"

    user_text += f"Align the following expressions from the claim with relevant substrings from the evidence text:\n"
    for expr in expressions:
        user_text += f"{separator} {expr}\n"

    user_text += f"\nThe aligned substrings should either support the expression, refute it, or simply refer to the same entity. "
    user_text += f"Where possible, provide an explanation following each alignment."
    user_text += f"If no relevant alignment exists, write \"None\". "

    prefix = f"Here are the aligned substrings:\n\n{separator} {expressions[0]}: \""

    return system_text, user_text, prefix
