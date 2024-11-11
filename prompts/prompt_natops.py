
import json
import os
import sys
from collections import defaultdict


########### EQUIVALENCE

def get_templates_EQUIVALENCE(span_evidence, span_claim):
    system_text = f"You are a helpful assistant."

    templates = [
        "Is \"{claim}\" a paraphrase of \"{evidence}\"?",
        "Are \"{claim}\" and \"{evidence}\" semantically equivalent in meaning?",
        "Is the meaning of \"{claim}\" effectively the same as \"{evidence}\"?",
        "Do \"{claim}\" and \"{evidence}\" function as synonyms or paraphrases of each other?",
        "Does \"{claim}\" serve as a paraphrase or an alternative expression for \"{evidence}\"?",
        "Are \"{claim}\" and \"{evidence}\" synonymous or nearly synonymous in meaning?",
        "Do \"{claim}\" and \"{evidence}\" mean the same, without using external knowledge or assumptions?",
        "Are \"{claim}\" and \"{evidence}\" semantically identical when considered independently of external knowledge?",
        "Considering just \"{claim}\" and \"{evidence}\", do these expressions have the same meaning?",
        "Comparing \"{claim}\" with \"{evidence}\", are they semantically equivalent based solely on their respective content?",
    ]

    prefix = None

    results = []
    for template in templates:
        user_text = template.format(evidence=span_evidence, claim=span_claim)
        results.append((system_text, user_text, prefix))

    return results



########### ENTAILMENT

def get_templates_ENTAILMENT(span_evidence, span_claim):
    system_text = f"You are a helpful assistant."

    templates = [
        "Given the premise \"{evidence}\" does the hypothesis \"{claim}\" hold?",
        "Does the expression \"{evidence}\" entail \"{claim}\"?",
        "Does the phrase \"{evidence}\" logically imply \"{claim}\"?",
        "Is it true that if \"{evidence}\" then \"{claim}\"?",
        "Is \"{claim}\" a valid inference from \"{evidence}\"?",
        "Can \"{claim}\" be inferred from the statement \"{evidence}\"?",
        "Given just the statements \"{evidence}\" and \"{claim}\", does the first statement logically and necessarily imply the second without any external information?",
        "Is it true that the statement \"{evidence}\" logically entails \"{claim}\" based solely on the information within the statements?",
        "Does \"{evidence}\" imply \"{claim}\" when only the information within these statements is considered?",
        "Is it accurate to say that \"{evidence}\" categorically entails \"{claim}\", without external interpretations?",
    ]

    prefix = None

    results = []
    for template in templates:
        user_text = template.format(evidence=span_evidence, claim=span_claim)
        results.append((system_text, user_text, prefix))

    return results



########### NEGATION

def get_templates_NEGATION(span_evidence, span_claim):
    system_text = f"You are a helpful assistant."

    templates = [
        "Is the phrase \"{claim}\" a negation of \"{evidence}\"?",
        "Do \"{claim}\" and \"{evidence}\" represent mutually exclusive states, where the presence of one negates the possibility of the other?",
        "Is the relationship between \"{claim}\" and \"{evidence}\" binary, such that if \"{claim}\" is true, \"{evidence}\" must necessarily be false, and vice versa?",
        "Do \"{claim}\" and \"{evidence}\" negate each other completely?",
        "Are \"{claim}\" and \"{evidence}\" in a dichotomous relationship, where the existence of one implies the non-existence of the other?",
        "Is there a mutually exclusive relationship between \"{claim}\" and \"{evidence}\", indicating that only one can be true at any given time?",
        "In the context of \"{claim}\" and \"{evidence}\", does the affirmation of one mean the automatic negation of the other?",
        "Do \"{claim}\" and \"{evidence}\" form a binary opposition, where one categorically negates the other?",
        "Are \"{claim}\" and \"{evidence}\" opposites in such a way that they cannot be true simultaneously?",
        "Is the relationship between \"{claim}\" and \"{evidence}\" characterized by a strict either/or dichotomy?",
    ]

    prefix = None

    results = []
    for template in templates:
        user_text = template.format(evidence=span_evidence, claim=span_claim)
        results.append((system_text, user_text, prefix))

    return results



########### ALTERNATION

def get_templates_ALTERNATION(span_evidence, span_claim):
    system_text = f"You are a helpful assistant."

    templates = [
        "Does \"{claim}\" exclude \"{evidence}\"?",
        "Do \"{claim}\" and \"{evidence}\" represent distinct alternatives, but not the only possibilities in their category?",
        "Are \"{claim}\" and \"{evidence}\" exclusively different without negating the existence of additional states or options?",
        "Do \"{claim}\" and \"{evidence}\" denote exclusive but not exhaustive options within a larger set of possibilities?",
        "In comparing \"{claim}\" and \"{evidence}\", are they distinct yet not limiting the possibility of other variations or alternatives?",
        "Are X and Y distinct entities or states that exclude each other without forming a complete, exhaustive set?",
        "Are \"{claim}\" and \"{evidence}\" different entities or states, but not in a way that negates the possibility of other, different entities or states?",
        "Are \"{claim}\" and \"{evidence}\" distinct entities or states that exclude each other without forming a complete, exhaustive set?",
        "In comparing \"{claim}\" and \"{evidence}\", are they exclusive in nature but not necessarily covering all possible alternatives?",
        "Do \"{claim}\" and \"{evidence}\" define separate, non-intersecting options, while not encompassing all possible scenarios?",
    ]

    prefix = None

    results = []
    for template in templates:
        user_text = template.format(evidence=span_evidence, claim=span_claim)
        results.append((system_text, user_text, prefix))

    return results



########### GET PROMPT

def get_prompt_templates(span_claim, span_evidence, symbol, swap_expressions=False):
    # Make sure that texts do not contain any double quotation marks
    span_claim = span_claim.replace('"', '\'')
    span_evidence = span_evidence.replace('"', '\'')

    if swap_expressions:
        span_evidence, span_claim = span_claim, span_evidence

    # Get prompt texts
    if symbol == "=":
        templates_data = get_templates_EQUIVALENCE(span_evidence=span_evidence, span_claim=span_claim)
    elif symbol == "<":
        templates_data = get_templates_ENTAILMENT(span_evidence=span_evidence, span_claim=span_claim)
    elif symbol == ">":
        templates_data = get_templates_ENTAILMENT(span_evidence=span_claim, span_claim=span_evidence)
    elif symbol == "!":
        templates_data = get_templates_NEGATION(span_evidence=span_claim, span_claim=span_evidence)
    elif symbol == "|":
        templates_data = get_templates_ALTERNATION(span_evidence=span_claim, span_claim=span_evidence)
    else:
        raise Exception(f"Symbol {symbol} is not supported!")

    # Build full prompts
    prompts = []
    for template_data in templates_data:
        system_text, user_text, prefix = template_data
        prompt = {
            'type': "yes/no",
            'system_text': system_text,
            'user_text': user_text,
            'params': {
                'temperature': 1.0,
                'top_p': 1.0,
                'max_new_tokens': 10,
                'stop_tokens': None
            },
            'prefix': prefix,
            'return_token_probs': True,
        }
        prompts.append(prompt)

    return prompts
