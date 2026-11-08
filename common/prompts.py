"""Every fixed prompt in the project lives here, so 'one fixed prompt' is checkable from git history."""

EXTRACT_RQ = """Read the title and abstract of a research paper. Write the single research question the paper sets out to answer, as one sentence ending with a question mark. Do not mention results, numbers or the paper itself ("this paper"). Output only the question.

Title: {title}
Abstract: {abstract}

Research question:"""

EXTRACT_CARD = """Read the title and abstract of a research paper and fill in a question card. Do NOT report any result, number, or "we show that". Answer in JSON with exactly these keys:
"problem": the problem being addressed (one sentence),
"gap": what existing work does not do (one sentence),
"gap_owner": which earlier line of work has that limitation (a few words),
"insight": the key idea of the approach (one sentence),
"why_now": why this is possible or needed now (one sentence),
"evidence_plan": one of "theory", "experiments", "both",
"contribution_type": one of "method", "theory", "empirical", "dataset", "system", "analysis",
"scope": the setting the question is limited to (a few words).

Title: {title}
Abstract: {abstract}

JSON:"""

EXTRACT_CLAIMS = """Read the title and abstract of a research paper. List between 4 and 10 atomic claims that together make up the paper's idea (what it proposes and what it asserts), one per line, each a short self-contained sentence. Do not include numbers. Output only the list, one claim per line, no bullets.

Title: {title}
Abstract: {abstract}

Claims:"""

FACETS = """For the research question below, answer four facet questions, each in a few words. Answer in JSON with keys "purpose" (what problem it addresses), "mechanism" (how it would be addressed), "evaluation" (how success would be judged), "domain" (where: data, field or setting).

Research question: {q}

JSON:"""

SAME_QUESTION_INSTRUCTION = ("Judge whether the Document asks exactly the same research question as the Query: "
                             "a complete answer to one would be a complete answer to the other. "
                             "Rewording does not matter; swapping roles, flipping a constraint, changing the goal, "
                             "the kind of inquiry or the direction makes it a different question.")

JUDGE_SAME = """Two research questions are given. Decide how Q2 relates to Q1. Answer with exactly one word:
same      - a complete answer to either is a complete answer to the other
narrower  - Q2 is a special case of Q1
broader   - Q2 contains Q1 as a special case
related   - same topic, but a different question
different - different topic

Q1: {q1}
Q2: {q2}
Answer:"""

PARAPHRASE = """Rewrite the following text so that it says exactly the same thing in different words. Keep every idea; change wording and sentence structure. Output only the rewritten text.

Text: {text}

Rewritten:"""

VAGUE = """Rewrite the following research question so that it becomes vague and general: keep the topic but remove the specific mechanism, setting and evaluation. Output only the question.

Question: {q}

Vague question:"""

FACET_EDIT = """Change only the {facet} of the following research question ({desc}) to something different but plausible; keep everything else the same. Output only the new question.

Question: {q}

New question:"""

EXTRACT_HYP = """Read the title and abstract of a research paper. Write the paper's main hypothesis: one testable sentence stating what the authors expect to be true (not the result numbers). Output only the sentence.

Title: {title}
Abstract: {abstract}

Main hypothesis:"""

JUDGE_PAIR = """Two research hypotheses were proposed from the same set of papers. Which one is the better research hypothesis, considering how novel, plausible and testable it is? Answer with exactly one letter, A or B.

A: {a}
B: {b}
Answer:"""

HYPO_PREDICT = """Use the hypothesis to predict the label of the example.

Hypothesis: {h}

Example: {x}
Possible labels: {labels}
Answer with the label only.
Label:"""

# ---- E6 generation prompts (one model, one decoding setting) ----
PAPERS_BLOCK = "{i}. {title}: {abstract}"

E6_DIRECT = """You are a researcher. Below are papers you have read.

{papers}

Propose {k} distinct, testable research hypotheses that build on these papers. Output exactly {k} lines, each starting with "H: "."""

E6_QUESTIONS = """You are a researcher. Below are papers you have read.

{papers}

Write {m} distinct research questions that these papers leave open. Each must be one sentence ending in "?". Output exactly {m} lines, each starting with "Q: "."""

E6_HYP_FOR_Q = """You are a researcher. Below are papers you have read.

{papers}

Research question: {q}

Propose {n} distinct, testable hypotheses that answer this question. Output exactly {n} lines, each starting with "H: "."""

E6_Q_FOR_H = """State, in one sentence ending in "?", the research question that the following hypothesis answers. Output only the question.

Hypothesis: {h}

Question:"""
