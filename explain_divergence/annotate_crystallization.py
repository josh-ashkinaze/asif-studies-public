"""
Author: Joshua Ashkinaze

Description: Labels each participant's open-ended responses (what worked and what did not,
combined) for whether their attitude about the issue was crystallized. Two types are judged
separately: prior familiarity (a view held, or awareness of the issue, before the study) and
reinforced (the task confirmed a belief already held). A response is crystallized when either
applies. Uses gpt-4.1-mini through LiteLLM at temperature 0. Only the AIM and vignette arms
answered these questions.

Inputs:
    - sci_fund_real/data/clean/analysis_data.csv: Study 1 (science funding)
    - satellite/data/clean/analysis_data.csv: Study 2 (satellite cybersecurity)
    - crystallization_sample_200.xlsx: hand-labeling sample (only with --sample)
    - .env: OPENAI_API_KEY

Outputs:
    - data/crystallization_annotations.csv: one row per participant with prior_familiarity,
      reinforced, is_crystallized (either type), an evidence quote per type, and empty
      human_prior_familiarity and human_reinforced columns for hand labels
    - crystallization_sample_200_llm.xlsx: the sample sheet with the LLM labels added
      (only with --sample; the original sheet is left untouched)

Notes: A full run is 1,264 calls. The script exits without calling the model if its output
    already exists; delete the output to re-annotate. The label is inferred from a response
    written after treatment, so it reports a claim about a prior state rather than a measurement
    taken beforehand. The manuscript's Supplementary Information reproduces PROMPT verbatim.
    Usage: python annotate_crystallization.py [--sample crystallization_sample_200.xlsx]
"""

import argparse
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent     # explain_divergence/
ROOT = HERE.parent                         # repo root
load_dotenv(ROOT / ".env")
import litellm
litellm.suppress_debug_info = True
warnings.filterwarnings("ignore")

MODEL = "gpt-4.1-mini"
OUT = HERE / "data" / "crystallization_annotations.csv"
STUDIES = {"Study 1": (ROOT / "sci_fund_real/data/clean/analysis_data.csv", "science funding cuts"),
           "Study 2": (ROOT / "satellite/data/clean/analysis_data.csv", "satellite cybersecurity")}
TYPES = ["prior_familiarity", "reinforced"]
PROMPT = """INSTRUCTIONS
You are coding open-ended survey responses from a study about {topic}. Participants wrote about
what was effective and ineffective in a task meant to make the issue feel personally close.
Decide whether the response shows that the participant's attitude about {topic} was
CRYSTALLIZED: settled before the task, so the task had little room to move it.

Judge each of the two types below separately. A response can show neither, one, or both.

Type 1: Prior familiarity
The participant held a settled or strong view on {topic} BEFORE the study. The view can be
supportive or opposed. This type also includes participants who say they were already aware of
or informed about {topic} or its risks, so the task told them nothing new.
Example signals: "I've always supported this", "I was already convinced", "my mind was made up",
"I've long believed", "I already cared a lot about this", "I already knew about this", "I was
already aware of these risks", "nothing here was new to me", "I work in this field, so I knew".

Type 2: Reinforced
The participant explicitly says the task confirmed, reaffirmed, or strengthened a belief they
already held, rather than changing it. If they describe that belief as vague, weak, or uncertain,
Type 2 does not apply.
Example signals: "it just reinforced what I believed", "it confirmed what I thought", "it reminded
me why I care", "it only strengthened my view".

Neither type applies when the participant only:
- says "it didn't change my attitude" with no reason, or with a reason about the story, the
  interface, the AI suggestions, or the scenario's plausibility rather than a view on {topic}
- doubts the scenario or its plausibility ("this would never happen")
- describes personal EXPERIENCE with the issue without a settled view or prior awareness
  ("I have chronic pain")
- discusses the story, the interface, or the AI suggestions

Return JSON:
{{"prior_familiarity": 0 or 1,
  "prior_familiarity_evidence": "<exact quote supporting Type 1, or empty>",
  "reinforced": 0 or 1,
  "reinforced_evidence": "<exact quote supporting Type 2, or empty>"}}

Response: \"\"\"{text}\"\"\""""

###############
# Wrangling functions
###############

def build_frame():
    """One row per AIM or vignette participant, both open-ended answers combined."""
    frames = []
    for study, (path, topic) in STUDIES.items():
        d = pd.read_csv(path).query("treat_status in ['treat', 'vignette']").copy()
        d["text"] = (d.why_why_not_pos.fillna("").astype(str) + " " +
                     d.why_why_not_neg.fillna("").astype(str)).str.strip()
        d = d[d.text.str.split().str.len() > 3]
        frames.append(pd.DataFrame({"study": study, "topic": topic, "arm": d.treat_status,
                                    "ResponseId": d.ResponseId, "text": d.text}))
    return pd.concat(frames, ignore_index=True)

###############
# Annotation functions
###############

def annotate_one(args):
    topic, text = args
    try:
        r = litellm.completion(model=MODEL, temperature=0, max_tokens=400,
                               messages=[{"role": "user", "content": PROMPT.format(topic=topic, text=text)}],
                               response_format={"type": "json_object"})
        return json.loads(r.choices[0].message.content)
    except Exception:
        return {}


def flag(o, key):
    """A 0/1 label from the model; anything else counts as 0."""
    try:
        return int(int(o.get(key, 0)) == 1)
    except (TypeError, ValueError):
        return 0


def annotate(d):
    """Add both type labels, their evidence, and is_crystallized (either type) to d."""
    with ThreadPoolExecutor(max_workers=12) as ex:
        out = list(ex.map(annotate_one, zip(d.topic, d.text)))
    for t in TYPES:
        d[t] = [flag(o, t) for o in out]
        d[f"{t}_evidence"] = [o.get(f"{t}_evidence", "") for o in out]
    d["is_crystallized"] = d[TYPES].max(axis=1)
    d["error"] = [int(not o) for o in out]
    return d

###############
# Apply
###############

parser = argparse.ArgumentParser()
parser.add_argument("--sample", help="hand-labeling sheet to annotate instead of the full data")
args = parser.parse_args()

if args.sample:
    src = Path(args.sample).resolve()
    out = src.with_name(f"{src.stem}_llm.xlsx")
    if out.exists():
        raise SystemExit(f"{out} already exists; delete it to re-annotate.")
    sheet = pd.read_excel(src)
    llm = annotate(sheet[["topic", "text"]].copy()).drop(columns=["topic", "text"]).add_prefix("llm_")
    pd.concat([sheet, llm], axis=1).to_excel(out, index=False)
    print(f"wrote {out}: {len(sheet)} responses, {llm.llm_error.sum()} errors")
elif OUT.exists():
    raise SystemExit(f"{OUT} already exists; delete it to re-annotate.")
else:
    d = annotate(build_frame())
    for t in TYPES:
        d[f"human_{t}"] = ""
    d.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(d)} participants, {d.error.sum()} errors")
