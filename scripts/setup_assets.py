"""
Author: Joshua Ashkinaze

Description: Downloads the model and corpus assets the analyses need but pip does not install.
pybiber tags text with a spaCy pipeline, the Empath feature code tokenizes and POS-tags with
NLTK, and the story-to-item similarity analysis embeds with sentence-transformers. Skips
anything already present, so it is safe to re-run.

Inputs:
    - none (downloads from PyPI, the NLTK index and the Hugging Face hub)

Outputs:
    - spaCy en_core_web_sm into the active environment
    - NLTK punkt, punkt_tab and averaged_perceptron_tagger_eng into the NLTK data path
    - sentence-transformers checkpoints into the Hugging Face cache
"""

import shutil
import subprocess
import sys

SPACY_MODEL = ("en_core_web_sm", "3.8.0")
NLTK_DATA = [("tokenizers/punkt", "punkt"),
             ("tokenizers/punkt_tab", "punkt_tab"),
             ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")]
# Only the models the tracked analyses actually load.
SBERT_MODELS = ["sentence-transformers/all-mpnet-base-v2", "hkunlp/instructor-large"]

###############
# Setup functions
###############

def install_spacy_model():
    name, want = SPACY_MODEL
    try:
        from importlib.metadata import version
        if version(name) == want:
            print(f"  spaCy {name}=={want}: present")
            return
    except Exception:
        pass
    url = (f"https://github.com/explosion/spacy-models/releases/download/"
           f"{name}-{want}/{name}-{want}-py3-none-any.whl")
    print(f"  spaCy {name}=={want}: installing")
    # A uv-built environment has no pip, so install through uv when it is available
    if shutil.which("uv"):
        command = ["uv", "pip", "install", "--quiet", "--python", sys.executable, url]
    else:
        command = [sys.executable, "-m", "pip", "install", "-q", url]
    subprocess.run(command, check=True)


def download_nltk():
    import nltk
    for path, package in NLTK_DATA:
        try:
            nltk.data.find(path)
            print(f"  nltk {package}: present")
        except LookupError:
            print(f"  nltk {package}: downloading")
            nltk.download(package, quiet=True)


def download_sbert():
    from huggingface_hub import snapshot_download
    for repo in SBERT_MODELS:
        print(f"  sbert {repo}: fetching (cached after the first run)")
        snapshot_download(repo_id=repo)

###############
# Apply
###############

if __name__ == "__main__":
    print("spaCy model (pybiber):")
    install_spacy_model()
    print("\nNLTK data (Empath features):")
    download_nltk()
    if "--skip-sbert" in sys.argv:
        print("\nsentence-transformers: skipped")
    else:
        print("\nsentence-transformers (story-to-item similarity, ~2.5 GB):")
        download_sbert()
    print("\nDone.")
