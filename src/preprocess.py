"""
preprocess.py

Usage:
    python preprocess.py --twitter data/raw_twitter.json --reddit data/raw_reddit.json --prices data/prices/ --out_dir data/processed/ 

Outputs:
    data/processed/posts.csv
    data/processed/users.csv
    (and copies normalized price files to data/processed/prices/)
"""

import os
import json
import re
import string
import argparse
from datetime import datetime
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

# Optional: fuzzy duplicate detection
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------
#  Configuration / Globals
# ---------------------------
# If running first time, uncomment nltk downloads or run them once separately.
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Ticker pattern: $TSLA or plain TSLA (we try to capture both when present)
TICKER_RE = re.compile(r'\$?([A-Z]{1,5})\b')

# A small seed list of bot usernames; expand this with your findings.
KNOWN_BOTS = {"AutoModerator", "VisualMod", "GME_bot", "BitcoinBot", "StockBot"}

# Minimal params (tuneable)
MIN_POSTS_PER_USER = 2       # filter low-activity users (set to 0 to keep all)
FUZZY_DUPLICATE_THRESHOLD = 95 if RAPIDFUZZ_AVAILABLE else None


# ---------------------------
#  Utility functions
# ---------------------------

def to_utc_timestamp(dt):
    """Return pandas.Timestamp in UTC or NaT on failure."""
    try:
        return pd.to_datetime(dt, utc=True)
    except Exception:
        return pd.NaT

def strip_nonprintable(s: str):
    """Remove non-ascii characters (emojis etc) safely."""
    if not isinstance(s, str):
        return ""
    # decode unicode escapes if present
    try:
        s = s.encode('latin1', 'ignore').decode('unicode_escape')
    except Exception:
        pass
    # remove non-ascii
    s = s.encode('ascii', 'ignore').decode('ascii')
    return s

def remove_urls(text: str):
    return re.sub(r"https?://\S+|www\.\S+", "", text or "")

def extract_tickers(text: str):
    """Return list of tickers found (uppercase). If none, empty list."""
    if not isinstance(text, str):
        return []
    # find $TSLA or TSLA-like uppercase tokens
    matches = TICKER_RE.findall(text)
    # heuristics: remove common words that match pattern accidentally, e.g., 'THE'
    filtered = [m.upper() for m in matches if len(m) <= 5 and m.isalpha()]
    # de-dup preserve order
    seen = set()
    out = []
    for t in filtered:
        if t in seen: continue
        seen.add(t)
        out.append(t)
    return out

def basic_clean(text: str):
    """Lowercase, strip, remove punctuation/digits, stopwords and lemmatize."""
    if not isinstance(text, str):
        return ""
    s = strip_nonprintable(text)
    s = remove_urls(s)
    # preserve $ for extraction earlier; remove punctuation/digits now
    s = s.translate(str.maketrans("", "", string.punctuation + string.digits))
    tokens = [tok.lower() for tok in s.split() if tok.lower() not in STOP_WORDS and len(tok) > 1]
    lem = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(lem)