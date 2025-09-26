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
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')