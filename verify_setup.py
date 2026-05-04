"""
Verification script for Swahili Phishing Detection environment.
"""

import sys
import importlib
from datetime import datetime

REQUIRED_PACKAGES = [
    ("numpy", "Numerical computing"),
    ("pandas", "Data manipulation"),
    ("sklearn", "Machine learning (scikit-learn)"),
    ("matplotlib", "Plotting"),
    ("seaborn", "Statistical plots"),
    ("nltk", "Natural language toolkit"),
    ("scipy", "Scientific computing"),
    ("joblib", "Model persistence"),
    ("tqdm", "Progress bars"),
    ("wordcloud", "Word cloud visualization"),
    ("emoji", "Emoji handling"),
    ("unidecode", "Unicode normalization"),
    ("openpyxl", "Excel file support"),
    ("jupyter", "Jupyter notebook"),
]


def main():
    print("=" * 60)
    print("Swahili Phishing Detection - Environment Verification")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"\nPython version: {sys.version}")
    print(f"Python executable: {sys.executable}\n")

    print("Checking required packages:")
    print("-" * 60)

    missing = []
    for pkg, description in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK]    {pkg:15s} v{version:12s}  ({description})")
        except ImportError:
            print(f"  [FAIL]  {pkg:15s} NOT INSTALLED  ({description})")
            missing.append(pkg)

    print("-" * 60)

    print("\nChecking NLTK data resources:")
    try:
        import nltk
        for resource in ["punkt", "stopwords"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if resource == "punkt"
                               else f"corpora/{resource}")
                print(f"  [OK]    NLTK '{resource}' available")
            except LookupError:
                print(f"  [INFO]  NLTK '{resource}' not downloaded yet")
    except ImportError:
        print("  [SKIP]  NLTK not installed")

    print("\nRunning quick functional test:")
    try:
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB

        sample_messages = [
            "Mpendwa mteja, akaunti yako itafungwa leo. Tuma namba ya siri.",
            "Hongera umeshinda zawadi! Bonyeza link hii kupokea.",
            "Habari za asubuhi, tutakutana saa nne ofisini.",
            "Mkutano umeahirishwa hadi kesho asubuhi.",
        ]
        labels = [1, 1, 0, 0]

        vec = TfidfVectorizer()
        X = vec.fit_transform(sample_messages)
        clf = MultinomialNB()
        clf.fit(X, labels)
        prediction = clf.predict(vec.transform(["Tuma pesa sasa hivi"]))
        print(f"  [OK]    TF-IDF + Naive Bayes pipeline works")
        print(f"          Test prediction for 'Tuma pesa sasa hivi': "
              f"{'PHISHING' if prediction[0] == 1 else 'LEGITIMATE'}")
    except Exception as e:
        print(f"  [FAIL]  Functional test failed: {e}")

    print("\n" + "=" * 60)
    if missing:
        print(f"RESULT: {len(missing)} package(s) missing: {', '.join(missing)}")
        sys.exit(1)
    else:
        print("RESULT: All packages installed correctly. Ready for Day 2!")
        print("=" * 60)


if __name__ == "__main__":
    main()