import re
import pandas as pd

NAME_ABBREVIATIONS = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\binc\b": "incorporated",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\band\b": "and",
}

LEGAL_SUFFIXES = [
    "private limited",
    "limited liability partnership",
    "limited liability company",
    "public limited",
    "limited",
    "incorporated",
    "corporation",
    "company",
    "llp",
    "llc",
    "inc",
    "corp",
    "ltd",
    "pvt",
    "co",
]

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bno\.?\b": "number",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
    r"\bsec\b": "sector",
}

_SUFFIX_REGEX = r"\b(" + "|".join(re.escape(s) for s in LEGAL_SUFFIXES) + r")\s*$"


def _lowercase_strip_punct(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_abbreviations(text: str, mapping: dict) -> str:
    for pattern, expansion in mapping.items():
        text = re.sub(pattern, expansion, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(raw_name: str) -> dict:
    text = _lowercase_strip_punct(raw_name)
    text = _expand_abbreviations(text, NAME_ABBREVIATIONS)
    full_normalized = text

    found_suffix = None
    for suffix in LEGAL_SUFFIXES:
        pattern = r"\b" + re.escape(suffix) + r"\s*$"
        if re.search(pattern, text):
            found_suffix = suffix
            text = re.sub(pattern, "", text).strip()
            break

    return {
        "clean_name": text,
        "has_legal_suffix": found_suffix is not None,
        "legal_suffix": found_suffix,
        "full_normalized": full_normalized,
    }


def normalize_address(raw_address: str) -> str:
    text = _lowercase_strip_punct(raw_address)
    text = _expand_abbreviations(text, ADDRESS_ABBREVIATIONS)
    return text


def normalize_name_series(series: pd.Series) -> pd.DataFrame:
    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    for pattern, expansion in NAME_ABBREVIATIONS.items():
        s = s.str.replace(pattern, expansion, regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    full_normalized = s
    extracted_suffix = s.str.extract(_SUFFIX_REGEX, expand=False)
    has_suffix = extracted_suffix.notna()
    clean_name = s.str.replace(_SUFFIX_REGEX, "", regex=True).str.strip()

    return pd.DataFrame(
        {
            "clean_name": clean_name,
            "has_legal_suffix": has_suffix,
            "legal_suffix": extracted_suffix,
            "full_normalized": full_normalized,
        },
        index=series.index,
    )


def normalize_address_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    for pattern, expansion in ADDRESS_ABBREVIATIONS.items():
        s = s.str.replace(pattern, expansion, regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    return s


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    name_info = normalize_name_series(df["business_name"])
    df["business_name"] = name_info["clean_name"]
    df["business_address"] = normalize_address_series(df["business_address"])
    return df


if __name__ == "__main__":
    test_names = [
        "Sharma & Sons Pvt. Ltd.",
        "ACME Corp",
        "Global Tech Solutions Private Limited",
        "Bob's Diner, Inc.",
    ]

    test_addresses = [
        "123 MG Rd, Nr. SBI ATM, Sec-14",
        "45 Main St., Apt 3, Bldg 2",
        "78 Park Ave, Floor 4",
        "12 Market Rd, Opp. Mall",
    ]

    print("--- Name normalization ---")
    for name in test_names:
        print(f"{name!r:45} -> {normalize_name(name)}")

    print("\n--- Address normalization ---")
    for address in test_addresses:
        print(f"{address!r:45} -> {normalize_address(address)!r}")

    print("\n--- Series normalization ---")
    test_df = pd.DataFrame(
        {
            "business_name": test_names,
            "business_address": test_addresses,
        }
    )
    name_result = normalize_name_series(test_df["business_name"])
    print(name_result)

    print("\n--- Address Series ---")
    address_result = normalize_address_series(test_df["business_address"])
    print(address_result)