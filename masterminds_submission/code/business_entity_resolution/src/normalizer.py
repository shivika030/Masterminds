import re
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Legal-suffix / abbreviation dictionaries
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Core cleanup helpers
# ---------------------------------------------------------------------------

def _lowercase_strip_punct(text: str) -> str:
    """
    Convert text to lowercase and replace punctuation with spaces.
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Replace punctuation with spaces.
    # Example:
    # "AT&T" -> "at t"
    # "Pvt. Ltd." -> "pvt ltd"
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove repeated spaces.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _expand_abbreviations(text: str, mapping: dict) -> str:
    """
    Expand known abbreviations.
    """

    for pattern, expansion in mapping.items():
        text = re.sub(pattern, expansion, text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# 3. Name normalization
# ---------------------------------------------------------------------------

def normalize_name(raw_name: str) -> dict:
    """
    Normalize a single business name.

    Returns:
        clean_name
        has_legal_suffix
        legal_suffix
        full_normalized
    """

    # Step 1: lowercase and remove punctuation
    text = _lowercase_strip_punct(raw_name)

    # Step 2: expand abbreviations
    text = _expand_abbreviations(
        text,
        NAME_ABBREVIATIONS
    )

    # Store complete normalized name before removing legal suffix
    full_normalized = text

    # Step 3: detect legal suffix
    found_suffix = None

    for suffix in LEGAL_SUFFIXES:

        pattern = r"\b" + re.escape(suffix) + r"\s*$"

        if re.search(pattern, text):

            found_suffix = suffix

            text = re.sub(
                pattern,
                "",
                text
            ).strip()

            break

    return {
        "clean_name": text,
        "has_legal_suffix": found_suffix is not None,
        "legal_suffix": found_suffix,
        "full_normalized": full_normalized,
    }


# ---------------------------------------------------------------------------
# 4. Address normalization
# ---------------------------------------------------------------------------

def normalize_address(raw_address: str) -> str:
    """
    Normalize a single business address.
    """

    text = _lowercase_strip_punct(raw_address)

    text = _expand_abbreviations(
        text,
        ADDRESS_ABBREVIATIONS
    )

    return text


# ---------------------------------------------------------------------------
# 5. Series normalization functions
# ---------------------------------------------------------------------------

def normalize_name_series(series: pd.Series) -> pd.DataFrame:
    """
    Normalize a pandas Series containing business names.

    Returns a DataFrame containing:

        clean_name
        has_legal_suffix
        legal_suffix
        full_normalized

    This is the structure expected by blocking.py.
    """

    series = series.fillna("")

    result = series.apply(normalize_name)

    result_df = pd.DataFrame(
        result.tolist(),
        index=series.index
    )

    return result_df


def normalize_address_series(series: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series containing business addresses.
    """

    return series.fillna("").apply(normalize_address)


# ---------------------------------------------------------------------------
# 6. Preprocessing helper
# ---------------------------------------------------------------------------

def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize business_name and business_address columns.
    """

    name_info = normalize_name_series(
        df["business_name"]
    )

    df["business_name"] = name_info["clean_name"]

    df["business_address"] = normalize_address_series(
        df["business_address"]
    )

    return df


# ---------------------------------------------------------------------------
# 7. Quick self-test
# ---------------------------------------------------------------------------

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
        print(
            f"{name!r:45} -> "
            f"{normalize_name(name)}"
        )

    print("\n--- Address normalization ---")

    for address in test_addresses:
        print(
            f"{address!r:45} -> "
            f"{normalize_address(address)!r}"
        )

    print("\n--- Series normalization ---")

    test_df = pd.DataFrame({
        "business_name": test_names,
        "business_address": test_addresses
    })

    name_result = normalize_name_series(
        test_df["business_name"]
    )

    print(name_result)

    print("\n--- Address Series ---")

    address_result = normalize_address_series(
        test_df["business_address"]
    )

    print(address_result)