"""Parse height and sex phenotype data from OpenSNP RSS feeds.

Outputs a TSV of genome_id, height_cm, sex for all individuals
with parseable height data.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path


DATA_DIR = Path("data")


def parse_rss(path: Path) -> list[dict]:
    """Parse an OpenSNP RSS file. Returns list of {genome_id, variation}."""
    tree = ET.parse(path)
    root = tree.getroot()
    entries = []
    for item in root.iter("item"):
        variation = item.findtext("variation", "").strip()
        dlink = item.findtext("dlink", "")
        # Extract genome ID from URL like http://opensnp.org/data/1.23andme.9
        genome_id = dlink.rsplit("/", 1)[-1] if "/" in dlink else ""
        if genome_id and variation:
            entries.append({"genome_id": genome_id, "variation": variation})
    return entries


def parse_height_cm(variation: str) -> float | None:
    """Try to parse a height string into centimeters."""
    v = variation.strip()
    # Normalize curly quotes and unicode
    v = v.replace("\u2018", "'").replace("\u2019", "'")
    v = v.replace("\u201c", '"').replace("\u201d", '"')
    v = v.replace("\u2032", "'").replace("\u2033", '"')
    v = v.replace("\u2034", "'")

    # "5'1" or 155cm" -> take the cm value
    m = re.search(r"(\d{3})\s*cm", v)
    if m and re.search(r"or|/", v):
        val = int(m.group(1))
        if 100 < val < 250:
            return float(val)

    # Direct cm: "169 cm", "175cm", "169"
    m = re.match(r"^(\d{3})\s*cm?\s*$", v)
    if m:
        return float(m.group(1))

    # Meters: "1.78m", "1.78 m", "1,78m", "1.78"
    m = re.match(r"^(\d)[.,](\d{2})\s*m?\s*$", v)
    if m:
        return float(m.group(1)) * 100 + float(m.group(2))

    # Feet and inches with possible fractional inches: 5'3", 5'10", 5' 10",
    # 5'3.5", 5' 6.5", 5'10', 5'3', 5'3''
    m = re.match(r"""^(\d)\s*[']\s*(\d{1,2})(?:\.(\d))?\s*['"]*\s*$""", v)
    if m:
        feet = int(m.group(1))
        inches = int(m.group(2))
        frac = float(f"0.{m.group(3)}") if m.group(3) else 0
        return feet * 30.48 + (inches + frac) * 2.54

    # "5 ft 10 in", "5ft10in", "5ft 10"
    m = re.match(r"^(\d)\s*ft\.?\s*(\d{1,2})(?:\.(\d))?\s*(?:in\.?)?\s*$", v, re.IGNORECASE)
    if m:
        feet = int(m.group(1))
        inches = int(m.group(2))
        frac = float(f"0.{m.group(3)}") if m.group(3) else 0
        return feet * 30.48 + (inches + frac) * 2.54

    # Just inches: 64", 72"
    m = re.match(r"^(\d{2})\s*[\"″]\s*$", v)
    if m:
        inches = int(m.group(1))
        if 48 <= inches <= 84:
            return inches * 2.54

    # Just feet: 6', 6 ft
    m = re.match(r"^(\d)\s*[']\s*$", v)
    if m:
        return int(m.group(1)) * 30.48

    # Bare number in cm range (140-220)
    m = re.match(r"^(\d{3})$", v)
    if m:
        val = int(m.group(1))
        if 140 <= val <= 220:
            return float(val)

    return None


def parse_sex(variation: str) -> str | None:
    """Parse sex variation into 'M' or 'F'."""
    v = variation.strip().lower()
    if v.startswith("male"):
        return "M"
    if v.startswith("female"):
        return "F"
    if v in ("m", "man", "xy"):
        return "M"
    if v in ("f", "woman", "xx"):
        return "F"
    return None


def main():
    height_path = DATA_DIR / "height.rss"
    sex_path = DATA_DIR / "sex.rss"

    print("Parsing height data...")
    height_entries = parse_rss(height_path)
    print(f"  {len(height_entries)} entries total")

    print("Parsing sex data...")
    sex_entries = parse_rss(sex_path)
    print(f"  {len(sex_entries)} entries total")

    # Build sex lookup
    sex_lookup = {}
    for entry in sex_entries:
        sex = parse_sex(entry["variation"])
        if sex:
            sex_lookup[entry["genome_id"]] = sex

    print(f"  {len(sex_lookup)} with parseable sex (M/F)")

    # Parse heights and join with sex
    out_path = DATA_DIR / "phenotypes.tsv"
    parsed = 0
    unparsed_examples = []

    with open(out_path, "w") as f:
        f.write("genome_id\theight_cm\tsex\n")
        for entry in height_entries:
            height = parse_height_cm(entry["variation"])
            if height is not None:
                # Sanity check
                if height < 100 or height > 250:
                    continue
                sex = sex_lookup.get(entry["genome_id"], "")
                f.write(f"{entry['genome_id']}\t{height:.1f}\t{sex}\n")
                parsed += 1
            else:
                if len(unparsed_examples) < 20:
                    unparsed_examples.append(entry["variation"])

    print(f"\nParsed {parsed} height entries -> {out_path}")
    print(f"Could not parse {len(height_entries) - parsed} entries")
    if unparsed_examples:
        print(f"\nExamples of unparsed heights:")
        for ex in unparsed_examples:
            print(f"  '{ex}'")


if __name__ == "__main__":
    main()
