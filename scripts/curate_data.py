"""
One-time script to curate ~200 documents from the Kaggle CompanyDocuments archive
into Data/Rag/ organized by category.
"""
import shutil
import random
from pathlib import Path

ARCHIVE_BASE = Path(__file__).resolve().parent.parent.parent.parent / "archive" / "CompanyDocuments"
TARGET_BASE = Path(__file__).resolve().parent.parent / "Data" / "Rag"

CATEGORIES = {
    "invoices": ARCHIVE_BASE / "invoices",
    "purchase_orders": ARCHIVE_BASE / "PurchaseOrders",
    "shipping_orders": ARCHIVE_BASE / "Shipping orders",
    "inventory_reports": ARCHIVE_BASE / "Inventory Report" / "monthly-Category" / "monthly-Category",
}

DOCS_PER_CATEGORY = 50


def curate():
    # Clean target
    if TARGET_BASE.exists():
        shutil.rmtree(TARGET_BASE)
    TARGET_BASE.mkdir(parents=True, exist_ok=True)

    total = 0
    for category_name, source_dir in CATEGORIES.items():
        target_dir = TARGET_BASE / category_name
        target_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            print(f"WARNING: Source not found: {source_dir}")
            continue

        files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
        selected = random.sample(files, min(DOCS_PER_CATEGORY, len(files)))

        for f in selected:
            shutil.copy2(f, target_dir / f.name)
            total += 1

        print(f"  {category_name}: copied {len(selected)} files")

    # Also copy the CSV for reference
    csv_source = ARCHIVE_BASE.parent / "company-document-text.csv"
    if csv_source.exists():
        shutil.copy2(csv_source, TARGET_BASE / "company-document-text.csv")
        print(f"  Copied company-document-text.csv")

    print(f"\nDone! Total: {total} documents in {TARGET_BASE}")


if __name__ == "__main__":
    random.seed(42)  # Reproducible selection
    curate()
