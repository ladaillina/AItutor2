from collections import Counter

from src.ingestion import (
    DATA_PATH,
    build_hierarchy,
    build_source_document,
    load_mineru,
)


data = load_mineru(DATA_PATH)
source = build_source_document(data)
documents = build_hierarchy(source)

levels = Counter(doc.meta["__level"] for doc in documents)

print("Total hierarchical documents:", len(documents))
print("Documents by level:", dict(levels))

leaves = [
    doc
    for doc in documents
    if not doc.meta["__children_ids"]
]

print("Leaf chunks:", len(leaves))

for i, doc in enumerate(leaves[:5], start=1):
    print("\n" + "=" * 80)
    print(f"CHILD {i}")
    print("page:", doc.meta.get("page_number"))
    print("parent:", doc.meta.get("__parent_id"))
    print()
    print(doc.content)