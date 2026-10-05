from collections import Counter

from src.ingestion import (
    DATA_PATH,
    build_children,
    build_parents,
    load_mineru,
)


data = load_mineru(DATA_PATH)
parents = build_parents(data)
children = build_children(parents)

sizes = [len(child.content.split()) for child in children]

print("Parents:", len(parents))
print("Children:", len(children))
print("Largest child:", max(sizes))
print("Average child:", round(sum(sizes) / len(sizes), 1))

for i, child in enumerate(children[:10], start=1):
    print("\n" + "=" * 80)
    print(f"CHILD {i}")
    print("chapter:", child.meta.get("chapter"))
    print("title:", child.meta.get("title"))
    print("words:", len(child.content.split()))
    print("parent:", child.meta.get("parent_id"))
    print()
    print(child.content)