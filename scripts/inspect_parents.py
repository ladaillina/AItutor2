from src.ingestion import DATA_PATH, build_parents, load_mineru


data = load_mineru(DATA_PATH)
parents = build_parents(data)

print("Parents:", len(parents))

for i, parent in enumerate(parents[:15], start=1):
    print("\n" + "=" * 80)
    print(f"PARENT {i}")
    print("title:", parent.meta["title"])
    print(
        "pages:",
        parent.meta["page_start"],
        "-",
        parent.meta["page_end"],
    )
    print("words:", len(parent.content.split()))
    print()
    print(parent.content[:700])