import json

with open("data/processed/nvidia/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

count = 0

for chunk in chunks:
    if chunk["chunk_type"] == "table":
        text = chunk["text"]

        # Skip short cover-page tables
        if len(text.split()) < 30:
            continue

        print("\n" + "=" * 70)
        print("SOURCE:", chunk["source_file"])
        print("SECTION:", chunk["section"])
        print("=" * 70)
        print(text[:1500])

        count += 1
        if count == 5:
            break

if count == 0:
    print("No substantial table chunks found.")