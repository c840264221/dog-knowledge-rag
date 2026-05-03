def filter_by_tags(docs, required_tags):
    results = []

    for doc in docs:
        tags = doc.metadata.get("tags", [])

        if any(tag in tags for tag in required_tags):
            results.append(doc)

    return results