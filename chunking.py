def chunk_paragraphs(structured_items):
    chunks = []
    current_heading = "Root"
    current_content = []

    for label, text in structured_items:
        if label == "heading":
            if current_content:
                chunks.append({
                    "heading": current_heading,
                    "content": " ".join(current_content).strip()
                })
                current_content = []
            current_heading = text
        else:
            if text:
                current_content.append(text)

    if current_content:
        chunks.append({
            "heading": current_heading,
            "content": " ".join(current_content).strip()
        })
    return chunks