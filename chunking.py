# chunking.py
from typing import List

def chunk_plain_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split plain text into overlapping chunks of roughly `chunk_size` words.
    Uses paragraphs as natural boundaries.
    """
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_word_count = 0
    for para in paragraphs:
        para_words = len(para.split())
        if current_word_count + para_words <= chunk_size:
            current_chunk.append(para)
            current_word_count += para_words
        else:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                # Overlap: keep last few paragraphs (approx overlap words)
                overlap_paras = []
                overlap_count = 0
                for p in reversed(current_chunk):
                    p_words = len(p.split())
                    if overlap_count + p_words <= overlap:
                        overlap_paras.insert(0, p)
                        overlap_count += p_words
                    else:
                        break
                current_chunk = overlap_paras + [para]
                current_word_count = overlap_count + para_words
            else:
                # If a single paragraph is too long, split by sentences.
                if para_words > chunk_size:
                    sentences = para.split('. ')
                    temp = []
                    temp_words = 0
                    for sent in sentences:
                        sent_words = len(sent.split())
                        if temp_words + sent_words <= chunk_size:
                            temp.append(sent)
                            temp_words += sent_words
                        else:
                            chunks.append('. '.join(temp) + '.')
                            temp = [sent]
                            temp_words = sent_words
                    if temp:
                        chunks.append('. '.join(temp) + '.')
                else:
                    current_chunk = [para]
                    current_word_count = para_words
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    return chunks