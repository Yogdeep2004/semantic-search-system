import re
import string

def clean_text(text: str) -> str:
    """
    Cleans email text by removing headers, quoted replies, signatures,
    and normalizing whitespace.
    """
    # 1. Remove Email headers (everything before the first blank line)
    parts = re.split(r'\n\s*\n', text, 1)
    if len(parts) > 1:
        text = parts[1]
    
    # 2. Remove Quoted replies (lines starting with >)
    text = re.sub(r'^[>].*', '', text, flags=re.MULTILINE)
    
    # 3. Remove Signatures (blocks starting with -- until end)
    text = re.split(r'^--\s*$', text, flags=re.MULTILINE)[0]
    
    # 4. Additional cleaning
    text = text.lower()
    # Replace multiple whitespaces/newlines with single space
    text = re.sub(r'[\s]+', ' ', text)
    # Remove special characters or keep only alphanumeric? Let's keep common punctuation for now.
    # But remove non-printable characters if any.
    text = "".join(filter(lambda x: x in string.printable, text))
    
    return text.strip()

def is_valid_doc(text: str, min_words: int = 30) -> bool:
    """Checks if a document has at least min_words."""
    return len(text.split()) >= min_words
