import re
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse

# Whitespace normalization
_WS_RE = re.compile(r"[ \t\u00A0\x0B\f]+")
_NL_RE = re.compile(r"\n{2,}")

# Junk hints (class/id substrings that often indicate non-article content)
_JUNK_HINTS = [
    "comment", "meta", "footer", "footnote", "sidebar", "subscribe", "signup",
    "advert", "ads", "cookie", "cookie-banner", "related", "promo", "social",
]

# Cutoff phrases that often signal the end of an article
_CUTOFF_PHRASES = [
    "related articles", "you might also like", "more from", "read next",
    "continue reading", "read more", "related stories",
]

# Boilerplate / noise phrases to filter out
_NOISE_PHRASES = [
    "follow us on", "sign up for", "subscribe to", "advertisement", "cookie policy",
]

# Headings that should trigger cutting off remaining blocks when encountered in plain text
_CUTOFF_HEADINGS = [
    "related articles", "more resources", "about the author", "you might also like",
    "further reading", "references",
]

# Bad line prefixes to drop in plaintext (copyright, legal boilerplate)
_BAD_LINE_PREFIXES = [
    "copyright", "©", "all rights reserved", "terms of service", "privacy policy",
]


def _collapse_ws(text: str) -> str:
    if not text:
        return ""
    s = _WS_RE.sub(" ", text)
    s = s.strip()
    s = _NL_RE.sub("\n\n", s)
    return s


def _looks_junk(tag) -> bool:
    if not tag or not getattr(tag, "attrs", None):
        return False
    cls = " ".join(tag.get("class", []) or [])
    idv = tag.get("id", "") or ""
    combined = f"{cls} {idv}".lower()
    for hint in _JUNK_HINTS:
        if hint in combined:
            return True
    return False


def _text_len(tag) -> int:
    if not tag:
        return 0
    text = tag.get_text(separator=" ") if hasattr(tag, "get_text") else str(tag)
    return len(text.strip())


def _densest_block(soup: BeautifulSoup):
    # Prefer <article>
    article = soup.find("article")
    if article and _text_len(article) > 100:
        return article

    # Score top-level containers
    candidates = []
    for tag in soup.find_all(["main", "div", "section", "article"], recursive=False):
        if _looks_junk(tag):
            continue
        score = _text_len(tag)
        candidates.append((score, tag))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # Fallback to body
    return soup.body or soup


def _is_heading_like(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if len(s) < 4 or len(s) > 200:
        return False
    # Heading-like if titlecase or all caps and short
    if s.isupper() and len(s.split()) < 6:
        return True
    if s.istitle() and len(s.split()) < 8:
        return True
    return False


def _dedupe_headings(lines):
    seen = set()
    out = []
    for ln in lines:
        key = ln.strip().lower()
        if not key:
            out.append(ln)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return out


def extract_article_content(html_bytes: bytes, url: str) -> dict:
    """Extracts a cleaned article body and metadata from HTML bytes.

    Returns: { 'title': str, 'url': str, 'site': str, 'body': str }
    """
    soup = BeautifulSoup(html_bytes, "html.parser")

    # Remove unwanted tags
    for sel in soup.find_all(["script", "style", "noscript", "template", "svg", "iframe", "meta", "header", "footer"]):
        sel.decompose()

    # Remove comments
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        c.extract()

    # Remove clearly junk containers
    for tag in list(soup.find_all(True)):
        if _looks_junk(tag):
            try:
                tag.decompose()
            except Exception:
                pass

    main = _densest_block(soup)

    # Within main, keep only semantic content
    # Convert lists to bullets
    for ul in main.find_all("ul"):
        items = []
        for li in ul.find_all("li"):
            text = _collapse_ws(li.get_text(" "))
            if text:
                items.append("- " + text)
        txt = "\n".join(items)
        new = soup.new_tag("p")
        new.string = txt
        ul.replace_with(new)

    # Unwrap anchors
    for a in main.find_all("a"):
        a.replace_with(a.get_text(" "))

    # Keep only headings and paragraphs; unwrap other layout tags
    for tag in list(main.find_all(True)):
        name = tag.name.lower()
        if name in ("h1", "h2", "h3", "h4", "p"):
            continue
        # replace br with newline
        if name == "br":
            tag.replace_with("\n")
            continue
        try:
            tag.unwrap()
        except Exception:
            pass

    # Linearize
    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p"]):
        text = _collapse_ws(el.get_text("\n"))
        if not text:
            continue
        if el.name.startswith("h"):
            lines.append(text)
            lines.append("")
        else:
            lines.append(text)

    # Deduplicate headings
    lines = _dedupe_headings(lines)

    body = "\n\n".join([l for l in lines if l is not None])
    body = _collapse_ws(body)

    # Title
    title_tag = soup.title.string if soup.title and soup.title.string else ""
    title = _collapse_ws(str(title_tag))
    # Clean common separators
    for sep in ["|", "-", "—"]:
        parts = [p.strip() for p in title.split(sep) if p.strip()]
        if len(parts) > 1:
            title = parts[0]
            break

    site = ""
    try:
        parsed = urlparse(url or "")
        site = parsed.netloc or ""
    except Exception:
        site = ""

    return {"title": title, "url": url, "site": site, "body": body}


def clean_plaintext_anysite(text: str) -> str:
    # Normalize line breaks
    if not text:
        return ""
    s = text.replace('\r\n', '\n').replace('\r', '\n')
    # Split into blocks on double newlines
    blocks = [b.strip() for b in re.split(r"\n{2,}", s) if b.strip()]
    out = []
    for b in blocks:
        low = b.lower()
        if any(h in low for h in _CUTOFF_HEADINGS):
            break
        if any(low.startswith(p) for p in _BAD_LINE_PREFIXES):
            continue
        if len(b) < 4:
            continue
        out.append(b)
    # Deduplicate
    seen = set()
    kept = []
    for b in out:
        key = b.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(b)
    joined = "\n\n".join(kept)
    return _collapse_ws(joined)
