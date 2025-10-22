# LMForge/lmforge/lmforge_core/utils/bs4_cleaner.py
import re
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse

_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")

# Common junk containers to drop by id/class hints
_JUNK_HINTS = [
    "cookie", "gdpr", "consent", "banner", "popover", "modal", "subscribe",
    "advert", "ad-", "ads-", "promo", "promo-", "footer", "nav", "breadcrumbs",
    "newsletter", "signup", "social", "share", "overlay", "paywall", "sidebar",
    "related", "recommended", "outbrain", "taboola", "toc", "table-of-contents",
    "comments", "comment", "discuss", "survey"
]

def _looks_junky(tag) -> bool:
    attr_text = " ".join([tag.get("id", "") or "", " ".join(tag.get("class", []))]).lower()
    return any(h in attr_text for h in _JUNK_HINTS)

def _text_len(tag) -> int:
    return len(tag.get_text(" ", strip=True))

def _densest_block(soup: BeautifulSoup):
    # Prefer article/main/role=main; else best-scoring section/div
    for selector in ("article", "main", "[role='main']"):
        node = soup.select_one(selector)
        if node and _text_len(node) > 200:
            return node
    candidates = soup.find_all(["section", "div", "article", "main"])
    if not candidates:
        return soup.body or soup
    best, best_score = None, -1
    for c in candidates:
        p_text = sum(_text_len(p) for p in c.find_all("p"))
        h_bonus = len(c.find_all(["h1","h2","h3"])) * 200
        score = p_text + h_bonus
        if score > best_score:
            best, best_score = c, score
    return best or soup

def _collapse_ws(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()

def _dedupe_headings(lines):
    seen = set(); out = []
    for line in lines:
        L = line.strip()
        if not L:
            out.append(L); continue
        key = L.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(L)
    return out

def clean_html_bytes(html_bytes: bytes, url: str) -> dict:
    """
    Return {title, url, site, body} cleaned for LLMs.
    """
    soup = BeautifulSoup(html_bytes, "html.parser")

    # Remove noise tags
    for t in soup(["script","style","noscript","template","svg","iframe"]):
        t.decompose()
    # Strip comments
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    # Kill junky containers
    for t in soup.find_all(True):
        if _looks_junky(t):
            t.decompose()

    main = _densest_block(soup)
    main = main.clone() if hasattr(main, "clone") else main

    # Drop junk again within main
    for t in main.find_all(True):
        if _looks_junky(t):
            t.decompose()

    # Convert <ul><li> to bullets
    for ul in main.find_all("ul"):
        items = []
        for li in ul.find_all("li", recursive=False):
            txt = li.get_text(" ", strip=True)
            if txt:
                items.append(f"- {txt}")
        if items:
            ul.replace_with("\n".join(items))

    # Unwrap anchors, keep text
    for a in main.find_all("a"):
        a.replace_with(a.get_text(" ", strip=True))

    # Keep headings + paragraphs; unwrap layout divs
    allowed = {"h1","h2","h3","h4","p"}
    for t in list(main.find_all(True)):
        if t.name not in allowed:
            t.unwrap()

    # Rebuild lines
    lines = []
    for node in main.children:
        name = getattr(node, "name", None)
        if name in {"h1","h2","h3","h4"}:
            lines.append(node.get_text(" ", strip=True))
            lines.append("")
        elif name == "p":
            txt = node.get_text(" ", strip=True)
            if txt:
                lines.append(txt)
        elif isinstance(node, str):
            txt = node.strip()
            if txt:
                lines.append(txt)

    lines = [_collapse_ws(x) for x in lines]
    lines = _dedupe_headings(lines)
    body = _collapse_ws("\n".join(lines))

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    # Drop suffixes like " | IBM"
    title = re.sub(r"\s+\|\s+.*$", "", title)
    site = urlparse(url).netloc

    return {"title": title or "", "url": url, "site": site, "body": body}
