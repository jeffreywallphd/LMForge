import re
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse

# Whitespace normalization
_WS_RE = re.compile(r"[ \t\u00A0\x0B\f]+")
_NL_RE = re.compile(r"\n{2,}")

# Junk hints (class/id substrings that often indicate non-article content)
# Keep these as lowercase substrings; _looks_junk matches them against class/id text
_JUNK_HINTS = [
    # generic rails/boilerplate
    "comment", "footer", "footnote", "sidebar", "subscribe", "signup",
    "advert", "ads", "cookie", "cookie-banner", "related", "promo", "social",
    # IBM/Carbon patterns
    "bx--", "ibm-masthead", "ibm-footer",
    # navigation / header / menu / breadcrumbs
    "breadcrumb", "navigation", "nav-", "menu", "site-header", "site-footer",
    # social / newsletter / cta patterns
    "share-", "social-share", "follow-us", "newsletter-", "signup-", "cta-", "call-to-action",
    # common recommendation/comment vendors
    "outbrain", "taboola", "disqus", "livefyre", "comments-section",
]


# Mid-article CTA hints (separators, should be skipped but not cause cutoff)
_MID_ARTICLE_CTA_HINTS = [
    "promo", "cta", "call-to-action", "learn-more", "try-now", "get-started",
]

# End-of-article rail hints (stop collecting after these)
_END_ARTICLE_HINTS = [
    "related-articles", "more-stories", "read-next", "you-might-like",
    "recommended", "further-reading",
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
    # Weighted text length: value paragraphs, lists, blockquotes and captions
    if not tag:
        return 0
    # gather text from semantic nodes
    pieces = []
    for p in tag.find_all("p"):
        pieces.append(p.get_text(" ", strip=True))
    for li in tag.find_all("li"):
        pieces.append(li.get_text(" ", strip=True))
    for bq in tag.find_all("blockquote"):
        pieces.append(bq.get_text(" ", strip=True))
    for f in tag.find_all(["figcaption", "td", "dd"]):
        pieces.append(f.get_text(" ", strip=True))
    base_text = " ".join([p for p in pieces if p])
    base_len = len(base_text.strip())
    if base_len == 0:
        raw = tag.get_text(" ", strip=True) if hasattr(tag, "get_text") else str(tag)
        return len(raw.strip())
    # apply modest additive bonuses for lists and blockquotes (avoid runaway by not scaling by base_len)
    li_count = len(tag.find_all("li"))
    bq_count = len(tag.find_all("blockquote"))
    # tuned additive constants — lists get smaller boost, blockquotes larger
    bonus = li_count * 40 + bq_count * 80
    # small heading presence boost
    h_count = len(tag.find_all(re.compile(r'^h[1-6]$')))
    bonus += h_count * 50
    return base_len + bonus


def _densest_block(soup: BeautifulSoup):
    # Prefer article/main/[role='main'] with threshold
    for sel in ("article", "main", "[role='main']"):
        node = soup.select_one(sel)
        if node and _text_len(node) > 200:
            return node

    # Scan recursively for candidate containers and score them by paragraph/list/heading content
    cands = soup.find_all(["section", "div", "article", "main"])
    if not cands:
        return soup.body or soup

    best, score = None, -1
    for c in cands:
        if _looks_junk(c):
            continue
        # base content length
        ptext = _text_len(c)
        # heading bonuses
        h2 = len(c.find_all("h2"))
        h3 = len(c.find_all("h3"))
        hbonus = h2 * 300 + h3 * 150
        # list bonuses
        list_bonus = 0
        for lst in c.find_all(["ul", "ol"]):
            items = lst.find_all("li")
            if len(items) >= 3:
                list_bonus += 200
        # semantic richness
        rich = 0
        if c.find("blockquote") or c.find("figure") or c.find("table"):
            rich += 100

        sc = ptext + hbonus + list_bonus + rich
        if sc > score:
            best, score = c, sc
    return best or soup


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


def _is_mid_article_cta(tag) -> bool:
    if not tag or not getattr(tag, "attrs", None):
        return False
    cls = " ".join(tag.get("class", []) or []) + " " + (tag.get("id", "") or "")
    low = cls.lower()
    if any(h in low for h in _MID_ARTICLE_CTA_HINTS):
        return True
    txt = (tag.get_text(" ", strip=True) or "").lower()
    if len(txt) < 300 and any(phrase in txt for phrase in ["learn more", "try now", "sign up", "get started"]):
        return True
    return False


def _is_end_of_article(tag) -> bool:
    if not tag:
        return False
    cls = " ".join(tag.get("class", []) or []) + " " + (tag.get("id", "") or "")
    low = cls.lower()
    if any(h in low for h in _END_ARTICLE_HINTS):
        return True
    heading_text = (tag.get_text(" ", strip=True) or "").lower()
    if any(h in heading_text for h in _CUTOFF_HEADINGS):
        return True
    return False


def _extract_metadata(soup: BeautifulSoup) -> dict:
    author = None
    publish_date = None
    publisher = None

    # meta tags
    meta_author = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", attrs={"property": "article:author"})
    if meta_author and meta_author.get("content"):
        author = meta_author.get("content").strip()

    meta_pub = soup.find("meta", attrs={"property": "article:published_time"})
    if meta_pub and meta_pub.get("content"):
        publish_date = meta_pub.get("content").strip()

    meta_site = soup.find("meta", attrs={"property": "og:site_name"})
    if meta_site and meta_site.get("content"):
        publisher = meta_site.get("content").strip()

    # <time> tag
    if not publish_date:
        t = soup.find("time")
        if t and t.get("datetime"):
            publish_date = t.get("datetime").strip()
        elif t:
            publish_date = t.get_text(" ", strip=True)

    # byline patterns near H1
    if not author:
        byline = soup.find(class_=lambda v: v and "byline" in v.lower()) or soup.find(class_=lambda v: v and "author" in v.lower())
        if byline:
            cand = byline.get_text(" ", strip=True)
            if cand:
                author = cand
        else:
            # scan a small portion of document for "By [Name]"
            text_head = " ".join([p.get_text(" ", strip=True) for p in soup.find_all(limit=20)])
            import re as _re
            m = _re.search(r"\b[Bb]y\s+([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,3})", text_head)
            if m:
                author = m.group(1)

    return {"author": author, "publish_date": publish_date, "publisher": publisher}


def _gather_content_blocks(soup: BeautifulSoup, main_container) -> list:
    if main_container is None:
        main_container = soup.body or soup

    blocks = []
    # If main contains multiple section/div children that look substantive, include them
    children = [c for c in main_container.find_all(recursive=False)]
    if children:
        for ch in children:
            if _looks_junk(ch):
                continue
            if _is_end_of_article(ch):
                break
            if _is_mid_article_cta(ch):
                continue
            if _text_len(ch) > 200 or ch.find(["h2", "h3"]) or ch.find(["ul", "ol"]):
                blocks.append(ch)
        if blocks:
            return blocks

    # Otherwise, include the main container and its meaningful sibling containers
    base = main_container
    blocks.append(base)
    parent = base.parent
    if parent:
        siblings = list(parent.find_all(recursive=False))
        try:
            idx = siblings.index(base)
        except ValueError:
            idx = None
        if idx is not None:
            for sib in siblings[idx + 1 :]:
                if _looks_junk(sib):
                    continue
                if _is_end_of_article(sib):
                    break
                if _is_mid_article_cta(sib):
                    continue
                if _text_len(sib) > 200 or sib.find(["h2", "h3"]) or sib.find(["ul", "ol"]):
                    blocks.append(sib)

    return blocks


def extract_article_content(html_bytes: bytes, url: str) -> dict:
    """Extracts a cleaned article body and metadata from HTML bytes.

    Returns: { 'title': str, 'url': str, 'site': str, 'body': str }
    """
    soup = BeautifulSoup(html_bytes, "html.parser")

    # Phase A: metadata extraction early (before DOM changes)
    meta = _extract_metadata(soup)

    # Remove low-level unwanted tags (scripts/styles/meta/templates) but keep header/footer until later
    for sel in soup.find_all(["script", "style", "noscript", "template", "svg", "iframe", "meta"]):
        sel.decompose()

    # Remove comments
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        c.extract()

    # Remove clearly junk containers site-wide
    for tag in list(soup.find_all(True)):
        if _looks_junk(tag):
            try:
                tag.decompose()
            except Exception:
                pass

    # Phase B: identify primary container(s)
    main = _densest_block(soup)

    # Handle header/footer carefully: unwrap inside main, decompose outside
    for hf in list(soup.find_all(["header", "footer"])):
        try:
            if main and hf in getattr(main, "descendants", []):
                hf.unwrap()
            else:
                hf.decompose()
        except Exception:
            try:
                hf.decompose()
            except Exception:
                pass

    # Gather multiple blocks when present
    blocks = _gather_content_blocks(soup, main)

    # Phase C: process each block and linearize
    all_lines = []
    for blk in blocks:
        # Convert lists to bullets
        for ul in list(blk.find_all(["ul", "ol"])):
            items = []
            for li in ul.find_all("li"):
                text = _collapse_ws(li.get_text(" "))
                if text:
                    items.append("- " + text)
            txt = "\n".join(items)
            new = soup.new_tag("p")
            new.string = txt
            try:
                ul.replace_with(new)
            except Exception:
                pass

        # Unwrap anchors
        for a in list(blk.find_all("a")):
            try:
                a.replace_with(a.get_text(" "))
            except Exception:
                pass

        # Keep only meaningful tags, unwrap others
        for tag in list(blk.find_all(True)):
            name = tag.name.lower()
            if name in ("h1", "h2", "h3", "h4", "p"):
                continue
            if name == "br":
                tag.replace_with("\n")
                continue
            try:
                tag.unwrap()
            except Exception:
                pass

        # Linearize block
        for el in blk.find_all(["h1", "h2", "h3", "h4", "p"]):
            text = _collapse_ws(el.get_text("\n"))
            if not text:
                continue
            if el.name.startswith("h"):
                all_lines.append(text)
                all_lines.append("")
            else:
                all_lines.append(text)

        # Add separator between blocks
        all_lines.append("")

    # Post-process lines
    all_lines = _dedupe_headings(all_lines)
    body = _collapse_ws("\n\n".join([l for l in all_lines if l is not None and l != ""]))

    # Phase D: prepend metadata if present
    meta_lines = []
    if meta.get("author"):
        meta_lines.append(f"Author: {meta.get('author')}")
    if meta.get("publish_date"):
        meta_lines.append(f"Published: {meta.get('publish_date')}")
    publisher = meta.get("publisher")
    if not publisher:
        try:
            parsed = urlparse(url or "")
            publisher = parsed.netloc or ""
        except Exception:
            publisher = ""
    if publisher:
        meta_lines.append(f"Source: {publisher}")

    if meta_lines:
        body = _collapse_ws("\n".join(meta_lines) + "\n\n" + body)

    # Title
    title = ""
    if soup.title:
        try:
            title = soup.title.get_text(" ", strip=True)
        except Exception:
            title = soup.title.string if soup.title and soup.title.string else ""
    title = _collapse_ws(str(title))
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

    result = {"title": title, "url": url, "site": site, "body": body}
    result.update({
        "author": meta.get("author"),
        "publish_date": meta.get("publish_date"),
        "publisher": meta.get("publisher") or publisher,
    })

    return result


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
