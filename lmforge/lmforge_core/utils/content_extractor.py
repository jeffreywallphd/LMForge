"""
Content extractor utilities

Capabilities:
- Multi-block aggregation with junk/CTA/end-of-article detection
- Metadata extraction (author, published, publisher)
- Preserves code blocks with markdown-style fenced formatting (```)
- Handles nested lists (ordered/unordered) with indentation and numbering
"""

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
    # Newsletter/roundup promotional content patterns
    "newsletter", "roundup", "news-roundup", "weekly-", "daily-", "newsletter-digest", "news-digest", "email-briefing", "site-briefing",
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
    # If a cutoff-like heading is present, only treat as end-of-article
    # when the section has little to no substantive content.
    if any(h in heading_text for h in _CUTOFF_HEADINGS):
        try:
            content_len = len(tag.get_text(" ", strip=True))
        except Exception:
            content_len = 0
        return content_len < 50
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
        # Remove extracted time element to avoid duplicate date in body
        if publish_date and t:
            try:
                # Only decompose small, metadata-like <time> elements
                if len((t.get_text(" ", strip=True) or "")) <= 120:
                    t.decompose()
            except Exception:
                pass

    # byline patterns near H1
    if not author:
        byline = soup.find(class_=lambda v: v and "byline" in v.lower()) or soup.find(class_=lambda v: v and "author" in v.lower())
        if byline:
            cand = byline.get_text(" ", strip=True)
            if cand:
                author = cand
                # Remove the byline container if it appears to be a dedicated byline/author element
                try:
                    clsid = (" ".join(byline.get("class", []) or []) + " " + (byline.get("id", "") or "")).lower()
                    short_text = len(cand) <= 120
                    looks_byline = any(k in clsid for k in ["byline", "author", "article__byline"]) or cand.lower().startswith("by ")
                    if looks_byline and short_text:
                        byline.decompose()
                except Exception:
                    pass
        else:
            # scan a small portion of document for "By [Name]"
            text_head = " ".join([p.get_text(" ", strip=True) for p in soup.find_all(limit=20)])
            import re as _re
            m = _re.search(r"\b[Bb]y\s+([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,3})", text_head)
            if m:
                author = m.group(1)
                # Attempt to remove the containing element if it primarily consists of the byline text
                try:
                    candidates = soup.find_all(["p", "span", "div"], limit=50)
                    for el in candidates:
                        txt = (el.get_text(" ", strip=True) or "")
                        low = txt.lower()
                        if low.startswith("by ") and len(txt) <= (len(m.group(0)) + 20):
                            # Ensure it's not a long paragraph with additional content
                            if len(txt) <= 140:
                                el.decompose()
                                break
                except Exception:
                    pass

    return {"author": author, "publish_date": publish_date, "publisher": publisher}


def _is_code_heavy_page(soup: BeautifulSoup, url: str) -> bool:
    """Detect documentation/technical pages with high code density.

    Heuristic: ratio of <pre>/<code> tags to total blocks, plus URL hints.
    """
    try:
        code_tags = len(soup.find_all(["pre", "code"]))
        blocks = len(soup.find_all(["p", "section", "article", "div"]))
        ratio = code_tags / max(blocks, 1)
    except Exception:
        ratio = 0.0

    url_l = (url or "").lower()
    url_hints = any(h in url_l for h in [
        "docs.", "github.com", "stackoverflow.com", "wiki", "tutorial", "documentation"
    ])
    return ratio > 0.15 or url_hints


def _process_code_block(tag) -> str:
    """Extract and format a block-level code element as fenced markdown.

    Inline <code> elements should not be processed by this function.
    """
    try:
        # Preserve exact whitespace/indentation
        code = tag.get_text()
        # Trim outer blank lines only
        code = code.strip("\n")
        if len(code) < 10 and "\n" not in code:
            # Very short, likely inline-like — skip special wrapping
            return None
        return "\n```\n" + code + "\n```\n"
    except Exception:
        return None


def _process_list_recursive(list_tag, depth: int = 0) -> list:
    """Recursively process a <ul>/<ol> into formatted text lines with indentation.

    - Unordered lists: "- " prefix
    - Ordered lists: "1. ", "2. ", etc. (respects 'start' attribute when present)
    - Indentation: 2 spaces per nesting level
    """
    lines = []
    indent = "  " * depth
    ordered = list_tag.name.lower() == "ol"
    start = 1
    try:
        start = int(list_tag.get("start", 1))
    except Exception:
        start = 1
    counter = start

    for li in list_tag.find_all("li", recursive=False):
        # Gather direct text of this <li> excluding nested lists
        parts = []
        for child in li.contents:
            # Skip nested lists here; they'll be processed recursively below
            if getattr(child, "name", None) in ("ul", "ol"):
                continue
            try:
                if hasattr(child, "get_text"):
                    parts.append(child.get_text(" ", strip=True))
                else:
                    parts.append(str(child).strip())
            except Exception:
                continue
        text = _collapse_ws(" ".join([p for p in parts if p])) if parts else ""

        if ordered:
            prefix = f"{counter}. "
            counter += 1
        else:
            prefix = "- "

        if text:
            lines.append(f"{indent}{prefix}{text}")
        else:
            # Allow bare list items with only nested lists
            pass

        # Process nested lists directly under this <li>
        for sub in li.find_all(["ul", "ol"], recursive=False):
            lines.extend(_process_list_recursive(sub, depth + 1))

    return lines


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

    # Detect code-heavy pages (docs, tutorials) for adjusted formatting
    is_code_heavy = _is_code_heavy_page(soup, url)

    # Remove low-level unwanted tags (scripts/styles/meta/templates) but keep header/footer until later
    for sel in soup.find_all(["script", "style", "noscript", "template", "svg", "iframe", "meta"]):
        sel.decompose()

    # Remove comments
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        c.extract()

    # Remove clearly junk containers site-wide (guard against removing large substantive sections)
    for tag in list(soup.find_all(True)):
        if _looks_junk(tag):
            try:
                # If the tag is very substantive, skip removal to avoid over-filtering
                if _text_len(tag) > 800:
                    continue
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
        # Step 1: Process code blocks first (before lists)
        for code_tag in list(blk.find_all(["pre", "code"])):
            # Skip nested code tags to avoid double-processing
            if code_tag.find_parent(["pre", "code"]):
                continue
            # Skip inline <code> inside paragraphs
            if code_tag.name == "code" and code_tag.find_parent("p") is not None:
                continue
            formatted_code = _process_code_block(code_tag)
            if formatted_code:
                new_p = soup.new_tag("p")
                new_p["data-code-block"] = "true"
                new_p.string = formatted_code
                try:
                    code_tag.replace_with(new_p)
                except Exception:
                    pass

        # Step 2: Process lists with nesting support (replace top-level lists only)
        for list_tag in list(blk.find_all(["ul", "ol"])):
            if list_tag.find_parent(["ul", "ol"]):
                continue
            formatted_lines = _process_list_recursive(list_tag, depth=0)
            if formatted_lines:
                list_text = "\n".join(formatted_lines)
                new = soup.new_tag("p")
                new.string = list_text
                try:
                    list_tag.replace_with(new)
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
        promo_heading_keywords = ["roundup", "newsletter", "weekly news", "daily digest", "briefing"]
        skip_next_short_para = False
        for el in blk.find_all(["h1", "h2", "h3", "h4", "p"]):
            text = _collapse_ws(el.get_text("\n"))
            if not text:
                continue
            # Optionally skip immediate short paragraph after a promotional heading
            if skip_next_short_para and el.name == "p":
                if len(text) < 200:
                    skip_next_short_para = False
                    continue
                # paragraph is substantive; include and clear flag
                skip_next_short_para = False
            if el.name.startswith("h"):
                # Filter out promotional headings with minimal content
                if el.name in ("h2", "h3"):
                    low = text.lower()
                    if _is_heading_like(text) and any(k in low for k in promo_heading_keywords):
                        # look ahead: mark to skip the next short paragraph
                        skip_next_short_para = True
                        continue
                all_lines.append(text)
                all_lines.append("")
            else:
                # Paragraph: if marked as code, preserve formatting exactly
                if el.get("data-code-block") == "true":
                    code_text = el.get_text()
                    all_lines.append(code_text)
                else:
                    all_lines.append(text)

        # Add separator between blocks
        all_lines.append("")

    # Post-process lines
    all_lines = _dedupe_headings(all_lines)
    joined = "\n\n".join([l for l in all_lines if l is not None and l != ""])

    # Collapse whitespace outside fenced code blocks only
    def _collapse_preserving_code(s: str) -> str:
        parts = re.split(r"(```[\s\S]*?```)", s)
        out_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                out_parts.append(part)
            else:
                out_parts.append(_collapse_ws(part))
        return "".join(out_parts)

    body = _collapse_preserving_code(joined)

    # Phase D: prepend metadata (always include all fields with N/A fallback)
    meta_lines = []
    # Use full canonical URL for Source field instead of domain only
    source_display = url or ""
    # Always include three lines
    meta_lines.append(f"Author: {meta.get('author') or 'N/A'}")
    meta_lines.append(f"Published: {meta.get('publish_date') or 'N/A'}")
    meta_lines.append(f"Source: {source_display or 'N/A'}")

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

    # Preserve returned metadata fields; for 'publisher' field in result, keep domain fallback behavior
    publisher_value = meta.get("publisher") or (urlparse(url or "").netloc or "")

    result = {"title": title, "url": url, "site": site, "body": body}
    result.update({
        "author": meta.get("author"),
        "publish_date": meta.get("publish_date"),
        "publisher": publisher_value,
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
