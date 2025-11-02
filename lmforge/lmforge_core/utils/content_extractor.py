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
    # Existing entries
    "related articles", "more resources", "about the author", "you might also like",
    "further reading", "references",
    # Single-word promotional headings
    "resources", "explore", "discover", "learn", "share", "subscribe", "follow", "connect",
    # Multi-word promotional headings
    "learn more", "explore more", "related content", "additional resources", "see also",
    "next steps", "recommended reading", "popular articles", "trending now",
    "what to read next", "continue exploring", "dive deeper", "keep reading",
    "more on this topic", "related topics", "similar articles", "you may also like",
    "recommended for you", "editor's picks", "featured content", "latest updates",
    "stay informed", "join the conversation", "get in touch", "contact us",
    "about us", "our team",
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
            content_text = tag.get_text(" ", strip=True)
            content_len = len(content_text)
            
            # Check for high link-to-text ratio (indicates promotional link list)
            links = tag.find_all('a') if hasattr(tag, 'find_all') else []
            link_text_len = sum(len((link.get_text(" ", strip=True) or "")) for link in links)
            if content_len > 0 and link_text_len / content_len > 0.7:
                return True
            
            # Check for common CTA phrases
            content_lower = content_text.lower()
            cta_phrases = ["click here", "read more", "sign up now", "learn more", "get started"]
            if any(phrase in content_lower for phrase in cta_phrases):
                return True
                
        except Exception:
            content_len = 0
        return content_len < 100
    return False


def _is_trailing_empty_heading(heading_text: str, remaining_elements: list) -> bool:
    """Check if a heading is a trailing promotional heading with no substantive content following it.
    
    Args:
        heading_text: The text content of the heading being evaluated
        remaining_elements: List of subsequent elements (tags) after this heading
        
    Returns:
        True if the heading appears to be a trailing promotional heading with no substantive content
    """
    if not heading_text:
        return False
    
    # Normalize heading text
    normalized = heading_text.strip().lower()
    
    # Check if heading matches cutoff patterns
    matches_cutoff = any(cutoff in normalized for cutoff in _CUTOFF_HEADINGS)
    if not matches_cutoff:
        return False
    
    # If no remaining elements, this is a trailing heading
    if not remaining_elements:
        return True
    
    # Analyze remaining content after the heading
    # Single-word headings require more substantive content
    is_single_word = len(normalized.split()) == 1
    threshold = 150 if is_single_word else 100
    
    # Look ahead at next 3-5 elements
    lookahead_count = min(5, len(remaining_elements))
    for i in range(lookahead_count):
        el = remaining_elements[i]
        
        # Skip other headings
        if hasattr(el, 'name') and el.name and el.name.startswith('h'):
            continue
        
        # Check paragraph/div elements for substantive text
        if hasattr(el, 'get_text'):
            text = el.get_text(" ", strip=True)
            # Ignore very short content
            if len(text) < 20:
                continue
            # Found substantive content
            if len(text) > threshold:
                return False
    
    # No substantive content found in lookahead
    return True



def _extract_metadata(soup: BeautifulSoup, url: str = None) -> dict:
    author = None
    publish_date = None
    publisher = None
    version = None
    last_updated = None

    # Extract version from URL
    if url:
        import re as _re
        # Pattern 1: /locale/{version}/ or /docs/{version}/ with end boundary (e.g., Django /en/5.2/)
        match = _re.search(r'/(?:[a-z]{2}(?:-[a-z]{2})?|docs)/([0-9]+\.[0-9]+(?:\.[0-9]+)?)(?:/|$)', url)
        if not match:
            # Pattern 2: /{major}.{minor}/ or /{major}.{minor}.{patch}/ with end boundary (e.g., Python /3.12/)
            match = _re.search(r'/([0-9]+\.[0-9]+(?:\.[0-9]+)?)(?:/|$)', url)
        if not match:
            # Pattern 3: /v{version}/ with end boundary (e.g., /v2.5/)
            match = _re.search(r'/v([0-9]+\.[0-9]+(?:\.[0-9]+)?)(?:/|$)', url)
        if not match:
            # Pattern 4: /version/{version}/ or /release/{version}/ with end boundary
            match = _re.search(r'/(?:version|release)/([0-9]+\.[0-9]+(?:\.[0-9]+)?)(?:/|$)', url)
        if not match:
            # Pattern 5: major-only versions with doc path hints (e.g., /v3/ or /3/)
            if _re.search(r'/(?:doc|api|ref|guide|manual)', url, _re.IGNORECASE):
                match = _re.search(r'/v([0-9]+)(?:/|$)', url)
                if not match:
                    match = _re.search(r'/([0-9]+)(?:/|$)', url)
        if match:
            version = match.group(1)

    # Extract last updated date
    # Check meta tags first (highest priority)
    meta_modified = soup.find("meta", attrs={"property": "article:modified_time"}) or soup.find("meta", attrs={"name": "last-modified"})
    if meta_modified and meta_modified.get("content"):
        last_updated = meta_modified.get("content").strip()
    
    # Additional meta tag fallbacks
    if not last_updated:
        meta_updated = soup.find("meta", attrs={"property": "og:updated_time"})
        if meta_updated and meta_updated.get("content"):
            last_updated = meta_updated.get("content").strip()
    
    if not last_updated:
        meta_revised = soup.find("meta", attrs={"name": "revised"})
        if meta_revised and meta_revised.get("content"):
            last_updated = meta_revised.get("content").strip()
    
    if not last_updated:
        # Check for itemprop='dateModified' on meta or time tags
        itemprop_modified = soup.find("meta", attrs={"itemprop": "dateModified"}) or soup.find("time", attrs={"itemprop": "dateModified"})
        if itemprop_modified:
            if itemprop_modified.get("content"):
                last_updated = itemprop_modified.get("content").strip()
            elif itemprop_modified.get("datetime"):
                last_updated = itemprop_modified.get("datetime").strip()
    
    # Search for dedicated "last updated" elements if meta tags not found
    if not last_updated:
        # Look for elements with "last-updated" or "last-modified" in class/id
        def class_has_last_updated(v):
            if not v:
                return False
            # Handle both list/tuple and string types
            if isinstance(v, (list, tuple)):
                normalized = ' '.join(v).lower()
            else:
                normalized = v.lower()
            return 'last-updated' in normalized or 'last-modified' in normalized
        
        last_updated_el = soup.find(class_=class_has_last_updated)
        if not last_updated_el:
            # Search for elements containing "last updated" or "last modified" text
            import re as _re
            for tag in soup.find_all(['footer', 'div', 'span', 'p', 'time']):
                text = tag.get_text(" ", strip=True)
                if _re.search(r'\b(?:last\s+(?:updated|modified)|updated\s+on|modified\s+on)\b', text, _re.IGNORECASE):
                    last_updated_el = tag
                    break
        
        if last_updated_el:
            # Prefer datetime attribute for <time> tags
            if last_updated_el.name == 'time' and last_updated_el.get('datetime'):
                last_updated = last_updated_el.get('datetime').strip()
            else:
                # Extract text and clean it
                text = last_updated_el.get_text(" ", strip=True)
                # Remove common prefixes
                import re as _re
                text = _re.sub(r'^(?:last\s+(?:updated|modified)|updated\s+on|modified\s+on)[\s:]*', '', text, flags=_re.IGNORECASE)
                last_updated = text.strip() if text else None

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

    return {"author": author, "publish_date": publish_date, "publisher": publisher, "version": version, "last_updated": last_updated}


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


def _detect_code_language(tag, code_text: str) -> str:
    """Detect programming language from tag classes or code content patterns.
    
    Returns normalized language name (e.g., 'python', 'bash', 'javascript') or empty string.
    """
    # Language normalization mapping
    lang_map = {
        'sh': 'bash', 'shell': 'bash', 'console': 'bash', 'terminal': 'bash',
        'ps1': 'powershell', 'ps': 'powershell', 'pwsh': 'powershell',
        'js': 'javascript', 'node': 'javascript',
        'ts': 'typescript',
        'py': 'python',
        'c++': 'cpp',
        'c#': 'csharp',
        'objective-c': 'objectivec',
        'f#': 'fsharp',
        'vb.net': 'vbnet',
    }
    
    # Known standalone language names
    known_langs = {
        'python', 'javascript', 'bash', 'java', 'cpp', 'ruby', 'go', 'rust',
        'typescript', 'sql', 'yaml', 'json', 'xml', 'html', 'css', 'php',
        'swift', 'kotlin', 'csharp', 'powershell', 'perl', 'r', 'scala',
        'dart', 'elixir', 'haskell', 'lua', 'matlab', 'objectivec', 'groovy',
        'fsharp', 'vbnet'
    }
    
    try:
        # Check tag classes first
        classes = tag.get('class', []) if hasattr(tag, 'get') else []
        for cls in classes:
            cls_lower = cls.lower()
            # Pattern: language-*, highlight-*, hljs-*, lang-*, brush:*
            for prefix in ['language-', 'highlight-', 'hljs-', 'lang-', 'brush:']:
                if cls_lower.startswith(prefix):
                    lang = cls_lower[len(prefix):].strip()
                    normalized = lang_map.get(lang, lang)
                    if normalized in known_langs or lang in known_langs:
                        return normalized if normalized in known_langs else lang
            # Direct language name as class
            if cls_lower in known_langs or cls_lower in lang_map:
                return lang_map.get(cls_lower, cls_lower)
        
        # Check parent <pre> if tag is <code>
        if tag.name == 'code':
            parent = tag.find_parent('pre')
            if parent:
                parent_classes = parent.get('class', [])
                for cls in parent_classes:
                    cls_lower = cls.lower()
                    for prefix in ['language-', 'highlight-', 'hljs-', 'lang-', 'brush:']:
                        if cls_lower.startswith(prefix):
                            lang = cls_lower[len(prefix):].strip()
                            normalized = lang_map.get(lang, lang)
                            if normalized in known_langs or lang in known_langs:
                                return normalized if normalized in known_langs else lang
                    if cls_lower in known_langs or cls_lower in lang_map:
                        return lang_map.get(cls_lower, cls_lower)
        
        # Analyze code content for shell prompts
        lines = [l.strip() for l in code_text.split('\n') if l.strip()]
        if len(lines) >= 2:
            dollar_count = sum(1 for l in lines[:10] if l.startswith('$'))
            # PowerShell: require PS> or C:\> patterns, not bare >
            powershell_count = sum(1 for l in lines[:10] if l.startswith('PS>') or l.startswith('C:\\>'))
            check_count = min(len(lines[:10]), 10)
            
            if check_count > 0:
                if dollar_count / check_count > 0.5:
                    return 'bash'
                if powershell_count / check_count > 0.5:
                    return 'powershell'
    except Exception:
        pass
    
    return ''


def _process_code_block(tag) -> str:
    """Extract and format a block-level code element as fenced markdown.

    Fences only <pre> or <code> with newlines; skips single-line inline code.
    Uses sentinels with language hints to generate language-aware fenced code blocks.
    """
    try:
        # Preserve exact whitespace/indentation
        code = tag.get_text()
        # Trim outer blank lines only
        code = code.strip("\n")
        # Fence only <pre> or multi-line <code>
        if tag.name == "pre" or "\n" in code:
            # Detect language from classes or content
            lang = _detect_code_language(tag, code)
            # Use sentinels with language embedded: __FENCE_START__{lang}__
            return f"\n__FENCE_START__{lang}__\n{code}\n__FENCE_END__\n"
        # Single-line <code> is inline; skip fencing
        return None
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
            # Respect per-item <li value> override to match HTML semantics
            value_override = li.get("value")
            if value_override:
                try:
                    counter = int(value_override)
                except Exception:
                    pass
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


def _strip_trailing_promotional_headings(lines: list) -> list:
    """Remove trailing promotional headings from the end of the content.
    
    Args:
        lines: List of text lines from the article
        
    Returns:
        Cleaned list with trailing promotional headings removed
    """
    if not lines:
        return lines
    
    # Only process the last 20-30 lines for efficiency
    process_count = min(30, len(lines))
    cutoff_index = len(lines) - process_count
    
    # Don't remove headings from the first 30% of the document
    min_keep_index = int(len(lines) * 0.3)
    
    # Iterate backwards to find trailing promotional content
    removal_start_index = len(lines)
    
    for i in range(len(lines) - 1, max(cutoff_index - 1, min_keep_index - 1), -1):
        line = lines[i].strip()
        
        # Empty line - continue searching backwards
        if not line:
            continue
        
        # Check if line matches cutoff heading
        line_lower = line.lower()
        is_cutoff_heading = any(cutoff in line_lower for cutoff in _CUTOFF_HEADINGS)
        
        if is_cutoff_heading:
            # Check if this is followed by substantive content
            has_substantive_content = False
            
            # Look at next 5-10 lines after this heading
            for j in range(i + 1, min(i + 11, len(lines))):
                next_line = lines[j].strip()
                if len(next_line) > 100:
                    has_substantive_content = True
                    break
                # Check for reference/citation patterns (URLs, formatted citations)
                if any(pattern in next_line for pattern in ['http://', 'https://', 'doi:', 'isbn:']):
                    has_substantive_content = True
                    break
            
            if not has_substantive_content:
                # Mark for removal
                removal_start_index = i
            else:
                # Found substantive content, stop removing
                break
        else:
            # Non-empty, non-heading line with substantive content
            if len(line) > 50 and not any(cutoff in line_lower for cutoff in _CUTOFF_HEADINGS):
                # Stop - this is real content
                break
    
    # Return cleaned list
    if removal_start_index < len(lines):
        return lines[:removal_start_index]
    return lines


def extract_article_content(html_bytes: bytes, url: str) -> dict:
    """Extracts a cleaned article body and metadata from HTML bytes.

    Returns: { 'title': str, 'url': str, 'site': str, 'body': str }
    """
    soup = BeautifulSoup(html_bytes, "html.parser")

    # Phase A: metadata extraction early (before DOM changes)
    meta = _extract_metadata(soup, url)

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
                # Relax threshold for code-heavy pages to preserve nav/sidebar structure
                threshold = 400 if is_code_heavy else 800
                if _text_len(tag) > threshold:
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
        # Step 0: For code-heavy pages, preserve blockquote/figure/table structure before unwrapping
        if is_code_heavy:
            # Convert blockquotes to prefixed paragraphs
            for bq in list(blk.find_all("blockquote")):
                try:
                    bq_text = bq.get_text("\n", strip=True)
                    lines = [l.strip() for l in bq_text.split("\n") if l.strip()]
                    quoted = "\n".join([f"> {line}" for line in lines])
                    new_p = soup.new_tag("p")
                    new_p.string = quoted
                    bq.replace_with(new_p)
                except Exception:
                    pass
            
            # Convert figures/figcaptions to labeled paragraphs
            for fig in list(blk.find_all("figure")):
                try:
                    caption = fig.find("figcaption")
                    if caption:
                        cap_text = caption.get_text(" ", strip=True)
                        if cap_text:
                            new_p = soup.new_tag("p")
                            new_p.string = f"Figure: {cap_text}"
                            fig.replace_with(new_p)
                        else:
                            fig.decompose()
                    else:
                        fig.decompose()
                except Exception:
                    pass
            
            # Convert tables to plaintext grid
            for table in list(blk.find_all("table")):
                try:
                    rows = []
                    header_row = None
                    for tr in table.find_all("tr"):
                        cells = []
                        for cell in tr.find_all(["th", "td"]):
                            cells.append(cell.get_text(" ", strip=True))
                        if cells:
                            row_text = " | ".join(cells)
                            if tr.find("th") and header_row is None:
                                header_row = row_text
                                rows.append(header_row)
                                rows.append("-" * len(header_row))
                            else:
                                rows.append(row_text)
                    if rows:
                        table_text = "\n".join(rows)
                        new_p = soup.new_tag("p")
                        new_p.string = table_text
                        table.replace_with(new_p)
                    else:
                        table.decompose()
                except Exception:
                    pass
        
        # Step 1: Process code blocks first (before lists)
        for code_tag in list(blk.find_all(["pre", "code"])):
            # Skip nested code tags to avoid double-processing
            if code_tag.find_parent(["pre", "code"]):
                continue
            # Skip inline <code> inside paragraphs
            if code_tag.name == "code" and code_tag.find_parent("p") is not None:
                continue
            # Skip single-line <code> (not in <pre>) — only fence multi-line code
            if code_tag.name == "code":
                code_text = code_tag.get_text()
                if "\n" not in code_text:
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
        promo_heading_keywords = [
            "roundup", "newsletter", "weekly news", "daily digest", "briefing",
            "news roundup", "weekly roundup", "daily roundup", "monthly digest",
            "email digest", "subscribe", "sign up", "join us", "stay updated",
            "get notified", "latest updates", "breaking news", "top stories",
            "this week", "this month"
        ]
        skip_next_short_para = False
        
        # Convert to indexed iteration for lookahead capability
        elements = blk.find_all(["h1", "h2", "h3", "h4", "p"])
        
        for idx, el in enumerate(elements):
            text = _collapse_ws(el.get_text("\n"))
            if not text:
                continue
            # Remove pilcrow (¶) anchor symbols from headings used in documentation sites
            if el.name.startswith("h"):
                text = text.replace('¶', '').strip()
            # Optionally skip immediate short paragraph after a promotional heading
            if skip_next_short_para and el.name == "p":
                if len(text) < 200:
                    skip_next_short_para = False
                    continue
                # paragraph is substantive; include and clear flag
                skip_next_short_para = False
            if el.name.startswith("h"):
                # Check if this is a trailing empty heading (NEW - before promotional filter)
                remaining_elements = elements[idx+1:] if idx+1 < len(elements) else []
                if _is_trailing_empty_heading(text, remaining_elements):
                    continue
                
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
    all_lines = _strip_trailing_promotional_headings(all_lines)
    joined = "\n\n".join([l for l in all_lines if l is not None and l != ""])

    # Collapse whitespace outside fenced code blocks only (using sentinels to protect code)
    def _collapse_preserving_code(s: str) -> str:
        """Split on sentinel markers, extract language, collapse non-code parts, inject language into fences."""
        parts = re.split(r"(__FENCE_START__.*?__[\s\S]*?__FENCE_END__)", s)
        out_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                # Code block — extract language from sentinel and generate fence
                try:
                    # Pattern: __FENCE_START__{lang}__\ncode\n__FENCE_END__
                    start_match = re.match(r"__FENCE_START__(.*?)__", part)
                    if start_match:
                        lang = start_match.group(1)
                        # Remove sentinel markers
                        code_body = part.replace(start_match.group(0), "").replace("__FENCE_END__", "")
                        # Generate fence with language hint
                        fence_open = f"```{lang}" if lang else "```"
                        code_content = f"{fence_open}{code_body}```"
                        out_parts.append(code_content)
                    else:
                        # Fallback: fully remove prefix pattern __FENCE_START__{lang}__
                        code_content = re.sub(r"__FENCE_START__[^\n]*?__", "```", part)
                        code_content = code_content.replace("__FENCE_END__", "```")
                        out_parts.append(code_content)
                except Exception:
                    # Fallback on error: fully remove prefix pattern
                    code_content = re.sub(r"__FENCE_START__[^\n]*?__", "```", part)
                    code_content = code_content.replace("__FENCE_END__", "```")
                    out_parts.append(code_content)
            else:
                # Non-code — apply whitespace collapsing
                out_parts.append(_collapse_ws(part))
        return "".join(out_parts)

    body = _collapse_preserving_code(joined)

    # Phase D: prepend metadata (always include all fields with N/A fallback)
    meta_lines = []
    # Use full canonical URL for Source field instead of domain only
    source_display = url or ""
    # Always include core metadata fields
    meta_lines.append(f"Author: {meta.get('author') or 'N/A'}")
    meta_lines.append(f"Published: {meta.get('publish_date') or 'N/A'}")
    meta_lines.append(f"Version: {meta.get('version') or 'N/A'}")
    # Only include Last Updated if present
    if meta.get('last_updated'):
        meta_lines.append(f"Last Updated: {meta.get('last_updated')}")
    meta_lines.append(f"Source: {source_display or 'N/A'}")

    # Collapse whitespace in metadata only, then prepend to body (preserving code fences)
    meta_str = _collapse_ws("\n".join(meta_lines))
    body = meta_str + "\n\n" + body

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
