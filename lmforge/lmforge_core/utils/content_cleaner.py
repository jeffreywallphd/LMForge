# General-purpose content cleaner for arbitrary websites.
# Works if you have HTML (best) or if you only have plaintext (fallback).

import re
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse

_WS = re.compile(r"[ \t]+")
_MANY_NL = re.compile(r"\n{3,}")

# Headings/phrases that typically mean "end of real article"
_CUTOFF_PHRASES = [
    "latest ai news", "mixture of experts", "related", "related content",
    "related solutions", "resources", "report", "training", "video",
    "ebook", "guide", "footnotes", "take the next step",
    "further reading", "you may also like", "sponsored", "newsletter",
    "subscribe"
]

_NOISE_PHRASES = [
    "authors", "ibm think", "share", "link copied", "subscribe",
    "watch all episodes", "the latest ai news", "mixture of experts",
    "ibm ai academy", "explore the series", "read the report",
    "read the ebook", "read the guide"
]


# Class/id/role hints for non-content chrome
_JUNK_HINTS = [
    "cookie", "gdpr", "consent", "banner", "popover", "modal", "subscribe",
    "advert", "ad-", "ads-", "promo", "promo-", "footer", "nav", "breadcrumbs",
    "newsletter", "signup", "social", "share", "overlay", "paywall", "sidebar",
    "related", "recommended", "outbrain", "taboola", "toc", "table-of-contents",
    "comments", "comment", "discuss", "survey"
]


def _collapse_ws(s: str) -> str:
    s = _WS.sub(" ", s)
    s = _MANY_NL.sub("\n\n", s)
    return s.strip()

def _looks_junk(tag) -> bool:
    t = " ".join([tag.get("id","") or "", " ".join(tag.get("class",[]))]).lower()
    return any(h in t for h in _JUNK_HINTS)

def _text_len(tag) -> int:
    return len(tag.get_text(" ", strip=True))

def _densest_block(soup: BeautifulSoup):
    for sel in ("article", "main", "[role='main']"):
        node = soup.select_one(sel)
        if node and _text_len(node) > 200:
            return node
    cands = soup.find_all(["section","div","article","main"])
    if not cands:
        return soup.body or soup
    best, score = None, -1
    for c in cands:
        ptext = sum(_text_len(p) for p in c.find_all("p"))
        hbonus = len(c.find_all(["h1","h2","h3"])) * 200
        sc = ptext + hbonus
        if sc > score:
            best, score = c, sc
    return best or soup

def _is_heading_like(s: str) -> bool:
    # Short-ish, Title Case or ALL CAPS → likely a heading
    if len(s) <= 90:
        if s == s.title() or s.isupper():
            return True
    return False

def _post_filter_lines(lines):
    # Normalize for comparisons
    def norm(s: str) -> str:
        s = s.strip().lower()
        s = " ".join(s.split())
        return s

    def looks_like_footnote_number(s: str) -> bool:
        ss = s.strip()
        return ss.isdigit() and len(ss) <= 2

    def has_any(hay: str, needles) -> bool:
        low = norm(hay)
        return any(needle in low for needle in needles)

    out = []
    seen_heads = set()
    cutoff_hit = False

    for ln in lines:
        if not ln:
            out.append(ln)
            continue

        n = norm(ln)

        # Cut at the first "rail" section
        if has_any(n, _CUTOFF_PHRASES):
            cutoff_hit = True
            break

        # Drop pure footnote markers like "1", "2"
        if looks_like_footnote_number(ln):
            continue

        # Drop common boilerplate
        if has_any(n, _NOISE_PHRASES):
            continue

        # Drop very short non-heading lines (leftover nav crumbs)
        word_count = len(n.split())
        if word_count < 3 and not _is_heading_like(ln):
            continue

        # Deduplicate exact headings/short blurbs
        if _is_heading_like(ln):
            key = n
            if key in seen_heads:
                continue
            seen_heads.add(key)

        out.append(ln)

    # Collapse multiple blank lines
    compact = []
    blank = False
    for ln in out:
        if ln.strip():
            compact.append(ln)
            blank = False
        else:
            if not blank:
                compact.append("")
                blank = True

    return compact


def clean_html_anysite(html_bytes: bytes, url: str) -> dict:
    """
    Universal HTML cleaner for any website.
    Extracts the main article text, removes ads, rails, footers, and other junk.
    Designed to produce clean, model-ready text.
    """
    soup = BeautifulSoup(html_bytes, "html.parser")

    # --- Remove non-content elements early ---
    for t in soup(["script", "style", "noscript", "template", "svg", "iframe", "meta", "header", "footer"]):
        t.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()

    # --- Identify the densest content block ---
    main = _densest_block(soup) 
    if not main:
        main = soup.body or soup

    # 1) INLINE REMOVALS: delete obvious promo/meta blocks anywhere in MAIN
    INLINE_NOISE_PHRASES = [
        "the latest ai news", "subscribe today", "watch all episodes",
        "link copied", "authors", "ibm think"
    ]
    for txtnode in list(main.find_all(string=True)):
        low = str(txtnode).strip().lower()
        if not low:
            continue
        if any(p in low for p in INLINE_NOISE_PHRASES):
            # climb to a reasonable container, but not the whole MAIN
            anc = txtnode.parent
            # stop climbing at direct child of MAIN or at semantic blocks
            stop_tags = {"section", "article", "div"}
            while anc and anc.parent is not main and anc.name not in stop_tags:
                anc = anc.parent
            try:
                (anc or txtnode.parent).decompose()
            except Exception:
                pass

    # 2) CHILD-LEVEL CUTOFF: if any *direct child* looks like a rail, remove it and everything after it
    RAIL_MARKERS = [
        "latest ai news", "mixture of experts", "newsletter", "subscribe",
        "related", "related content", "related solutions", "resources",
        "report", "training", "video", "ebook", "guide",
        "footnotes", "take the next step", "further reading",
        "you may also like", "sponsored"
    ]
    children = list(main.children)
    cut_index = None
    for i, ch in enumerate(children):
        if not getattr(ch, "name", None):
            continue
        text = ch.get_text(" ", strip=True).lower() if ch else ""
        if any(m in text for m in RAIL_MARKERS):
            cut_index = i
            break

    if cut_index is not None:
        # delete the matched child and all following siblings
        for ch in children[cut_index:]:
            try:
                ch.decompose()
            except Exception:
                pass


    # --- Nuke sidebar / rail / ads containers ---
    junk_selectors = [
        "[role='complementary']", "[data-testid='rail']",
        ".sidebar", ".side-rail", ".rail", ".related", ".recommended",
        ".promo", ".ads", ".ad-container", ".newsletter", ".cookie-banner",
        ".outbrain", ".taboola", ".share", ".footer", ".disclaimer",
    ]
    for sel in junk_selectors:
        for el in main.select(sel):
            el.decompose()

    # --- Heuristic cutoff: remove trailing junk sections ---
    def _txt(el):
        try:
            return el.get_text(" ", strip=True).lower()
        except Exception:
            return ""

    RAIL_MARKERS = [
        "latest news", "mixture of experts", "related", "resources",
        "report", "training", "video", "ebook", "guide", "footnotes",
        "take the next step", "further reading", "you may also like",
        "sponsored", "newsletter", "subscribe", "read the report",
        "read the ebook", "watch all episodes", "explore the series"
    ]

    cut_node = None
    for el in main.descendants:
        if not getattr(el, "name", None):
            continue
        if not el.get_text(strip=True):
            continue
        low = _txt(el)
        if any(m in low for m in RAIL_MARKERS):
            # Ascend to direct main child
            anc = el
            while anc and anc.parent is not main:
                anc = anc.parent
            cut_node = anc or el
            break

    if cut_node:
        cur = cut_node
        while cur:
            nxt = cur.next_sibling
            try:
                cur.decompose()
            except Exception:
                pass
            cur = nxt

    # --- Extra deep clean inside MAIN ---
    for el in main.find_all(True):
        if _looks_junk(el):
            el.decompose()

    # --- Normalize structure ---
    # Convert <ul><li> lists to text bullets
    for ul in main.find_all("ul"):
        items = [f"- {li.get_text(' ', strip=True)}" for li in ul.find_all("li", recursive=False)]
        ul.replace_with("\n".join([i for i in items if i]))

    # Unwrap links
    for a in main.find_all("a"):
        a.replace_with(a.get_text(" ", strip=True))

    # Keep only meaningful tags
    allowed = {"h1", "h2", "h3", "h4", "p"}
    for t in list(main.find_all(True)):
        if t.name not in allowed:
            t.unwrap()

    # --- Linearize to lines ---
    lines = []
    for node in main.children:
        name = getattr(node, "name", None)
        if name in {"h1", "h2", "h3", "h4"}:
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(text)
                lines.append("")
        elif name == "p":
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(text)
        elif isinstance(node, str):
            text = node.strip()
            if text:
                lines.append(text)

    # --- Trim top boilerplate ---
    def _is_real_para(s: str) -> bool:
        s2 = s.strip()
        return len(s2) >= 120 or len(s2.split()) >= 20

    cleaned_lines = []
    started = False
    for ln in lines:
        if not started:
            low = ln.strip().lower()
            if (
                not ln.strip() or
                low.startswith("authors") or
                "share" in low or
                "link copied" in low or
                "subscribe" in low or
                "ibm think" in low or
                len(low) <= 3
            ):
                continue
            if _is_real_para(ln):
                started = True
                cleaned_lines.append(ln)
            else:
                if ln == ln.title() or ln.isupper():
                    cleaned_lines.append(ln)
        else:
            cleaned_lines.append(ln)

    lines = [_collapse_ws(x) for x in cleaned_lines]
    lines = _post_filter_lines(lines)
    body = _collapse_ws("\n".join(lines))

    # --- TEXT-LEVEL FINAL PASS: guaranteed cleanup ---
    # 1) Cut off everything after the first rail/promo marker in the text
    TEXT_RAIL_MARKERS = [
        "the latest ai news", "mixture of experts", "newsletter", "subscribe",
        "related", "related content", "related solutions", "resources",
        "report", "training", "video", "ebook", "guide", "footnotes",
        "take the next step", "further reading", "you may also like",
        "sponsored", "explore the series", "watch all episodes",
        "explore watsonx.ai", "explore ai solutions"
    ]
    bl = body.lower()
    cut_positions = [bl.find(m) for m in TEXT_RAIL_MARKERS if m in bl]
    if cut_positions:
        cut_at = min(p for p in cut_positions if p >= 0)
        if cut_at > 120:   # don't cut if the page is basically only promos
            body = body[:cut_at].rstrip()

    # 2) Drop top boilerplate lines (authors/share/etc.) that sometimes slip through
    DROP_PREFIXES = [
        "authors", "ibm think", "share", "link copied", "subscribe today"
    ]
    clean_lines = []
    started = False
    for ln in body.splitlines():
        low = ln.strip().lower()
        if not started:
            if not low or any(low.startswith(p) for p in DROP_PREFIXES):
                continue
            # start once we see a real paragraph or a heading-y line
            if len(ln) >= 120 or len(ln.split()) >= 20 or ln == ln.title() or ln.isupper():
                started = True
                clean_lines.append(ln.strip())
        else:
            # skip stray promo one-liners that appear mid-text
            if any(p in low for p in ["subscribe", "link copied"]):
                continue
            clean_lines.append(ln.strip())
    body = "\n\n".join([l for l in clean_lines if l])

    # 3) Final whitespace collapse
    body = _collapse_ws(body)

    # --- Metadata ---
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    title = re.sub(r"\s+\|\s+.*$", "", title)
    site = urlparse(url).netloc

    return {"title": title, "site": site, "url": url, "body": body}


def clean_plaintext_anysite(text: str) -> str:
    """
    Fallback if you only have raw plaintext for the whole page.
    Heuristically removes chrome/rails and tidies structure.
    """
    # Normalize breaks and spaces
    txt = text.replace("\r\n", "\n")
    # split on blank lines to get paragraphs/headings-ish blocks
    blocks = [b.strip() for b in txt.split("\n\n") if b.strip()]
    out = []
    for b in blocks:
        b_norm = _collapse_ws(b)
        low = b_norm.lower()
        # drop obvious rails/promos
        if any(low.startswith(h) for h in _CUTOFF_HEADINGS):
            break
        if any(low.startswith(p) for p in _BAD_LINE_PREFIXES):
            continue
        # drop pure links lists / navigationy short blocks
        if len(b_norm) < 4:
            continue
        out.append(b_norm)
    # dedupe and join
    seen = set(); kept = []
    for p in out:
        k = p.lower()
        if k in seen: 
            continue
        seen.add(k)
        kept.append(p)
    return _collapse_ws("\n\n".join(kept))