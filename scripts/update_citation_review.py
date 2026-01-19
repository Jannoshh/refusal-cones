#!/usr/bin/env python3
"""
Update `docs/citation_review.md` with papers that cite the base paper.

Primary source: OpenAlex (open metadata + cursor pagination).
Optional source: Semantic Scholar Graph API (often has additional metadata/links).

This script only rewrites the section between:
  <!-- BEGIN AUTO-CITATION-REVIEW -->
  <!-- END AUTO-CITATION-REVIEW -->
Everything outside those markers is preserved for manual notes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


OPENALEX_API = "https://api.openalex.org"
S2_API = "https://api.semanticscholar.org/graph/v1"

AUTO_BEGIN = "<!-- BEGIN AUTO-CITATION-REVIEW -->"
AUTO_END = "<!-- END AUTO-CITATION-REVIEW -->"


@dataclass(frozen=True)
class Paper:
    source_ids: Dict[str, str]  # e.g. {"openalex": "...", "s2": "...", "doi": "...", "arxiv": "..."}
    title: str
    year: Optional[int]
    venue: Optional[str]
    authors: List[str]
    url: Optional[str]
    abstract: Optional[str]
    cited_by_count: Optional[int]

    def dedupe_key(self) -> str:
        doi = (self.source_ids.get("doi") or "").strip().lower()
        if doi:
            doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
            return f"doi:{doi}"
        norm_title = re.sub(r"[^a-z0-9]+", " ", self.title.lower()).strip()
        return f"title_year:{norm_title}:{self.year or 'unknown'}"


def _http_get_json(url: str, *, headers: Optional[Dict[str, str]] = None, timeout_s: int = 30) -> Any:
    merged_headers = {
        "User-Agent": "refusal-cones-citation-review/0.1 (+https://github.com/)",
        "Accept": "application/json",
    }
    if headers:
        merged_headers.update(headers)
    req = Request(url, headers=merged_headers)
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _best_effort_request_json(url: str, *, headers: Optional[Dict[str, str]] = None) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return _http_get_json(url, headers=headers), None
    except HTTPError as e:
        return None, f"HTTPError {e.code} for {url}"
    except URLError as e:
        return None, f"URLError for {url}: {e}"
    except Exception as e:  # noqa: BLE001 - best-effort diagnostics for a CLI tool
        return None, f"Error for {url}: {type(e).__name__}: {e}"


def _openalex_normalize_work_id(openalex_id_or_url: str) -> str:
    s = openalex_id_or_url.strip()
    if not s:
        raise ValueError("Empty OpenAlex work id")
    if s.startswith("https://openalex.org/"):
        return s
    if re.match(r"^W\d+$", s):
        return f"https://openalex.org/{s}"
    return s


def _openalex_inverted_index_to_text(index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not index:
        return None
    positions: Dict[int, str] = {}
    max_pos = -1
    for word, pos_list in index.items():
        for pos in pos_list:
            positions[pos] = word
            if pos > max_pos:
                max_pos = pos
    if max_pos < 0:
        return None
    words = [positions.get(i, "") for i in range(max_pos + 1)]
    text = " ".join(w for w in words if w)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s+'\s+", "'", text)
    return text.strip() or None


def _openalex_search_base_work(query: str, *, mailto: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    params = {"search": query, "per-page": 5}
    if mailto:
        params["mailto"] = mailto
    url = f"{OPENALEX_API}/works?{urlencode(params)}"
    data, err = _best_effort_request_json(url)
    if err or not data or not data.get("results"):
        return None, err or "No results from OpenAlex"
    results = data["results"]
    best = results[0]
    # If OpenAlex provides a relevance_score field, prefer that.
    if any("relevance_score" in r for r in results):
        best = max(results, key=lambda r: r.get("relevance_score", 0.0))
    return best, None


def _openalex_fetch_citers(
    base_openalex_work_url: str,
    *,
    mailto: Optional[str],
    max_results: int,
    per_page: int = 200,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    results: List[Dict[str, Any]] = []
    cursor = "*"
    base_openalex_work_url = _openalex_normalize_work_id(base_openalex_work_url)

    while cursor and len(results) < max_results:
        params = {
            "filter": f"cites:{base_openalex_work_url}",
            "per-page": min(per_page, max_results - len(results)),
            "cursor": cursor,
        }
        if mailto:
            params["mailto"] = mailto
        url = f"{OPENALEX_API}/works?{urlencode(params)}"
        data, err = _best_effort_request_json(url)
        if err:
            errors.append(err)
            break
        if not data:
            break
        page_results = data.get("results") or []
        results.extend(page_results)
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not page_results:
            break
    return results[:max_results], errors


def _openalex_work_to_paper(work: Dict[str, Any]) -> Paper:
    ids = work.get("ids") or {}
    authorships = work.get("authorships") or []
    authors = []
    for a in authorships:
        name = ((a.get("author") or {}).get("display_name") or "").strip()
        if name:
            authors.append(name)

    venue = None
    host_venue = work.get("host_venue") or {}
    if host_venue.get("display_name"):
        venue = host_venue["display_name"]

    abstract = _openalex_inverted_index_to_text(work.get("abstract_inverted_index"))

    source_ids: Dict[str, str] = {}
    if ids.get("openalex"):
        source_ids["openalex"] = ids["openalex"]
    if ids.get("doi"):
        source_ids["doi"] = ids["doi"]
    if ids.get("arxiv"):
        source_ids["arxiv"] = ids["arxiv"]
    if work.get("doi"):
        source_ids.setdefault("doi", work["doi"])

    return Paper(
        source_ids=source_ids,
        title=(work.get("title") or "").strip() or "<missing title>",
        year=work.get("publication_year"),
        venue=venue,
        authors=authors,
        url=work.get("id") or ids.get("openalex"),
        abstract=abstract,
        cited_by_count=work.get("cited_by_count"),
    )


def _s2_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"User-Agent": "refusal-cones-citation-review/0.1"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _s2_search_base_paper(query: str, *, api_key: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    params = {
        "query": query,
        "limit": 5,
        "fields": "title,year,authors,paperId,externalIds,url,venue",
    }
    url = f"{S2_API}/paper/search?{urlencode(params)}"
    data, err = _best_effort_request_json(url, headers=_s2_headers(api_key))
    if err or not data or not data.get("data"):
        return None, err or "No results from Semantic Scholar"
    results = data["data"]
    return results[0], None


def _s2_fetch_citers(
    paper_id: str,
    *,
    api_key: Optional[str],
    max_results: int,
    page_size: int = 100,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    results: List[Dict[str, Any]] = []
    offset = 0
    headers = _s2_headers(api_key)

    fields = (
        "citingPaper.paperId,citingPaper.title,citingPaper.year,citingPaper.venue,"
        "citingPaper.authors,citingPaper.externalIds,citingPaper.url,citingPaper.abstract"
    )

    while offset < max_results:
        params = {"fields": fields, "limit": min(page_size, max_results - offset), "offset": offset}
        url = f"{S2_API}/paper/{paper_id}/citations?{urlencode(params)}"
        data, err = _best_effort_request_json(url, headers=headers)
        if err:
            errors.append(err)
            break
        if not data:
            break
        page = data.get("data") or []
        if not page:
            break
        results.extend(page)
        offset += len(page)
    return results[:max_results], errors


def _s2_citation_to_paper(citation_row: Dict[str, Any]) -> Optional[Paper]:
    citing = citation_row.get("citingPaper") or {}
    title = (citing.get("title") or "").strip()
    if not title:
        return None
    external = citing.get("externalIds") or {}
    source_ids: Dict[str, str] = {"s2": citing.get("paperId", "")}
    if external.get("DOI"):
        source_ids["doi"] = f"https://doi.org/{external['DOI']}"
    if external.get("ArXiv"):
        source_ids["arxiv"] = external["ArXiv"]

    authors = []
    for a in citing.get("authors") or []:
        name = (a.get("name") or "").strip()
        if name:
            authors.append(name)

    return Paper(
        source_ids={k: v for k, v in source_ids.items() if v},
        title=title,
        year=citing.get("year"),
        venue=(citing.get("venue") or None),
        authors=authors,
        url=citing.get("url") or None,
        abstract=(citing.get("abstract") or None),
        cited_by_count=None,
    )


def _extract_key_sentences(text: str, *, max_sentences: int = 3) -> List[str]:
    if not text:
        return []
    # Simple sentence segmentation (good enough for abstracts)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return []
    triggers = (
        "we show",
        "we find",
        "we demonstrate",
        "we propose",
        "we introduce",
        "we present",
        "our results",
        "we prove",
        "we uncover",
        "we identify",
    )
    selected = [p for p in parts if any(t in p.lower() for t in triggers)]
    if not selected:
        selected = parts[: max_sentences]
    return selected[:max_sentences]


def _suggest_experiment_tags(text: str, experiment_codes: List[str]) -> List[str]:
    t = text.lower()
    tags: List[str] = []

    def has_any(*needles: str) -> bool:
        return any(n in t for n in needles)

    # Heuristic mappings to PAPER_PLAN experiments (best-effort)
    if has_any("gaussian process", "bayesian optimization", "sample efficiency", "measurements", "acquisition"):
        for code in ("E1.1", "E1.2"):
            if code in experiment_codes:
                tags.append(code)
    if has_any("cone", "polyhedral", "concept cone"):
        for code in ("E1.2",):
            if code in experiment_codes:
                tags.append(code)
    if has_any("layer", "per-layer", "depth", "middle layer"):
        for code in ("E2.1", "E2.2", "E2.3"):
            if code in experiment_codes:
                tags.append(code)
    if has_any("affine", "projection", "ablation", "addition"):
        for code in ("E3.1", "E3.2"):
            if code in experiment_codes:
                tags.append(code)
    if has_any("reinforcement", "rl", "ppo", "grpo", "reward"):
        for code in ("E4.1", "E4.2", "E4.3"):
            if code in experiment_codes:
                tags.append(code)
    if has_any("ablation study", "ablation studies", "sensitivity", "hyperparameter"):
        for code in ("E5.1", "E5.2"):
            if code in experiment_codes:
                tags.append(code)

    # Avoid duplicates, preserve order
    seen = set()
    out: List[str] = []
    for tag in tags:
        if tag not in seen:
            out.append(tag)
            seen.add(tag)
    return out


def _parse_planned_experiments(paper_plan_path: Path) -> List[Tuple[str, str]]:
    if not paper_plan_path.exists():
        return []
    experiments: List[Tuple[str, str]] = []
    for line in paper_plan_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^####\s+(E\d+\.\d+):\s*(.+?)\s*$", line)
        if m:
            experiments.append((m.group(1), m.group(2)))
    return experiments


def _format_links(p: Paper) -> str:
    links: List[str] = []
    doi = p.source_ids.get("doi")
    if doi:
        links.append(f"[doi]({doi})")
    arxiv = p.source_ids.get("arxiv")
    if arxiv:
        links.append(f"[arXiv](https://arxiv.org/abs/{arxiv})")
    openalex = p.source_ids.get("openalex")
    if openalex:
        links.append(f"[OpenAlex]({openalex})")
    s2 = p.source_ids.get("s2")
    if s2:
        links.append(f"[S2](https://www.semanticscholar.org/paper/{s2})")
    if p.url and not links:
        links.append(f"[link]({p.url})")
    return " · ".join(links) if links else ""


def _render_auto_block(
    *,
    base_title: str,
    base_openalex: Optional[Paper],
    base_s2: Optional[Dict[str, Any]],
    citers: List[Paper],
    planned_experiments: List[Tuple[str, str]],
    fetch_warnings: List[str],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    exp_codes = [c for c, _ in planned_experiments]

    lines: List[str] = []
    lines.append(AUTO_BEGIN)
    lines.append(f"_Last updated: {now}_")
    lines.append("")
    lines.append("## Base paper (auto)")
    lines.append(f"- Query: {base_title}")
    if base_openalex:
        lines.append(f"- OpenAlex: {base_openalex.source_ids.get('openalex','')}")
        if base_openalex.source_ids.get("doi"):
            lines.append(f"- DOI: {base_openalex.source_ids['doi']}")
        if base_openalex.source_ids.get("arxiv"):
            lines.append(f"- arXiv: {base_openalex.source_ids['arxiv']}")
    if base_s2:
        paper_id = base_s2.get("paperId") or ""
        if paper_id:
            lines.append(f"- Semantic Scholar: https://www.semanticscholar.org/paper/{paper_id}")
    lines.append("")

    lines.append("## Planned experiments (auto, from `docs/PAPER_PLAN.md`)")
    if planned_experiments:
        for code, title in planned_experiments:
            lines.append(f"- {code}: {title}")
    else:
        lines.append("- (Could not parse any experiments.)")
    lines.append("")

    if fetch_warnings:
        lines.append("## Fetch warnings (auto)")
        for w in fetch_warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append(f"## Citing works (auto, n={len(citers)})")
    lines.append("")
    lines.append("| Year | Title | Venue | Links | Suggested experiments | Auto findings |")
    lines.append("|---:|---|---|---|---|---|")

    # Sort newest first, then title.
    def sort_key(p: Paper) -> Tuple[int, str]:
        return (-(p.year or 0), p.title.lower())

    for p in sorted(citers, key=sort_key):
        venue = (p.venue or "").replace("|", "\\|")
        title = p.title.replace("|", "\\|")
        links = _format_links(p).replace("|", "\\|")
        text_for_tags = f"{p.title}\\n{p.abstract or ''}"
        tags = ", ".join(_suggest_experiment_tags(text_for_tags, exp_codes))
        findings = "; ".join(_extract_key_sentences(p.abstract or "", max_sentences=2)).replace("|", "\\|")
        lines.append(f"| {p.year or ''} | {title} | {venue} | {links} | {tags} | {findings} |")

    lines.append("")
    lines.append("### Details (auto)")
    lines.append("")
    for p in sorted(citers, key=sort_key):
        lines.append(f"#### {p.year or '????'} — {p.title}")
        if p.authors:
            shown = ", ".join(p.authors[:8]) + (" et al." if len(p.authors) > 8 else "")
            lines.append(f"- Authors: {shown}")
        if p.venue:
            lines.append(f"- Venue: {p.venue}")
        link_line = _format_links(p)
        if link_line:
            lines.append(f"- Links: {link_line}")
        if p.cited_by_count is not None:
            lines.append(f"- OpenAlex cited_by_count: {p.cited_by_count}")

        text_for_tags = f"{p.title}\\n{p.abstract or ''}"
        tags = _suggest_experiment_tags(text_for_tags, exp_codes)
        if tags:
            lines.append(f"- Suggested experiments (heuristic): {', '.join(tags)}")

        key = p.dedupe_key()
        lines.append(f"- Key: `{key}`")

        findings = _extract_key_sentences(p.abstract or "", max_sentences=3)
        if findings:
            lines.append("- Auto findings (from abstract):")
            for s in findings:
                lines.append(f"  - {s}")
        else:
            lines.append("- Auto findings: (no abstract available)")
        lines.append("")

    lines.append(AUTO_END)
    return "\n".join(lines) + "\n"


def _replace_or_insert_auto_block(existing_md: str, new_block: str) -> str:
    if AUTO_BEGIN in existing_md and AUTO_END in existing_md:
        pattern = re.compile(re.escape(AUTO_BEGIN) + r".*?" + re.escape(AUTO_END), re.S)
        return pattern.sub(new_block.strip("\n"), existing_md).rstrip() + "\n"
    # Insert near top, after first heading if present.
    lines = existing_md.splitlines()
    insert_at = 0
    for i, line in enumerate(lines[:50]):
        if line.startswith("# "):
            insert_at = i + 1
            break
    out = lines[:insert_at] + ["", new_block.strip("\n"), ""] + lines[insert_at:]
    return "\n".join(out).rstrip() + "\n"


def _merge_papers(primary: Iterable[Paper], secondary: Iterable[Paper]) -> List[Paper]:
    merged: Dict[str, Paper] = {}

    def merge_into(p: Paper) -> None:
        key = p.dedupe_key()
        if key not in merged:
            merged[key] = p
            return
        existing = merged[key]
        # Prefer richer metadata: abstract, venue, authors, urls.
        source_ids = dict(existing.source_ids)
        source_ids.update(p.source_ids)
        abstract = existing.abstract or p.abstract
        venue = existing.venue or p.venue
        authors = existing.authors or p.authors
        url = existing.url or p.url
        cited_by_count = existing.cited_by_count if existing.cited_by_count is not None else p.cited_by_count
        merged[key] = Paper(
            source_ids=source_ids,
            title=existing.title or p.title,
            year=existing.year or p.year,
            venue=venue,
            authors=authors,
            url=url,
            abstract=abstract,
            cited_by_count=cited_by_count,
        )

    for p in primary:
        merge_into(p)
    for p in secondary:
        merge_into(p)
    return list(merged.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Update docs/citation_review.md from OpenAlex + Semantic Scholar.")
    parser.add_argument(
        "--base-query",
        default="The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence",
        help="Title/DOI/arXiv query used to locate the base paper (default: repo base paper title).",
    )
    parser.add_argument("--openalex-work", default="", help="OpenAlex work id/url for the base paper (e.g. W... or https://openalex.org/W...).")
    parser.add_argument("--s2-paper-id", default="", help="Semantic Scholar paperId for the base paper (skips search).")
    parser.add_argument("--mailto", default=os.environ.get("OPENALEX_MAILTO", ""), help="Email for OpenAlex polite pool (or set OPENALEX_MAILTO).")
    parser.add_argument("--s2-api-key", default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""), help="Semantic Scholar API key (optional).")
    parser.add_argument("--no-s2", action="store_true", help="Disable Semantic Scholar fetching.")
    parser.add_argument("--max-results", type=int, default=300, help="Max number of citing works to fetch per source.")
    parser.add_argument("--paper-plan", default="docs/PAPER_PLAN.md", help="Path to PAPER_PLAN.md for experiment extraction.")
    parser.add_argument("--output", default="docs/citation_review.md", help="Path to citation review markdown to update.")
    args = parser.parse_args()

    paper_plan_path = Path(args.paper_plan)
    planned_experiments = _parse_planned_experiments(paper_plan_path)

    warnings: List[str] = []

    # OpenAlex base work discovery
    base_openalex_work: Optional[Dict[str, Any]] = None
    if args.openalex_work.strip():
        base_openalex_work_url = _openalex_normalize_work_id(args.openalex_work)
        openalex_work_id = base_openalex_work_url.split("/")[-1]
        params = {"mailto": args.mailto} if args.mailto else {}
        url = f"{OPENALEX_API}/works/{openalex_work_id}" + (f"?{urlencode(params)}" if params else "")
        data, err = _best_effort_request_json(url)
        if err:
            warnings.append(err)
        elif isinstance(data, dict):
            base_openalex_work = data
    else:
        base_openalex_work, err = _openalex_search_base_work(args.base_query, mailto=args.mailto or None)
        if err:
            warnings.append(err)

    base_openalex_paper: Optional[Paper] = _openalex_work_to_paper(base_openalex_work) if base_openalex_work else None

    openalex_citers: List[Paper] = []
    if base_openalex_paper and base_openalex_paper.source_ids.get("openalex"):
        raw, errs = _openalex_fetch_citers(
            base_openalex_paper.source_ids["openalex"],
            mailto=args.mailto or None,
            max_results=args.max_results,
        )
        warnings.extend(errs)
        openalex_citers = [_openalex_work_to_paper(w) for w in raw]
    else:
        warnings.append("OpenAlex base paper not found; skipping OpenAlex citers.")

    # Semantic Scholar base work discovery
    s2_base: Optional[Dict[str, Any]] = None
    s2_citers: List[Paper] = []
    if not args.no_s2:
        if args.s2_paper_id.strip():
            s2_base = {"paperId": args.s2_paper_id.strip()}
        else:
            s2_base, err = _s2_search_base_paper(args.base_query, api_key=args.s2_api_key or None)
            if err:
                warnings.append(err)
        if s2_base and s2_base.get("paperId"):
            raw, errs = _s2_fetch_citers(
                s2_base["paperId"],
                api_key=args.s2_api_key or None,
                max_results=args.max_results,
            )
            warnings.extend(errs)
            for row in raw:
                p = _s2_citation_to_paper(row)
                if p:
                    s2_citers.append(p)
        else:
            warnings.append("Semantic Scholar base paper not found; skipping S2 citers.")

    merged_citers = _merge_papers(openalex_citers, s2_citers)

    output_path = Path(args.output)
    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
    else:
        existing = "# Citation Review\n\n" + "(Run `python3 scripts/update_citation_review.py` to populate.)\n"

    auto_block = _render_auto_block(
        base_title=args.base_query,
        base_openalex=base_openalex_paper,
        base_s2=s2_base,
        citers=merged_citers,
        planned_experiments=planned_experiments,
        fetch_warnings=warnings,
    )

    updated = _replace_or_insert_auto_block(existing, auto_block)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(updated, encoding="utf-8")

    print(f"Updated {output_path} with {len(merged_citers)} citing works.")
    if warnings:
        print("Warnings:")
        for w in warnings[:10]:
            print(" -", w)
        if len(warnings) > 10:
            print(f" - (+{len(warnings) - 10} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
