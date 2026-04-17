#!/usr/bin/env python3
"""Generate query index pages from YAML frontmatter in sources/ and wiki/."""

import re
import yaml
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent
SOURCES_DIR = REPO_ROOT / "sources"
WIKI_DIR = REPO_ROOT / "wiki"
QUERIES_DIR = REPO_ROOT / "queries"


def extract_frontmatter(filepath):
    """Extract YAML frontmatter from a markdown file."""
    with open(filepath) as f:
        content = f.read()
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None


def collect_all_pages():
    """Collect frontmatter from all source and wiki pages."""
    pages = []
    for search_dir in [SOURCES_DIR, WIKI_DIR]:
        if not search_dir.exists():
            continue
        for md_file in sorted(search_dir.rglob("*.md")):
            fm = extract_frontmatter(md_file)
            if fm and isinstance(fm, dict):
                fm["_path"] = str(md_file.relative_to(REPO_ROOT))
                fm["_dir"] = str(search_dir.relative_to(REPO_ROOT))
                pages.append(fm)
    return pages


def generate_by_problem(pages):
    """Generate by-problem.md from wiki/patterns/ pages."""
    lines = [
        "# Query: By Problem / Symptom",
        "",
        "> Auto-generated from `wiki/patterns/*.md` frontmatter. Do not edit manually.",
        "> Regenerate: `python scripts/generate-indices.py`",
        "",
    ]

    patterns = [p for p in pages if p.get("type") == "pattern"]
    if not patterns:
        lines.append("*No pattern pages found.*")
        return "\n".join(lines) + "\n"

    lines.append("| Symptom | Pattern Page | Candidate Techniques | Sources |")
    lines.append("|---------|-------------|---------------------|---------|")

    for p in sorted(patterns, key=lambda x: x.get("title", "")):
        symptoms = ", ".join(p.get("symptoms", []))
        title = p.get("title", "Untitled")
        path = p.get("_path", "")
        techniques = ", ".join(p.get("candidate_techniques", []))
        src_count = len(p.get("sources", []))
        lines.append(f"| {symptoms} | [{title}]({path}) | {techniques} | {src_count} sources |")

    return "\n".join(lines) + "\n"


def generate_by_technique(pages):
    """Generate by-technique.md from wiki/techniques/ pages."""
    lines = [
        "# Query: By Technique",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
    ]

    techniques = [p for p in pages if p.get("type") == "technique"]
    if not techniques:
        lines.append("*No technique pages found.*")
        return "\n".join(lines) + "\n"

    lines.append("| Technique | Tags | Architectures | Confidence | Reproducibility | Sources |")
    lines.append("|-----------|------|--------------|------------|-----------------|---------|")

    for t in sorted(techniques, key=lambda x: x.get("title", "")):
        title = t.get("title", "Untitled")
        path = t.get("_path", "")
        tags = ", ".join(t.get("tags", [])[:4])
        archs = ", ".join(t.get("architectures", []))
        conf = t.get("confidence", "")
        repro = t.get("reproducibility", "")
        src_count = len(t.get("sources", []))
        lines.append(f"| [{title}]({path}) | {tags} | {archs} | {conf} | {repro} | {src_count} |")

    return "\n".join(lines) + "\n"


def generate_by_hardware_feature(pages):
    """Generate by-hardware-feature.md mapping features to wiki pages."""
    lines = [
        "# Query: By Hardware Feature",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
    ]

    # Load valid hardware feature tags
    tags_path = REPO_ROOT / "data" / "tags.yaml"
    hw_tag_set = set()
    if tags_path.exists():
        with open(tags_path) as f:
            tag_data = yaml.safe_load(f)
        hw_tag_set = set(tag_data.get("hardware_features", []))

    # Collect hardware features from all pages (only actual hardware tags)
    feature_pages = defaultdict(list)
    for p in pages:
        for feat in p.get("hardware_features", []):
            feature_pages[feat].append(p)
        # Also check tags, but only if they're actual hardware features
        for tag in p.get("tags", []):
            if tag in hw_tag_set:
                feature_pages[tag].append(p)

    # Also add dedicated hardware pages
    hw_pages = [p for p in pages if p.get("type") == "hardware"]
    for hp in hw_pages:
        fid = hp.get("id", "").replace("hw-", "")
        if fid:
            feature_pages[fid].append(hp)

    if not feature_pages:
        lines.append("*No hardware feature references found.*")
        return "\n".join(lines) + "\n"

    lines.append("| Feature | Related Pages |")
    lines.append("|---------|--------------|")

    for feat in sorted(feature_pages.keys()):
        page_links = []
        seen = set()
        for p in feature_pages[feat]:
            pid = p.get("id", p.get("_path", ""))
            if pid not in seen:
                seen.add(pid)
                title = p.get("title", pid)
                path = p.get("_path", "")
                page_links.append(f"[{title}]({path})")
        lines.append(f"| `{feat}` | {', '.join(page_links[:5])} |")

    return "\n".join(lines) + "\n"


def generate_by_repo(pages):
    """Generate by-repo.md listing source PRs grouped by repository."""
    lines = [
        "# Query: By Repository",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
    ]

    repos = defaultdict(list)
    for p in pages:
        if "repo" in p and "pr" in p:
            repos[p["repo"]].append(p)

    if not repos:
        lines.append("*No PR sources found.*")
        return "\n".join(lines) + "\n"

    for repo in sorted(repos.keys()):
        prs = sorted(repos[repo], key=lambda x: x.get("date", ""), reverse=True)
        lines.append(f"## {repo} ({len(prs)} PRs)")
        lines.append("")
        lines.append("| PR | Title | Date | Techniques | Tags |")
        lines.append("|-----|-------|------|------------|------|")
        for pr in prs:
            pr_num = pr.get("pr", "?")
            title = pr.get("title", "Untitled")
            date = pr.get("date", "")
            path = pr.get("_path", "")
            techniques = ", ".join(pr.get("techniques", [])[:3])
            tags = ", ".join(pr.get("tags", [])[:3])
            lines.append(f"| [#{pr_num}]({path}) | {title} | {date} | {techniques} | {tags} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_by_kernel_type(pages):
    """Generate by-kernel-type.md mapping kernel types to wiki pages."""
    lines = [
        "# Query: By Kernel Type",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
    ]

    type_pages = defaultdict(list)
    for p in pages:
        for kt in p.get("kernel_types", []):
            type_pages[kt].append(p)

    if not type_pages:
        lines.append("*No kernel type references found.*")
        return "\n".join(lines) + "\n"

    lines.append("| Kernel Type | Pages |")
    lines.append("|-------------|-------|")

    for kt in sorted(type_pages.keys()):
        page_links = []
        seen = set()
        for p in type_pages[kt]:
            pid = p.get("id", p.get("_path", ""))
            if pid not in seen:
                seen.add(pid)
                title = p.get("title", pid)
                path = p.get("_path", "")
                page_links.append(f"[{title}]({path})")
        lines.append(f"| `{kt}` | {', '.join(page_links[:8])} |")

    return "\n".join(lines) + "\n"


def generate_by_language(pages):
    """Generate by-language.md mapping languages to relevant pages."""
    lines = [
        "# Query: By Language / DSL",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
    ]

    lang_pages = defaultdict(list)

    # Include dedicated language pages first (wiki/languages/)
    for p in pages:
        if p.get("type") == "language":
            # Map language page to its language tag based on id
            lang_id = p.get("id", "")
            # e.g. lang-cute-dsl -> cute-dsl, lang-cuda-cpp -> cuda-cpp
            lang_tag = lang_id.replace("lang-", "") if lang_id.startswith("lang-") else ""
            if lang_tag:
                lang_pages[lang_tag].insert(0, p)  # first-class page first
            # Also use tags
            for tag in p.get("tags", []):
                if tag not in lang_pages or p not in lang_pages[tag]:
                    lang_pages[tag].append(p)

    # Then include all pages that use each language
    for p in pages:
        if p.get("type") != "language":  # avoid double-counting
            for lang in p.get("languages", []):
                lang_pages[lang].append(p)

    if not lang_pages:
        lines.append("*No language references found.*")
        return "\n".join(lines) + "\n"

    lines.append("| Language | Guide | Related Pages |")
    lines.append("|----------|-------|--------------|")

    for lang in sorted(lang_pages.keys()):
        guide_link = ""
        related_links = []
        seen = set()
        for p in lang_pages[lang]:
            pid = p.get("id", p.get("_path", ""))
            if pid in seen:
                continue
            seen.add(pid)
            title = p.get("title", pid)
            path = p.get("_path", "")
            if p.get("type") == "language":
                guide_link = f"[{title}]({path})"
            else:
                related_links.append(f"[{title}]({path})")
        lines.append(f"| `{lang}` | {guide_link} | {', '.join(related_links[:6])} |")

    return "\n".join(lines) + "\n"


def main():
    QUERIES_DIR.mkdir(exist_ok=True)
    pages = collect_all_pages()
    print(f"Collected {len(pages)} pages from sources/ and wiki/")

    generators = {
        "by-problem.md": generate_by_problem,
        "by-technique.md": generate_by_technique,
        "by-hardware-feature.md": generate_by_hardware_feature,
        "by-repo.md": generate_by_repo,
        "by-kernel-type.md": generate_by_kernel_type,
        "by-language.md": generate_by_language,
    }

    for filename, gen_func in generators.items():
        content = gen_func(pages)
        outpath = QUERIES_DIR / filename
        outpath.write_text(content)
        print(f"  Generated {outpath.relative_to(REPO_ROOT)}")

    print("Done.")


if __name__ == "__main__":
    main()
