"""Resumable Elsevier/Scopus acquisition without embedded credentials."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET

from agew_heabert.config import require_environment
from agew_heabert.stage_utils import check_result

SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
ARTICLE_URL = "https://api.elsevier.com/content/article/doi/"


def _safe_filename(title: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", title).strip(" ._")
    return cleaned[:180] or "untitled"


def _load_progress(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        return {"successful_downloads": [], "failed_downloads": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("progress file must contain an object")
    data.setdefault("successful_downloads", [])
    data.setdefault("failed_downloads", [])
    return data


def _save_progress(path: Path, progress: dict[str, Any]) -> None:
    path.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")


def run(config: dict[str, Any], *, check_only: bool = False) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    if check_only:
        return check_result(
            "acquire-literature",
            [output_dir],
            credential="ELSEVIER_API_KEY",
            note="Downloaded XML is licensed material and is excluded from this repository.",
        )

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Literature acquisition requires the 'extraction' dependency group"
        ) from exc

    api_key = require_environment("ELSEVIER_API_KEY")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "download_progress.json"
    progress = _load_progress(progress_path)
    completed = {item["doi"] for item in progress["successful_downloads"]}
    session = requests.Session()
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    count = int(config.get("page_size", 100))
    limit = int(config.get("limit", 100))
    start = 0

    while start < limit:
        response = session.get(
            SCOPUS_SEARCH_URL,
            headers=headers,
            params={"query": config["query"], "count": min(count, limit - start), "start": start},
            timeout=60,
        )
        response.raise_for_status()
        entries = response.json().get("search-results", {}).get("entry", [])
        if not entries:
            break
        for entry in entries:
            doi = entry.get("prism:doi")
            title = entry.get("dc:title", "Untitled")
            if not doi or doi in completed:
                continue
            try:
                article = session.get(
                    ARTICLE_URL + doi,
                    headers={"X-ELS-APIKey": api_key, "Accept": "application/xml"},
                    timeout=120,
                )
                article.raise_for_status()
                ET.fromstring(article.content)
                stem = _safe_filename(str(title))
                (output_dir / f"{stem}.xml").write_bytes(article.content)
                metadata = {"doi": doi, "title": title, "source": "Elsevier"}
                (output_dir / f"{stem}.metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                progress["successful_downloads"].append(metadata)
                completed.add(doi)
            except Exception as exc:  # preserve partial success and resumability
                progress["failed_downloads"].append({"doi": doi, "title": title, "error": str(exc)})
            _save_progress(progress_path, progress)
            time.sleep(float(config.get("request_delay_seconds", 0.5)))
        start += len(entries)
    return {"stage": "acquire-literature", "status": "completed", "progress": str(progress_path)}
