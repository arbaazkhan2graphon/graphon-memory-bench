"""Fetch the LOCOMO and LongMemEval-S datasets to ``data/``.

Sources:
  * LOCOMO  -- ``locomo10.json`` from github.com/snap-research/locomo, pinned
    to the commit SHA resolved at download time (recorded in the manifest).
  * LongMemEval-S -- official cleaned release on HuggingFace
    (``xiaowu0162/longmemeval-cleaned``).

Datasets are NOT redistributed with this harness; this script documents and
reproduces the expected local layout:

    data/
      locomo10.json
      longmemeval_s.json
      manifest.json          # pinned SHAs / URLs / sizes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

try:
    import certifi

    _VERIFY = certifi.where()
except ImportError:  # pragma: no cover
    _VERIFY = True

logger = logging.getLogger("membench_download")

PROJECT_DIR = Path(__file__).resolve().parent

GITHUB_API = "https://api.github.com/repos/{repo}/commits/{ref}"
GITHUB_RAW = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# Candidate file names for the LongMemEval-S release (the cleaned repo has
# used a couple of layouts over time; first hit wins).
LME_CANDIDATES = [
    "longmemeval_s_cleaned.json",
    "longmemeval_s.json",
    "data/longmemeval_s_cleaned.json",
]


def _get(url: str, attempts: int = 4, timeout: int = 300) -> requests.Response:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, timeout=timeout, verify=_VERIFY, headers={
                "User-Agent": "graphon-memory-bench",
            })
            if resp.status_code in (429, 502, 503) and attempt < attempts:
                wait = 10 * attempt
                logger.warning("HTTP %s from %s; retrying in %ss", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # noqa: PERF203
            last = exc
            if attempt < attempts:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to GET {url}: {last}")


def download_locomo(cfg: dict, data_dir: Path, force: bool) -> dict:
    lcfg = cfg["data"]["locomo"]
    out = data_dir / "locomo10.json"
    if out.exists() and not force:
        logger.info("LOCOMO already present: %s", out)
        return {"file": str(out), "skipped": True}

    sha_resp = _get(GITHUB_API.format(repo=lcfg["repo"], ref=lcfg.get("ref", "main")))
    sha = sha_resp.json()["sha"]
    url = GITHUB_RAW.format(repo=lcfg["repo"], sha=sha, path=lcfg["file"])
    logger.info("Fetching LOCOMO @ %s", sha[:12])
    resp = _get(url)
    out.write_bytes(resp.content)
    data = json.loads(resp.content)
    n_qa = sum(len(s.get("qa", [])) for s in data)
    logger.info("LOCOMO: %d conversations, %d QA pairs -> %s", len(data), n_qa, out)
    return {"file": str(out), "url": url, "sha": sha, "conversations": len(data), "qa_pairs": n_qa}


def download_longmemeval(cfg: dict, data_dir: Path, force: bool) -> dict:
    mcfg = cfg["data"]["longmemeval"]
    out = data_dir / "longmemeval_s.json"
    if out.exists() and not force:
        logger.info("LongMemEval-S already present: %s", out)
        return {"file": str(out), "skipped": True}

    candidates = [mcfg["file"]] + [c for c in LME_CANDIDATES if c != mcfg["file"]]
    last_err = ""
    for cand in candidates:
        url = HF_RESOLVE.format(repo=mcfg["hf_repo"], path=cand)
        try:
            logger.info("Trying %s", url)
            resp = _get(url)
        except RuntimeError as exc:
            last_err = str(exc)
            continue
        out.write_bytes(resp.content)
        data = json.loads(resp.content)
        logger.info("LongMemEval-S: %d questions -> %s", len(data), out)
        return {"file": str(out), "url": url, "questions": len(data)}
    raise RuntimeError(f"Could not fetch LongMemEval-S from any candidate path: {last_err}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load((PROJECT_DIR / "config.yaml").read_text(encoding="utf-8"))
    data_dir = PROJECT_DIR / cfg["data"]["dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest["locomo"] = download_locomo(cfg, data_dir, args.force)
    manifest["longmemeval"] = download_longmemeval(cfg, data_dir, args.force)
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest written to %s", data_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
