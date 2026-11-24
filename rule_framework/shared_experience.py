"""Locate reviewed RAG and skills materials or the preceding frozen three-library release."""
from pathlib import Path

from rule_framework import verified_resume as vr
from rule_framework.flat_skill_rag import require

DIRECTORY = "shared_experience"
PROTOCOL = "shared_first_review_20261107"
MATERIALS = ("cases.json", "skills.json", "catalog.json")
RELEASE_SHA256 = "e088c9322e74d0b263de7a009bc1df40e614b5eb95863e839687fce74daf9bf8"
REVIEWED_DIRECTORY = "library_source_rerun_20261124"
REVIEWED_FLAT_SHA256 = "348f50eb546219820b35fdb4097265c721802316def05fff59206022a527f85d"


def release_index(directory):
    path = directory / "release.json"
    require(vr.sha(path) == RELEASE_SHA256, "Different first-review release")
    return vr.unseal(path)


def library_path(results_root, name="n2000"):
    root = Path(results_root) / DIRECTORY
    index = release_index(root)
    require(index["protocol"] == PROTOCOL and name in index["libraries"], "Unknown shared-review library")
    require(vr.sha(root / "source_reviews.json") == index["source_reviews_sha256"], "Source reviews changed")
    directory = root / name
    receipt = directory / "construction_acceptance.json"
    require(vr.sha(receipt) == index["libraries"][name]["acceptance_sha256"], "Library receipt changed")
    accepted = vr.read(receipt)
    require(accepted["passed"] and accepted["protocol"] == PROTOCOL and accepted["name"] == name
            and accepted["source_reviews_sha256"] == index["source_reviews_sha256"]
            and set(accepted["materials"]) == set(MATERIALS), "Wrong library or review pool")
    require(vr.sha(directory / "sources.json") == accepted["sources_sha256"]
            and vr.sha(directory / "material_isolation.json") == accepted["material_isolation_sha256"]
            and all(vr.sha(directory / "materials" / n) == h for n, h in accepted["materials"].items()),
            "Published library files changed")
    return directory


def reviewed_flat_library(results_root, name="n2000"):
    """Locate just the two regenerated arms; the rule bundle is an organizer input."""
    work = Path(results_root) / REVIEWED_DIRECTORY
    root = work / "flat_reviewed"
    index_path = root / "flat_release.json"
    require(vr.sha(index_path) == REVIEWED_FLAT_SHA256, "Different reviewed flat release")
    index = vr.read(index_path)
    require(index["source_mode"] == "reviewed" and index["released_materials"] == ["cases.json", "skills.json"]
            and name in index["groups"], "Unknown reviewed flat library")
    replay = work / "source_replay.json"
    require(vr.sha(replay) == index["source_replay_sha256"], "Source replay changed")
    require(vr.sha(work / "source_reviews.json") == vr.unseal(replay)["files"]["source_reviews.json"],
            "First-review evidence changed")
    directory = root / name
    receipt = directory / "flat_acceptance.json"
    group = index["groups"][name]
    require(vr.sha(receipt) == group["acceptance_sha256"] and group["name"] == name
            and vr.read(receipt) == {k: v for k, v in group.items() if k != "acceptance_sha256"},
            "Wrong flat library or receipt")
    require(all(vr.sha(directory / file) == expected for file, expected in group["files"].items()),
            "Flat library files changed")
    return directory
