"""Read a verifier library whose three arms share one frozen first-review record."""
from pathlib import Path

from rule_framework import verified_resume as vr
from rule_framework.flat_skill_rag import require

DIRECTORY = "shared_experience"
PROTOCOL = "shared_first_review_20261107"
MATERIALS = ("cases.json", "skills.json", "catalog.json")
RELEASE_SHA256 = "e088c9322e74d0b263de7a009bc1df40e614b5eb95863e839687fce74daf9bf8"


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
