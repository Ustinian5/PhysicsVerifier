import json

import pytest

from rule_framework import shared_experience as shared
from rule_framework import verified_resume as vr
from scripts.release_shared_experience import source_case


def test_same_question_and_answer_do_not_make_an_independent_review_equivalent():
    sample = dict(id="q", question="Question", prediction="30B answer", answer="", context="")
    audit = {"status": "error_found", "key_errors": [{"id": "E1", "message": "first diagnosis"}]}
    case = dict(id="case_q", source_ids=["q"], text="历史题目\nQuestion\n\n原30B回答（可能有错，不是正确范例）\n30B answer"
                + "\n\n120B原始核查说明\n" + json.dumps(audit))
    source_case(case, sample, audit)
    other = {"status": "error_found", "key_errors": [{"id": "E1", "message": "a different diagnosis"}]}
    with pytest.raises(ValueError, match="first review"):
        source_case(case, sample, other)
    with pytest.raises(ValueError, match="first review"):
        source_case(case, dict(sample, prediction="another answer"), audit)


def test_lookup_rejects_an_old_flat_library_substituted_after_release(tmp_path, monkeypatch):
    directory = tmp_path / shared.DIRECTORY
    library = directory / "n2000"
    for filename in shared.MATERIALS:
        vr.atomic(library / "materials" / filename, [])
    vr.atomic(directory / "source_reviews.json", {"q": "first"})
    vr.atomic(library / "sources.json", {"draw": ["q"]})
    vr.atomic(library / "material_isolation.json", {"passed": True})
    digest = vr.sha(directory / "source_reviews.json")
    receipt = library / "construction_acceptance.json"
    vr.atomic(receipt, dict(passed=True, protocol=shared.PROTOCOL, name="n2000", source_reviews_sha256=digest,
                            sources_sha256=vr.sha(library / "sources.json"),
                            material_isolation_sha256=vr.sha(library / "material_isolation.json"),
                            materials={n: vr.sha(library / "materials" / n) for n in shared.MATERIALS}))
    vr.seal(directory / "release.json", dict(protocol=shared.PROTOCOL, source_reviews_sha256=digest,
            libraries={"n2000": {"acceptance_sha256": vr.sha(receipt)}}))
    monkeypatch.setattr(shared, "RELEASE_SHA256", vr.sha(directory / "release.json"))
    assert shared.library_path(tmp_path) == library
    vr.atomic(library / "materials/skills.json", [{"id": "old-skill", "source_ids": ["q"]}])
    with pytest.raises(ValueError, match="files changed"):
        shared.library_path(tmp_path)
    vr.seal(directory / "release.json", dict(protocol=shared.PROTOCOL, libraries={}))
    with pytest.raises(ValueError, match="Different first-review release"):
        shared.release_index(directory)
