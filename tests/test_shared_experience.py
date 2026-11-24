import json

import pytest

from rule_framework import shared_experience as shared
from rule_framework import verified_resume as vr
from scripts.release_shared_experience import source_case
from scripts import rerun_reviewed_libraries as reviewed


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


def test_only_an_adjudicated_source_may_ignore_the_missing_parent_phrase():
    source = 'reviewed'
    sentinel = dict(heldout_id='unresolved_parent', reason='dependent_question_parent_not_resolved')
    reviewed.check_matches([sentinel], source, {source})
    with pytest.raises(ValueError, match='Held-out evidence'):
        reviewed.check_matches([sentinel], 'other', {source})
    for hit in [dict(heldout_id='final:q', reason='numeric_template'),
                dict(heldout_id='unresolved_parent', reason='shared_subproblem_prose')]:
        with pytest.raises(ValueError, match='Held-out evidence'):
            reviewed.check_matches([sentinel, hit], source, {source})


def test_rag_reconstruction_preserves_the_first_review_even_if_it_is_wrong():
    sample = dict(id='q', question='物体的高度？', prediction='2 米', answer='', context='')
    original = dict(status='error_found', key_errors=[dict(id='e', reason='2.48 米')])
    case = reviewed.rebuild_case(sample, original)
    source_case(case, sample, original)
    with pytest.raises(ValueError, match='first review'):
        source_case(case, sample, dict(status='no_error_found', key_errors=[]))


def test_projecting_exactly_deduplicated_skills_keeps_only_in_draw_sources():
    skill = dict(id='s', source_ids=['a', 'b'], title='method', text='procedure')
    original = [skill, dict(id='other', source_ids=['c'])]
    assert reviewed.project_skills(original, {'b'}) == [dict(skill, source_ids=['b'])]
    assert skill['source_ids'] == ['a', 'b']


def test_reviewed_flat_lookup_checks_both_source_and_material_hashes(tmp_path, monkeypatch):
    work = tmp_path / shared.REVIEWED_DIRECTORY
    root = work / 'flat_reviewed'
    group = root / 'n2000'
    vr.atomic(work / 'source_reviews.json', {'q': {'audit_sha256': 'first'}})
    vr.seal(work / 'source_replay.json', dict(files={'source_reviews.json': vr.sha(work / 'source_reviews.json')}))
    files = {}
    for name in ('materials/cases.json', 'materials/skills.json', 'group_sources.json', 'tree_bundle.json'):
        vr.atomic(group / name, [])
        files[name] = vr.sha(group / name)
    entry = dict(name='n2000', files=files)
    vr.atomic(group / 'flat_acceptance.json', entry)
    index = root / 'flat_release.json'
    vr.atomic(index, dict(source_mode='reviewed', released_materials=['cases.json', 'skills.json'],
                          source_replay_sha256=vr.sha(work / 'source_replay.json'),
                          groups={'n2000': dict(entry, acceptance_sha256=vr.sha(group / 'flat_acceptance.json'))}))
    monkeypatch.setattr(shared, 'REVIEWED_FLAT_SHA256', vr.sha(index))
    assert shared.reviewed_flat_library(tmp_path) == group
    vr.atomic(group / 'materials/skills.json', ['independent second review'])
    with pytest.raises(ValueError, match='Flat library files changed'):
        shared.reviewed_flat_library(tmp_path)
    vr.atomic(work / 'source_reviews.json', {})
    with pytest.raises(ValueError, match='First-review evidence changed'):
        shared.reviewed_flat_library(tmp_path)
