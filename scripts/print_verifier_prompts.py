"""Print the current English verifier messages for human review, without model calls or dataset access."""
import json

from rule_framework import step_match_lines, bounded_tree, final_answer_revision as final, flat_skill_rag as step
from rule_framework import organization_followup, semantic_index_retrieval as selection, verifier_english as en
from scripts import continuous_context_v2 as quotes, pilot_retrieval_fusion as fusion
from scripts import rebuild_qwen30b_corpus as solver, run_bench_reannotation as annotator
from scripts import run_combined_source_screen as screen, run_corrected_extraction as build
from scripts import run_timed_semantic_retrieval as timed, replay_retrieval_rank_role as rank_role


def render():
    row = dict(id='example', question='A cart travels 12 m in 3 s at constant speed. Find its speed.',
               prediction='The speed is 3 m/s.', context='')
    answer = ['4 m/s']
    taxonomy = {'domains': [dict(id='mechanics', name='Mechanics', summary='Forces and motion',
                                 topics=[dict(id='kinematics', name='Kinematics', summary='Motion over time')])]}
    profile = dict(prompt_profile=en.PROFILE, audit_system_prompt=build.AUDIT_PROMPT_FILE.read_text().rstrip('\n'))
    original = dict(sample=dict(row, official_reference={'label': answer}), taxonomy=taxonomy, max_rules_per_sample=2)
    audit = {'status':'error_found', 'summary':'The response multiplied current and voltage to obtain resistance.'}
    case = dict(id='example_case', title='Resistance from current and voltage',
                text=en.case_text('A resistor carries 2 A at 3 V. Find its resistance.', 'Its resistance is 6 ohms.', audit))
    queries = {'queries':['constant speed distance elapsed time']}
    lines = ['# English verifier prompts', '',
             'These are the messages built by the current request functions for an illustrative physics problem. '
             'Repeated conversation prefixes appear once. This document can be regenerated offline with '
             '`python -m scripts.print_verifier_prompts` in the physicsverifier conda environment.', '',
             'For both tasks the pure 120B reads the problem and the student response and checks them directly. '
             'RAG, flat skills and the rule tree share one English PREPARE, retrieve from their own libraries, '
             'and CHECK in that conversation. All four use the same student response and scoring criteria. '
             'The official answer guides the first 120B review of each construction problem. All four online checkers '
             'are blind to the test problem\'s official answer and benchmark labels. Only the three knowledge arms '
             'receive historical material from that review process. Scoring uses the frozen benchmark annotations. '
             'The annotation examples below are for development; this evaluation reuses the published benchmark.', '']

    def show(title, messages):
        lines.extend(['## ' + title, ''])
        for message in messages:
            lines.extend([message['role'].capitalize(), '', '````text', message['content'].rstrip('\n'), '````', ''])

    show('Source eligibility', screen.screen_messages([row]))
    show('30B solution', solver.solver_messages(row))
    show('Catalog summaries', build.request_messages('taxonomy', taxonomy, profile))
    show('First 120B review and reusable rules', build.request_messages('extract_experience', original, profile))
    repair = dict(original=original, draft={'_unparsed_response_text':'<previous draft>'},
                  validation_error='Invalid source error ID')
    show('Structural repair after the same first review', build.request_messages('repair_experience', repair, profile)[2:])
    show('Skills from the first-review case used by RAG', organization_followup.skill_messages(case, english=True))
    rule = dict(id='I1', title='Check mean speed', trigger='Travel time and distance are given',
                check_logic='Compare the committed speed with distance divided by elapsed time.', error_type='calculation')
    for stage, payload in [
        ('route_topics', dict(items=[rule], topics=bounded_tree.topic_list(taxonomy))),
        ('scan_groups', dict(items=[rule], groups=[], previous_best=[], best_by_item={'I1':None})),
        ('place_groups', dict(items=[rule], groups=[])),
    ]:
        show('Rule tree / ' + stage, bounded_tree.messages(stage, payload, english=True))
    show('Missing tree decisions after the preserved draft', bounded_tree.messages('complete_missing', dict(
        stage='route_topics', original=dict(items=[rule], topics=bounded_tree.topic_list(taxonomy)),
        draft={'decisions':[]}, missing=['I1']), english=True)[2:])
    show('Independent step annotations / shared system', [dict(role='instructions', content=annotator.annotation_system(profile))])
    annotation = dict(row, answer=json.dumps(answer))
    for stage in annotator.STAGES:
        previous = {name:'<previous annotation>' for name in
                    ({'review':('initial','blind'), 'final':('review',)}.get(stage, ()))}
        show('Step annotation / ' + stage, [dict(role='input', content=annotator.canonical(
            annotator.task_body(stage, annotation, previous)))])
    show('Independent final-answer labels', final.original.label_messages(row, answer))
    show('Independent final-answer label review', final.review_messages(row, answer))
    for name, module in [('Step',step), ('Final answer',final)]:
        show(name + ' / pure direct check', module.direct_messages(row))
        show(name + ' / shared PREPARE for the three knowledge arms', module.query_messages(row, english=True))
        item = dict(id='S1', title='Mean speed', text='Check distance divided by elapsed time.')
        show(name + ' / skills or rule-tree CHECK', module.continuation_messages(row, queries, [item], arm='flat_skills', english=True)[2:])
        rag = module.continuation_messages(row, queries, [case], arm='case_rag', english=True)
        show(name + ' / CHECK with a historical RAG case', rag[-1:])
    show('Step evidence-quote repair', [dict(role='user', content=quotes.repair_message(
        {'diagnostics':[{'evidence':{'quote':'3 metres per second'}}]}, [0], english=True))])
    judge_row = dict(row, roots=[dict(root_id='E1', claim='The numerical speed is incorrect.',
        reason='12 m divided by 3 s gives 4 m/s.', correct_relation='v = d/t',
        premises='The given speed is constant.', quote=row['prediction'])])
    show('Independent post-publication step matching with original line numbers', step_match_lines.messages(
        judge_row, dict(message='The stated speed is inconsistent with distance divided by time.',
                        evidence={'quote': row['prediction']})))
    prefix = final.query_messages(row, english=True) + [dict(role='assistant', content=json.dumps(queries))]
    show('Retrieval development / English query rewrite', fusion.rewrite_messages(row,prefix,english=True)[-1:])
    show('Semantic candidate ranking / page or merge', timed.model_text(selection.RANK+'24.',
         dict(query=queries['queries'], candidates=[[1,'Mean speed','Check distance and travel time.']])))
    show('Semantic tree routing', timed.model_text(selection.ROUTE,
         dict(query=queries['queries'], children=[[1,'Mechanics','Kinematics','Motion over time']])))
    show('Fusion reranking', timed.model_text(selection.RANK+'6.', dict(query=queries['queries'], candidates=[])))
    show('Development candidate-pool reranking', [dict(role='system',content=rank_role.ROLE),
                                                 dict(role='user',content=fusion.RANK)])
    return '\n'.join(lines)


if __name__ == '__main__':
    print(render(), end='')
