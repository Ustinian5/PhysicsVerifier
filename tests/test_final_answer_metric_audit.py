import unittest

from scripts.audit_final_answer_metrics import measure


class MetricTests(unittest.TestCase):
    def test_all_incorrect_is_not_a_useful_discriminator(self):
        gold = {'a':'incorrect', 'b':'correct'}
        predictions = {k:dict(status='complete',value={'verdict':'incorrect'}) for k in gold}
        result = measure(gold,predictions)
        self.assertAlmostEqual(result['original_binary_view']['f1'],2/3)
        self.assertEqual(result['binary_supplement']['balanced_accuracy'],.5)
        self.assertEqual(result['binary_supplement']['mcc'],0)

    def test_failure_does_not_disappear_from_whole_cohort_detection(self):
        gold = {'a':'incorrect','b':'incorrect','c':'correct'}
        predictions = {'a':dict(status='complete',value={'verdict':'incorrect'}),
                       'b':dict(status='content_failure'),
                       'c':dict(status='complete',value={'verdict':'uncertain'})}
        r = measure(gold,predictions)
        self.assertEqual(r['original_binary_view']['recall'],1)
        self.assertEqual(r['whole_cohort_view']['recall'],.5)
        self.assertEqual(r['counts']['TN'],0)
        self.assertEqual(r['counts']['failed_positive'],1)
        self.assertEqual(r['counts']['uncertain_negative'],1)

    def test_class_orientation_and_one_result_per_question(self):
        gold = dict(a='incorrect',b='correct',c='incorrect',d='correct')
        predictions = {k:dict(status='complete',value={'verdict':v})
                       for k,v in dict(a='incorrect',b='incorrect',c='correct',d='correct').items()}
        r=measure(gold,predictions)
        self.assertEqual([r['counts'][k] for k in ('TP','FP','FN','TN')],[1,1,1,1])
        self.assertEqual(r['original_binary_view'],dict(precision=.5,recall=.5,f1=.5))
        with self.assertRaises(ValueError): measure(gold,{k:v for k,v in predictions.items() if k!='a'})


if __name__ == '__main__': unittest.main()
