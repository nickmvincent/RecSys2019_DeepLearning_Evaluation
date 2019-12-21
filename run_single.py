from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects

from pprint import pprint
from Conferences.SIGIR.CMN_our_interface.Pinterest.PinterestICCVReader import PinterestICCVReader
from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample
import pandas as pd


dataset_name = 'pinterest'
row_dicts = []

configs = []
fracs = [0.01, 0.05, .1, .2, 0.3, .4, .5]
seeds = [0,1,3]
for frac in fracs:
    for seed in seeds:
        configs.append({
            'frac': frac,
            'seed': seed,
            'recsys': ItemKNNCFRecommender,
            'company': 'large',
            'scenario': f'{frac}_random{seed}_large'
        })

        configs.append({
            'frac': frac,
            'seed': seed,
            'recsys': ItemKNNCFRecommender,
            'company': 'small',
            'scenario': f'{frac}_random{seed}_small'
        })

configs.append({
    'frac': None,
    'seed': None,
    'recsys': ItemKNNCFRecommender,
    'company': 'large',
    'scenario': 'full_itemknn',

})

configs.append({
    'frac': None,
    'seed': None,
    'recsys': TopPop,
    'company': 'large',
    'scenario': 'full_toppop',
})

configs.append({
    'frac': None,
    'seed': None,
    'recsys': Random,
    'company': 'large',
    'scenario': 'full_toppop',
})

for config in configs:
    scenario = config['scenario']

    if dataset_name == "pinterest":
        path = False
        if 'full' not in scenario:
            path = scenario
        dataset = PinterestICCVReader(path)
    else:
        raise ValueError('not yet implemented')

    URM_train = dataset.URM_train.copy()
    #URM_validation = dataset.URM_validation.copy()
    URM_test = dataset.URM_test.copy()
    URM_test_negative = dataset.URM_test_negative.copy()

    model = config['recsys'](URM_train)
    try:
        model.loadModel('model_cache/', scenario)
    except Exception as e:
        print(e)
        model.fit(topK=800, shrink=1000, similarity='cosine', normalize=True, feature_weighting='BM25')
        model.saveModel('model_cache/', scenario)
    # {'topK': 800, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'}


    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[5, 10])

    print(scenario)
    ret = evaluator_test.evaluateRecommender(model)[0]
    print()
    row_dict = config.copy()
    row_dict['hitrate5'] =  ret[5]['HIT_RATE']
    row_dict['ndcg5'] =  ret[5]['NDCG']
    row_dicts.append(row_dict)

res = pd.DataFrame(row_dicts)
res.to_csv('itemknn_pinterest_rows.csv', index=None)