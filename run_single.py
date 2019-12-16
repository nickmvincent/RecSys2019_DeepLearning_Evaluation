from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from pprint import pprint
from Conferences.SIGIR.CMN_our_interface.Pinterest.PinterestICCVReader import PinterestICCVReader

dataset_name = 'pinterest'
for scenario in [False, '0.1_random0_large']:


    if dataset_name == "pinterest":
        dataset = PinterestICCVReader(scenario)

    output_folder_path = "result_experiments/temp.csv"


    URM_train = dataset.URM_train.copy()
    #URM_validation = dataset.URM_validation.copy()
    URM_test = dataset.URM_test.copy()
    URM_test_negative = dataset.URM_test_negative.copy()

    model = ItemKNNCFRecommender(URM_train)
    model.fit(topK=800, shrink=1000, similarity='cosine', normalize=True, feature_weighting='BM25')
    # {'topK': 800, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'}

    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    #evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=[5])
    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[5, 10])

    #print(evaluator_validation.evaluateRecommender(model))
    print(scenario)
    ret = evaluator_test.evaluateRecommender(model)[0]
    print(ret[10]['HIT_RATE'])