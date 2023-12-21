import numpy as np


def prettifier(res: dict):
    """training result"""
    print("=" * 50)
    try:
        size = len(res["Acc"])
        print(f'Avg. ACC: {sum(res["Acc"])/size}')
        print(f'Avg. TPR: {sum(res["TPR"])/size}')
        print(f'Avg. TNR: {sum(res["TNR"])/size}')
        print(f'Avg. number of selected features: {sum(res["n_feature"])/size}')
        print(f'std. number of selected features: {np.std(res["n_feature"])}')
    except:
        print("Not valid result for feature selection")
    finally:
        print("=" * 50)
