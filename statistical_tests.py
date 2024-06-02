from scipy.stats import ttest_rel
import pandas as pd

def compare_classifiers(scores1, scores2, metric='accuracy', label1='Set 1', label2='Set 2'):
    """
    Compare classifiers using paired t-test on the specified metric.

    :param scores1: A dictionary of scores for each classifier from the first dataset.
    :type scores1: dict
    :param scores2: A dictionary of scores for each classifier from the second dataset.
    :type scores2: dict
    :param metric: The performance metric to compare (default is 'accuracy').
    :type metric: str
    :param label1: Label for the first set of scores.
    :type label1: str
    :param label2: Label for the second set of scores.
    :type label2: str
    :return: A DataFrame with p-values of the paired t-tests between classifiers.
    :rtype: pandas.DataFrame
    """
    classifiers = list(scores1.keys())
    comparisons = []

    for clf in classifiers:
        score1 = scores1[clf][metric]
        score2 = scores2[clf][metric]
        t_stat, p_value = ttest_rel(score1, score2)
        comparisons.append({
            'Classifier': clf,
            f'{label1} {metric}': score1,
            f'{label2} {metric}': score2,
            't-statistic': t_stat,
            'p-value': p_value
        })

    return pd.DataFrame(comparisons)
