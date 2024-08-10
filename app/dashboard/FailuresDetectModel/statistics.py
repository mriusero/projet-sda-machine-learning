import pandas as pd
from scipy.stats import shapiro, ttest_ind, mannwhitneyu


class StatisticalTests:
    def __init__(self, df):
        self.df = df

    def test_normality(self, column_name):
        """
        Test de normalité de Shapiro-Wilk
        """
        stat, p_value = shapiro(self.df[column_name])
        alpha = 0.05
        return p_value > alpha, p_value

    def test_t_test(self, col1, col2):
        """
        Test t de Student pour deux échantillons indépendants
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        stat, p_value = ttest_ind(group1, group2)
        alpha = 0.05
        return p_value > alpha, p_value

    def test_mannwhitney(self, col1, col2):
        """
        Test de Mann-Whitney pour deux échantillons indépendants
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        stat, p_value = mannwhitneyu(group1, group2)
        alpha = 0.05
        return p_value > alpha, p_value


def run_statistical_test(df, test_type, *args):
    tester = StatisticalTests(df)

    if test_type == 'normality':
        result, p_value = tester.test_normality(args[0])
        return f"Normality Test - p-value: {p_value}, Result: {'Normal' if result else 'Not Normal'}"

    elif test_type == 'ttest':
        result, p_value = tester.test_t_test(args[0], args[1])
        return f"T-Test - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}"

    elif test_type == 'mannwhitney':
        result, p_value = tester.test_mannwhitney(args[0], args[1])
        return f"Mann-Whitney Test - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}"

    else:
        return "Unknown test type"
