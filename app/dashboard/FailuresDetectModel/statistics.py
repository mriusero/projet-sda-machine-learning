import pandas as pd
import streamlit as st
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, f_oneway, chi2_contingency, pearsonr, wilcoxon


class StatisticalTests:
    def __init__(self, df):
        self.df = df

    def test_normality(self, column_name):
        """
        Test de normalité de Shapiro-Wilk
        """
        stat, p_value = shapiro(self.df[column_name].dropna())
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

    def test_anova(self, *args):
        """
        Test ANOVA (Analyse de la Variance) pour comparer les moyennes de plusieurs échantillons
        """
        groups = [self.df[arg] for arg in args]
        stat, p_value = f_oneway(*groups)
        alpha = 0.05
        return p_value > alpha, p_value

    def test_chi2(self, col1, col2):
        """
        Test du Chi-carré pour tester l'indépendance entre deux variables catégorielles
        """
        contingency_table = pd.crosstab(self.df[col1], self.df[col2])
        stat, p_value, dof, expected = chi2_contingency(contingency_table)
        alpha = 0.05
        return p_value > alpha, p_value

    def test_correlation(self, col1, col2):
        """
        Test de corrélation de Pearson pour mesurer la relation linéaire entre deux variables quantitatives
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        corr_coef, p_value = pearsonr(group1, group2)
        return corr_coef, p_value

    def test_wilcoxon(self, col1, col2):
        """
        Test de Wilcoxon pour comparer deux échantillons appariés
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        stat, p_value = wilcoxon(group1, group2)
        alpha = 0.05
        return p_value > alpha, p_value

def run_statistical_test(df, test_type, *args):
    tester = StatisticalTests(df)

    if test_type == 'normality':         # Test de normalité pour vérifier si la distribution des données suit une loi normale.
        result, p_value = tester.test_normality(args[0])
        return st.write(f"Normality Test on {args[0]} - p-value: {p_value}, Result: {'Normal' if result else 'Not Normal'}")

    elif test_type == 'ttest':          # Test T (Student) pour comparer les moyennes de deux échantillons indépendants ou appariés.
        result, p_value = tester.test_t_test(args[0], args[1])
        return st.write(f"T-Test between {args[0]} and {args[1]} - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'mannwhitney':        # Test de Mann-Whitney pour comparer les distributions de deux échantillons indépendants.
        result, p_value = tester.test_mannwhitney(args[0], args[1])
        return st.write(f"Mann-Whitney Test between {args[0]} and {args[1]} - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'anova':              # Test ANOVA (Analyse de la Variance) pour comparer les moyennes de trois échantillons ou plus.
        result, p_value = tester.test_anova(*args)
        return st.write(f"ANOVA Test - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'chi2':               # Test du Chi-carré pour tester l'indépendance entre deux variables catégorielles.
        result, p_value = tester.test_chi2(args[0], args[1])
        return st.write(f"Chi-Square Test between {args[0]} and {args[1]} - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'correlation':         # Test de corrélation (Pearson) pour mesurer la force et la direction de la relation linéaire entre deux variables.
        corr_coef, p_value = tester.test_correlation(args[0], args[1])
        return st.write(f"Correlation Test between {args[0]} and {args[1]} - p-value: {p_value}, Correlation coefficient: {corr_coef}")

    elif test_type == 'wilcoxon':           # Test de Wilcoxon pour comparer les paires de valeurs lorsque la distribution n'est pas normale.
        result, p_value = tester.test_wilcoxon(args[0], args[1])
        return st.write(f"Wilcoxon Test - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    else:
        return st.write("Unknown test type")
