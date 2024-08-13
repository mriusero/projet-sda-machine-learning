import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.model_selection import learning_curve
import scipy.stats as stats

class DisplayData:
    def __init__(self, data):
        self.data = data

    def plot_correlation_matrix(self):
        corr = self.data.corr()
        fig = px.imshow(corr, color_continuous_scale='Viridis', text_auto=True)
        fig.update_layout(title='Correlation Matrix', title_x=0.5)
        return fig

    def plot_learning_curve(self, model, X, y, cv=5):
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode='lines+markers', name='Training score',
                                 line=dict(color='red')))
        fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean, mode='lines+markers', name='Cross-validation score',
                                 line=dict(color='blue')))
        fig.update_layout(title='Learning Curve', xaxis_title='Training Size', yaxis_title='Score')
        return fig

    def plot_scatter_real_vs_predicted(self, y_test, predictions):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=y_test, y=predictions, mode='markers', marker=dict(color='cyan'), name='Predicted vs Actual'))
        fig.add_trace(
            go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', line=dict(color='red'),
                       name='Ideal Line'))
        fig.update_layout(title='Scatter Plot of Real vs Predicted', xaxis_title='Real Values',
                          yaxis_title='Predicted Values')
        return fig

    def plot_residuals_vs_predicted(self, y_test, predictions):
        residuals = y_test - predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predictions, y=residuals, mode='markers', marker=dict(color='cyan'),
                                 name='Residuals vs Predicted'))
        fig.add_trace(go.Scatter(x=[min(predictions), max(predictions)], y=[0, 0], mode='lines', line=dict(color='red'),
                                 name='Zero Line'))
        fig.update_layout(title='Residuals vs Predicted Values', xaxis_title='Predicted Values',
                          yaxis_title='Residuals')
        return fig

    def plot_histogram_of_residuals(self, residuals):
        fig = px.histogram(residuals, nbins=30, color_discrete_sequence=['green'])
        fig.update_layout(title='Histogram of Residuals', xaxis_title='Residuals', yaxis_title='Frequency')
        return fig

    def plot_density_curve_of_residuals(self, residuals):
        fig = px.density_contour(x=residuals, color_discrete_sequence=['green'])
        fig.update_layout(title='Density Curve of Residuals', xaxis_title='Residuals', yaxis_title='Density')
        return fig

    def plot_qq_diagram(self, residuals):
        fig = go.Figure()

        # Calculer le QQ plot
        qq = stats.probplot(residuals, dist="norm", plot=None)
        x = qq[0][0]  # Quantiles théoriques
        y = qq[0][1]  # Quantiles observés

        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='QQ Plot'))
        fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[min(x), max(x)], mode='lines', line=dict(color='red'),
                                 name='Line of Equality'))

        fig.update_layout(title='QQ Plot des résidus', xaxis_title='Quantiles Théoriques',
                          yaxis_title='Quantiles Observés')

        return fig

