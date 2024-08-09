from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_models(train_item):
    """
    Entrainement des models :
    - Linear Regression for 'rul (months)'
    - Linear Regression for 'Time to failure (months)'
    - RandomForestClassifier for 'Failure mode'
    """
    X_train = train_item[['time (months)', 'crack length (arbitary unit)']]
    y_train_rul = train_item['rul (months)']
    y_train_ttf = train_item['Time to failure (months)']
    y_train_failure_mode = train_item['Failure mode']

    # Modèle de régression pour RUL
    model_rul = make_pipeline(StandardScaler(), LinearRegression())
    model_rul.fit(X_train, y_train_rul)

    # Modèle de régression pour Time to Failure
    model_ttf = make_pipeline(StandardScaler(), LinearRegression())
    model_ttf.fit(X_train, y_train_ttf)

    # Modèle de classification pour Failure Mode
    le_failure_mode = LabelEncoder()
    y_train_failure_mode_encoded = le_failure_mode.fit_transform(y_train_failure_mode)

    model_failure_mode = make_pipeline(StandardScaler(), RandomForestClassifier())
    model_failure_mode.fit(X_train, y_train_failure_mode_encoded)

    return model_rul, model_ttf, model_failure_mode, le_failure_mode







