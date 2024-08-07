import streamlit as st


def page_0():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)
    st.markdown('<div class="header">Marius Ayrault - SDA 2024/2025</div>', unsafe_allow_html=True)
    st.text("------------------------------------------------------------------------------------------------------------------------")
    context = """ 
# Phase I : Remaining Useful Life (RUL)_
--> https://www.kaggle.com/competitions/predictive-maintenance-for-industrial-robots-i

## I.1) Context
The purpose consist to predict remaining useful life (RUL) of an industrial robot based on monitored data from three failure modes.
The test dataset consisting of 50 robots, each measured for crack growth at a monthly frequency. The purpose consist to predict the remaining useful life (RUL) of each robot in the test set and decide whether the robot will survive the next 6 months.

Based on historical knowledge, the robot has three main failure modes:

- **Infant Mortality** : *Failures due to manufacturing defects that occur very early in the robot’s life.*
- **Failure of Control Boards** : *Random failures described by a probabilistic distribution.*
- **Fatigue Crack Growth** : *Failures due to the growth of cracks over time. (measured every month).*

**Target:** Predict if each robot can survive the next 6 months

## I.2) Knowledges
### a) Physical model for the crack growth process

On suppose que le processus de croissance des fissures suit une fonction logistique :

$$
y(t) = {β2} / {1 + e^{-(β0 + β1 t)}}
$$

où $y(t)$ est la longueur de la fissure en unités arbitraires, $t$ est le temps en mois et $β0$, $β1$, $β2$ sont des paramètres liés aux propriétés du matériau.

Une défaillance se produit lorsque

$$
y(t) > y_{th}
$$

où
$y_{th} = 0.85$.


### b) Process noise, observation noise, and state space models


En raison du bruit de processus, les paramètres $β0$, $β1$, $β2$ peuvent légèrement varier au fil du temps. De plus, en raison des limitations de l'équipement de mesure, la longueur de fissure mesurée est affectée par un bruit de mesure important.

La longueur de fissure est mesurée tous les $1$ mois. Le modèle d'espace d'état suivant est utilisé pour capturer l'incertitude du processus et de l'observation :

$$
z_k = y_k + \epsilon_k
$$

where

$$
y_k = {β{2,k}} / {1 + e^{-(β{0,k} + β{1,k} t_k)}}
$$

and

$$
β{2,k} = β{2,k-1} + \omega_{2,k-1}
$$

$$
β{1,k} = β{1,k-1} + \omega_{1,k-1}
$$

$$
β{0,k} = β{0,k-1} + \omega_{0,k-1}
$$

#### Observation Noise:

$$
\epsilon_k \sim \text{Normal}(0, 0.05)
$$

#### Process Noise:

$$
\omega_{2,k-1} \sim \text{Normal}(0, 0.01)
$$

$$
\omega_{1,k-1} \sim \text{Normal}(0, 0.001)
$$

$$
\omega_{0,k-1} \sim \text{Normal}(0, 0.01)
$$


## I.3) Prediction and metrics

Le participant doit prédire si le RUL d'un élément est inférieur à $6$ mois :

- Étiquette = $1$, si RUL ≤ 6  (signifie que le robot échouera dans les $6$ mois suivants)
- Étiquette = $0$, sinon. (signifie le contraire)  

Si l'étiquette prédite correspond à la vérité terrain, une récompense de $2$ sera attribuée.

Si elle ne correspond pas, alors :

- Une pénalité de -$4$ sera attribuée, si la vérité est $1$ et la prédiction est $0$ ;
- Une pénalité de -$1/60 \times \text{true\_rul}$ sera attribuée, si la vérité est $0$ et la prédiction est $1$.

La métrique d'évaluation est calculée comme suit :

$$
\text{perf} = \sum_{i=1}^{n} \text{Reward}_i
$$

où $\text{Reward}_i$ est calculé comme mentionné précédemment.

La fonction suivante calcule la métrique d'évaluation :

    import pandas as pd
    import pandas.api.types

    def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
        '''
        This metric is customized to measure the performance of remaining useful life prediction. 
        The participant is asked to predict whether the RUL of an item is less than 6 months: 1 - if RUL<=6 and 0 otherwise.
        In the ground truth file "Solution.csv", there will be a column "true_rul" as well as a column "label".
        If the predicted label matches the ground truth, a reward of 2 will be given.
        If it does not match, then,
        - A penalty of -4 will be given, if truth is 1 and prediction is 0;
        - A penalty of -1/60*true_rul will be given, if truth is 0 and prediction is 1.

        TODO: Add unit tests. We recommend using doctests so your tests double as usage demonstrations for competition hosts.
        https://docs.python.org/3/library/doctest.html
        # This example doctest works for mean absolute error:
        >>> import pandas as pd
        >>> row_id_column_name = "item_index"
        >>> solution_data = {'item_index': [0, 1, 2, 3], 'label': [1, 0, 1, 0], 'true_rul': [5, 20, 1, 6]}
        >>> submission_data = {'item_index': [0, 1, 2, 3], 'label': [1, 0, 0, 0]}
        >>> solution = pd.DataFrame(solution_data)
        >>> submission = pd.DataFrame(submission_data)
        >>> score(solution.copy(), submission.copy(), row_id_column_name)
        2
        '''

        # Initialize rewards and penalties
        reward = 2
        penalty_false_positive = -1/60
        penalty_false_negative = -4

        # Compare labels and calculate rewards/penalties
        rewards_penalties = []
        for _, (sol_label, sub_label, true_rul) in enumerate(zip(solution['label'], submission['label'], solution['true_rul'])):
            if sol_label == sub_label:
                rewards_penalties.append(reward)
            elif sol_label == 1 and sub_label == 0:
                rewards_penalties.append(penalty_false_negative)
            elif sol_label == 0 and sub_label == 1:
                rewards_penalties.append(penalty_false_positive * true_rul)
            else:
                rewards_penalties.append(0)  # No reward or penalty if labels don't match   

        return sum(rewards_penalties)




# Phase II : _

--> https://www.kaggle.com/competitions/predictive-maintenance-of-a-robot-ii

    """
    st.markdown(context)
  


