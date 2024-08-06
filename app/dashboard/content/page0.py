import streamlit as st


def page_0():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)
    st.markdown('<div class="header">Marius Ayrault - SDA 2024/2025</div>', unsafe_allow_html=True)
    st.text("------------------------------------------------------------------------------------------------------------------------")
    context = """ 
# #Phase I : Remaining Useful Life (RUL)_
**Objectif :** prédire la durée de vie restante d'un robot industriel d'après les données monitorées de trois modes de défaillance :
* **Mortalité infantile** : échecs dus à des défauts de fabrication qui se produisent très tôt dans la vie du robot.
* **Échec des cartes de contrôle** : échecs aléatoires décrits par une distribution probabiliste.
* **Croissance des fissures de fatigue** : échecs dus à la croissance des fissures dans le temps. Ces fissures peuvent être mesurées régulièrement (chaque mois).

**Target :**  prédire si chaque robot peut survivre aux 6 mois suivants

# #Phase II : _
    
    """
    st.markdown(context)
  