import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64

st.set_page_config(page_title="Systeme de Decision Clinique", layout="wide")

API_URL = "http://localhost:8000/api/v1"

# 1. Initialisation de la memoire cache
if "current_page" not in st.session_state:
    st.session_state.current_page = "Accueil"

if "prep_data" not in st.session_state:
    st.session_state.prep_data = None

# 2. Navigation Laterale
PAGES = [
    "Accueil", 
    "Module 1 : Exploration", 
    "Module 2 : Preparation", 
    "Module 3 : Modelisation", 
    "Module 4 : Evaluation", 
    "Module 5 : XAI"
]

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Choisissez un module :", PAGES, index=PAGES.index(st.session_state.current_page))

if menu != st.session_state.current_page:
    st.session_state.current_page = menu
    st.rerun()

st.sidebar.divider()
st.sidebar.info("Projet Challenge 1 : Application Web Interactive de Classification Explicable.\nDeveloppe par Salma.")

# 3. Page Accueil
if st.session_state.current_page == "Accueil":
    st.title("Prediction des Readmissions Hospitalieres")
    st.markdown("Bienvenue sur le systeme d'aide a la decision medicale.")
    
    st.write("### Acces Rapide :")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Module 1 : Exploration", use_container_width=True):
            st.session_state.current_page = "Module 1 : Exploration"
            st.rerun()
    with col2:
        if st.button("Module 2 : Preparation", use_container_width=True):
            st.session_state.current_page = "Module 2 : Preparation"
            st.rerun()
    with col3:
        if st.button("Module 3 : Modelisation", use_container_width=True):
            st.session_state.current_page = "Module 3 : Modelisation"
            st.rerun()
    with col4:
        if st.button("Module 4 : Evaluation", use_container_width=True):
            st.session_state.current_page = "Module 4 : Evaluation"
            st.rerun()
    with col5:
        if st.button("Module 5 : XAI", use_container_width=True):
            st.session_state.current_page = "Module 5 : XAI"
            st.rerun()

# 4. Page Module 1 : Exploration
elif st.session_state.current_page == "Module 1 : Exploration":
    st.title("Module 1 : Dashboard d'Exploration (EDA)")
    
    st.subheader("1. Apercu du jeu de donnees (50 premieres lignes)")
    response_sample = requests.get(f"{API_URL}/data/sample")
    if response_sample.status_code == 200:
        st.dataframe(pd.DataFrame(response_sample.json()), use_container_width=True)
    else:
        st.error("Le backend n'est pas accessible.")
    
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2. Valeurs Manquantes")
        resp_missing = requests.get(f"{API_URL}/data/missing-values")
        if resp_missing.status_code == 200 and resp_missing.json():
            df_missing = pd.DataFrame(list(resp_missing.json().items()), columns=["Variable", "Pourcentage"])
            st.plotly_chart(px.bar(df_missing, x="Variable", y="Pourcentage", color="Pourcentage", color_continuous_scale="Reds"), use_container_width=True)

    with col2:
        st.subheader("3. Variable Cible (Readmitted)")
        resp_target = requests.get(f"{API_URL}/data/target-distribution")
        if resp_target.status_code == 200:
            df_target = pd.DataFrame(list(resp_target.json().items()), columns=["Categorie", "Patients"])
            st.plotly_chart(px.pie(df_target, names="Categorie", values="Patients"), use_container_width=True)

    st.divider()
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("4. Detection d'Outliers")
        resp_outliers = requests.get(f"{API_URL}/data/outliers")
        if resp_outliers.status_code == 200:
            df_outliers = pd.DataFrame(list(resp_outliers.json().items()), columns=["Variable", "Nombre d'Outliers"])
            st.plotly_chart(px.bar(df_outliers, x="Variable", y="Nombre d'Outliers"), use_container_width=True)

    with col4:
        st.subheader("5. Audit Ethique : Biais par age")
        resp_bias = requests.get(f"{API_URL}/data/bias-analysis")
        if resp_bias.status_code == 200:
            df_bias = pd.DataFrame(list(resp_bias.json().items()), columns=["Tranche d'age", "Taux de readmission"])
            st.plotly_chart(px.line(df_bias, x="Tranche d'age", y="Taux de readmission", markers=True), use_container_width=True)

# 5. Page Module 2 : Preparation
elif st.session_state.current_page == "Module 2 : Preparation":
    st.title("Module 2 : Pipeline de Data Engineering")
    
    col_ui1, col_ui2 = st.columns(2)
    
    with col_ui1:
        st.subheader("Parametrage du Pipeline ML")
        test_size = st.slider("Taille du jeu de Test (%)", min_value=10, max_value=40, value=20, step=5)
        
        if st.button("Lancer le Pipeline", type="primary"):
            with st.spinner("Application des regles et sous-echantillonnage..."):
                payload = {"test_size": test_size / 100.0}
                resp = requests.post(f"{API_URL}/prep/process", json=payload)
                
                if resp.status_code == 200:
                    st.session_state.prep_data = resp.json()
                else:
                    st.error("Erreur lors du pretraitement.")

        if st.session_state.prep_data:
            data = st.session_state.prep_data
            st.success("Pipeline execute avec succes.")
            
            st.write("### Impact du Random Under-Sampling")
            col_a, col_b = st.columns(2)
            col_a.metric("Taille Train Set AVANT", data['train_size_before'])
            col_b.metric("Taille Train Set APRES", data['train_size_after'])
            
            st.write("### Dataset Final")
            col_c, col_d = st.columns(2)
            col_c.metric("Variables (Features)", data['features_count'])
            col_d.metric("Taille du Test Set", data['test_size'])

    with col_ui2:
        st.subheader("Regles automatisees :")
        st.markdown("""
        * Nettoyage strict (suppression variables vides > 80%).
        * Traduction sémantique des identifiants.
        * Conservation du Top 10 pour la maladie principale.
        * Creation de la cible binaire.
        * Transformation du texte (One-Hot Encoding).
        * Mise a l'echelle des donnees numeriques (StandardScaler).
        * Sous-echantillonnage aleatoire de la classe majoritaire.
        """)

# 6. Page Module 3 : Modelisation
elif st.session_state.current_page == "Module 3 : Modelisation":
    st.title("Module 3 : Modelisation et Hyperparametres")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration du Modele")
        model_choice = st.selectbox(
            "Choisissez l'algorithme :", 
            ["Logistic Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost"]
        )
        
        hyperparams = {}
        
        if model_choice == "Logistic Regression":
            st.caption("Recommandation : C=0.1 pour reduire le surapprentissage.")
            c_val = st.slider("Parametre C", 0.01, 10.0, 1.0)
            hyperparams["C"] = c_val
        elif model_choice == "Decision Tree":
            st.caption("Recommandation : Une profondeur de 5 a 7 evite un arbre trop complexe.")
            max_depth = st.slider("Profondeur maximale", 1, 30, 5)
            hyperparams["max_depth"] = max_depth
        elif model_choice == "KNN":
            st.caption("Recommandation : K=9 ou 11 donne souvent de bons resultats.")
            n_neighbors = st.slider("Nombre de voisins (K)", 1, 21, 5, step=2)
            hyperparams["n_neighbors"] = n_neighbors
        elif model_choice == "Random Forest":
            st.caption("Recommandation : 150 arbres et une profondeur de 10.")
            hyperparams["n_estimators"] = st.slider("Nombre d'arbres", 10, 300, 100, step=10)
            hyperparams["max_depth"] = st.slider("Profondeur maximale", 1, 30, 10)
        elif model_choice == "XGBoost":
            st.caption("Recommandation : 200 arbres, taux a 0.1, profondeur 4.")
            hyperparams["n_estimators"] = st.slider("Nombre d'arbres", 10, 300, 100, step=10)
            hyperparams["learning_rate"] = st.select_slider("Taux d'apprentissage", options=[0.001, 0.01, 0.1, 0.2, 0.3], value=0.1)
            hyperparams["max_depth"] = st.slider("Profondeur maximale", 1, 15, 6)

        if st.button(f"Entrainer {model_choice}", type="primary"):
            with st.spinner("Entrainement en cours..."):
                payload = {"model_name": model_choice, "params": hyperparams}
                resp = requests.post(f"{API_URL}/models/train", json=payload)
                if resp.status_code == 200:
                    st.success(resp.json()["message"])
                else:
                    st.error("Erreur d'entrainement.")

    with col2:
        st.subheader("Console")
        st.markdown("Selection des modeles et hyperparametres depuis l'interface conformement au cahier des charges.")
        st.info("Les modeles sont sauvegardes avec leurs parametres pour pouvoir les comparer au Module 4.")

# 7. Page Module 4 : Evaluation
elif st.session_state.current_page == "Module 4 : Evaluation":
    st.title("Module 4 : Evaluation et Comparaison")
    
    with st.spinner("Analyse des modeles..."):
        resp = requests.get(f"{API_URL}/eval/compare")
    
    if resp.status_code != 200:
        st.warning("Aucun modele trouve. Entrainez un modele au Module 3.")
    else:
        data = resp.json()
        models_available = list(data.keys())
        
        st.subheader("1. Tableau Comparatif (Au Seuil Optimal)")
        
        df_metrics = pd.DataFrame({
            "Modele": models_available,
            "AUC-ROC": [data[m]["roc_auc"] for m in models_available],
            "PR-AUC": [data[m]["pr_auc"] for m in models_available],
            "Seuil Optimal": [data[m]["optimal_threshold"] for m in models_available],
            "Accuracy": [data[m]["accuracy"] for m in models_available],
            "Precision": [data[m]["precision"] for m in models_available],
            "Recall": [data[m]["recall"] for m in models_available],
            "F1-Score": [data[m]["f1_score"] for m in models_available]
        }).set_index("Modele")
        
        st.dataframe(df_metrics.style.highlight_max(axis=0, color="#2ecc71"), use_container_width=True)
        
        st.divider()
        st.subheader("2. Analyse Graphique")
        col_roc, col_pr = st.columns(2)
        
        with col_roc:
            fig_roc = go.Figure()
            for m in models_available:
                fig_roc.add_trace(go.Scatter(x=data[m]["fpr"], y=data[m]["tpr"], mode='lines', name=m))
            fig_roc.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with col_pr:
            fig_pr = go.Figure()
            for m in models_available:
                fig_pr.add_trace(go.Scatter(x=data[m]["recall_curve"], y=data[m]["precision_curve"], mode='lines', name=m))
            fig_pr.update_layout(title="Courbe Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig_pr, use_container_width=True)
            
        st.divider()
        st.subheader("3. Simulation du Seuil de Decision")
        
        col_slider, col_cm = st.columns(2)
        
        with col_slider:
            selected_model = st.selectbox("Selectionnez le modele :", models_available)
            opt_thresh = float(data[selected_model]["optimal_threshold"])
            threshold = st.slider("Seuil de decision", min_value=0.01, max_value=0.99, value=opt_thresh, step=0.01)
            
            payload = {"model_name": selected_model, "threshold": threshold}
            resp_thresh = requests.post(f"{API_URL}/eval/threshold", json=payload)
            
            if resp_thresh.status_code == 200:
                thresh_data = resp_thresh.json()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{thresh_data['accuracy']:.2f}")
                c2.metric("Precision", f"{thresh_data['precision']:.2f}")
                c3.metric("Recall", f"{thresh_data['recall']:.2f}")
                c4.metric("F1-Score", f"{thresh_data['f1']:.2f}")
        
        with col_cm:
            if resp_thresh.status_code == 200:
                cm = thresh_data['confusion_matrix']
                fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Prediction", y="Verite", color="Patients"), color_continuous_scale='Blues')
                st.plotly_chart(fig_cm, use_container_width=True)

# 8. Page Module 5 : XAI
elif st.session_state.current_page == "Module 5 : XAI":
    st.title("Module 5 : Explainable AI (SHAP & LIME)")
    
    resp_models = requests.get(f"{API_URL}/eval/compare")
    if resp_models.status_code != 200:
        st.warning("Aucun modele disponible. Entrainez un modele au Module 3.")
    else:
        models_available = list(resp_models.json().keys())
        selected_model = st.selectbox("Modele a analyser", models_available)

        tab1, tab2 = st.tabs(["Explications Globales (SHAP)", "Explications Locales (Patient)"])

        with tab1:
            st.subheader("Comportement Global du Modele")
            if st.button("Generer l'analyse globale"):
                with st.spinner("Calcul des valeurs SHAP globales (Cela peut prendre quelques secondes)..."):
                    resp_global = requests.get(f"{API_URL}/xai/global/{selected_model}")
                    if resp_global.status_code == 200:
                        xai_data = resp_global.json()
                        st.markdown(f"**Variable la plus influente sur le modele :** `{xai_data['top_feature']}`")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("Graphique SHAP Summary")
                            st.image(base64.b64decode(xai_data['summary_plot']))
                        with c2:
                            st.write(f"Dependance de la variable {xai_data['top_feature']}")
                            st.image(base64.b64decode(xai_data['dependence_plot']))
                    else:
                        st.error("Erreur lors du calcul XAI global.")

        with tab2:
            st.subheader("Analyse individuelle d'un patient")
            patient_id = st.number_input("Index du patient dans le jeu de test", min_value=0, max_value=2000, value=0)
            
            if st.button("Expliquer la prediction pour ce patient"):
                with st.spinner("Execution de SHAP Waterfall et LIME..."):
                    resp_local = requests.get(f"{API_URL}/xai/local/{selected_model}/{patient_id}")
                    if resp_local.status_code == 200:
                        loc_data = resp_local.json()
                        
                        st.info(loc_data['synthesis'])
                        
                        c3, c4 = st.columns(2)
                        with c3:
                            st.write("SHAP Waterfall")
                            st.image(base64.b64decode(loc_data['waterfall_plot']))
                        with c4:
                            st.write("LIME Explanation")
                            components.html(loc_data['lime_html'], height=400)
                    else:
                        st.error("Erreur lors de l'explication locale.")