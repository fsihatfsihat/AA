import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Configuration de la page
st.set_page_config(page_title="MStat Moderne", layout="wide")
st.title("📊 Statistiques Descriptives et Visualisation")
st.markdown("Une version moderne du module MStat.")

# --- BARRE LATÉRALE : IMPORTATION DES DONNÉES ---
st.sidebar.header("📁 Données")
uploaded_file = st.sidebar.file_uploader("Importer des données (CSV)", type="csv")


# Fonction pour générer des données de démonstration
@st.cache_data
def load_demo_data():
    np.random.seed(42)
    return pd.DataFrame({
        'A': np.random.normal(50, 15, 100),
        'B': np.random.normal(60, 10, 100),
        'C': np.random.uniform(20, 80, 100),
        'D': np.random.choice(['Groupe 1', 'Groupe 2', 'Groupe 3'], 100),
        'E': np.random.normal(40, 5, 100)
    })


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Utilisation des données de démonstration.")
    df = load_demo_data()

# --- NAVIGATION PRINCIPALE ---
# Reproduction des trois sous-menus de MStat
onglets = st.tabs(["⚙️ Préparation Des Données", "📈 Statistiques Descriptives", "🎨 Visualisation"])

# ---------------------------------------------------------
# ONGLET 1 : PRÉPARATION DES DONNÉES
# ---------------------------------------------------------
with onglets[0]:
    st.header("Préparation Des Données")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Imputation de Données Manquantes")
        if st.checkbox("Activer l'imputation"):
            method = st.selectbox("Méthode", ["Moyenne", "Médiane", "Suppression des lignes"])
            if method == "Moyenne":
                df = df.fillna(df.mean(numeric_only=True))
            elif method == "Médiane":
                df = df.fillna(df.median(numeric_only=True))
            else:
                df = df.dropna()
            st.success("Imputation appliquée.")

    with col2:
        st.subheader("Normalisation des Données")
        if st.checkbox("Activer la normalisation"):
            norm_method = st.selectbox("Méthode de normalisation", ["Min-Max (0 à 1)", "Standardisation (Z-score)"])
            num_cols = df.select_dtypes(include=np.number).columns
            if norm_method == "Min-Max (0 à 1)":
                df[num_cols] = (df[num_cols] - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min())
            else:
                df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
            st.success("Normalisation appliquée.")

    with col3:
        st.subheader("Détecter Les Outliers")
        if st.checkbox("Filtrer les Outliers (Méthode IQR)"):
            num_cols = df.select_dtypes(include=np.number).columns
            Q1 = df[num_cols].quantile(0.25)
            Q3 = df[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            condition = ~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            df = df[condition]
            st.success("Outliers filtrés.")

    st.write("**Aperçu des données actuelles :**")
    st.dataframe(df.head())

# ---------------------------------------------------------
# ONGLET 2 : STATISTIQUES DESCRIPTIVES[cite: 4]
# ---------------------------------------------------------
with onglets[1]:
    st.header("Statistiques Descriptives")

    choix_stat = st.radio("Sélectionnez l'analyse :",
                          ["Calculer les Statistiques Descriptives", "Calculer l'Alpha de Cronbach"])

    if choix_stat == "Calculer les Statistiques Descriptives":
        st.subheader("Statistiques Quantitatives et Qualitatives")
        colonnes_select = st.multiselect("Sélectionnez les variables", df.columns, default=df.columns.tolist())
        if colonnes_select:
            st.dataframe(df[colonnes_select].describe(include='all').T)

    elif choix_stat == "Calculer l'Alpha de Cronbach":
        st.subheader("Paramètres Alpha de Cronbach")
        num_cols = df.select_dtypes(include=np.number).columns
        items_inclus = st.multiselect("Items à inclure :", num_cols, default=list(num_cols)[:3])

        if len(items_inclus) > 1:
            items = df[items_inclus].dropna()
            k = items.shape[1]
            var_items = items.var(axis=0, ddof=1).sum()
            var_totale = items.sum(axis=1).var(ddof=1)

            if var_totale == 0:
                st.warning("La variance totale est nulle.")
            else:
                alpha = (k / (k - 1)) * (1 - (var_items / var_totale))
                st.metric("Coefficient Alpha de Cronbach", round(alpha, 4))
                st.info("Un alpha > 0.7 est généralement considéré comme acceptable.")
        else:
            st.warning("Veuillez sélectionner au moins 2 variables quantitatives.")

# ---------------------------------------------------------
# ONGLET 3 : VISUALISATION
# ---------------------------------------------------------
with onglets[2]:
    st.header("Visualisation")

    # Liste exacte des graphiques de MStat
    type_graph = st.selectbox("Choisissez le type de graphique :", [
        "Créer un Histogramme",
        "Créer une Boîte à Moustaches (Boxplot)",
        "Créer un Diagramme en Barres",
        "Créer un Diagramme Circulaire (Pie Chart)",
        "Créer un Nuage de Points (Scatter Plot)",
        "Créer une Courbe de Distribution",
        "Créer une Courbe Temporelle (Line Plot)",
        "Créer une Carte Thermique (Heatmap)",
        "Créer un Diagramme de Violon (Violin Plot)",
        "Créer un Graphique Q-Q",
        "Comparaison de plusieurs variables"
    ])

    fig, ax = plt.subplots(figsize=(10, 6))
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if type_graph in ["Créer un Histogramme", "Créer une Courbe de Distribution",
                      "Créer une Boîte à Moustaches (Boxplot)", "Créer un Diagramme de Violon (Violin Plot)",
                      "Créer un Graphique Q-Q"]:
        var_x = st.selectbox("Variable quantitative", num_cols)

        if type_graph == "Créer un Histogramme":
            sns.histplot(df[var_x], kde=False, ax=ax, color='steelblue')
        elif type_graph == "Créer une Courbe de Distribution":
            sns.histplot(df[var_x], kde=True, ax=ax, color='purple')
        elif type_graph == "Créer une Boîte à Moustaches (Boxplot)":
            sns.boxplot(y=df[var_x], ax=ax, color='lightgreen')
        elif type_graph == "Créer un Diagramme de Violon (Violin Plot)":
            sns.violinplot(y=df[var_x], ax=ax, color='coral')
        elif type_graph == "Créer un Graphique Q-Q":
            stats.probplot(df[var_x].dropna(), dist="norm", plot=ax)

    elif type_graph in ["Créer un Diagramme en Barres", "Créer un Diagramme Circulaire (Pie Chart)"]:
        var_cat = st.selectbox("Variable catégorielle", cat_cols if len(cat_cols) > 0 else df.columns)
        counts = df[var_cat].value_counts()

        if type_graph == "Créer un Diagramme en Barres":
            sns.barplot(x=counts.index, y=counts.values, ax=ax, palette='viridis')
        elif type_graph == "Créer un Diagramme Circulaire (Pie Chart)":
            ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')

    elif type_graph in ["Créer un Nuage de Points (Scatter Plot)", "Créer une Courbe Temporelle (Line Plot)"]:
        col1, col2 = st.columns(2)
        with col1:
            var_x = st.selectbox("Axe X", df.columns)
        with col2:
            var_y = st.selectbox("Axe Y", num_cols)

        if type_graph == "Créer un Nuage de Points (Scatter Plot)":
            sns.scatterplot(x=df[var_x], y=df[var_y], ax=ax)
        elif type_graph == "Créer une Courbe Temporelle (Line Plot)":
            sns.lineplot(x=df[var_x], y=df[var_y], ax=ax)

    elif type_graph == "Créer une Carte Thermique (Heatmap)":
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)

    elif type_graph == "Comparaison de plusieurs variables":
        vars_comp = st.multiselect("Variables à comparer", num_cols, default=num_cols[:2])
        if len(vars_comp) > 0:
            sns.boxplot(data=df[vars_comp], ax=ax)

    st.pyplot(fig)
