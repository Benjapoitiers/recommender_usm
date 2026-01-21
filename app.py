import streamlit as st
import pandas as pd
import pyalex
from pyalex import Authors, Works
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuración de PyAlex
pyalex.config.email = "benjapoitiers@gmail.com"

# --- FUNCIONES AUXILIARES (Backend) ---

def reconstruct_abstract(inverted_index):
    """Reconstruye el abstract desde el formato invertido de OpenAlex."""
    if not inverted_index:
        return ""
    max_index = max([max(positions) for positions in inverted_index.values()])
    abstract_list = [""] * (max_index + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            abstract_list[pos] = word
    return " ".join(abstract_list)

def get_research_text(name):
    """
    Busca a un autor en OpenAlex y concatena todo su texto relevante
    (Títulos + Keywords + Abstracts) en un solo string gigante.
    """
    try:
        # Buscamos al autor
        authors = Authors().search(name).get()
        if not authors:
            return None, None
        
        author = authors[0] # Tomamos el mejor match
        author_id = author['id']
        
        # Buscamos sus trabajos
        works = Works().filter(author={"id": author_id}).get()
        
        full_text = []
        for work in works:
            title = work.get('title', '') or ""
            
            # Keywords
            concepts = [c['display_name'] for c in work.get('concepts', [])]
            keywords = " ".join(concepts)
            
            # Abstract
            abstract = reconstruct_abstract(work.get('abstract_inverted_index')) or ""
            
            # Unimos todo con peso extra a los keywords repitiéndolos
            text_chunk = f"{title} {keywords} {keywords} {abstract}"
            full_text.append(text_chunk)
            
        return " ".join(full_text), author['display_name']
    
    except Exception as e:
        st.error(f"Error conectando con OpenAlex: {e}")
        return None, None

def load_local_database():
    """Carga y prepara la base de datos de profesores USM."""
    try:
        # 1. Cargar Papers (El archivo que generaste con el script anterior)
        #Esto se debe cambiar eventualmente: 
        df_papers = pd.read_excel("Output_OpenAlex_Papers_final.xlsx")
        
        # 2. Cargar Datos Administrativos (Para obtener Correo y Depto)
        df_admin = pd.read_excel("ACAD-DOC 19_ENE_26.xlsx")
        
        # Crear columnas clave para el cruce (limpieza básica)
        # Asumimos que 'Nombre_Busqueda' en papers coincide con la construcción de nombres del admin
        # Ojo: Aquí hacemos una agrupación por Profesor en la base de papers
        
        # Agrupamos todo el texto por profesor existente
        df_papers['Full_Content'] = df_papers['Paper_Title'].fillna('') + " " + \
                                    df_papers['Keywords'].fillna('') + " " + \
                                    df_papers['Abstract'].fillna('')
        
        df_corpus = df_papers.groupby('Nombre_Busqueda')['Full_Content'].apply(' '.join).reset_index()
        
        return df_corpus, df_admin
        
    except FileNotFoundError as e:
        st.error(f"Falta un archivo vital: {e}")
        return None, None

# --- INTERFAZ DE USUARIO (Frontend) ---

# Configuración de la página
st.set_page_config(page_title="Recommender USM Research", layout="wide")

# Estilos CSS personalizados para el botón rojo y el logo
st.markdown("""
    <style>
    .stButton > button {
        background-color: #d93025; /* Rojo */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #a50e0e; /* Rojo más oscuro al pasar mouse */
        color: white;
    }
    .header-container {
        display: flex;
        align-items: center;
    }
    .logo-img {
        width: 60px;
        margin-left: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Encabezado con Logo
col_header1, col_header2 = st.columns([0.8, 0.2])
with col_header1:
    st.markdown("""
        <div class='header-container'>
            <h1>Recommender USM Research <img src='https://upload.wikimedia.org/wikipedia/commons/4/47/Logo_UTFSM.png' class='logo-img'></h1>
        </div>
        """, unsafe_allow_html=True)
    st.subheader("Authors Search")

# Inputs
col1, col2 = st.columns(2)
with col1:
    first_name = st.text_input("First Name")
with col2:
    last_name = st.text_input("Last Name")

# Botón de Análisis
if st.button("Analyze Connection"):
    if not first_name or not last_name:
        st.warning("Please enter both First Name and Last Name.")
    else:
        with st.spinner('Fetching data from OpenAlex and calculating similarities...'):
            
            # 1. Obtener datos del USUARIO NUEVO (Input)
            query_name = f"{first_name} {last_name}"
            user_text, user_display_name = get_research_text(query_name)
            
            if not user_text:
                st.error(f"No research data found for '{query_name}' in OpenAlex.")
            else:
                st.success(f"Found author: **{user_display_name}**. Analyzing connection...")
                
                # 2. Cargar base de datos USM
                df_corpus, df_admin = load_local_database()
                
                if df_corpus is not None:
                    # 3. Motor de Recomendación (TF-IDF + Cosine Similarity)
                    
                    # Lista de todos los textos: [Texto_Usuario_Nuevo, Texto_Prof_1, Texto_Prof_2, ...]
                    all_texts = [user_text] + df_corpus['Full_Content'].tolist()
                    professor_names = df_corpus['Nombre_Busqueda'].tolist()
                    
                    # Vectorización
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(all_texts)
                    
                    # Calcular similitud del primero (usuario) contra todos los demás
                    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                    
                    # 4. Obtener Top 5
                    # Obtener índices de los 5 valores más altos
                    top_indices = cosine_sim.argsort()[-5:][::-1]
                    
                    results = []
                    for idx in top_indices:
                        prof_name = professor_names[idx]
                        score = cosine_sim[idx]
                        
                        # Buscar metadatos en el archivo ACAD-DOC (Admin)
                        # Intentamos hacer match difuso o directo. Aquí usaremos una lógica simple de string containment
                        # para cruzar 'Nombre_Busqueda' (que viene del excel OpenAlex) con los datos del excel Admin.
                        
                        # Estrategia de búsqueda en ACAD-DOC:
                        # Buscamos la fila donde el nombre coincida mejor.
                        # Asumimos que el script anterior usó los nombres del Excel, así que deberían ser idénticos.
                        
                        # Filtramos filas en df_admin
                        # Nota: Esto depende de cómo construiste 'Nombre_Busqueda' en el script anterior.
                        # Si es exacto:
                        admin_info = df_admin[
                            (df_admin['NOMBRES'].astype(str) + " " + 
                             df_admin['PRIMER_APELLIDO'].astype(str) + " " + 
                             df_admin['SEGUNDO_APELLIDO'].astype(str)).str.strip() == prof_name
                        ]
                        
                        # Si no hay match exacto (por espacios o NaN), fallback:
                        if admin_info.empty:
                             # Intentar solo Nombre + Primer Apellido
                             admin_info = df_admin[
                                (df_admin['NOMBRES'].astype(str) + " " + df_admin['PRIMER_APELLIDO'].astype(str)).apply(lambda x: x in prof_name)
                             ]

                        dept = "Unknown Department"
                        email = "No Email Found"
                        
                        if not admin_info.empty:
                            dept = admin_info.iloc[0]['DEPARTAMENTO']
                            email = admin_info.iloc[0]['CORREO_INSTITUCIONAL'] # Asegúrate que la columna se llame así en tu Excel

                        results.append({
                            "Professor": prof_name,
                            "Similarity Score": f"{score:.2%}",
                            "Department": dept,
                            "Email": email
                        })
                    
                    # 5. Mostrar Resultados
                    st.markdown("### Top 5 Research Matches")
                    st.write("Based on keywords, paper titles, and abstracts content similarity.")
                    
                    df_results = pd.DataFrame(results)
                    
                    # Mostramos tabla estilizada
                    st.dataframe(
                        df_results,
                        column_config={
                            "Professor": st.column_config.TextColumn("Professor Name", width="medium"),
                            "Similarity Score": st.column_config.ProgressColumn("Match Score", format="%.2f%%", min_value=0, max_value=1),
                            "Email": st.column_config.LinkColumn("Email"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )