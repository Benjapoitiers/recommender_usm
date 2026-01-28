'''Codigo creado por: Benjamin Fuentes Valdebenito'''

import streamlit as st
import pandas as pd
import pyalex
from pyalex import Authors, Works
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import time
import unicodedata
from urllib.parse import quote
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from io import BytesIO  
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Cambiar el email si es necesario para entrar en el polite pool.
pyalex.config.email = "benjapoitiers@gmail.com"

# --- FUNCIONES AUXILIARES DE LIMPIEZA Y VALIDACIÓN ---

def normalizar(texto):
    if not isinstance(texto, str): 
        return ""
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) 
                    if unicodedata.category(c) != 'Mn')
    texto = texto.replace('.', ' ').replace(',', ' ')
    return " ".join(texto.split())

def reconstruct_abstract(inverted_index):
    if not inverted_index: 
        return ""
    try:
        max_index = max([max(positions) for positions in inverted_index.values()])
        abstract_list = [""] * (max_index + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                abstract_list[pos] = word
        return " ".join(abstract_list)
    except:
        return ""

# Función para generar la figura del Word Cloud
def generate_wordcloud_fig(text, title):
    if not text or len(text.strip()) == 0:
        return None
    
    # 1. Definir lista de palabras a ignorar (Misma que en PyVis)
    custom_ignore_words = [
        'non', 'color', 'study', 'analysis', 'using', 'method', 'result', 
        'paper', 'based', 'data', 'new', 'time', 'case', 'model', 
        'approach', 'used', 'results', 'proposed', 'et', 'al', 'doi',
        'https', 'high', 'low', 'different', 'use', 'development'
    ]
    
    # 2. Fusionar con las stopwords por defecto de la librería
    # Usamos un set para evitar duplicados y mejorar velocidad
    final_stopwords = set(STOPWORDS)
    final_stopwords.update(custom_ignore_words)
    
    # 3. Crear el objeto WordCloud con el filtro aplicado
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis', 
        max_words=100,
        stopwords=final_stopwords  # <--- AQUÍ SE APLICA EL FILTRO
    ).generate(text)

    # Crear la figura con Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=15, pad=20)
    return fig

# Validación estricta de identidad BIDIRECCIONAL.
def verificar_identidad_completa(nombre_openalex, row_nombres, row_apellido1, row_apellido2=""):
    
    if not nombre_openalex: return False

    oa_norm = normalizar(nombre_openalex)
    parts_oa = oa_norm.split()

    if len(parts_oa) < 2: return False 

    nombres_norm = normalizar(row_nombres)
    ape1_norm = normalizar(row_apellido1)
    
    if not ape1_norm: return False 

    # PASO 1: Buscar apellido paterno 
    idx_ape = -1
    for i, token in enumerate(parts_oa):
        if token == ape1_norm:
            idx_ape = i
            break

    if idx_ape == -1: return False 

    # PASO 2: Aislar nombres de pila del candidato
    if idx_ape == 0 and len(parts_oa) > 1:
        nombres_pila_cand = parts_oa[1:]
    else:
        nombres_pila_cand = parts_oa[:idx_ape]

    if not nombres_pila_cand: return False

    # PASO 3: VALIDACIÓN CRUZADA 
    nombres_busqueda = nombres_norm.split() if nombres_norm else []
    
    for token_q in nombres_busqueda:
        match_encontrado = False
        
        for nombre_token_cand in nombres_pila_cand:
            if nombre_token_cand == token_q:
                match_encontrado = True
                break
            if len(nombre_token_cand) == 1 and token_q.startswith(nombre_token_cand):
                match_encontrado = True
                break
            if len(token_q) == 1 and nombre_token_cand.startswith(token_q):
                match_encontrado = True
                break
        
        if not match_encontrado:
            return False 

    return True

# --- MOTORES DE BÚSQUEDA (OPENALEX + ORCID) ---

def buscar_input_openalex(nombres, apellido1):
    intentos = []
    if nombres and apellido1:
        primer_nombre = nombres.split()[0]
        intentos.append(f"{primer_nombre} {apellido1}")
        intentos.append(f"{nombres} {apellido1}")
    
    intentos = list(dict.fromkeys(intentos)) 
    top_author = None

    # Búsqueda Global
    for query in intentos:
        try:
            candidates = Authors().search(query).get()
            if candidates:
                for cand in candidates[:10]:
                    if verificar_identidad_completa(cand['display_name'], nombres, apellido1):
                        top_author = cand
                        break
            if top_author: break
        except Exception:
            pass

    if not top_author:
        return [], None, None

    # Extraer ORCID si existe
    orcid_path = None
    oa_orcid_url = top_author.get('ids', {}).get('orcid')
    if oa_orcid_url:
        orcid_path = oa_orcid_url.replace("https://orcid.org/", "")

    # Extraer Papers
    try:
        works = Works().filter(author={"id": top_author['id']}).get()
        papers = []
        for w in works:
            abstract = reconstruct_abstract(w.get('abstract_inverted_index'))
            keywords = [c['display_name'] for c in w.get('concepts', [])]
            
            papers.append({
                'Title': w.get('title'),
                'DOI': w.get('doi'),
                'Keywords': " ".join(keywords[:10]),
                'Abstract': abstract,
                'Source': 'OpenAlex'
            })
        return papers, top_author['display_name'], orcid_path
    except:
        return [], top_author['display_name'], orcid_path

def buscar_input_orcid(orcid_path, nombres, apellido1):
    headers = {'Accept': 'application/json'}
    target_orcid = orcid_path
    found_name = None

    if not target_orcid and nombres and apellido1:
        primer_nombre = nombres.split()[0]
        query = f"given-names:{primer_nombre} AND family-name:{apellido1}"
        try:
            url_search = f"https://pub.orcid.org/v3.0/search/?q={quote(query)}"
            resp = requests.get(url_search, headers=headers, timeout=5)
            if resp.status_code == 200:
                results = resp.json().get('result', [])
                for res in results[:15]:
                    oid = res.get('orcid-identifier', {}).get('path')
                    if oid:
                        rp = requests.get(f"https://pub.orcid.org/v3.0/{oid}/person", headers=headers, timeout=5)
                        if rp.status_code == 200:
                            dp = rp.json()
                            g = dp.get('name', {}).get('given-names', {}).get('value', '')
                            f = dp.get('name', {}).get('family-name', {}).get('value', '')
                            full_n = f"{g} {f}"
                            if verificar_identidad_completa(full_n, nombres, apellido1):
                                target_orcid = oid
                                found_name = full_n
                                break
        except:
            pass

    if not target_orcid:
        return [], None

    papers = []
    try:
        url_works = f"https://pub.orcid.org/v3.0/{target_orcid}/works"
        r_w = requests.get(url_works, headers=headers, timeout=10)
        if r_w.status_code == 200:
            groups = r_w.json().get('group', [])
            for g in groups:
                summaries = g.get('work-summary', [])
                if not summaries: continue
                work = summaries[0]
                
                title = work.get('title', {}).get('title', {}).get('value', '')
                doi = None
                for eid in work.get('external-ids', {}).get('external-id', []):
                    if eid.get('external-id-type') == 'doi':
                        doi = eid.get('external-id-value')
                        if doi and 'https://doi.org/' not in doi: doi = f"https://doi.org/{doi}"
                        break
                
                papers.append({
                    'Title': title,
                    'DOI': doi,
                    'Keywords': "", 
                    'Abstract': "", 
                    'Source': 'ORCID'
                })
        
        return papers, (found_name if found_name else f"ORCID:{target_orcid}")
    except:
        return [], None

def get_hybrid_research_text(first_name, last_name):
    oa_papers, oa_name, linked_orcid = buscar_input_openalex(first_name, last_name)
    orcid_papers, orcid_name = buscar_input_orcid(linked_orcid, first_name, last_name)
    
    display_name = oa_name if oa_name else (orcid_name if orcid_name else None)
    
    if not display_name:
        return None, None

    final_papers = {}

    for p in oa_papers:
        key = p['DOI'] if p['DOI'] else normalizar(p['Title'])
        if key: final_papers[key] = p
    
    for p in orcid_papers:
        key = p['DOI'] if p['DOI'] else normalizar(p['Title'])
        if key and key not in final_papers:
            final_papers[key] = p

    full_text_list = []
    for p in final_papers.values():
        t = p['Title'] or ""
        k = p['Keywords'] or ""
        a = p['Abstract'] or ""
        
        chunk = f"{t} {t} {t} {k} {k} {a}"
        full_text_list.append(chunk)
    
    return " ".join(full_text_list), display_name



def create_term_network(texts, max_terms=60, n_clusters=4):
    # 1. LISTA DE PALABRAS A IGNORAR (Custom Stop Words)

    custom_ignore_words = [
        'non', 'color', 'study', 'analysis', 'using', 'method', 'result', 
        'paper', 'based', 'data', 'new', 'time', 'case', 'model', 
        'approach', 'used', 'results', 'proposed', 'et', 'al', 'doi',
        'https', 'high', 'low', 'different', 'use', 'development'
    ]
    
    # Fusionamos las stop words de inglés con las nuestras
    my_stop_words = list(ENGLISH_STOP_WORDS.union(custom_ignore_words))

    # --- CORRECCIÓN CRÍTICA PARA EVITAR "NOT ENOUGH DATA" ---
    if len(texts) > 1:
        current_min_df = 2
        current_max_df = 0.85
    else:
        current_min_df = 1
        current_max_df = 1.0 

    # 2. Extracción de términos (Usando la lista personalizada)
    vec = CountVectorizer(
        stop_words=my_stop_words,  # <--- AQUÍ USAMOS LA LISTA MEJORADA
        max_features=max_terms, 
        ngram_range=(1, 2),
        min_df=current_min_df,   
        max_df=current_max_df
    )
    
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return nx.Graph()

    terms = vec.get_feature_names_out()
    if len(terms) == 0:
        return nx.Graph()

    # 3. Matriz de Co-ocurrencia
    co_occurrence_matrix = (X.T * X)
    co_occurrence_matrix.setdiag(0)
    co_occurrence_array = co_occurrence_matrix.toarray()
    
    # 4. Clustering (Colores por temática)
    actual_clusters = min(n_clusters, len(terms))
    if len(terms) > actual_clusters and actual_clusters > 1:
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(co_occurrence_array)
    else:
        clusters = np.zeros(len(terms), dtype=int)
    
    # 5. Paleta de Colores (VOSviewer style)
    color_palette = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'
    ]
    
    G = nx.Graph()
    
    # Frecuencias para tamaño de nodos
    term_freqs = np.asarray(X.sum(axis=0)).flatten()
    min_freq = term_freqs.min()
    max_freq = term_freqs.max()
    
    # 6. Añadir Nodos
    for i, term in enumerate(terms):
        # Escalar tamaño
        if max_freq > min_freq:
            normalized_size = 10 + 30 * ((term_freqs[i] - min_freq) / (max_freq - min_freq))
        else:
            normalized_size = 20
        
        cluster_id = int(clusters[i])
        node_color = color_palette[cluster_id % len(color_palette)]
        
        G.add_node(
            term,
            size=float(normalized_size),
            title=f"{term}\nFreq: {int(term_freqs[i])}\nCluster: {cluster_id + 1}",
            color=node_color,
            group=cluster_id,
            font={'size': 16, 'face': 'Tahoma', 'color': 'black'},
            shadow={'enabled': True, 'color': 'rgba(0,0,0,0.2)'}
        )
    
    # 7. Añadir Aristas (Conexiones)
    non_zero_values = co_occurrence_array[co_occurrence_array > 0]
    
    if len(non_zero_values) > 0:
        percentile = 60 if len(texts) > 1 else 50 
        threshold = np.percentile(non_zero_values, percentile)
    else:
        threshold = 0
    
    cx = co_occurrence_matrix.tocoo()
    edge_weights = []
    
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if i < j and v >= threshold:
            edge_weights.append(float(v))
            
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        
        for i, j, v in zip(cx.row, cx.col, cx.data):
            if i < j and v >= threshold:
                if max_weight > min_weight:
                    norm_w = 1 + 5 * ((v - min_weight) / (max_weight - min_weight))
                else:
                    norm_w = 2
                
                G.add_edge(
                    terms[i], 
                    terms[j], 
                    value=float(norm_w),
                    title=f"Co-occurrence: {int(v)}",
                    width=float(norm_w)
                )
    
    return G
# --- CARGA DE DATOS LOCALES ---

@st.cache_data
def load_local_database():
    try:
        archivo_papers = "Output_Papers_Hibrido_20260122_100831.xlsx" 
        df_papers = pd.read_excel(archivo_papers)
        
        # Asegurarse que este archivo exista en la carpeta
        df_admin = pd.read_excel("ACAD-DOC_PUBLICO_USM.xlsx") 
        
        column_mapping = {
            'Keywords': 'Paper_Keywords',
            'Title': 'Paper_Title',
            'Abstract': 'Paper_Abstract',
            'DOI': 'Paper_DOI',
            'Year': 'Publication_Year',    
            'Paper_Year': 'Publication_Year' 
        }
        df_papers.rename(columns=column_mapping, inplace=True)
        
        required_cols = ['Paper_Title', 'Paper_Keywords', 'Paper_Abstract']
        for col in required_cols:
            if col not in df_papers.columns:
                df_papers[col] = ""

        df_papers['Full_Content'] = df_papers['Paper_Title'].fillna('') + " " + \
                                    df_papers['Paper_Keywords'].fillna('') + " " + \
                                    df_papers['Paper_Abstract'].fillna('')
        
        df_corpus = df_papers.groupby('Nombre_Busqueda')['Full_Content'].apply(' '.join).reset_index()
        
        return df_corpus, df_admin, df_papers
        
    except FileNotFoundError as e:
        st.error(f"Error cargando base de datos: {e}")
        return None, None, None
    
    
# --- INTERFAZ DE USUARIO (Frontend) ---

st.set_page_config(page_title="Recommender USM Research", layout="wide")

st.markdown("""
    <style>
    .stButton > button {
        background-color: #d93025; 
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #a50e0e; 
        color: white;
    }
    .header-container { display: flex; align-items: center; }
    .logo-img { width: 60px; margin-left: 15px; }
    h3 { color: #004b85; }
    </style>
    """, unsafe_allow_html=True)

col_header1, col_header2 = st.columns([0.8, 0.2])
with col_header1:
    st.markdown("""
        <div class='header-container'>
            <h1>Recommender USM Research <img src='https://upload.wikimedia.org/wikipedia/commons/4/47/Logo_UTFSM.png' class='logo-img'></h1>
        </div>
        """, unsafe_allow_html=True)
    st.subheader("Hybrid Authors Search (OpenAlex + ORCID)")

col1, col2 = st.columns(2)
with col1:
    first_name = st.text_input("First Name")
with col2:
    last_name = st.text_input("Last Name")

# --- CONTROL DE SESIÓN (SOLUCIÓN AL REINICIO) ---
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False

# Botón para EJECUTAR el análisis (Actualiza el estado)
if st.button("Analyze Connection"):
    st.session_state.search_performed = True
    st.session_state.first_name = first_name
    st.session_state.last_name = last_name

# --- LÓGICA DE VISUALIZACIÓN (Depende del estado, no del botón) ---
if st.session_state.search_performed:
    
    # Recuperamos los nombres guardados en sesión para mantener consistencia
    f_name = st.session_state.first_name
    l_name = st.session_state.last_name

    if not f_name or not l_name:
        st.warning("Please enter both First Name and Last Name.")
        st.session_state.search_performed = False # Resetear si falta info
    else:
        with st.spinner('Performing hybrid search (OpenAlex + ORCID) and calculating matches...'):
            
            # 1. Obtener datos HÍBRIDOS del Usuario
            # Nota: Esto se ejecuta en cada refresh, pero como los inputs son los mismos, 
            # PyAlex debería responder rápido o usar caché interno si estuviera configurado.
            # Para mayor optimización se podría cachear get_hybrid_research_text.
            user_text, user_display_name = get_hybrid_research_text(f_name, l_name)
            
            if not user_text:
                st.error(f"No research data found for '{f_name} {l_name}' in OpenAlex or ORCID.")
            else:
                st.success(f"Identity Verified: **{user_display_name}**. Analyzing research compatibility...")
                
                # 2. Cargar base de datos (Cacheada por st.cache_data)
                df_corpus, df_admin, df_raw_papers = load_local_database()
                
                if df_corpus is not None:
                    # 3. Motor de Recomendación
                    all_texts = [user_text] + df_corpus['Full_Content'].tolist()
                    professor_names = df_corpus['Nombre_Busqueda'].tolist()
                    
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(all_texts)
                    
                    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                    
                    top_indices = cosine_sim.argsort()[-5:][::-1]
                    
                    results = []
                    
                    # --- SECCIÓN 1: TABLA RESUMEN ---
                    for idx in top_indices:
                        prof_name = professor_names[idx]
                        score = cosine_sim[idx]
                        
                        # Match con datos administrativos
                        admin_info = df_admin[
                            (df_admin['NOMBRES'].astype(str) + " " + 
                             df_admin['PRIMER_APELLIDO'].astype(str) + " " + 
                             df_admin['SEGUNDO_APELLIDO'].astype(str)).str.strip() == prof_name
                        ]
                        
                        if admin_info.empty:
                             admin_info = df_admin[
                                (df_admin['NOMBRES'].astype(str) + " " + df_admin['PRIMER_APELLIDO'].astype(str)).apply(lambda x: x in prof_name)
                             ]

                        dept = "Unknown Department"
                        email = "No Email Found"
                        
                        if not admin_info.empty:
                            dept = admin_info.iloc[0]['DEPARTAMENTO']
                            email = admin_info.iloc[0]['CORREO_INSTITUCIONAL']

                        results.append({
                            "Professor": prof_name,
                            "Similarity Score": score,
                            "Department": dept,
                            "Email": email
                        })
                    
                    st.markdown("### Top 5 Collaborative Matches")
                    st.write("Matches based on consolidated research from OpenAlex and ORCID.")
                    
                    df_results = pd.DataFrame(results)
                    
                    st.dataframe(
                        df_results,
                        column_config={
                            "Professor": st.column_config.TextColumn("Professor Name", width="medium"),
                            "Similarity Score": st.column_config.ProgressColumn("Match Score", format="%.2f%%", min_value=0, max_value=1),
                            "Email": st.column_config.LinkColumn("Email"),
                        },
                        hide_index=True,
                        width="stretch"
                    )

                    # --- SECCIÓN 2: DETALLE EXTENDIDO ---
                    st.markdown("---")
                    st.markdown("### Detailed Academic Profiles & Publications")
                    st.write("Explore specific papers and contact details for your top matches.")

                    for res in results:
                        prof_name = res["Professor"]
                        score = res["Similarity Score"]
                        dept = res["Department"]
                        email = res["Email"]

                        # Crear Expander por profesor
                        with st.expander(f" {prof_name}  |  Similarity: {score:.1%}"):
                            
                            c1, c2 = st.columns([1, 2])
                            
                            # Columna Izquierda: Datos de Contacto
                            with c1:
                                st.markdown(f"**Department:**\n{dept}")
                                st.markdown(f"**Email:**\n[{email}](mailto:{email})")
                                st.info("This match is calculated using title keywords and abstract semantics.")

                            # Columna Derecha: Lista de Papers (Usando df_raw_papers)
                            with c2:
                                st.markdown("##### Selected Publications")
                                
                                # Filtrar papers solo de este profesor desde el DF crudo
                                prof_papers = df_raw_papers[df_raw_papers['Nombre_Busqueda'] == prof_name] \
                                                .sort_values(by='Publication_Year', ascending=False) \
                                                .head(5)
                                
                                if not prof_papers.empty:
                                    for _, paper in prof_papers.iterrows():
                                        title = paper.get('Paper_Title', 'Untitled')
                                        year = int(paper['Publication_Year']) if pd.notnull(paper.get('Publication_Year')) else "N/A"
                                        doi = paper.get('Paper_DOI', None) 
                                        
                                        if doi and isinstance(doi, str) and doi.startswith('http'):
                                            link_md = f"[Read Paper]({doi})"
                                        else:
                                            link_md = "(No Link)"
                                        
                                        st.markdown(f"- **{year}** | {title} | {link_md}")
                                else:
                                    st.warning("No specific paper details found in local database.")
                                    
                    # --- SECCIÓN 3: WORD CLOUDS (DESPLEGABLE CON DESCARGA) ---
                    st.markdown("---")
                    st.markdown("### Research Topic Word Clouds")
                    st.write("Visualizing the most frequent research terms (Titles, Keywords, Abstracts).")
                    
                    # 1. Word Cloud del INPUT AUTHOR (Desplegable)
                    with st.expander(f" Show Word Cloud: {user_display_name} (Input Author)"):
                        fig_input = generate_wordcloud_fig(user_text, f"Research Topics: {user_display_name}")
                        if fig_input:
                            st.pyplot(fig_input)
                            
                            # --- Lógica de Descarga Input Author ---
                            buf = BytesIO()
                            fig_input.savefig(buf, format="png", bbox_inches='tight')
                            buf.seek(0)
                            
                            st.download_button(
                                label=" Download Image",
                                data=buf,
                                file_name=f"wordcloud_input_{f_name}.png",
                                mime="image/png"
                            )
                            plt.close(fig_input) # Cerrar para liberar memoria
                        else:
                            st.warning("Not enough text data to generate Word Cloud for input author.")
                    
                    # 2. Word Clouds de los MATCHES (Desplegable General)
                    with st.expander(" Show Word Clouds: Top 5 Matches"):
                        # Usamos columnas para que no sea una lista eterna hacia abajo
                        cols = st.columns(2) 
                        
                        for i, idx in enumerate(top_indices):
                            prof_name = professor_names[idx]
                            
                            # Extraemos el texto completo usado para el TF-IDF
                            prof_text = df_corpus.iloc[idx]['Full_Content']
                            
                            fig_prof = generate_wordcloud_fig(prof_text, f"{prof_name}")
                            
                            # Alternar columnas
                            with cols[i % 2]:
                                if fig_prof:
                                    st.pyplot(fig_prof)
                                    
                                    # --- Lógica de Descarga por Profesor ---
                                    buf_prof = BytesIO()
                                    fig_prof.savefig(buf_prof, format="png", bbox_inches='tight')
                                    buf_prof.seek(0)
                                    
                                    st.download_button(
                                        label=f" Download {prof_name}",
                                        data=buf_prof,
                                        file_name=f"wordcloud_{prof_name}.png",
                                        mime="image/png",
                                        key=f"dl_btn_{i}" # ¡IMPORTANTE! Key única para evitar errores
                                    )
                                    
                                    plt.close(fig_prof)
                                else:
                                    st.warning(f"No text data for {prof_name}")
                    
                    # --- SECCIÓN 4: RED DE CONCEPTOS INDIVIDUALES (VOSviewer style) ---
                    # --- SECCIÓN 4: RED DE CONCEPTOS INDIVIDUALES (VOSviewer style) ---
                    st.markdown("---")
                    st.markdown("### Interactive Research Networks (VOSviewer Style)")
                    st.write("Visualizing distinct conceptual clusters for the Input Author and Top Matches.")

                    # Configuración de física y estilo (VOSviewer-like)
                    # Configuración optimizada para ESTABILIDAD (Quieto estilo VOSviewer)
                    # Configuración optimizada: Líneas FINAS y visualización ESTÁTICA
                    PYVIS_OPTIONS = """
                    var options = {
                    "nodes": {
                        "font": { "size": 16, "face": "tahoma", "color": "#000000" },
                        "borderWidth": 2,
                        "shadow": { "enabled": true, "color": "rgba(0,0,0,0.2)" },
                        "shape": "dot"
                    },
                    "edges": {
                        "color": { 
                        "color": "#848484", 
                        "highlight": "#000000", 
                        "opacity": 0.3  
                        },
                        "width": 0.5,
                        "scaling": {
                        "min": 0.5,
                        "max": 3
                        },
                        "smooth": { "enabled": true, "type": "continuous" }
                    },
                    "physics": {
                        "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08,
                        "damping": 0.4,
                        "avoidOverlap": 0
                        },
                        "maxVelocity": 50,
                        "minVelocity": 0.1,
                        "solver": "forceAtlas2Based",
                        "stabilization": {
                        "enabled": true,
                        "iterations": 1000,
                        "updateInterval": 25,
                        "onlyDynamicEdges": false,
                        "fit": true
                        }
                    },
                    "interaction": {
                        "hover": true,
                        "navigationButtons": true,
                        "keyboard": true,
                        "tooltipDelay": 200
                    }
                    }
                    """

                    def render_pyvis_graph(text_data, element_id, graph_height="600px"):

                        try:
                            # IMPORTANTE: Convertimos el string a lista [text_data] para que CountVectorizer funcione
                            # Esto activa la lógica min_df=1 en create_term_network
                            G = create_term_network([text_data], max_terms=55, n_clusters=5)
                            
                            if len(G.nodes) == 0:
                                st.info("Not enough data to generate a network for this author.")
                                return

                            # Configuración PyVis
                            net = Network(height=graph_height, width="100%", bgcolor="#ffffff", font_color="black")
                            net.from_nx(G)
                            
                            # Aplicar opciones VOSviewer
                            net.set_options(PYVIS_OPTIONS)
                            
                            # Guardar archivo HTML único
                            path_html = f"network_{element_id}.html"
                            net.save_graph(path_html)
                            
                            # Leer e incrustar
                            with open(path_html, 'r', encoding='utf-8') as f:
                                source_code = f.read()
                            components.html(source_code, height=int(graph_height.replace('px',''))+10, scrolling=False)
                            
                        except Exception as e:
                            st.error(f"Error generating network: {e}")

                    # 1. GRAFO DEL INPUT AUTHOR
                    with st.expander(f"Show Network: {user_display_name} (Input Author)", expanded=True):
                        st.caption("Conceptual map based on Input Author's titles, keywords, and abstracts.")
                        render_pyvis_graph(user_text, "input_author")

                    # 2. GRAFOS DE LOS MATCHES
                    with st.expander("Show Networks: Top 5 Matches"):
                        st.caption("Individual conceptual maps for each recommended professor.")
                        
                        # Creamos pestañas para cada profesor
                        match_tabs = st.tabs([f"{professor_names[i]}" for i in top_indices])
                        
                        for i, tab in enumerate(match_tabs):
                            idx = top_indices[i]
                            prof_name = professor_names[idx]
                            
                            # Extraemos el texto completo del profesor
                            prof_text = df_corpus.iloc[idx]['Full_Content']
                            
                            with tab:
                                st.markdown(f"#### Concept Network: {prof_name}")
                                # Generamos el grafo con ID único
                                render_pyvis_graph(prof_text, f"match_{i}")