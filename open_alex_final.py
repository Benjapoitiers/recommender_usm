import pandas as pd
import pyalex
from pyalex import Authors, Works
import time
import unicodedata
import datetime

# Configuración
pyalex.config.email = "benjapoitiers@gmail.com" 

# CONSTANTE: ID de la Universidad Técnica Federico Santa María en OpenAlex
# Esto es más seguro que buscar por string cada vez.
USM_INSTITUTION_ID = "I8606547" 

# --- FUNCIONES AUXILIARES ---

#Reconstruye el abstract desde el índice invertido.
def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None
    max_index = max([max(positions) for positions in inverted_index.values()])
    abstract_list = [""] * (max_index + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            abstract_list[pos] = word
    return " ".join(abstract_list)

#Elimina tildes y pone minúsculas.
def normalizar(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')


#Verifica coincidencia de apellidos considerando:
#Apellidos completos
#Iniciales
#Apellidos compuestos
def verificar_match_segundo_apellido(nombre_openalex, apellido_paterno, apellido_materno):
    parts = normalizar(nombre_openalex).split()
    if not parts: 
        return False
    
    paterno_norm = normalizar(apellido_paterno) if apellido_paterno else ""
    materno_norm = normalizar(apellido_materno) if apellido_materno else ""
    
    # Si no tenemos apellidos para comparar, aceptamos (modo permisivo)
    if not paterno_norm:
        return True
    
    parts_set = set(parts)
    
    # 1. Coincidencia exacta en cualquier posición
    if paterno_norm in parts_set:
        return True
    if materno_norm and materno_norm in parts_set:
        return True
    
    # 2. Coincidencia por inicial del segundo apellido
    for part in parts:
        if len(part) == 1:  # Es una inicial
            if materno_norm and part == materno_norm[0]:
                return True
            if paterno_norm and part == paterno_norm[0]:
                return True
    
    # 3. Si tiene materno y el último apellido no coincide con ninguno -> RECHAZAR
    #Aqui se soluciona el caso de Felipe Escudero 
    if materno_norm:
        ultimo = parts[-1]
        if ultimo != paterno_norm and ultimo != materno_norm:
            return False
    
    return True  # En duda, aceptamos (modo permisivo)

# --- FUNCIÓN DE BÚSQUEDA HÍBRIDA (NÚCLEO DE LA MEJORA) ---


#Estrategia USM (Estricta): Filtra por afiliación a la USM.
#Estrategia Global (Respaldo): Busca en todo el mundo pero valida apellidos.
def buscar_profesor_hibrido(query_nombre, apellido1, apellido2):
    # --- ESTRATEGIA 1: Filtrar por Afiliación USM (Alta Confianza) ---
    try:
        candidates_usm = Authors().search(query_nombre).filter(last_known_institution={"id": USM_INSTITUTION_ID}).get()
        if candidates_usm:
            # Si encontramos a alguien filtrando por USM, confiamos en el primer resultado (normalmente el más relevante)
            return candidates_usm[0], "USM_AFFILIATION_MATCH"
    except Exception:
        pass # Si falla la API o no encuentra, pasamos a la estrategia 2

    # --- ESTRATEGIA 2: Búsqueda Global + Validación Apellidos (Respaldo) ---
    try:
        candidates_global = Authors().search(query_nombre).get()
        for cand in candidates_global:
            # Aquí aplicamos tu preocupación: NO tomamos el primero ciegamente.
            # Verificamos que los apellidos coincidan.
            if verificar_match_segundo_apellido(cand['display_name'], apellido1, apellido2):
                
                # Opcional: Scoring extra simple
                # Si en su afiliación (string) dice "Santa Maria" aunque el ID no sea el principal, es un plus.
                affiliations = str(cand.get('last_known_institution', {})).lower()
                if 'santa maria' in affiliations or 'usm' in affiliations:
                    return cand, "GLOBAL_MATCH_WITH_USM_TEXT"
                
                return cand, "GLOBAL_MATCH_VERIFIED"
                
    except Exception:
        pass

    return None, "NOT_FOUND"

# --- MAIN ---

def main():
    archivo_entrada = "ACAD-DOC 19_ENE_26.xlsx"
    try:
        df_input = pd.read_excel(archivo_entrada)
        print(f"Iniciando proceso HÍBRIDO con {len(df_input)} profesores...")
    except FileNotFoundError:
        print(f"Error: No se encontró '{archivo_entrada}'.")
        return

    resultados_papers = []

    for index, row in df_input.iterrows():
        # 1. Preparar datos
        nombres = str(row.get('NOMBRES', '')).strip()
        apellido1 = str(row.get('PRIMER_APELLIDO', '')).strip()
        apellido2 = str(row.get('SEGUNDO_APELLIDO', '')).strip()
        nombre_pref = str(row.get('NOMBRE_PREFERIDO', '')).strip()
        
        nombre_completo_reporte = f"{nombres} {apellido1} {apellido2}".strip()
        
        # 2. Variaciones de nombre
        intentos_busqueda = []
        if nombres and apellido1 and apellido2: intentos_busqueda.append(f"{nombres} {apellido1} {apellido2}")
        if nombre_pref and nombre_pref.lower() != "nan": intentos_busqueda.append(nombre_pref)
        if nombres and apellido1: intentos_busqueda.append(f"{nombres} {apellido1}")
        primer_nombre = nombres.split(' ')[0] if nombres else ""
        if primer_nombre and apellido1 and primer_nombre != nombres: intentos_busqueda.append(f"{primer_nombre} {apellido1}")
        intentos_busqueda = list(dict.fromkeys(intentos_busqueda))

        encontrado = False
        top_author = None
        metodo_hallazgo = ""

        print(f"[{index+1}/{len(df_input)}] Buscando: {nombre_completo_reporte}...")

        # 3. Ejecutar Búsqueda Híbrida
        for query in intentos_busqueda:
            author, metodo = buscar_profesor_hibrido(query, apellido1, apellido2)
            
            if author:
                top_author = author
                encontrado = True
                print(f"   -> ENCONTRADO ({metodo}): {top_author['display_name']}")
                break # Dejamos de probar variaciones de nombre si ya lo encontramos

        if not encontrado:
            print(f"   -> NO encontrado.")
            continue 

        # 4. Extracción de Papers
        try:
            author_id = top_author['id']
            author_display_name = top_author['display_name']
            
            works = Works().filter(author={"id": author_id}).get()
            print(f"   -> Extrayendo {len(works)} papers...")

            for work in works:
                concepts = [c['display_name'] for c in work.get('concepts', [])]
                keywords_str = ", ".join(concepts[:5])
                abstract_text = reconstruct_abstract(work.get('abstract_inverted_index'))

                resultados_papers.append({
                    'Nombre_Busqueda': nombre_completo_reporte,
                    'OpenAlex_Author_Name': author_display_name,
                    'OpenAlex_Author_ID': author_id,
                    'Match_Method': metodo, # Útil para depurar
                    'Paper_Title': work.get('title'),
                    'Publication_Year': work.get('publication_year'),
                    'DOI': work.get('doi'),
                    'Cited_By_Count': work.get('cited_by_count'),
                    'Keywords': keywords_str,
                    'Abstract': abstract_text
                })
            time.sleep(0.2)

        except Exception as e:
            print(f"   -> Error extrayendo papers: {e}")

        # 5. Respaldo Parcial
        if (index + 1) % 50 == 0:
            try:
                pd.DataFrame(resultados_papers).to_excel(f"backup_parcial_{index+1}.xlsx", index=False)
            except: pass

    # 6. Guardado Final Seguro
    if resultados_papers:
        df_output = pd.DataFrame(resultados_papers)
        nombre_base = "Output_OpenAlex_Papers_final.xlsx"
        try:
            df_output.to_excel(nombre_base, index=False)
            print(f"\n¡ÉXITO! Datos guardados en '{nombre_base}'")
        except PermissionError:
            ts = datetime.datetime.now().strftime("%H%M%S")
            df_output.to_excel(f"Output_OpenAlex_Papers_SAFE_{ts}.xlsx", index=False)
            print(f"\n¡Archivo guardado con nombre alternativo por bloqueo!")

if __name__ == "__main__":
    main()