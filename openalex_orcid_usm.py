'''Codigo creado por: Benjamin Fuentes Valdebenito'''

import pandas as pd
import pyalex
from pyalex import Authors, Works
import requests
import time
import unicodedata
import re
from urllib.parse import quote
import datetime


pyalex.config.email = "benjapoitiers@gmail.com"
USM_INSTITUTION_ID = "I8606547"

# Normalización robusta: minúsculas, sin tildes, sin puntuación extra.
def normalizar(texto):
    if not isinstance(texto, str): 
        return ""
    texto = texto.lower()
    # Eliminar tildes
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) 
                    if unicodedata.category(c) != 'Mn')
    # Reemplazar puntos y comas por espacios
    texto = texto.replace('.', ' ').replace(',', ' ')
    # Quitar espacios múltiples
    return " ".join(texto.split())

# Reconstruye el abstract legible a partir del índice invertido.
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


#Validación estricta de identidad:
#Si el candidato tiene 2+ nombres, TODOS deben coincidir
#Si tiene 1 nombre, debe coincidir con uno de los dos nombres del autor buscado
#Valida obligatoriamente apellido paterno

def verificar_identidad_completa(nombre_openalex, row_nombres, row_apellido1, row_apellido2):

    if not nombre_openalex:
        return False

    oa_norm = normalizar(nombre_openalex)
    parts_oa = oa_norm.split()

    if len(parts_oa) < 2:
        return False 

    # Preparar datos buscados
    nombres_norm = normalizar(row_nombres)
    ape1_norm = normalizar(row_apellido1)
    ape2_norm = normalizar(row_apellido2)

    if not ape1_norm:
        return False  

    # PASO 1: IDENTIFICAR POSICIÓN DEL APELLIDO PATERNO ---
    idx_ape = -1
    for i, token in enumerate(parts_oa):
        if token == ape1_norm:
            idx_ape = i
            break

    if idx_ape == -1:
        return False 

    # PASO 2: AISLAR NOMBRES DE PILA ---
    if idx_ape == 0 and len(parts_oa) > 1:
        nombres_pila_cand = parts_oa[1:]
    else:
        nombres_pila_cand = parts_oa[:idx_ape]

    if not nombres_pila_cand:
        return False

    #  PASO 3: VALIDACIÓN ESTRICTA DE NOMBRES 
    # Obtener los nombres de búsqueda
    nombres_busqueda = nombres_norm.split() if nombres_norm else []
    
    if not nombres_busqueda:
        return False

    # REGLA CRÍTICA: Cada nombre del candidato debe coincidir con alguno de la búsqueda
    # Esto evita falsos positivos como "Danny Smith" vs "Benjamin Smith"
    for nombre_token_cand in nombres_pila_cand:
        match_encontrado = False
        
        for token_q in nombres_busqueda:
            # A) Coincidencia Exacta
            if nombre_token_cand == token_q:
                match_encontrado = True
                break
            
            # B) Iniciales (Cand="b" vs Query="benjamin")
            if len(nombre_token_cand) == 1 and token_q.startswith(nombre_token_cand):
                match_encontrado = True
                break
            
            # C) Iniciales Inversa (Cand="benjamin" vs Query="b")
            if len(token_q) == 1 and nombre_token_cand.startswith(token_q):
                match_encontrado = True
                break
        
        if not match_encontrado:
            return False 

    # PASO 4: VALIDACIÓN SEGUNDO APELLIDO 
    if ape2_norm and len(parts_oa) > idx_ape + 1:
        segundo_apellido = parts_oa[idx_ape + 1]
        # Si es sustancial y NO coincide, rechazar
        if len(segundo_apellido) > 2 and segundo_apellido != ape2_norm:
            return False

    return True

# MOTOR DE BÚSQUEDA EN OPENALEX 

def buscar_profesor_openalex(nombres, apellido1, apellido2):

    # Preparar intentos de búsqueda
    intentos = []
    
    if nombres and apellido1:
        primer_nombre = nombres.split()[0]
        intentos.append(f"{primer_nombre} {apellido1}")  
        intentos.append(f"{nombres} {apellido1}")
    
    if nombres and apellido1 and apellido2:
        intentos.append(f"{nombres} {apellido1} {apellido2}")

    # Eliminar duplicados
    intentos = list(dict.fromkeys(intentos))

    top_author = None
    
    for query in intentos:
        try:
            # ESTRATEGIA 1: Filtro por afiliación USM
            candidates_usm = Authors().search(query).filter(
                last_known_institutions={"id": USM_INSTITUTION_ID}
            ).get()
            
            if candidates_usm:
                for cand in candidates_usm:
                    if verificar_identidad_completa(
                        cand['display_name'], 
                        nombres, 
                        apellido1, 
                        apellido2
                    ):
                        top_author = cand
                        break
            
            if top_author:
                break

        except Exception as e:
            pass
    
    # ESTRATEGIA 2: Búsqueda global
    if not top_author:
        for query in intentos:
            try:
                candidates_global = Authors().search(query).get()
                
                if candidates_global:
                    for cand in candidates_global[:10]:  
                        if verificar_identidad_completa(
                            cand['display_name'],
                            nombres,
                            apellido1,
                            apellido2
                        ):
                            top_author = cand
                            break
                
                if top_author:
                    break
                    
            except Exception as e:
                pass

    if not top_author:
        return [], None, None

    # Extraer ORCID vinculado
    orcid_path = None
    oa_orcid_url = top_author.get('ids', {}).get('orcid')
    if oa_orcid_url:
        orcid_path = oa_orcid_url.replace("https://orcid.org/", "")

    # Extraer papers
    try:
        works = Works().filter(author={"id": top_author['id']}).get()
        papers = []
        
        for w in works:
            abstract = reconstruct_abstract(w.get('abstract_inverted_index'))
            keywords = [c['display_name'] for c in w.get('concepts', [])]
            
            papers.append({
                'Title': w.get('title'),
                'Year': w.get('publication_year'),
                'DOI': w.get('doi'),
                'Keywords': ", ".join(keywords[:6]),
                'Abstract': abstract,
                'Source': 'OpenAlex',
                'Author_ID': top_author['id']
            })
        
        return papers, top_author['display_name'], orcid_path
        
    except Exception as e:
        return [], top_author['display_name'], orcid_path

# MOTOR DE BÚSQUEDA EN ORCID 

def buscar_orcid_api(orcid_path=None, nombres=None, apellido1=None, apellido2=None):

    headers = {'Accept': 'application/json'}
    target_orcid = orcid_path
    found_name_orcid = None

    # CASO 1: Si tenemos ORCID vinculado de OpenAlex, úsalo directamente
    if target_orcid:
        found_name_orcid = f"ORCID:{target_orcid}"
    else:
        # CASO 2: Búsqueda por nombre si no tenemos ORCID previo
        if not nombres or not apellido1:
            return [], None

        primer_nombre = nombres.split()[0] if nombres else ""
        query = f"given-names:{primer_nombre} AND family-name:{apellido1}"
        
        try:
            url_search = f"https://pub.orcid.org/v3.0/search/?q={quote(query)}"
            resp = requests.get(url_search, headers=headers, timeout=10)
            
            if resp.status_code != 200:
                return [], None

            results = resp.json().get('result', [])
            
            #Si se queda congelado entonces, prueba con: 
            # for res in results[:15]: Asi solo busca los primeros 15 de ORCID y no todos. 
            for res in results:
                oid = res.get('orcid-identifier', {}).get('path')
                if not oid:
                    continue

                # Verificar que sea el correcto
                url_p = f"https://pub.orcid.org/v3.0/{oid}/person"
                rp = requests.get(url_p, headers=headers, timeout=10)
                
                if rp.status_code == 200:
                    dp = rp.json()
                    g = dp.get('name', {}).get('given-names', {}).get('value', '')
                    f = dp.get('name', {}).get('family-name', {}).get('value', '')
                    full_n = f"{g} {f}"

                    # Validar identidad
                    if verificar_identidad_completa(full_n, nombres, apellido1, apellido2):
                        target_orcid = oid
                        found_name_orcid = full_n
                        break
                
                time.sleep(0.3)

        except Exception as e:
            return [], None

    if not target_orcid:
        return [], None

    # EXTRAER PAPERS DE ORCID
    papers = []
    url_works = f"https://pub.orcid.org/v3.0/{target_orcid}/works"
    
    try:
        r_w = requests.get(url_works, headers=headers, timeout=10)
        
        if r_w.status_code == 200:
            groups = r_w.json().get('group', [])
            
            for g in groups:
                summaries = g.get('work-summary', [])
                if not summaries:
                    continue

                work = summaries[0]
                title = work.get('title', {}).get('title', {}).get('value', '')
                year_val = work.get('publication-date', {}).get('year', {}).get('value', None)
                
                doi = None
                for eid in work.get('external-ids', {}).get('external-id', []):
                    if eid.get('external-id-type') == 'doi':
                        doi = eid.get('external-id-value')
                        if doi and 'https://doi.org/' not in doi:
                            doi = f"https://doi.org/{doi}"
                        break

                papers.append({
                    'Title': title,
                    'Year': year_val,
                    'DOI': doi,
                    'Keywords': "",
                    'Abstract': "",
                    'Source': 'ORCID',
                    'Author_ID': target_orcid
                })

        return papers, found_name_orcid

    except Exception as e:
        return [], found_name_orcid


def procesar_todo():
    archivo = "ACAD-DOC_PUBLICO.xlsx"
    
    try:
        df = pd.read_excel(archivo)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{archivo}'")
        return
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return

    # Filtro de observaciones
    if 'OBSERVACIONES' in df.columns:
        patron_excluir = r'renuncia|fallecimiento|jubila'
        filtro = ~df['OBSERVACIONES'].astype(str).str.contains(patron_excluir, case=False, na=False)
        df = df[filtro]

    all_data = []
    contador_procesados = 0
    contador_encontrados_oa = 0
    contador_mejorados_orcid = 0

    print("\n" + "="*80)
    print("BÚSQUEDA HÍBRIDA: OpenAlex + ORCID")
    print("="*80)
    print(f"Total de profesores a procesar: {len(df)}\n")

    for idx, row in df.iterrows():
        nombres = str(row.get('NOMBRES', '')).strip()
        ape1 = str(row.get('PRIMER_APELLIDO', '')).strip()
        ape2 = str(row.get('SEGUNDO_APELLIDO', '')).strip()

        if not nombres or not ape1 or nombres.lower() == 'nan' or ape1.lower() == 'nan':
            continue

        contador_procesados += 1
        full_name_busqueda = f"{nombres} {ape1} {ape2}".strip()

        #  BÚSQUEDA EN OPENALEX 
        oa_papers, oa_name_found, linked_orcid = buscar_profesor_openalex(nombres, ape1, ape2)

        if oa_name_found:
            contador_encontrados_oa += 1
            print(f"[{contador_procesados}] {full_name_busqueda}")
            print(f"    ✓ OpenAlex:     {oa_name_found} ({len(oa_papers)} papers)")
        else:
            print(f"[{contador_procesados}] {full_name_busqueda}")
            print(f"    ✗ OpenAlex:     No encontrado")

        #  BÚSQUEDA EN ORCID 
        orcid_papers, orcid_name_found = buscar_orcid_api(
            orcid_path=linked_orcid,
            nombres=nombres,
            apellido1=ape1,
            apellido2=ape2
        )

        #  CONSOLIDACIÓN 
        final_papers = {}

        # Agregar papers de OpenAlex
        for p in oa_papers:
            key = p['DOI'] if p['DOI'] else normalizar(p['Title'])
            if key:
                final_papers[key] = p

        # Agregar papers de ORCID (sin duplicados)
        added_from_orcid = 0
        for p in orcid_papers:
            key = p['DOI'] if p['DOI'] else normalizar(p['Title'])
            if key and key not in final_papers:
                final_papers[key] = p
                added_from_orcid += 1

        # OUTPUT A CONSOLA 
        if len(orcid_papers) > 0:
            contador_mejorados_orcid += 1
            print(f"    ✓ ORCID:        {orcid_name_found} (+{added_from_orcid} papers nuevos)")
        
        print(f"    → TOTAL:        {len(final_papers)} papers consolidados")
        print()

        # --- GUARDAR DATOS ---
        for p in final_papers.values():
            all_data.append({
                'Nombre_Busqueda': full_name_busqueda,
                'OpenAlex_Encontrado': oa_name_found if oa_name_found else 'No',
                'OpenAlex_Papers': len([x for x in final_papers.values() if x['Source'] == 'OpenAlex']),
                'ORCID_Integrado': orcid_name_found if orcid_name_found else 'No',
                'ORCID_Papers_Nuevos': added_from_orcid if added_from_orcid > 0 else 0,
                'Total_Consolidado': len(final_papers),
                'Paper_Title': p['Title'],
                'Paper_Year': p['Year'],
                'Paper_DOI': p['DOI'],
                'Paper_Keywords': p['Keywords'],
                'Paper_Source': p['Source'],
                'Paper_Abstract': p['Abstract'] if p['Abstract'] else ""
            })

        time.sleep(0.3)  # Rate limiting

        # Se ejecuta cada 100 profesores PROCESADOS (tengan papers o no)
        if contador_procesados % 100 == 0 and all_data:
            try:
                # Crea un nombre de archivo único para este lote (ej: backup_100.xlsx)
                nombre_backup = f"backup_progreso_{contador_procesados}.xlsx"
                pd.DataFrame(all_data).to_excel(nombre_backup, index=False)
                
                print("*" * 60)
                print(f"  [AUTO-GUARDADO] Backup generado: {nombre_backup}")
                print(f"  Registros acumulados hasta ahora: {len(all_data)}")
                print("*" * 60 + "\n")
            except Exception as e:
                print(f"  [ERROR] No se pudo crear el backup: {e}\n")

    # REPORTE FINAL 
    print("\n" + "="*80)
    print("RESUMEN DEL PROCESO")
    print("="*80)
    print(f"Profesores procesados:           {contador_procesados}")
    print(f"Encontrados en OpenAlex:         {contador_encontrados_oa}")
    print(f"Mejorados con ORCID:             {contador_mejorados_orcid}")
    print(f"Total de registros datos:        {len(all_data)}")
    print("="*80 + "\n")

    # Guardado final
    if all_data:
        df_out = pd.DataFrame(all_data)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"Output_Papers_Hibrido_{ts}.xlsx"
        
        try:
            df_out.to_excel(nombre_archivo, index=False)
            print(f"✓ Archivo guardado: {nombre_archivo}")
        except PermissionError:
            nombre_archivo = f"Output_SAFE_{ts}.xlsx"
            df_out.to_excel(nombre_archivo, index=False)
            print(f"✓ Archivo guardado como: {nombre_archivo}")
    else:
        print("✗ No se encontraron papers para procesar")

if __name__ == "__main__":
    procesar_todo()
