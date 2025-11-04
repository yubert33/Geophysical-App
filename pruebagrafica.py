import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from io import BytesIO
import re
from datetime import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings
import tempfile
import os
warnings.filterwarnings('ignore')

# Verificar disponibilidad de lasio
try:
    import lasio
    LASIO_AVAILABLE = True
except ImportError:
    LASIO_AVAILABLE = False

# Configuración de la página
st.set_page_config(layout="wide", page_title="Sistema Avanzado de Análisis Petrofísico")
st.title("🛢️ Sistema Avanzado de Análisis Petrofísico")
st.write("**Herramienta integral para visualización, análisis e interpretación de registros de pozo**")

# Mostrar advertencia si lasio no está disponible
if not LASIO_AVAILABLE:
    st.warning("""
    ⚠️ **Para mejor procesamiento de archivos LAS, instala lasio:**
    ```bash
    pip install lasio
    ```
    """)
    # ================================
# 📁 FUNCIONES MEJORADAS CON LASIO PARA ARCHIVOS LAS
# ================================
def procesar_las_con_lasio(archivo):
    """
    Función mejorada para procesar archivos LAS usando la librería lasio
    """
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as temp_file:
            temp_file.write(archivo.getvalue())
            temp_path = temp_file.name
        
        try:
            import lasio
            st.info("🔍 Procesando archivo LAS con lasio...")
            
            # Leer archivo LAS con lasio
            las = lasio.read(temp_path)
            
            # Convertir a DataFrame
            df = las.df()
            df.reset_index(inplace=True)  # La profundidad viene como índice
            
            # Obtener información de las curvas (CORREGIDO)
            curve_info = []
            for mnemonic, curve in las.curves.items():
                curve_info.append({
                    'MNEMONIC': mnemonic,
                    'UNIT': curve.unit if curve.unit else '',
                    'DESCRIPTION': curve.descr if curve.descr else '',
                    'VALUE': curve.value if curve.value else '',
                    'API_CODE': curve.API_code if hasattr(curve, 'API_code') else ''  # CORREGIDO
                })
            
            # Obtener metadatos del pozo
            well_info = []
            if hasattr(las, 'well') and las.well:
                for mnemonic, item in las.well.items():
                    well_info.append({
                        'MNEMONIC': mnemonic,
                        'UNIT': item.unit if hasattr(item, 'unit') else '',
                        'VALUE': item.value if hasattr(item, 'value') else '',
                        'DESCRIPTION': item.descr if hasattr(item, 'descr') else ''
                    })
            
            # Obtener información de parámetros
            parameter_info = []
            if hasattr(las, 'params') and las.params:
                for mnemonic, param in las.params.items():
                    parameter_info.append({
                        'MNEMONIC': mnemonic,
                        'UNIT': param.unit if hasattr(param, 'unit') else '',
                        'VALUE': param.value if hasattr(param, 'value') else '',
                        'DESCRIPTION': param.descr if hasattr(param, 'descr') else ''
                    })
            
            sections = {
                'curve': curve_info,
                'well': well_info,
                'parameter': parameter_info,
                'version': getattr(las, 'version', [])
            }
            
            # Limpiar archivo temporal
            os.unlink(temp_path)
            
            st.success(f"✅ Archivo LAS procesado con lasio: {len(df)} filas, {len(df.columns)} columnas")
            return df, sections
            
        except ImportError:
            st.warning("⚠️ Lasio no está instalado. Usando procesamiento manual...")
            os.unlink(temp_path)
            return procesar_las_manual(archivo)
            
    except Exception as e:
        st.error(f"❌ Error procesando LAS con lasio: {str(e)}")
        import traceback
        st.error(f"Detalles: {traceback.format_exc()}")
        # Fallback al método manual
        return procesar_las_manual(archivo)

def procesar_las_manual(archivo):
    """
    Función de respaldo para procesar archivos LAS manualmente
    """
    try:
        # Leer todo el contenido
        contenido = archivo.getvalue().decode('utf-8', errors='ignore')
        lineas = contenido.split('\n')
        
        st.info("🔍 Procesando archivo LAS (enfoque manual)...")
        
        # Buscar la sección de datos (~A)
        inicio_datos = None
        for i, linea in enumerate(lineas):
            if linea.strip().startswith('~A'):
                inicio_datos = i + 1
                break
        
        if inicio_datos is None:
            st.error("❌ No se encontró la sección ~A en el archivo")
            return None, None
        
        # Extraer datos después de ~A
        datos = []
        for i in range(inicio_datos, len(lineas)):
            linea = lineas[i].strip()
            if not linea or linea.startswith('#'):
                continue
                
            # Dividir por espacios/tabs y filtrar elementos vacíos
            partes = re.split(r'\s+', linea)
            partes = [p for p in partes if p]
            
            # Verificar si tiene números
            if partes and any(parte.replace('.', '').replace('-', '').isdigit() for parte in partes):
                datos.append(partes)
        
        if not datos:
            st.error("❌ No se encontraron datos numéricos después de ~A")
            return None, None
        
        st.info(f"✅ {len(datos)} filas de datos encontradas")
        
        # USAR NOMBRES DE COLUMNAS MANUALES
        column_names = ['DEPTH', 'CALI', 'SP', 'ILM', 'ILD', 'LAT', 'CILD']
        
        # Verificar que tenemos suficientes columnas
        num_columnas_datos = len(datos[0])
        if num_columnas_datos != len(column_names):
            st.warning(f"⚠️ Número de columnas esperado: {len(column_names)}, encontrado: {num_columnas_datos}")
            # Ajustar nombres si es necesario
            if num_columnas_datos > len(column_names):
                column_names.extend([f'COL_{i}' for i in range(len(column_names), num_columnas_datos)])
            else:
                column_names = column_names[:num_columnas_datos]
        
        # Crear DataFrame
        data_records = []
        for fila in datos:
            processed_values = []
            for val in fila:
                try:
                    processed_values.append(float(val))
                except ValueError:
                    processed_values.append(val)
            data_records.append(processed_values)
        
        df = pd.DataFrame(data_records, columns=column_names)
        
        # Reemplazar valores NULL
        null_values = [-999.25, -999.250000, -999.2500]
        for null_val in null_values:
            df.replace(null_val, np.nan, inplace=True)
        
        # Extraer información de secciones para metadata
        sections = {}
        current_section = None
        
        for linea in lineas:
            linea_limpia = linea.strip()
            
            if linea_limpia.startswith('~'):
                if 'V' in linea_limpia:
                    current_section = 'version'
                elif 'W' in linea_limpia:
                    current_section = 'well'
                elif 'C' in linea_limpia:
                    current_section = 'curve'
                elif 'P' in linea_limpia:
                    current_section = 'parameter'
                else:
                    current_section = 'other'
                
                if current_section not in sections:
                    sections[current_section] = []
                continue
            
            if current_section and linea_limpia and not linea_limpia.startswith('#'):
                sections[current_section].append(linea_limpia)
        
        st.success(f"✅ Archivo LAS procesado: {len(df)} filas, {len(df.columns)} columnas")
        return df, sections
        
    except Exception as e:
        st.error(f"❌ Error procesando LAS manualmente: {str(e)}")
        return None, None
def descargar_las_como_excel(df, sections, nombre_archivo="datos_las_convertidos.xlsx"):
    """Crea archivo Excel descargable con información completa de curvas usando lasio"""
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # ===== HOJA 1: DATOS PRINCIPALES =====
            df.to_excel(writer, index=False, sheet_name='DATOS')
            
            # ===== HOJA 2: METADATOS Y INFORMACIÓN DE CURVAS =====
            metadata_rows = []
            
            # Agregar información de la sección de curvas primero (más importante)
            if 'curve' in sections and sections['curve']:
                metadata_rows.append("=== CURVE INFORMATION SECTION ===")
                metadata_rows.append("MNEMONIC\tUNIT\tDESCRIPTION\tVALUE\tAPI_CODE")
                metadata_rows.append("--------\t----\t-----------\t-----\t--------")
                
                # Verificar si es una lista de diccionarios (lasio) o strings (manual)
                if isinstance(sections['curve'][0], dict):
                    for curve_info in sections['curve']:
                        # CORREGIDO: Usar get() para evitar errores si falta algún campo
                        row = f"{curve_info.get('MNEMONIC', '')}\t{curve_info.get('UNIT', '')}\t{curve_info.get('DESCRIPTION', '')}\t{curve_info.get('VALUE', '')}\t{curve_info.get('API_CODE', '')}"
                        metadata_rows.append(row)
                else:
                    for line in sections['curve']:
                        if line.strip() and not line.startswith('#'):
                            metadata_rows.append(line)
                metadata_rows.append("")  # Línea en blanco
            
            # Agregar información del pozo
            if 'well' in sections and sections['well']:
                metadata_rows.append("=== WELL INFORMATION SECTION ===")
                metadata_rows.append("MNEMONIC\tUNIT\tVALUE\tDESCRIPTION")
                metadata_rows.append("--------\t----\t-----\t-----------")
                
                if isinstance(sections['well'][0], dict):
                    for well_info in sections['well']:
                        row = f"{well_info.get('MNEMONIC', '')}\t{well_info.get('UNIT', '')}\t{well_info.get('VALUE', '')}\t{well_info.get('DESCRIPTION', '')}"
                        metadata_rows.append(row)
                else:
                    for line in sections['well']:
                        if line.strip() and not line.startswith('#'):
                            metadata_rows.append(line)
                metadata_rows.append("")  # Línea en blanco
            
            # Agregar información de parámetros
            if 'parameter' in sections and sections['parameter']:
                metadata_rows.append("=== PARAMETER INFORMATION SECTION ===")
                metadata_rows.append("MNEMONIC\tUNIT\tVALUE\tDESCRIPTION")
                metadata_rows.append("--------\t----\t-----\t-----------")
                
                if isinstance(sections['parameter'][0], dict):
                    for param_info in sections['parameter']:
                        row = f"{param_info.get('MNEMONIC', '')}\t{param_info.get('UNIT', '')}\t{param_info.get('VALUE', '')}\t{param_info.get('DESCRIPTION', '')}"
                        metadata_rows.append(row)
                else:
                    for line in sections['parameter']:
                        if line.strip() and not line.startswith('#'):
                            metadata_rows.append(line)
                metadata_rows.append("")  # Línea en blanco
            
            # Crear DataFrame de metadatos
            if metadata_rows:
                metadata_df = pd.DataFrame(metadata_rows, columns=['METADATA'])
                metadata_df.to_excel(writer, sheet_name='METADATOS', index=False)
            
            # ===== HOJA 3: RESUMEN ESTADÍSTICO =====
            if not df.empty:
                stats_df = df.describe(include='all')
                stats_df.to_excel(writer, sheet_name='ESTADISTICAS')
        
        output.seek(0)
        
        st.download_button(
            label="📥 Descargar como Excel (Completo)",
            data=output,
            file_name=nombre_archivo,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        return True
    except Exception as e:
        st.error(f"Error al crear Excel: {e}")
        return False

def descargar_como_excel(df, nombre_archivo="datos_convertidos.xlsx"):
    """Crea archivo Excel descargable simple"""
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Datos_Pozo')
        output.seek(0)
        
        st.download_button(
            label="📥 Descargar como Excel",
            data=output,
            file_name=nombre_archivo,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        return True
    except Exception as e:
        st.error(f"Error al crear Excel: {e}")
        return False

def detect_file_type(archivo):
    """Detecta tipo de archivo de manera más robusta"""
    try:
        # Guardar posición actual
        current_pos = archivo.tell()
        
        # Leer primeros bytes para verificar signature
        primeros_bytes = archivo.read(10)
        archivo.seek(current_pos)  # Volver al inicio
        
        # Verificar si es Excel por signature
        if primeros_bytes.startswith(b'PK'):  # ZIP signature (Excel es ZIP)
            return 'EXCEL'
        
        # Leer contenido para verificar LAS
        contenido = archivo.read(2000).decode('utf-8', errors='ignore')
        archivo.seek(current_pos)  # Volver al inicio
        
        # Verificar patrones LAS
        las_patterns = ['~VERSION', '~WELL', '~CURVE', '~A']
        if any(pattern in contenido for pattern in las_patterns):
            return 'LAS'
        
        # Verificar por extensión como fallback
        nombre_archivo = archivo.name.lower()
        if nombre_archivo.endswith('.las'):
            return 'LAS'
        elif nombre_archivo.endswith(('.xlsx', '.xls')):
            return 'EXCEL'
        else:
            return 'UNKNOWN'
            
    except Exception as e:
        # Fallback por extensión
        try:
            nombre_archivo = archivo.name.lower()
            if nombre_archivo.endswith('.las'):
                return 'LAS'
            elif nombre_archivo.endswith(('.xlsx', '.xls')):
                return 'EXCEL'
            else:
                return 'UNKNOWN'
        except:
            return 'UNKNOWN'
# ================================
# 🔄 FUNCIONES DE MULTICARGA DE ARCHIVOS
# ================================
def procesar_multiple_archivos(archivos):
    """Procesa múltiples archivos y retorna DataFrames combinados"""
    dataframes = []
    info_archivos = []
    
    for i, archivo in enumerate(archivos):
        with st.spinner(f"Procesando archivo {i+1}/{len(archivos)}: {archivo.name}..."):
            file_type = detect_file_type(archivo)
            
            if file_type == 'LAS':
                # Para LAS, procesar y convertir
                if LASIO_AVAILABLE:
                    df_temp, sections = procesar_las_con_lasio(archivo)
                else:
                    df_temp, sections = procesar_las_manual(archivo)
                
                if df_temp is not None:
                    # Agregar columna de identificador
                    df_temp['ARCHIVO_ORIGEN'] = archivo.name
                    dataframes.append(df_temp)
                    info_archivos.append({
                        'nombre': archivo.name,
                        'tipo': 'LAS',
                        'filas': len(df_temp),
                        'columnas': len(df_temp.columns)
                    })
                    
            elif file_type == 'EXCEL':
                # Para Excel, cargar directamente
                try:
                    archivo.seek(0)
                    if archivo.name.endswith('.xlsx'):
                        df_temp = pd.read_excel(archivo, engine='openpyxl')
                    elif archivo.name.endswith('.xls'):
                        df_temp = pd.read_excel(archivo, engine='xlrd')
                    else:
                        df_temp = pd.read_excel(archivo, engine='openpyxl')
                    
                    # Agregar columna de identificador
                    df_temp['ARCHIVO_ORIGEN'] = archivo.name
                    dataframes.append(df_temp)
                    info_archivos.append({
                        'nombre': archivo.name,
                        'tipo': 'Excel',
                        'filas': len(df_temp),
                        'columnas': len(df_temp.columns)
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error cargando {archivo.name}: {str(e)}")
    
    if not dataframes:
        st.error("❌ No se pudieron procesar los archivos")
        return None, None
    
    # Combinar DataFrames
    try:
        # Encontrar columnas comunes para el merge
        common_columns = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_columns = common_columns.intersection(set(df.columns))
        
        # Convertir a lista y ordenar
        common_columns = sorted(list(common_columns))
        
        if not common_columns:
            st.error("❌ No hay columnas comunes entre los archivos")
            return None, None
        
        # Combinar DataFrames
        df_combinado = pd.concat(dataframes, axis=0, ignore_index=True)
        
        st.success(f"✅ {len(dataframes)} archivos combinados: {len(df_combinado)} filas totales")
        return df_combinado, info_archivos
        
    except Exception as e:
        st.error(f"❌ Error combinando archivos: {str(e)}")
        return None, None

# ================================
# 🎯 INTERFAZ MEJORADA PARA LAS CON LASIO
# ================================
def interfaz_las_mejorada(archivo):
    """Interfaz mejorada para archivos LAS con información de curvas usando lasio"""
    
    st.success(f"📁 Archivo LAS detectado: {archivo.name}")
    
    # Mostrar opciones de procesamiento
    st.subheader("🔄 Opciones de Procesamiento")
    
    if LASIO_AVAILABLE:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 **Procesamiento Profesional con Lasio** (Recomendado)
            **Incluye:**
            • Información completa de curvas con metadatos
            • Nombres reales de columnas del archivo LAS
            • Unidades y descripciones de curvas
            • Metadatos estructurados del pozo
            • Manejo robusto de formatos LAS
            """)
            
            if st.button("🔄 Procesar con Lasio", type="primary", use_container_width=True, key="procesar_lasio"):
                with st.spinner("Procesando archivo LAS con Lasio..."):
                    # Resetear archivo a posición inicial
                    archivo.seek(0)
                    df, sections = procesar_las_con_lasio(archivo)
                    
                    if df is not None and sections is not None:
                        mostrar_resultados_las(df, sections, archivo.name, "Lasio")
    
    else:
        st.markdown("""
        ### ⚠️ **Procesamiento Básico** (Lasio no disponible)
        **Instala Lasio para mejor experiencia:**
        ```bash
        pip install lasio
        ```
        """)
    
    # Opción de procesamiento manual (siempre disponible)
    col_manual1, col_manual2 = st.columns(2) if LASIO_AVAILABLE else st.columns(1)
    
    with col_manual1:
        st.markdown("""
        ### 🔧 **Procesamiento Manual**
        **Características:**
        • Extracción básica de datos
        • Nombres de columnas predefinidos
        • Formato simple pero funcional
        • Siempre disponible
        """)
        
        if st.button("🔧 Procesamiento Manual", use_container_width=True, 
                    type="primary" if not LASIO_AVAILABLE else "secondary",
                    key="procesar_manual"):
            with st.spinner("Procesando archivo LAS (manual)..."):
                # Resetear archivo a posición inicial
                archivo.seek(0)
                df, sections = procesar_las_manual(archivo)
                
                if df is not None and sections is not None:
                    mostrar_resultados_las(df, sections, archivo.name, "Manual")

def mostrar_resultados_las(df, sections, nombre_archivo, metodo):
    """Muestra los resultados del procesamiento LAS"""
    st.balloons()
    st.success(f"✅ ¡Archivo procesado exitosamente con {metodo}!")
    
    # Mostrar información de curvas
    st.subheader("📋 Información de Curvas")
    if 'curve' in sections and sections['curve']:
        with st.expander("🔍 Ver detalles de curvas", expanded=True):
            # Verificar el tipo de datos en sections
            if isinstance(sections['curve'][0], dict):
                # Datos de lasio (diccionarios)
                curve_df = pd.DataFrame(sections['curve'])
                st.dataframe(curve_df, use_container_width=True)
            else:
                # Datos manuales (strings)
                for line in sections['curve']:
                    if line.strip() and not line.startswith('#'):
                        st.code(line)
    
    # Mostrar información del pozo
    if 'well' in sections and sections['well']:
        with st.expander("🏭 Información del Pozo", expanded=False):
            if isinstance(sections['well'][0], dict):
                well_df = pd.DataFrame(sections['well'])
                st.dataframe(well_df, use_container_width=True)
            else:
                for line in sections['well']:
                    if line.strip() and not line.startswith('#'):
                        st.code(line)
    
    # Mostrar datos
    st.subheader("📊 Datos Procesados")
    st.dataframe(df.head(10))
    st.info(f"**Estructura:** {len(df)} filas × {len(df.columns)} columnas")
    st.info(f"**Columnas:** {list(df.columns)}")
    
    # Estadísticas básicas
    with st.expander("📈 Estadísticas Básicas", expanded=False):
        st.dataframe(df.describe(), use_container_width=True)
    
    # Descargar
    st.subheader("💾 Descargar Resultado")
    nombre_descarga = nombre_archivo.replace('.las', f'_{metodo.upper()}.xlsx').replace('.LAS', f'_{metodo.upper()}.xlsx')
    
    if descargar_las_como_excel(df, sections, nombre_descarga):
        st.success(f"""
        **📝 Siguientes pasos ({metodo}):**
        1. **Descarga** el archivo Excel completo
        2. **Vuelve** a cargar el Excel descargado  
        3. **Usa** los módulos de análisis
        4. **Consulta** la hoja METADATOS para información de curvas
        """)
# ================================
# 🎯 FUNCIÓN DE CARGA PRINCIPAL MEJORADA CON LASIO Y MULTICARGA
# ================================
def cargar_datos(archivo=None, archivos=None):
    """Función principal para cargar datos - MEJORADA CON LASIO Y MULTICARGA"""
    
    # Manejar multicarga
    if archivos and len(archivos) > 1:
        st.info(f"📁 {len(archivos)} archivos seleccionados para procesamiento múltiple")
        
        # Mostrar información de archivos
        with st.expander("📋 Archivos seleccionados", expanded=True):
            for i, archivo in enumerate(archivos):
                file_type = detect_file_type(archivo)
                st.write(f"{i+1}. **{archivo.name}** - Tipo: {file_type}")
        
        # Procesar múltiples archivos
        if st.button("🔄 Procesar Múltiples Archivos", type="primary", use_container_width=True):
            df_combinado, info_archivos = procesar_multiple_archivos(archivos)
            
            if df_combinado is not None:
                st.session_state.df_actual = df_combinado
                st.session_state.multicarga_info = info_archivos
                
                # Mostrar resumen de multicarga
                st.subheader("📊 Resumen de Multicarga")
                info_df = pd.DataFrame(info_archivos)
                st.dataframe(info_df, use_container_width=True)
                
                # Mostrar datos combinados
                st.subheader("📈 Datos Combinados")
                st.dataframe(df_combinado.head(15))
                st.info(f"**Estructura total:** {len(df_combinado)} filas × {len(df_combinado.columns)} columnas")
                
                # Opción para descargar datos combinados
                st.subheader("💾 Descargar Datos Combinados")
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_combinado.to_excel(writer, sheet_name='DATOS_COMBINADOS', index=False)
                    info_df.to_excel(writer, sheet_name='INFO_ARCHIVOS', index=False)
                
                output.seek(0)
                st.download_button(
                    label="📥 Descargar Datos Combinados",
                    data=output,
                    file_name=f"datos_combinados_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                return df_combinado
        
        return None
    
    # Manejar archivo único (comportamiento original)
    elif archivo:
        try:
            file_type = detect_file_type(archivo)
            
            st.info(f"📊 Tipo de archivo detectado: {file_type}")
            
            if file_type == 'LAS':
                # Para archivos LAS, mostrar la interfaz de procesamiento mejorada
                interfaz_las_mejorada(archivo)
                return None
                
            elif file_type == 'EXCEL':
                # Para Excel, carga normal con manejo de errores
                try:
                    # Resetear archivo a posición inicial
                    archivo.seek(0)
                    
                    if archivo.name.endswith('.xlsx'):
                        df = pd.read_excel(archivo, engine='openpyxl')
                    elif archivo.name.endswith('.xls'):
                        df = pd.read_excel(archivo, engine='xlrd')
                    else:
                        df = pd.read_excel(archivo, engine='openpyxl')
                    
                    st.success(f"✅ Excel cargado: {len(df)} filas, {len(df.columns)} columnas")
                    return df
                    
                except Exception as excel_error:
                    st.error(f"❌ Error cargando Excel: {str(excel_error)}")
                    st.info("💡 ¿Estás seguro de que es un archivo Excel válido?")
                    return None
                    
            else:
                st.error("❌ Tipo de archivo no reconocido")
                st.info("""
                **Formatos soportados:**
                • Excel (.xlsx, .xls)
                • LAS (.las)
                
                **Por favor verifica:**
                • Que el archivo no esté corrupto
                • Que sea uno de los formatos soportados
                """)
                return None
                
        except Exception as e:
            st.error(f"❌ Error general al cargar archivo: {str(e)}")
            return None
    
    return None

# ================================
# 🎛️ FUNCIÓN PARA MOSTRAR CARGADOR CON MULTICARGA (CORREGIDA)
# ================================
def mostrar_cargador_datos(modulo_nombre, instrucciones_especificas="", permitir_multicarga=True):
    st.header(f"{modulo_nombre}")
    st.subheader("📂 Carga de Datos")
    
    # Inicializar variables para evitar UnboundLocalError
    archivos = None
    archivo = None
    
    # Configuración de multicarga
    if permitir_multicarga:
        col_multicarga1, col_multicarga2 = st.columns(2)
        
        with col_multicarga1:
            modo_carga = st.radio(
                "Modo de carga:",
                ["Archivo único", "Múltiples archivos"],
                horizontal=True
            )
        
        with col_multicarga2:
            if modo_carga == "Múltiples archivos":
                st.info("🔗 Los archivos se combinarán automáticamente")
    
    # Cargador de archivos
    if permitir_multicarga and modo_carga == "Múltiples archivos":
        archivos = st.file_uploader(
            f"Sube tus archivos Excel o LAS", 
            type=["xlsx", "xls", "las"],
            key=f"cargador_multiple_{modulo_nombre}",
            accept_multiple_files=True
        )
        
        if archivos:
            resultado = cargar_datos(archivos=archivos)
            if resultado is not None:
                st.session_state.df_actual = resultado
                return resultado
            return None
            
    else:
        archivo = st.file_uploader(
            f"Sube tu archivo Excel o LAS", 
            type=["xlsx", "xls", "las"],
            key=f"cargador_{modulo_nombre}"
        )
        
        if archivo is not None:
            file_type = detect_file_type(archivo)
            
            if file_type == 'LAS':
                st.info("""
                **🔄 Procesamiento LAS MEJORADO:**
                • Extracción completa con información de curvas
                • Nombres reales de columnas
                • Metadatos del pozo incluidos
                • Estructura profesional en Excel
                """)
            
            resultado = cargar_datos(archivo=archivo)
            
            # Solo procesar si es Excel y tenemos un DataFrame
            if file_type != 'LAS' and resultado is not None:
                st.session_state.df_actual = resultado
                st.success(f"✅ Datos cargados: {len(resultado)} filas, {len(resultado.columns)} columnas")
                
                with st.expander("📊 Vista previa", expanded=True):
                    st.dataframe(resultado.head(10))
                    st.info(f"**Columnas:** {list(resultado.columns)}")
                    
                return resultado
            
            # Para LAS, no retornamos DataFrame porque el usuario debe descargar el Excel primero
            return None
    
    # Mostrar instrucciones si no hay archivos (CORREGIDO)
    # Ahora ambas variables están siempre definidas
    if not archivos and not archivo:
        st.info(f"""
        **📝 Instrucciones para {modulo_nombre}:**
        
        {instrucciones_especificas}
        
        **📋 Formatos:**
        • **Excel (.xlsx, .xls)** - Carga directa
        • **LAS (.las)** - Conversión automática a Excel
        
        **💡 Para archivos LAS (MEJORADO):**
        Se convertirán automáticamente a Excel con información completa de curvas.
        
        **🔄 Multicarga disponible:** Puedes cargar múltiples archivos y se combinarán automáticamente.
        """)
        return None
    
    return None
# ================================
# 📊 MÓDULO 1: VISUALIZACIÓN BÁSICA (ACTUALIZADO CON FUNCIONALIDADES AVANZADAS)
# ================================
def modulo_visualizacion_basica():
    
    # Instrucciones específicas para visualización
    instrucciones_visualizacion = """
    1. **Selecciona la columna de profundidad** que será tu eje Y
    2. **Elige las curvas** que quieres visualizar en los ejes X
    3. **Personaliza** colores, estilos y configuración de ejes
    4. **Ajusta el rango** de profundidad si es necesario
    5. **Agrega marcadores estratigráficos** para identificar formaciones
    6. **Usa plantillas predefinidas** para diferentes ambientes geológicos
    7. **Compara múltiples pozos** (modo multicarga)
    8. **Aplica normalizaciones** para mejorar la visualización
    9. **Analiza correlaciones** entre curvas seleccionadas
    """
    
    df = mostrar_cargador_datos(
        "Visualización Básica", 
        instrucciones_visualizacion,
        permitir_multicarga=True
    )
    
    if df is None:
        return

    # Verificar si es multicarga
    es_multicarga = 'ARCHIVO_ORIGEN' in df.columns if df is not None else False
    
    if es_multicarga:
        st.info("🎯 **Modo Multicarga Activado** - Visualizando datos combinados de múltiples archivos")
        
        # Selector de archivo para filtrar
        archivos_unicos = df['ARCHIVO_ORIGEN'].unique()
        archivo_seleccionado = st.selectbox(
            "Filtrar por archivo (opcional):",
            options=["Todos los archivos"] + list(archivos_unicos)
        )
        
        # Aplicar filtro si se selecciona un archivo específico
        if archivo_seleccionado != "Todos los archivos":
            df = df[df['ARCHIVO_ORIGEN'] == archivo_seleccionado]
            st.info(f"📁 Mostrando datos de: {archivo_seleccionado}")

    # ================================
    # 🎨 PLANTILLAS PREDEFINIDAS
    # ================================
    st.subheader("🎨 Plantillas de Visualización")
    
    col_temp1, col_temp2, col_temp3 = st.columns(3)
    
    with col_temp1:
        plantilla_seleccionada = st.selectbox(
            "Plantilla geológica:",
            options=["Personalizada", "Clástico (Triple Combo)", "Carbonato", "Lutitas (Unconventional)", "Aguas Profundas", "Básica"],
            help="Selecciona una plantilla predefinida para configuración automática"
        )
    
    with col_temp2:
        # Aplicar plantilla
        if st.button("🔄 Aplicar Plantilla", use_container_width=True):
            st.session_state.aplicar_plantilla = True
    
    with col_temp3:
        # Nombre personalizado para la gráfica
        nombre_grafica = st.text_input(
            "Nombre de la gráfica:",
            value=f"Registros_{datetime.now().strftime('%H%M')}",
            help="Asigna un nombre personalizado a tu gráfica"
        )

    # Configuración de plantillas
    configuraciones_plantillas = {
        "Clástico (Triple Combo)": {
            "curvas_sugeridas": ['GR', 'RT', 'RHOB', 'NPHI', 'CALI'],
            "colores": ['green', 'red', 'blue', 'purple', 'orange'],
            "escalas": {'GR': (0, 150), 'RT': (0.2, 2000), 'RHOB': (1.8, 2.8), 'NPHI': (0.45, -0.15)},
            "eje_secundario": ['RT']
        },
        "Carbonato": {
            "curvas_sugeridas": ['GR', 'RT', 'RHOB', 'NPHI', 'DT', 'PEF'],
            "colores": ['darkgreen', 'darkred', 'navy', 'darkviolet', 'brown', 'gray'],
            "escalas": {'GR': (0, 100), 'RT': (1, 1000), 'RHOB': (2.3, 2.8), 'NPHI': (0.3, -0.05), 'DT': (40, 140)},
            "eje_secundario": ['RT', 'DT']
        },
        "Lutitas (Unconventional)": {
            "curvas_sugeridas": ['GR', 'RT', 'RHOB', 'NPHI', 'CALI', 'SP'],
            "colores": ['brown', 'red', 'blue', 'purple', 'orange', 'teal'],
            "escalas": {'GR': (50, 200), 'RT': (1, 100), 'RHOB': (2.4, 2.9), 'NPHI': (0.25, 0.05)},
            "eje_secundario": ['RT']
        },
        "Aguas Profundas": {
            "curvas_sugeridas": ['GR', 'RD', 'RS', 'RHOB', 'NPHI', 'CALI'],
            "colores": ['green', 'red', 'darkred', 'blue', 'purple', 'orange'],
            "escalas": {'GR': (0, 150), 'RD': (0.2, 200), 'RS': (0.2, 200), 'RHOB': (1.8, 2.8), 'NPHI': (0.45, -0.15)},
            "eje_secundario": ['RD', 'RS']
        },
        "Básica": {
            "curvas_sugeridas": ['GR', 'RT', 'RHOB'],
            "colores": ['green', 'red', 'blue'],
            "escalas": {},
            "eje_secundario": ['RT']
        }
    }

    # --- Selección de columna de profundidad (Eje Y) ---
    st.subheader("🎯 Configuración de Ejes")
    
    # Filtrar columnas numéricas para profundidad
    columnas_numericas = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not columnas_numericas:
        st.error("❌ No se encontraron columnas numéricas en el archivo.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Selecciona la columna para el eje Y (Profundidad):**")
        columna_profundidad = st.selectbox(
            "Columna de profundidad:",
            options=columnas_numericas,
            index=0,
            help="Selecciona la columna que representa la profundidad o tiempo"
        )
        
        # Guardar en sesión
        st.session_state.columna_profundidad = columna_profundidad
        
        # Mostrar información de la columna seleccionada
        st.info(f"""
        **Información de {columna_profundidad}:**
        - Mínimo: {df[columna_profundidad].min():.2f}
        - Máximo: {df[columna_profundidad].max():.2f}
        - Valores únicos: {df[columna_profundidad].nunique()}
        """)
    
    with col2:
        st.markdown("**Opciones de profundidad:**")
        invertir_eje_y = st.checkbox("Invertir eje Y", value=True, 
                                   help="Invertir el eje Y para que la profundidad aumente hacia abajo")
        
        # Renombrar columna temporalmente para consistencia
        df_temp = df.rename(columns={columna_profundidad: "PROFUNDIDAD"})

    # ================================
    # 🔧 NORMALIZACIÓN DE CURVAS
    # ================================
    st.subheader("🔧 Normalización de Curvas")
    
    col_norm1, col_norm2, col_norm3 = st.columns(3)
    
    with col_norm1:
        aplicar_normalizacion = st.checkbox(
            "Aplicar normalización", 
            value=False,
            help="Normalizar curvas para mejor comparación visual"
        )
    
    with col_norm2:
        if aplicar_normalizacion:
            metodo_normalizacion = st.selectbox(
                "Método de normalización:",
                options=["Z-score", "Min-Max", "Robust (IQR)", "Percentil (5-95%)"],
                help="Selecciona el método de normalización"
            )
    
    with col_norm3:
        if aplicar_normalizacion:
            normalizar_por_archivo = st.checkbox(
                "Normalizar por archivo individual", 
                value=True,
                disabled=not es_multicarga,
                help="Aplicar normalización separada para cada archivo en multicarga"
            )

    # --- Filtrar columnas para curvas (Eje X) ---
    columnas_para_usar = [col for col in columnas_numericas if col != columna_profundidad]

    if len(columnas_para_usar) == 0:
        st.error("No se encontraron columnas numéricas válidas para graficar (además de la columna de profundidad).")
        return

    # ================================
    # 🔍 SELECCIÓN DE RANGO DE PROFUNDIDAD
    # ================================
    st.subheader("🔍 Selecciona el rango de profundidad para graficar")

    # Asegurar que la columna de profundidad sea numérica
    df_temp["PROFUNDIDAD"] = pd.to_numeric(df_temp["PROFUNDIDAD"], errors="coerce")
    df_temp = df_temp.dropna(subset=["PROFUNDIDAD"])

    prof_min_global = float(df_temp["PROFUNDIDAD"].min())
    prof_max_global = float(df_temp["PROFUNDIDAD"].max())

    # Redondear para mejor visualización
    prof_min_red = round(prof_min_global, 1)
    prof_max_red = round(prof_max_global, 1)

    profundidad_min, profundidad_max = st.slider(
        f"Rango de {columna_profundidad}",
        min_value=prof_min_red,
        max_value=prof_max_red,
        value=(prof_min_red, prof_max_red),
        step=0.1,
        format="%.1f"
    )

    # Filtrar el DataFrame al rango seleccionado
    df_filtrado = df_temp[(df_temp["PROFUNDIDAD"] >= profundidad_min) & (df_temp["PROFUNDIDAD"] <= profundidad_max)].copy()

    if df_filtrado.empty:
        st.error("❌ No hay datos en el rango de profundidad seleccionado.")
        return

    # Reemplazar `df_temp` por `df_filtrado` para el resto del flujo
    df_temp = df_filtrado

    # ================================
    # 🎯 CONFIGURACIÓN DE MARCADORES ESTRATIGRÁFICOS
    # ================================
    st.subheader("🎯 Marcadores Estratigráficos (Topes)")
    
    col_marc1, col_marc2 = st.columns([2, 1])
    
    with col_marc1:
        agregar_marcadores = st.checkbox(
            "Agregar marcadores estratigráficos", 
            value=False,
            help="Agrega líneas horizontales para identificar formaciones geológicas"
        )
    
    with col_marc2:
        if agregar_marcadores:
            num_marcadores = st.number_input(
                "Número de marcadores:", 
                min_value=1, 
                max_value=20, 
                value=3,
                help="Cantidad de topes estratigráficos a agregar"
            )

    # Configuración de marcadores
    marcadores = []
    if agregar_marcadores:
        st.markdown("#### 📍 Configuración de Marcadores")
        
        # Colores para marcadores
        colores_marcadores = [
            'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray',
            'olive', 'cyan', 'magenta', 'darkred', 'darkblue', 'darkgreen',
            'darkviolet', 'gold', 'lime', 'teal', 'navy', 'maroon'
        ]
        
        # Estilos de línea para marcadores
        estilos_marcadores = ['-', '--', '-.', ':']
        
        for i in range(int(num_marcadores)):
            st.markdown(f"---")
            st.markdown(f"#### 📌 Marcador {i+1}")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                profundidad_marcador = st.number_input(
                    f"Profundidad del tope {i+1}",
                    min_value=float(profundidad_min),
                    max_value=float(profundidad_max),
                    value=float(profundidad_min + (i * (profundidad_max - profundidad_min) / num_marcadores)),
                    step=0.1,
                    key=f"prof_marc_{i}"
                )
            
            with col_m2:
                nombre_marcador = st.text_input(
                    f"Nombre formación {i+1}",
                    value=f"Formación {i+1}",
                    key=f"nombre_marc_{i}"
                )
            
            with col_m3:
                color_marcador = st.selectbox(
                    f"Color {i+1}",
                    options=colores_marcadores,
                    index=i % len(colores_marcadores),
                    key=f"color_marc_{i}"
                )
            
            with col_m4:
                estilo_marcador = st.selectbox(
                    f"Estilo línea {i+1}",
                    options=estilos_marcadores,
                    format_func=lambda x: {"-": "Sólida", "--": "Punteada", "-.": "Trazo-punto", ":": "Puntos"}[x],
                    index=i % len(estilos_marcadores),
                    key=f"estilo_marc_{i}"
                )
            
            # Agregar anotación opcional
            anotacion_marcador = st.text_area(
                f"Anotación para {nombre_marcador} (opcional)",
                value="",
                placeholder="Descripción de la formación, características, etc.",
                key=f"anot_marc_{i}"
            )
            
            marcadores.append({
                "profundidad": profundidad_marcador,
                "nombre": nombre_marcador,
                "color": color_marcador,
                "estilo": estilo_marcador,
                "anotacion": anotacion_marcador
            })

    # ================================
    # 📈 CONFIGURACIÓN DE CURVAS (ACTUALIZADO CON PLANTILLAS)
    # ================================
    st.subheader("📈 Configuración de curvas")

    # Número de curvas a mostrar
    num_curvas = st.number_input("Número de curvas a mostrar", min_value=1, max_value=10, value=2)

    # Aplicar configuración de plantilla si se seleccionó
    if plantilla_seleccionada != "Personalizada" and st.session_state.get('aplicar_plantilla', False):
        plantilla = configuraciones_plantillas[plantilla_seleccionada]
        curvas_sugeridas = [c for c in plantilla['curvas_sugeridas'] if c in columnas_para_usar]
        num_curvas = min(len(curvas_sugeridas), num_curvas)
        st.session_state.aplicar_plantilla = False
        st.success(f"✅ Plantilla {plantilla_seleccionada} aplicada")

    # Listas y opciones
    colores_validos = [
        'black', 'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink',
        'gray', 'olive', 'cyan', 'magenta', 'darkred', 'darkblue', 'darkgreen',
        'darkviolet', 'gold', 'lime', 'teal', 'navy', 'maroon'
    ]

    estilos_sugeridos = ['-', '--', '-.', ':']

    orden_z = {"Fondo": 1, "Medio": 3, "Delante": 5}

    curvas = []

    for i in range(int(num_curvas)):
        st.markdown(f"---")
        st.markdown(f"#### 📌 Curva {i+1}")

        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

        with col1:
            # Sugerir curvas basadas en plantilla
            if plantilla_seleccionada != "Personalizada" and i < len(curvas_sugeridas):
                default_col = curvas_sugeridas[i] if curvas_sugeridas[i] in columnas_para_usar else columnas_para_usar[min(i, len(columnas_para_usar)-1)]
            else:
                default_col = columnas_para_usar[min(i, len(columnas_para_usar)-1)]
            
            col_seleccionada = st.selectbox(
                f"Selecciona columna (Curva {i+1})",
                options=columnas_para_usar,
                index=columnas_para_usar.index(default_col) if default_col in columnas_para_usar else min(i, len(columnas_para_usar)-1),
                key=f"col_{i}"
            )

        usos_previos = sum(1 for c in curvas if c["columna"] == col_seleccionada)
        etiqueta_default = f"{col_seleccionada} ({usos_previos + 1})" if usos_previos > 0 else col_seleccionada

        with col2:
            # Color basado en plantilla
            if plantilla_seleccionada != "Personalizada" and i < len(plantilla['colores']):
                color_default = plantilla['colores'][i]
                color_index = colores_validos.index(color_default) if color_default in colores_validos else i % len(colores_validos)
            else:
                color_index = i % len(colores_validos)
            
            color = st.selectbox(f"Color (Curva {i+1})", options=colores_validos, index=color_index, key=f"color_{i}")
        
        with col3:
            estilo = st.selectbox(
                f"Estilo de línea (Curva {i+1})",
                options=estilos_sugeridos,
                format_func=lambda x: {"-": "Sólida", "--": "Punteada", "-.": "Trazo-punto", ":": "Puntos"}[x],
                index=i % len(estilos_sugeridos),
                key=f"estilo_{i}"
            )
        
        with col4:
            etiqueta = st.text_input(f"Etiqueta en leyenda", value=etiqueta_default, key=f"etiqueta_{i}")
        
        with col5:
            # Eje secundario basado en plantilla
            if plantilla_seleccionada != "Personalizada" and col_seleccionada in plantilla['eje_secundario']:
                eje_secundario_default = True
            else:
                eje_secundario_default = False
            
            eje_secundario = st.checkbox(f"Eje X superior", value=eje_secundario_default, key=f"eje_sec_{i}")
        
        with col6:
            suavizar = st.checkbox(f"Suavizar", key=f"suavizar_{i}")
        
        with col7:
            ventana = st.number_input(f"Ventana (puntos)", min_value=2, max_value=50, value=5, step=1, key=f"ventana_{i}", disabled=not suavizar)
        
        with col8:
            invertir_x = st.checkbox(f"Invertir eje X", key=f"invertir_x_{i}")
        
        with col9:
            capa = st.selectbox(f"Posición", options=list(orden_z.keys()), index=1, key=f"capa_{i}")

        nombre_base_limpio = re.sub(r'[^a-zA-Z0-9_]', '_', col_seleccionada)
        nombre_interno_limpio = f"curva_{nombre_base_limpio}_{i+1}"

        curvas.append({
            "columna": col_seleccionada,
            "color": color,
            "estilo": estilo,
            "etiqueta": etiqueta,
            "eje_secundario": eje_secundario,
            "suavizar": suavizar,
            "ventana": ventana if suavizar else 1,
            "invertir_x": invertir_x,
            "zorder": orden_z[capa],
            "nombre_interno": nombre_interno_limpio
        })
    
    # --- Alinear todas las curvas en profundidad común ---
    st.subheader("⚙️ Procesando datos...")

    # Aplicar normalización si está activada
    if aplicar_normalizacion:
        df_procesado = df_temp.copy()
        
        if es_multicarga and normalizar_por_archivo:
            # Normalizar por archivo individual
            df_procesado_normalizado = pd.DataFrame()
            for archivo in df_procesado['ARCHIVO_ORIGEN'].unique():
                df_archivo = df_procesado[df_procesado['ARCHIVO_ORIGEN'] == archivo].copy()
                
                for curva in curvas:
                    col_data = df_archivo[curva['columna']].dropna()
                    if len(col_data) > 0:
                        if metodo_normalizacion == "Z-score":
                            df_archivo[f"{curva['columna']}_NORM"] = (col_data - col_data.mean()) / col_data.std()
                        elif metodo_normalizacion == "Min-Max":
                            df_archivo[f"{curva['columna']}_NORM"] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                        elif metodo_normalizacion == "Robust (IQR)":
                            Q1 = col_data.quantile(0.25)
                            Q3 = col_data.quantile(0.75)
                            IQR = Q3 - Q1
                            df_archivo[f"{curva['columna']}_NORM"] = (col_data - col_data.median()) / IQR
                        elif metodo_normalizacion == "Percentil (5-95%)":
                            p5 = col_data.quantile(0.05)
                            p95 = col_data.quantile(0.95)
                            df_archivo[f"{curva['columna']}_NORM"] = (col_data - p5) / (p95 - p5)
                
                df_procesado_normalizado = pd.concat([df_procesado_normalizado, df_archivo])
            
            df_procesado = df_procesado_normalizado
            # Actualizar nombres de columnas en curvas
            for curva in curvas:
                curva['columna_original'] = curva['columna']
                curva['columna'] = f"{curva['columna']}_NORM"
                curva['etiqueta'] = f"{curva['etiqueta']} (Norm)"
        
        else:
            # Normalizar todo el dataset
            for curva in curvas:
                col_data = df_procesado[curva['columna']].dropna()
                if len(col_data) > 0:
                    if metodo_normalizacion == "Z-score":
                        df_procesado[f"{curva['columna']}_NORM"] = (col_data - col_data.mean()) / col_data.std()
                    elif metodo_normalizacion == "Min-Max":
                        df_procesado[f"{curva['columna']}_NORM"] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                    elif metodo_normalizacion == "Robust (IQR)":
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        df_procesado[f"{curva['columna']}_NORM"] = (col_data - col_data.median()) / IQR
                    elif metodo_normalizacion == "Percentil (5-95%)":
                        p5 = col_data.quantile(0.05)
                        p95 = col_data.quantile(0.95)
                        df_procesado[f"{curva['columna']}_NORM"] = (col_data - p5) / (p95 - p5)
            
            # Actualizar nombres de columnas en curvas
            for curva in curvas:
                curva['columna_original'] = curva['columna']
                curva['columna'] = f"{curva['columna']}_NORM"
                curva['etiqueta'] = f"{curva['etiqueta']} (Norm)"
    else:
        df_procesado = df_temp.copy()

    # Continuar con el procesamiento normal
    df_comun = df_procesado[["PROFUNDIDAD"]].copy()
    for curva in curvas:
        col_data = df_procesado[["PROFUNDIDAD", curva["columna"]]].dropna()
        col_data = col_data.rename(columns={curva["columna"]: curva["nombre_interno"]})
        df_comun = df_comun.merge(col_data, on="PROFUNDIDAD", how="inner")

    if len(df_comun) == 0:
        st.error("❌ No hay datos comunes en profundidad para todas las curvas seleccionadas.")
        return

    profundidad_comun = df_comun["PROFUNDIDAD"].values

    # --- Función de suavizado ---
    def suavizar_serie(serie, ventana):
        return serie.rolling(window=ventana, center=True, min_periods=1).mean()

    # ================================
    # 📊 ANÁLISIS DE CORRELACIÓN
    # ================================
    st.subheader("📊 Análisis de Correlación")
    
    if len(curvas) >= 2:
        col_corr1, col_corr2 = st.columns(2)
        
        with col_corr1:
            st.markdown("**Selecciona dos curvas para analizar correlación:**")
            curva1_corr = st.selectbox(
                "Curva 1:",
                options=[c["columna"] for c in curvas],
                key="curva1_corr"
            )
            curva2_corr = st.selectbox(
                "Curva 2:",
                options=[c["columna"] for c in curvas if c["columna"] != curva1_corr],
                key="curva2_corr"
            )
            
            # Calcular correlación
            if curva1_corr and curva2_corr:
                datos_curva1 = df_comun[next(c["nombre_interno"] for c in curvas if c["columna"] == curva1_corr)]
                datos_curva2 = df_comun[next(c["nombre_interno"] for c in curvas if c["columna"] == curva2_corr)]
                
                # Filtrar datos válidos
                mask = (~datos_curva1.isna()) & (~datos_curva2.isna())
                datos_curva1_clean = datos_curva1[mask]
                datos_curva2_clean = datos_curva2[mask]
                
                if len(datos_curva1_clean) > 1:
                    # Calcular coeficientes
                    correlacion_pearson, p_value_pearson = stats.pearsonr(datos_curva1_clean, datos_curva2_clean)
                    r_cuadrado = correlacion_pearson ** 2
                    
                    # Mostrar resultados
                    st.metric("Coeficiente de Correlación (Pearson)", f"{correlacion_pearson:.4f}")
                    st.metric("R² (Coeficiente de determinación)", f"{r_cuadrado:.4f}")
                    st.metric("Valor p", f"{p_value_pearson:.4e}")
                    
                    # Interpretación
                    if abs(correlacion_pearson) > 0.7:
                        st.success("✅ **Fuerte correlación** entre las curvas")
                    elif abs(correlacion_pearson) > 0.3:
                        st.info("🟡 **Correlación moderada** entre las curvas")
                    else:
                        st.warning("🟠 **Correlación débil** entre las curvas")
        
        with col_corr2:
            # Gráfico de dispersión para correlación
            if curva1_corr and curva2_corr and len(datos_curva1_clean) > 1:
                fig_dispersion = px.scatter(
                    x=datos_curva1_clean,
                    y=datos_curva2_clean,
                    title=f"Dispersión: {curva1_corr} vs {curva2_corr}",
                    labels={'x': curva1_corr, 'y': curva2_corr}
                )
                fig_dispersion.update_layout(height=400)
                st.plotly_chart(fig_dispersion, use_container_width=True)

    # ================================
    # 📈 GRÁFICO PRINCIPAL (ACTUALIZADO CON MARCADORES Y NOMBRE PERSONALIZADO)
    # ================================
    st.subheader("📈 Gráfico de Registros")

    # Configuración del gráfico
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        altura_grafico = st.slider("Altura del gráfico (píxeles)", 400, 1200, 600)
        mostrar_grid = st.checkbox("Mostrar grid", value=True)
        mostrar_leyenda = st.checkbox("Mostrar leyenda", value=True)
        mostrar_marcadores = st.checkbox("Mostrar marcadores estratigráficos", value=True) if agregar_marcadores else False
        
    with col_config2:
        escala_log = st.checkbox("Usar escala logarítmica para curvas seleccionadas")
        mostrar_puntos = st.checkbox("Mostrar puntos en curvas", value=False)
        mostrar_nombres_marcadores = st.checkbox("Mostrar nombres en marcadores", value=True) if agregar_marcadores else False

    # Crear figura
    fig, ax1 = plt.subplots(figsize=(12, altura_grafico/100))
    
    # Configurar título personalizado
    if nombre_grafica:
        plt.title(nombre_grafica, fontsize=14, fontweight='bold', pad=20)
    
    # Configurar eje Y (profundidad)
    ax1.set_ylabel("Profundidad")
    if invertir_eje_y:
        ax1.set_ylim(profundidad_max, profundidad_min)
    else:
        ax1.set_ylim(profundidad_min, profundidad_max)

    # Crear eje secundario si es necesario
    ax2 = None
    tiene_eje_secundario = any(curva["eje_secundario"] for curva in curvas)
    
    if tiene_eje_secundario:
        ax2 = ax1.twiny()

    # Graficar cada curva
    for curva in curvas:
        datos_curva = df_comun[curva["nombre_interno"]].values
        
        # Aplicar suavizado si está activado
        if curva["suavizar"] and curva["ventana"] > 1:
            datos_curva = suavizar_serie(pd.Series(datos_curva), curva["ventana"]).values
        
        # Aplicar inversión si está activado
        if curva["invertir_x"]:
            datos_curva = -datos_curva
        
        # Seleccionar eje
        eje_actual = ax2 if curva["eje_secundario"] and ax2 is not None else ax1
        
        # Graficar
        line = eje_actual.plot(
            datos_curva,
            profundidad_comun,
            color=curva["color"],
            linestyle=curva["estilo"],
            label=curva["etiqueta"],
            zorder=curva["zorder"],
            linewidth=1.5,
            marker='o' if mostrar_puntos else None,
            markersize=3 if mostrar_puntos else 0
        )[0]
        
        # Configurar escala logarítmica si está activada
        if escala_log:
            eje_actual.set_xscale('log')

    # ================================
    # 🎯 AGREGAR MARCADORES ESTRATIGRÁFICOS AL GRÁFICO
    # ================================
    if agregar_marcadores and mostrar_marcadores:
        for i, marcador in enumerate(marcadores):
            # Agregar línea horizontal para cada marcador
            ax1.axhline(
                y=marcador["profundidad"], 
                color=marcador["color"], 
                linestyle=marcador["estilo"],
                linewidth=2,
                alpha=0.8,
                zorder=10  # Alto zorder para que esté por encima de las curvas
            )
            
            # Agregar etiqueta si está activado
            if mostrar_nombres_marcadores:
                ax1.text(
                    0.02,  # Posición x (2% del ancho del gráfico)
                    marcador["profundidad"], 
                    f"  {marcador['nombre']}",
                    verticalalignment='center',
                    horizontalalignment='left',
                    transform=ax1.get_yaxis_transform(),  # Usar transformación del eje y
                    fontsize=10,
                    fontweight='bold',
                    color=marcador["color"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=marcador["color"])
                )

    # Configurar ejes y leyenda
    ax1.set_xlabel("Curvas principales")
    if ax2 is not None:
        ax2.set_xlabel("Curvas secundarias")
    
    if mostrar_leyenda:
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        if ax2 is not None:
            ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    if mostrar_grid:
        ax1.grid(True, alpha=0.3)
        if ax2 is not None:
            ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # ================================
    # 📋 TABLA DE MARCADORES ESTRATIGRÁFICOS
    # ================================
    if agregar_marcadores and marcadores:
        st.subheader("📋 Resumen de Marcadores Estratigráficos")
        
        # Crear tabla de marcadores
        marcadores_df = pd.DataFrame(marcadores)
        st.dataframe(marcadores_df, use_container_width=True)
        
        # Mostrar información adicional si hay anotaciones
        marcadores_con_anotaciones = [m for m in marcadores if m["anotacion"]]
        if marcadores_con_anotaciones:
            st.markdown("#### 📝 Anotaciones de Formaciones")
            for marcador in marcadores_con_anotaciones:
                st.markdown(f"**{marcador['nombre']}** ({marcador['profundidad']} m): {marcador['anotacion']}")

    # ================================
    # 💾 EXPORTACIÓN PROFESIONAL MEJORADA
    # ================================
    st.subheader("💾 Exportación Profesional")
    
    col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
    
    with col_exp1:
        # Exportar gráfico como PNG de alta calidad
        buf_png = BytesIO()
        fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
        st.download_button(
            label="📥 PNG Alta Calidad",
            data=buf_png.getvalue(),
            file_name=f"{nombre_grafica}_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col_exp2:
        # Exportar como PDF para reportes
        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format='pdf', dpi=300, bbox_inches='tight')
        st.download_button(
            label="📄 PDF para Reportes",
            data=buf_pdf.getvalue(),
            file_name=f"{nombre_grafica}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    with col_exp3:
        # Exportar datos procesados en Excel
        output_excel = BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            # Hoja principal de datos
            df_comun.to_excel(writer, sheet_name='Datos_Procesados', index=False)
            
            # Hoja de configuración de curvas
            config_data = []
            for curva in curvas:
                config_row = {
                    'Columna': curva['columna'],
                    'Etiqueta': curva['etiqueta'],
                    'Color': curva['color'],
                    'Estilo_Linea': curva['estilo'],
                    'Eje_Secundario': curva['eje_secundario'],
                    'Suavizado': curva['suavizar'],
                    'Ventana_Suavizado': curva['ventana'],
                    'Invertir_Eje_X': curva['invertir_x']
                }
                if aplicar_normalizacion:
                    config_row['Columna_Original'] = curva.get('columna_original', curva['columna'])
                    config_row['Metodo_Normalizacion'] = metodo_normalizacion
                config_data.append(config_row)
            
            config_df = pd.DataFrame(config_data)
            config_df.to_excel(writer, sheet_name='Configuracion_Curvas', index=False)
            
            # Hoja con marcadores estratigráficos si existen
            if agregar_marcadores and marcadores:
                marcadores_export = []
                for marcador in marcadores:
                    marcadores_export.append({
                        'Nombre_Formacion': marcador['nombre'],
                        'Profundidad': marcador['profundidad'],
                        'Color': marcador['color'],
                        'Estilo_Linea': marcador['estilo'],
                        'Anotacion': marcador['anotacion']
                    })
                marcadores_df = pd.DataFrame(marcadores_export)
                marcadores_df.to_excel(writer, sheet_name='Marcadores_Estratigraficos', index=False)
            
            # Hoja de metadatos
            metadata = {
                'Campo': [nombre_grafica],
                'Fecha_Generacion': [datetime.now().strftime('%Y-%m-%d %H:%M')],
                'Plantilla_Utilizada': [plantilla_seleccionada],
                'Rango_Profundidad': [f"{profundidad_min} - {profundidad_max}"],
                'Normalizacion_Aplicada': [metodo_normalizacion if aplicar_normalizacion else 'Ninguna'],
                'Numero_Curvas': [len(curvas)],
                'Numero_Marcadores': [len(marcadores) if agregar_marcadores else 0]
            }
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_excel(writer, sheet_name='Metadatos', index=False)
        
        st.download_button(
            label="📊 Excel Completo",
            data=output_excel.getvalue(),
            file_name=f"{nombre_grafica}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_exp4:
        # Exportar configuración como template reutilizable
        config_str = f"# CONFIGURACIÓN DE VISUALIZACIÓN - {nombre_grafica}\n"
        config_str += f"# Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        config_str += f"PLANTILLA: {plantilla_seleccionada}\n"
        config_str += f"RANGO_PROFUNDIDAD: {profundidad_min},{profundidad_max}\n"
        config_str += f"NORMALIZACION: {metodo_normalizacion if aplicar_normalizacion else 'NINGUNA'}\n"
        config_str += f"INVERTIR_EJE_Y: {invertir_eje_y}\n\n"
        
        config_str += "## CURVAS CONFIGURADAS:\n"
        for i, curva in enumerate(curvas):
            config_str += f"CURVA_{i+1}: {curva['columna']}\n"
            config_str += f"  ETIQUETA: {curva['etiqueta']}\n"
            config_str += f"  COLOR: {curva['color']}\n"
            config_str += f"  ESTILO: {curva['estilo']}\n"
            config_str += f"  EJE_SECUNDARIO: {curva['eje_secundario']}\n"
            config_str += f"  SUAVIZADO: {curva['suavizar']}\n"
            if curva['suavizar']:
                config_str += f"  VENTANA: {curva['ventana']}\n"
            config_str += f"  INVERTIR_X: {curva['invertir_x']}\n\n"
        
        if agregar_marcadores and marcadores:
            config_str += "## MARCADORES ESTRATIGRÁFICOS:\n"
            for i, marcador in enumerate(marcadores):
                config_str += f"MARCADOR_{i+1}: {marcador['nombre']}\n"
                config_str += f"  PROFUNDIDAD: {marcador['profundidad']}\n"
                config_str += f"  COLOR: {marcador['color']}\n"
                config_str += f"  ESTILO: {marcador['estilo']}\n"
                if marcador['anotacion']:
                    config_str += f"  ANOTACION: {marcador['anotacion']}\n"
                config_str += "\n"
        
        st.download_button(
            label="⚙️ Template Config",
            data=config_str.encode(),
            file_name=f"template_{nombre_grafica}_{datetime.now().strftime('%Y%m%d_%H%M')}.cfg",
            mime="text/plain",
            use_container_width=True
        )

    # ================================
    # 📊 COMPARACIÓN MULTI-POZO (VISUALIZACIÓN AVANZADA)
    # ================================
    if es_multicarga and len(archivos_unicos) > 1:
        st.subheader("📊 Comparación Multi-pozo")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            mostrar_comparacion = st.checkbox(
                "Mostrar comparación side-by-side",
                value=False,
                help="Mostrar gráficas individuales para cada pozo"
            )
        
        with col_comp2:
            if mostrar_comparacion:
                pozos_comparar = st.multiselect(
                    "Seleccionar pozos para comparar:",
                    options=archivos_unicos,
                    default=archivos_unicos[:min(3, len(archivos_unicos))]
                )
        
        if mostrar_comparacion and pozos_comparar:
            st.markdown("#### 📈 Vista Comparativa")
            
            # Crear subplots para comparación
            n_pozos = len(pozos_comparar)
            fig_comparacion, axes = plt.subplots(1, n_pozos, figsize=(5*n_pozos, altura_grafico/100))
            if n_pozos == 1:
                axes = [axes]
            
            for idx, pozo in enumerate(pozos_comparar):
                df_pozo = df_temp[df_temp['ARCHIVO_ORIGEN'] == pozo]
                ax = axes[idx]
                
                # Configurar eje Y
                if invertir_eje_y:
                    ax.set_ylim(profundidad_max, profundidad_min)
                else:
                    ax.set_ylim(profundidad_min, profundidad_max)
                
                # Graficar curvas para este pozo
                for curva in curvas:
                    if curva['columna'] in df_pozo.columns:
                        datos_curva = df_pozo[curva['columna']].values
                        prof_pozo = df_pozo['PROFUNDIDAD'].values
                        
                        # Aplicar suavizado
                        if curva["suavizar"] and curva["ventana"] > 1:
                            datos_curva = suavizar_serie(pd.Series(datos_curva), curva["ventana"]).values
                        
                        # Aplicar inversión
                        if curva["invertir_x"]:
                            datos_curva = -datos_curva
                        
                        ax.plot(
                            datos_curva,
                            prof_pozo,
                            color=curva["color"],
                            linestyle=curva["estilo"],
                            label=curva["etiqueta"],
                            linewidth=1.5
                        )
                
                ax.set_title(f"{pozo}", fontsize=10)
                ax.grid(True, alpha=0.3)
                if idx == 0:
                    ax.set_ylabel("Profundidad")
                ax.set_xlabel("Valores")
            
            plt.tight_layout()
            st.pyplot(fig_comparacion)
# ================================
# 🔍 MÓDULO 2: ANÁLISIS ESTADÍSTICO (SIMPLIFICADO)
# ================================
def modulo_analisis_estadistico():
    """Módulo de análisis estadístico mejorado"""
    
    instrucciones_estadisticas = """
    1. **Selecciona las columnas** numéricas para análisis
    2. **Configura** los parámetros estadísticos
    3. **Analiza** distribuciones y correlaciones
    4. **Exporta** reportes estadísticos completos
    """
    
    df = mostrar_cargador_datos(
        "Análisis Estadístico", 
        instrucciones_estadisticas,
        permitir_multicarga=True
    )
    
    if df is None:
        return
    
    st.header("📊 Análisis Estadístico Avanzado")
    
    # Verificar si es multicarga
    es_multicarga = 'ARCHIVO_ORIGEN' in df.columns if df is not None else False
    
    if es_multicarga:
        st.info("🎯 **Modo Multicarga Activado** - Analizando datos combinados")
        
        # Estadísticas por archivo
        st.subheader("📈 Estadísticas por Archivo")
        stats_por_archivo = df.groupby('ARCHIVO_ORIGEN').agg({
            'ARCHIVO_ORIGEN': 'count'
        }).rename(columns={'ARCHIVO_ORIGEN': 'Filas'})
        
        st.dataframe(stats_por_archivo, use_container_width=True)
    
    # Selección de columnas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columnas_numericas:
        st.error("❌ No se encontraron columnas numéricas para análisis")
        return
    
    st.subheader("🎯 Selección de Variables")
    
    columnas_analisis = st.multiselect(
        "Selecciona las columnas para análisis:",
        options=columnas_numericas,
        default=columnas_numericas[:min(5, len(columnas_numericas))]
    )
    
    if not columnas_analisis:
        st.warning("⚠️ Selecciona al menos una columna para análisis")
        return
    
    # Estadísticas descriptivas
    st.subheader("📋 Estadísticas Descriptivas")
    stats_df = df[columnas_analisis].describe().T
    stats_df['Varianza'] = df[columnas_analisis].var()
    stats_df['Asimetría'] = df[columnas_analisis].skew()
    stats_df['Curtosis'] = df[columnas_analisis].kurtosis()
    
    st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)
    
    # Matriz de correlación
    if len(columnas_analisis) > 1:
        st.subheader("🔗 Matriz de Correlación")
        corr_matrix = df[columnas_analisis].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.3f')
        st.pyplot(fig)
# ================================
# 🛢️ MÓDULO 3: ANÁLISIS PETROFÍSICO
# ================================
def modulo_analisis_petrofisico():
    st.header("🛢️ Análisis Petrofísico Avanzado")
    
    # Cargar datos
    df = mostrar_cargador_datos(
        "Análisis Petrofísico", 
        "Carga datos con curvas de registros para análisis petrofísico (RESISTIVIDAD, POROSIDAD, etc.)",
        permitir_multicarga=True
    )
    
    if df is None:
        return

    st.subheader("🎯 Configuración de Análisis Petrofísico")

    # Selección de curvas disponibles
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columnas_numericas:
        st.error("❌ No se encontraron columnas numéricas para análisis")
        return

    # ================================
    # 📊 CONFIGURACIÓN DE CURVAS
    # ================================
    col_curvas1, col_curvas2 = st.columns(2)
    
    with col_curvas1:
        st.markdown("#### 📈 Curvas de Resistividad")
        col_rt = st.selectbox("Resistividad Profunda (RT):", options=columnas_numericas, index=0)
        col_rs = st.selectbox("Resistividad Someras (RS):", options=columnas_numericas, index=min(1, len(columnas_numericas)-1))
        
        st.markdown("#### 📊 Curvas de Porosidad")
        col_density = st.selectbox("Densidad (RHOB):", options=columnas_numericas, index=min(2, len(columnas_numericas)-1))
        col_neutron = st.selectbox("Neutrón (NPHI):", options=columnas_numericas, index=min(3, len(columnas_numericas)-1))
        col_sonic = st.selectbox("Sónico (DT):", options=columnas_numericas, index=min(4, len(columnas_numericas)-1))
    
    with col_curvas2:
        st.markdown("#### 🎯 Parámetros Petrofísicos")
        rw = st.number_input("Resistividad del Agua (Rw) [Ωm]:", value=0.1, min_value=0.001, max_value=10.0, step=0.01)
        a_value = st.number_input("Constante a (Archie):", value=1.0, min_value=0.1, max_value=2.0, step=0.1)
        m_value = st.number_input("Exponente de cementación m:", value=2.0, min_value=1.5, max_value=3.0, step=0.1)
        n_value = st.number_input("Exponente de saturación n:", value=2.0, min_value=1.5, max_value=3.0, step=0.1)
        
        st.markdown("#### ⚙️ Configuración de Litologías")
        vshale_cutoff = st.slider("Cutoff de Lutita (Vshale):", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        sw_cutoff = st.slider("Cutoff de Saturación de Agua (Sw):", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    # ================================
    # 🧮 CÁLCULOS PETROFÍSICOS
    # ================================
    st.subheader("🧮 Cálculos Petrofísicos")
    
    # Crear copia para cálculos
    df_petro = df.copy()
    
    # Inicializar columnas de resultados
    calculos_realizados = []
    
    # 1. Cálculo de Porosidad
    if st.checkbox("Calcular Porosidad a partir de Densidad", value=True):
        if col_density in df_petro.columns:
            # Parámetros de matriz y fluido
            rhoma = st.number_input("Densidad de Matriz (g/cc):", value=2.65, min_value=2.0, max_value=3.0, step=0.05)
            rhof = st.number_input("Densidad de Fluido (g/cc):", value=1.0, min_value=0.8, max_value=1.2, step=0.05)
            
            df_petro['PHI_DENS'] = (rhoma - df_petro[col_density]) / (rhoma - rhof)
            df_petro['PHI_DENS'] = df_petro['PHI_DENS'].clip(0, 1)  # Limitar entre 0 y 1
            calculos_realizados.append('PHI_DENS')
    
    # 2. Cálculo de Saturación de Agua (Archie)
    if st.checkbox("Calcular Saturación de Agua (Ecuación de Archie)", value=True):
        if all(col in df_petro.columns for col in [col_rt]) and 'PHI_DENS' in df_petro.columns:
            # Calcular Ro (resistividad de formación 100% saturada con agua)
            df_petro['RO'] = a_value * rw / (df_petro['PHI_DENS'] ** m_value)
            
            # Calcular Sw (saturación de agua)
            df_petro['SW_ARCHIE'] = ((a_value * rw) / (df_petro[col_rt] * df_petro['PHI_DENS'] ** m_value)) ** (1/n_value)
            df_petro['SW_ARCHIE'] = df_petro['SW_ARCHIE'].clip(0, 1)  # Limitar entre 0 y 1
            
            # Calcular Hidrocarburo móvil
            df_petro['SHC_ARCHIE'] = 1 - df_petro['SW_ARCHIE']
            df_petro['SHC_ARCHIE'] = df_petro['SHC_ARCHIE'].clip(0, 1)
            
            calculos_realizados.extend(['SW_ARCHIE', 'SHC_ARCHIE'])

    # 3. Cálculo de Volumen de Lutita
    if st.checkbox("Calcular Volumen de Lutita", value=True):
        # Usar curva de rayos gamma si está disponible
        col_gamma = st.selectbox("Curva de Rayos Gamma (GR):", 
                               options=[''] + columnas_numericas,
                               help="Selecciona la curva de rayos gamma para cálculo de Vshale")
        
        if col_gamma and col_gamma in df_petro.columns:
            # Obtener valores de GR limpio y GR lutita
            gr_min = st.number_input("GR Min (arena limpia):", 
                                   value=float(df_petro[col_gamma].quantile(0.05)), 
                                   step=1.0)
            gr_max = st.number_input("GR Max (lutita):", 
                                   value=float(df_petro[col_gamma].quantile(0.95)), 
                                   step=1.0)
            
            # Calcular Vshale (método lineal)
            df_petro['VSHALE'] = (df_petro[col_gamma] - gr_min) / (gr_max - gr_min)
            df_petro['VSHALE'] = df_petro['VSHALE'].clip(0, 1)
            calculos_realizados.append('VSHALE')

    # ================================
    # 📈 VISUALIZACIÓN DE RESULTADOS
    # ================================
    if calculos_realizados:
        st.subheader("📈 Resultados del Análisis Petrofísico")
        
        # Mostrar estadísticas de los cálculos
        st.markdown("#### 📊 Estadísticas de los Cálculos")
        stats_petro = df_petro[calculos_realizados].describe()
        st.dataframe(stats_petro, use_container_width=True)
        
        # Gráficos de resultados
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Histograma de porosidad
            if 'PHI_DENS' in calculos_realizados:
                fig_phi, ax_phi = plt.subplots(figsize=(8, 6))
                ax_phi.hist(df_petro['PHI_DENS'].dropna(), bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                ax_phi.set_xlabel('Porosidad (PHI_DENS)')
                ax_phi.set_ylabel('Frecuencia')
                ax_phi.set_title('Distribución de Porosidad')
                ax_phi.grid(True, alpha=0.3)
                st.pyplot(fig_phi)
                plt.close(fig_phi)
        
        with col_viz2:
            # Histograma de saturación de agua
            if 'SW_ARCHIE' in calculos_realizados:
                fig_sw, ax_sw = plt.subplots(figsize=(8, 6))
                ax_sw.hist(df_petro['SW_ARCHIE'].dropna(), bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
                ax_sw.set_xlabel('Saturación de Agua (SW_ARCHIE)')
                ax_sw.set_ylabel('Frecuencia')
                ax_sw.set_title('Distribución de Saturación de Agua')
                ax_sw.grid(True, alpha=0.3)
                st.pyplot(fig_sw)
                plt.close(fig_sw)
        
        # Crossplot Porosidad vs Resistividad
        if all(col in calculos_realizados for col in ['PHI_DENS', 'SW_ARCHIE']):
            st.markdown("#### 📊 Crossplot Porosidad vs Saturación de Agua")
            fig_cross, ax_cross = plt.subplots(figsize=(10, 6))
            
            scatter = ax_cross.scatter(df_petro['PHI_DENS'], df_petro['SW_ARCHIE'], 
                                     c=df_petro[col_rt] if col_rt in df_petro.columns else None,
                                     alpha=0.6, s=30, cmap='viridis')
            ax_cross.set_xlabel('Porosidad (PHI_DENS)')
            ax_cross.set_ylabel('Saturación de Agua (SW_ARCHIE)')
            ax_cross.set_title('Porosidad vs Saturación de Agua')
            ax_cross.grid(True, alpha=0.3)
            
            if col_rt in df_petro.columns:
                plt.colorbar(scatter, ax=ax_cross, label='Resistividad (RT)')
            
            st.pyplot(fig_cross)
            plt.close(fig_cross)

        # ================================
        # 🎯 IDENTIFICACIÓN DE ZONAS INTERESANTES
        # ================================
        st.subheader("🎯 Identificación de Zonas Potenciales")
        
        # Definir criterios para zona interesante
        criterios_zona = {
            'Porosidad mínima': st.number_input("Porosidad mínima:", value=0.1, min_value=0.0, max_value=0.5, step=0.01),
            'Saturación máxima de agua': st.number_input("Saturación máxima de agua:", value=0.5, min_value=0.0, max_value=1.0, step=0.05),
            'Volumen máximo de lutita': st.number_input("Volumen máximo de lutita:", value=0.3, min_value=0.0, max_value=1.0, step=0.05)
        }
        
        # Aplicar criterios
        mascara_zona = pd.Series(True, index=df_petro.index)
        
        if 'PHI_DENS' in calculos_realizados:
            mascara_zona &= (df_petro['PHI_DENS'] >= criterios_zona['Porosidad mínima'])
        
        if 'SW_ARCHIE' in calculos_realizados:
            mascara_zona &= (df_petro['SW_ARCHIE'] <= criterios_zona['Saturación máxima de agua'])
        
        if 'VSHALE' in calculos_realizados:
            mascara_zona &= (df_petro['VSHALE'] <= criterios_zona['Volumen máximo de lutita'])
        
        zonas_potenciales = df_petro[mascara_zona]
        
        st.success(f"**Zonas potenciales identificadas:** {len(zonas_potenciales)} intervalos")
        
        if not zonas_potenciales.empty:
            st.dataframe(zonas_potenciales[calculos_realizados].describe(), use_container_width=True)

        # ================================
        # 💾 EXPORTAR RESULTADOS
        # ================================
        st.subheader("💾 Exportar Resultados Petrofísicos")
        
        if st.button("📥 Generar Reporte Petrofísico", use_container_width=True):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Hoja de datos originales + cálculos
                df_petro.to_excel(writer, sheet_name='Datos_Petrofisicos', index=False)
                
                # Hoja de zonas potenciales
                if not zonas_potenciales.empty:
                    zonas_potenciales.to_excel(writer, sheet_name='Zonas_Potenciales', index=False)
                
                # Hoja de parámetros y configuración
                config_data = {
                    'Parámetro': ['Rw', 'a', 'm', 'n', 'Vshale Cutoff', 'Sw Cutoff', 'Zonas Identificadas'],
                    'Valor': [rw, a_value, m_value, n_value, vshale_cutoff, sw_cutoff, len(zonas_potenciales)]
                }
                config_df = pd.DataFrame(config_data)
                config_df.to_excel(writer, sheet_name='Configuracion', index=False)
            
            output.seek(0)
            
            st.download_button(
                label="⬇️ Descargar Reporte Petrofísico",
                data=output,
                file_name=f"analisis_petrofisico_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    else:
        st.warning("⚠️ No se han realizado cálculos. Activa al menos una opción de cálculo arriba.")

# ================================
# 🔍 MÓDULO 4: INTERPRETACIÓN AVANZADA
# ================================
def modulo_interpretacion_avanzada():
    st.header("🔍 Interpretación Avanzada")
    
    # Cargar datos
    df = mostrar_cargador_datos(
        "Interpretación Avanzada", 
        "Carga datos para análisis de facies, clusterización y interpretación avanzada",
        permitir_multicarga=True
    )
    
    if df is None:
        return

    st.subheader("🎯 Configuración de Interpretación")

    # Selección de variables para análisis
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columnas_numericas:
        st.error("❌ No se encontraron columnas numéricas para análisis")
        return

    st.markdown("#### 📊 Selección de Variables para Análisis")
    variables_analisis = st.multiselect(
        "Selecciona las variables para el análisis:",
        options=columnas_numericas,
        default=columnas_numericas[:min(4, len(columnas_numericas))],
        help="Selecciona las curvas que representen diferentes propiedades de la formación"
    )

    if not variables_analisis:
        st.warning("⚠️ Por favor selecciona al menos 2 variables para análisis")
        return

    if len(variables_analisis) < 2:
        st.error("❌ Necesitas al menos 2 variables para análisis de clusterización")
        return

    # Preparar datos
    df_analysis = df[variables_analisis].dropna()
    
    if df_analysis.empty:
        st.error("❌ No hay datos válidos después de eliminar valores nulos")
        return

    # ================================
    # 🎯 ANÁLISIS DE CLUSTERIZACIÓN
    # ================================
    st.subheader("🎯 Análisis de Clusterización")
    
    col_cluster1, col_cluster2 = st.columns(2)
    
    with col_cluster1:
        n_clusters = st.slider("Número de clusters:", min_value=2, max_value=10, value=3)
        algoritmo = st.selectbox("Algoritmo de clusterización:", 
                               options=['KMeans', 'DBSCAN', 'Agglomerative'])
    
    with col_cluster2:
        normalizar = st.checkbox("Normalizar datos antes de clusterización", value=True)
        random_state = st.number_input("Semilla aleatoria:", value=42, min_value=0, max_value=100)

    # Normalizar datos si es necesario
    if normalizar:
        scaler = StandardScaler()
        datos_escalados = scaler.fit_transform(df_analysis)
    else:
        datos_escalados = df_analysis.values

    # Aplicar clusterización
    try:
        if algoritmo == 'KMeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
            labels = clusterer.fit_predict(datos_escalados)
            
        elif algoritmo == 'DBSCAN':
            eps = st.slider("Parámetro EPS (DBSCAN):", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            min_samples = st.slider("Mínimo de muestras (DBSCAN):", min_value=2, max_value=20, value=5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(datos_escalados)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
        else:  # Agglomerative
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(datos_escalados)
        
        # Añadir labels al DataFrame
        df_analysis_clustered = df_analysis.copy()
        df_analysis_clustered['CLUSTER'] = labels
        df_analysis_clustered['CLUSTER'] = df_analysis_clustered['CLUSTER'].astype(str)
        
        st.success(f"✅ Clusterización completada: {n_clusters} clusters identificados")
        
    except Exception as e:
        st.error(f"❌ Error en clusterización: {str(e)}")
        return

    # ================================
    # 📊 VISUALIZACIÓN DE CLUSTERS
    # ================================
    st.subheader("📊 Visualización de Clusters")
    
    # Scatter plot matrix
    if len(variables_analisis) >= 2:
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            x_var = st.selectbox("Variable X:", options=variables_analisis, key="cluster_x")
        with col_viz2:
            y_var = st.selectbox("Variable Y:", options=variables_analisis, key="cluster_y")
        
        if x_var != y_var:
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            
            scatter = ax_scatter.scatter(df_analysis_clustered[x_var], 
                                       df_analysis_clustered[y_var], 
                                       c=df_analysis_clustered['CLUSTER'].astype('category').cat.codes,
                                       cmap='viridis', alpha=0.6, s=30)
            ax_scatter.set_xlabel(x_var)
            ax_scatter.set_ylabel(y_var)
            ax_scatter.set_title(f'Clusters: {x_var} vs {y_var}')
            ax_scatter.grid(True, alpha=0.3)
            
            # Añadir leyenda de clusters
            plt.colorbar(scatter, ax=ax_scatter, label='Cluster')
            
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)

    # ================================
    # 📈 ANÁLISIS DE CLUSTERS
    # ================================
    st.subheader("📈 Estadísticas por Cluster")
    
    # Estadísticas descriptivas por cluster
    stats_clusters = df_analysis_clustered.groupby('CLUSTER').describe()
    st.dataframe(stats_clusters, use_container_width=True)
    
    # Interpretación de clusters
    st.markdown("#### 🔍 Interpretación de Clusters")
    
    # Calcular promedios por cluster para interpretación
    promedios_cluster = df_analysis_clustered.groupby('CLUSTER').mean()
    
    # Mostrar características de cada cluster
    for cluster_id in promedios_cluster.index:
        st.markdown(f"**Cluster {cluster_id}:**")
        cluster_data = promedios_cluster.loc[cluster_id]
        
        col_interp1, col_interp2 = st.columns(2)
        
        with col_interp1:
            # Encontrar variable con valor máximo
            var_max = cluster_data.idxmax()
            val_max = cluster_data.max()
            st.write(f"• **Mayor valor:** {var_max} = {val_max:.3f}")
        
        with col_interp2:
            # Encontrar variable con valor mínimo
            var_min = cluster_data.idxmin()
            val_min = cluster_data.min()
            st.write(f"• **Menor valor:** {var_min} = {val_min:.3f}")
        
        # Interpretación básica basada en los valores
        st.write("• **Posible interpretación:** ", end="")
        
        # Aquí puedes agregar lógica de interpretación específica para tus datos
        if 'GR' in variables_analisis:
            gr_val = cluster_data.get('GR', 0)
            if gr_val > promedios_cluster['GR'].mean():
                st.write("Posible zona lutítica")
            else:
                st.write("Posible zona arenosa")
        else:
            st.write("Analizar patrones específicos de las curvas seleccionadas")

    # ================================
    # 💾 EXPORTAR RESULTADOS
    # ================================
    st.subheader("💾 Exportar Resultados de Interpretación")
    
    if st.button("📥 Generar Reporte de Interpretación", use_container_width=True):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Hoja de datos con clusters
            df_analysis_clustered.to_excel(writer, sheet_name='Datos_Clusters', index=False)
            
            # Hoja de estadísticas por cluster
            stats_clusters.to_excel(writer, sheet_name='Estadisticas_Clusters')
            
            # Hoja de interpretación
            interpretacion_data = []
            for cluster_id in promedios_cluster.index:
                cluster_row = {'Cluster': cluster_id}
                for var in variables_analisis:
                    cluster_row[var] = promedios_cluster.loc[cluster_id, var]
                cluster_row['Muestras'] = len(df_analysis_clustered[df_analysis_clustered['CLUSTER'] == cluster_id])
                interpretacion_data.append(cluster_row)
            
            interpretacion_df = pd.DataFrame(interpretacion_data)
            interpretacion_df.to_excel(writer, sheet_name='Interpretacion', index=False)
        
        output.seek(0)
        
        st.download_button(
            label="⬇️ Descargar Reporte de Interpretación",
            data=output,
            file_name=f"interpretacion_avanzada_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
# ================================
# 📋 MÓDULO 5: REPORTES AUTOMÁTICOS
# ================================
def modulo_reportes_automaticos():
    st.header("📋 Generación de Reportes Automáticos")
    
    st.info("""
    **📊 Módulo de Reportes Automáticos:**
    Genera reportes profesionales con análisis completo de los datos de pozo.
    Incluye estadísticas, gráficos, interpretaciones y recomendaciones.
    """)
    
    # Cargar datos
    df = mostrar_cargador_datos(
        "Reportes Automáticos", 
        "Carga datos para generar reporte automático completo",
        permitir_multicarga=True
    )
    
    if df is None:
        return

    # Configuración del reporte
    st.subheader("⚙️ Configuración del Reporte")
    
    col_report1, col_report2 = st.columns(2)
    
    with col_report1:
        nombre_pozo = st.text_input("Nombre del Pozo:", value="Pozo_Ejemplo")
        operadora = st.text_input("Operadora:", value="Compañía Ejemplo")
        formato_reporte = st.selectbox("Formato de salida:", options=["Excel Completo", "PDF Resumen"])
        
    with col_report2:
        incluir_estadisticas = st.checkbox("Incluir análisis estadístico", value=True)
        incluir_correlaciones = st.checkbox("Incluir matriz de correlaciones", value=True)
        incluir_graficos = st.checkbox("Incluir gráficos principales", value=True)
        incluir_interpretacion = st.checkbox("Incluir interpretación", value=True)

    # Selección de variables para el reporte
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columnas_numericas:
        st.error("❌ No se encontraron columnas numéricas para el reporte")
        return

    variables_reporte = st.multiselect(
        "Variables a incluir en el reporte:",
        options=columnas_numericas,
        default=columnas_numericas,
        help="Selecciona las curvas que quieres incluir en el análisis del reporte"
    )

    if not variables_reporte:
        st.warning("⚠️ Por favor selecciona al menos una variable para el reporte")
        return

    # Generar reporte
    if st.button("🚀 Generar Reporte Automático", type="primary", use_container_width=True):
        with st.spinner("Generando reporte automático..."):
            
            # Crear reporte en Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                
                # ===== HOJA 1: PORTADA Y METADATOS =====
                portada_data = {
                    'Campo': [nombre_pozo],
                    'Operadora': [operadora],
                    'Fecha de Generación': [datetime.now().strftime('%Y-%m-%d %H:%M')],
                    'Número de Muestras': [len(df)],
                    'Número de Curvas': [len(variables_reporte)],
                    'Rango de Profundidad': [f"{df.select_dtypes(include=[np.number]).iloc[:, 0].min():.2f} - {df.select_dtypes(include=[np.number]).iloc[:, 0].max():.2f}"]
                }
                portada_df = pd.DataFrame(portada_data)
                portada_df.to_excel(writer, sheet_name='PORTADA', index=False)
                
                # ===== HOJA 2: DATOS ORIGINALES =====
                df[variables_reporte].to_excel(writer, sheet_name='DATOS_ORIGINALES', index=False)
                
                # ===== HOJA 3: ESTADÍSTICAS DESCRIPTIVAS =====
                if incluir_estadisticas:
                    stats_df = df[variables_reporte].describe().T
                    stats_df['Varianza'] = df[variables_reporte].var()
                    stats_df['Asimetría'] = df[variables_reporte].skew()
                    stats_df['Curtosis'] = df[variables_reporte].kurtosis()
                    stats_df['Valores Nulos'] = df[variables_reporte].isnull().sum()
                    stats_df['% Nulos'] = (df[variables_reporte].isnull().sum() / len(df)) * 100
                    stats_df.to_excel(writer, sheet_name='ESTADISTICAS')
                
                # ===== HOJA 4: CORRELACIONES =====
                if incluir_correlaciones and len(variables_reporte) > 1:
                    corr_matrix = df[variables_reporte].corr()
                    corr_matrix.to_excel(writer, sheet_name='CORRELACIONES')
                
                # ===== HOJA 5: INTERPRETACIÓN =====
                if incluir_interpretacion:
                    interpretacion_data = []
                    
                    for variable in variables_reporte:
                        datos_var = df[variable].dropna()
                        if len(datos_var) > 0:
                            interpretacion_data.append({
                                'Variable': variable,
                                'Media': datos_var.mean(),
                                'Mediana': datos_var.median(),
                                'Mínimo': datos_var.min(),
                                'Máximo': datos_var.max(),
                                'Rango': datos_var.max() - datos_var.min(),
                                'Interpretación': generar_interpretacion(variable, datos_var)
                            })
                    
                    interpretacion_df = pd.DataFrame(interpretacion_data)
                    interpretacion_df.to_excel(writer, sheet_name='INTERPRETACION', index=False)
                
                # ===== HOJA 6: RECOMENDACIONES =====
                recomendaciones = generar_recomendaciones(df[variables_reporte])
                recomendaciones_df = pd.DataFrame(recomendaciones, columns=['Recomendaciones'])
                recomendaciones_df.to_excel(writer, sheet_name='RECOMENDACIONES', index=False)
            
            output.seek(0)
            
            # Descargar reporte
            st.success("✅ Reporte generado exitosamente!")
            
            st.download_button(
                label="📥 Descargar Reporte Completo",
                data=output,
                file_name=f"reporte_{nombre_pozo}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

def generar_interpretacion(variable, data):
    """Genera interpretación automática basada en estadísticas de la variable"""
    mean_val = data.mean()
    std_val = data.std()
    skew_val = data.skew()
    
    interpretations = []
    
    # Interpretación basada en valores típicos de curvas de pozo
    if 'GR' in variable.upper():
        if mean_val < 50:
            interpretations.append("Zona potencialmente arenosa")
        elif mean_val > 100:
            interpretations.append("Zona potencialmente lutítica")
        else:
            interpretations.append("Zona intermedia arena-lutita")
    
    if 'RES' in variable.upper() or 'RT' in variable.upper():
        if mean_val > 100:
            interpretations.append("Alta resistividad - posible zona productora")
        elif mean_val < 10:
            interpretations.append("Baja resistividad - posible zona acuífera")
    
    if 'NPHI' in variable.upper() or 'PHI' in variable.upper():
        if mean_val > 0.15:
            interpretations.append("Buena porosidad - favorable")
        elif mean_val < 0.05:
            interpretations.append("Baja porosidad - desfavorable")
    
    if 'RHOB' in variable.upper():
        if mean_val < 2.3:
            interpretations.append("Baja densidad - posible porosidad")
        elif mean_val > 2.6:
            interpretations.append("Alta densidad - posible compactación")
    
    # Interpretación basada en distribución
    if abs(skew_val) > 1:
        interpretations.append("Distribución muy asimétrica")
    elif abs(skew_val) > 0.5:
        interpretations.append("Distribución moderadamente asimétrica")
    else:
        interpretations.append("Distribución relativamente simétrica")
    
    return "; ".join(interpretations) if interpretations else "Análisis estándar - revisar valores específicos"

def generar_recomendaciones(data):
    """Genera recomendaciones automáticas basadas en el análisis de datos"""
    recommendations = []
    
    # Análisis de completitud de datos
    null_percentage = (data.isnull().sum() / len(data) * 100).max()
    if null_percentage > 50:
        recommendations.append("⚠️ ALTA PROPORCIÓN DE DATOS FALTANTES - Considerar adquisición adicional de datos")
    elif null_percentage > 20:
        recommendations.append("📋 DATOS INCOMPLETOS - Sugerir complementar con registros adicionales")
    
    # Análisis de variabilidad
    numeric_data = data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        cv_values = numeric_data.std() / numeric_data.mean()
        high_variability = (cv_values > 1).any()
        
        if high_variability:
            recommendations.append("📊 ALTA VARIABILIDAD EN DATOS - Recomendable análisis detallado por zonas")
        else:
            recommendations.append("📈 DATOS CONSISTENTES - Favorable para interpretación integrada")
    
    # Recomendaciones generales
    recommendations.extend([
        "✅ VALIDAR DATOS CON NÚCLEOS Y PRUEBAS DE POZO",
        "🔍 REALIZAR ANÁLISIS INTEGRADO CON DATOS SÍSMICOS",
        "📝 CONSIDERAR CONTEXTO GEOLÓGICO REGIONAL",
        "🎯 PRIORIZAR ZONAS CON MEJORES CARACTERÍSTICAS PETROFÍSICAS"
    ])
    
    return recommendations

# ================================
# 🧠 MÓDULO 6: MACHINE LEARNING AVANZADO
# ================================
def modulo_machine_learning():
    st.header("🧠 Machine Learning para Análisis Petrofísico")
    
    st.markdown("""
    **🔮 Predicción inteligente de facies y propiedades de formación usando algoritmos de ML**
    
    **Funcionalidades:**
    • Clasificación automática de facies
    • Predicción de propiedades petrofísicas
    • Clusterización no supervisada
    • Análisis de componentes principales (PCA)
    • Optimización de hiperparámetros
    """)
    
    # Cargar datos
    df = mostrar_cargador_datos(
        "Machine Learning", 
        "Carga datos con curvas de registros para entrenar modelos predictivos",
        permitir_multicarga=True
    )
    
    if df is None:
        return

    # Selección de modo de operación
    st.subheader("🎯 Modo de Operación")
    modo_ml = st.radio(
        "Selecciona el tipo de análisis:",
        ["Clasificación Supervisada", "Clusterización No Supervisada", "Predicción de Propiedades"],
        horizontal=True
    )

    # Selección de variables
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columnas_numericas:
        st.error("❌ No se encontraron columnas numéricas para análisis")
        return

    st.subheader("📊 Selección de Variables")
    
    col_ml1, col_ml2 = st.columns(2)
    
    with col_ml1:
        features = st.multiselect(
            "Variables predictoras (Features):",
            options=columnas_numericas,
            default=columnas_numericas[:min(5, len(columnas_numericas))],
            help="Selecciona las curvas que usarás para predecir"
        )
    
    with col_ml2:
        if modo_ml == "Clasificación Supervisada":
            # Buscar columna de facies o crear selector
            posibles_facies = [col for col in df.columns if any(x in col.upper() for x in ['FACIES', 'LITHO', 'CLASS', 'ZONE'])]
            if posibles_facies:
                target = st.selectbox("Variable objetivo (Facies):", options=posibles_facies)
            else:
                st.warning("No se encontró columna de facies. Usa clusterización no supervisada.")
                modo_ml = "Clusterización No Supervisada"
        
        elif modo_ml == "Predicción de Propiedades":
            target = st.selectbox(
                "Variable a predecir:",
                options=[col for col in columnas_numericas if col not in features]
            )

    if not features:
        st.warning("⚠️ Selecciona al menos una variable predictora")
        return

    # Preparar datos
    df_ml = df[features].copy().dropna()
    
    if df_ml.empty:
        st.error("❌ No hay datos válidos después del preprocesamiento")
        return

    # ================================
    # 🎯 CLASIFICACIÓN SUPERVISADA
    # ================================
    if modo_ml == "Clasificación Supervisada" and 'target' in locals():
        
        st.subheader("🎯 Configuración de Clasificación")
        
        # Preparar datos target
        y = df[target].copy()
        
        # Codificar labels si es necesario
        if y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            st.info(f"Facies codificadas: {dict(zip(le.classes_, range(len(le.classes_))))}")
        else:
            y_encoded = y.values
        
        # Combinar features y target
        df_combined = df_ml.copy()
        df_combined = df_combined.loc[y.index]  # Alinear índices
        df_combined['TARGET'] = y_encoded
        df_combined = df_combined.dropna()
        
        if df_combined.empty:
            st.error("❌ No hay datos alineados entre features y target")
            return
        
        X = df_combined[features]
        y_final = df_combined['TARGET']
        
        # Configuración del modelo
        col_model1, col_model2 = st.columns(2)
        
        with col_model1:
            algoritmo = st.selectbox(
                "Algoritmo de clasificación:",
                ["Random Forest", "Gradient Boosting", "SVM", "Red Neuronal"]
            )
            
            test_size = st.slider("Porcentaje para test:", 0.1, 0.4, 0.2, 0.05)
        
        with col_model2:
            cv_folds = st.slider("Folds para validación cruzada:", 3, 10, 5)
            random_state = st.number_input("Semilla aleatoria:", 42, key="ml_rand")
        
        # Entrenar modelo
        if st.button("🚀 Entrenar Modelo de Clasificación", type="primary"):
            with st.spinner("Entrenando modelo..."):
                
                # Dividir datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_final, test_size=test_size, random_state=random_state, stratify=y_final
                )
                
                # Escalar features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Seleccionar y configurar modelo
                if algoritmo == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                elif algoritmo == "Gradient Boosting":
                    model = GradientBoostingClassifier(random_state=random_state)
                elif algoritmo == "SVM":
                    model = SVC(probability=True, random_state=random_state)
                else:  # Red Neuronal
                    model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=random_state)
                
                # Entrenar
                model.fit(X_train_scaled, y_train)
                
                # Predecir
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)
                
                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
                
                # Mostrar resultados
                st.subheader("📊 Resultados del Modelo")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric("Accuracy Test", f"{accuracy:.3f}")
                with col_res2:
                    st.metric("Accuracy CV Mean", f"{cv_scores.mean():.3f}")
                with col_res3:
                    st.metric("Accuracy CV Std", f"{cv_scores.std():.3f}")
                
                # Matriz de confusión
                st.markdown("#### 🎯 Matriz de Confusión")
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel('Predicción')
                ax_cm.set_ylabel('Real')
                ax_cm.set_title('Matriz de Confusión')
                st.pyplot(fig_cm)
                
                # Reporte de clasificación
                st.markdown("#### 📋 Reporte de Clasificación")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                
                # Importancia de características
                if hasattr(model, 'feature_importances_'):
                    st.markdown("#### 📈 Importancia de Características")
                    feature_importance = pd.DataFrame({
                        'feature': features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax_fi)
                    ax_fi.set_title('Importancia de Características')
                    st.pyplot(fig_fi)
                
                # Guardar modelo en session state
                st.session_state.ml_model = model
                st.session_state.ml_scaler = scaler
                st.session_state.ml_features = features
                
                st.success("✅ Modelo entrenado y guardado exitosamente!")

    # ================================
    # 🔍 CLUSTERIZACIÓN NO SUPERVISADA
    # ================================
    elif modo_ml == "Clusterización No Supervisada":
        
        st.subheader("🔍 Configuración de Clusterización")
        
        col_clust1, col_clust2 = st.columns(2)
        
        with col_clust1:
            metodo_cluster = st.selectbox(
                "Método de clusterización:",
                ["K-Means", "DBSCAN", "Agrupamiento Jerárquico"]
            )
            
            if metodo_cluster == "K-Means":
                n_clusters = st.slider("Número de clusters:", 2, 10, 3)
            elif metodo_cluster == "DBSCAN":
                eps = st.slider("Parámetro EPS:", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("Mínimo de muestras:", 2, 20, 5)
        
        with col_clust2:
            usar_pca = st.checkbox("Usar PCA para visualización", value=True)
            random_state = st.number_input("Semilla aleatoria:", 42, key="cluster_rand")
        
        if st.button("🎯 Ejecutar Clusterización", type="primary"):
            with st.spinner("Realizando clusterización..."):
                
                # Escalar datos
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_ml)
                
                # Aplicar clusterización
                if metodo_cluster == "K-Means":
                    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
                elif metodo_cluster == "DBSCAN":
                    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                else:  # Agglomerative
                    from sklearn.cluster import AgglomerativeClustering
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                
                labels = clusterer.fit_predict(X_scaled)
                
                # Añadir labels al DataFrame
                df_clustered = df_ml.copy()
                df_clustered['CLUSTER'] = labels
                df_clustered['CLUSTER'] = df_clustered['CLUSTER'].astype(str)
                
                # Resultados
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                st.success(f"✅ Clusterización completada: {n_clusters_found} clusters identificados")
                
                # Visualización con PCA
                if usar_pca and len(features) > 1:
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    df_viz = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Cluster': labels.astype(str)
                    })
                    
                    fig_pca = px.scatter(
                        df_viz, x='PC1', y='PC2', color='Cluster',
                        title='Visualización PCA de Clusters',
                        opacity=0.7
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    # Varianza explicada
                    var_explained = pca.explained_variance_ratio_.sum()
                    st.info(f"**Varianza explicada por PCA:** {var_explained:.2%}")
                
                # Estadísticas por cluster
                st.markdown("#### 📊 Estadísticas por Cluster")
                stats_clusters = df_clustered.groupby('CLUSTER').describe()
                st.dataframe(stats_clusters, use_container_width=True)
                
                # Interpretación de clusters
                st.markdown("#### 🔍 Interpretación de Clusters")
                
                promedios_cluster = df_clustered.groupby('CLUSTER').mean()
                
                for cluster_id in promedios_cluster.index:
                    if cluster_id != '-1':  # Excluir outliers de DBSCAN
                        st.markdown(f"**Cluster {cluster_id}:**")
                        cluster_data = promedios_cluster.loc[cluster_id]
                        
                        # Características distintivas
                        var_max = cluster_data.idxmax()
                        var_min = cluster_data.idxmin()
                        
                        col_interp1, col_interp2 = st.columns(2)
                        with col_interp1:
                            st.write(f"• **Alto:** {var_max} = {cluster_data[var_max]:.3f}")
                        with col_interp2:
                            st.write(f"• **Bajo:** {var_min} = {cluster_data[var_min]:.3f}")
                        
                        # Interpretación petrofísica
                        interpretacion = interpretar_cluster_petrofisico(cluster_data, features)
                        st.write(f"• **Interpretación:** {interpretacion}")
                
                # Guardar resultados
                st.session_state.clustering_results = df_clustered
                st.session_state.clustering_scaler = scaler

def interpretar_cluster_petrofisico(cluster_data, features):
    """Interpreta un cluster basado en valores promedio de características petrofísicas"""
    
    interpretaciones = []
    
    # Análisis basado en curvas típicas
    for feature, value in cluster_data.items():
        if 'GR' in feature.upper():
            if value > 100:
                interpretaciones.append("Lutítico")
            elif value < 50:
                interpretaciones.append("Arenoso")
                
        elif 'RES' in feature.upper() or 'RT' in feature.upper():
            if value > 50:
                interpretaciones.append("Resistivo")
            elif value < 10:
                interpretaciones.append("Conductor")
                
        elif 'NPHI' in feature.upper() or 'PHI' in feature.upper():
            if value > 0.20:
                interpretaciones.append("Poroso")
            elif value < 0.10:
                interpretaciones.append("Compacto")
                
        elif 'RHOB' in feature.upper():
            if value < 2.3:
                interpretaciones.append("Baja densidad")
            elif value > 2.6:
                interpretaciones.append("Alta densidad")
    
    # Eliminar duplicados y unir
    interpretaciones_unicas = list(set(interpretaciones))
    
    if interpretaciones_unicas:
        return " | ".join(interpretaciones_unicas)
    else:
        return "Cluster con características mixtas - análisis detallado requerido"
# ================================
# 🎛️ FUNCIÓN PRINCIPAL ACTUALIZADA
# ================================
def main():
    """Función principal de la aplicación"""
    
    # Inicializar session state
    if 'df_actual' not in st.session_state:
        st.session_state.df_actual = None
    if 'multicarga_info' not in st.session_state:
        st.session_state.multicarga_info = None
    
    # Sidebar para navegación
    st.sidebar.title("🛢️ Navegación")
    st.sidebar.markdown("---")
    
    # Selección de módulo
    modulo = st.sidebar.selectbox(
        "Selecciona el módulo:",
        options=[
            "🏠 Inicio",
            "📊 Visualización Básica", 
            "📈 Análisis Estadístico",
            "🛢️ Análisis Petrofísico",
            "🔍 Interpretación Avanzada",
            "📋 Reportes Automáticos",
            "🧠 Machine Learning Avanzado"
        ]
    )
    
    # Información en sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Novedades")
    st.sidebar.success("""
    **✨ Nuevas características:**
    • **Lasio integrado** para mejor procesamiento LAS
    • **Multicarga de archivos** 
    • **Combinación automática** de datos
    • **Metadatos completos** de curvas
    """)
    
    # Navegación a módulos
    if modulo == "🏠 Inicio":
        st.header("🏠 Bienvenido al Sistema Avanzado de Análisis Petrofísico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 **Características Principales**
            
            **📁 Carga Inteligente:**
            • **Lasio integrado** - Procesamiento profesional de archivos LAS
            • **Multicarga** - Combina múltiples archivos automáticamente
            • **Detección automática** de formatos
            • **Metadatos completos** de curvas y pozos
            
            **📊 Módulos de Análisis:**
            • **Visualización avanzada** - Gráficos profesionales de registros
            • **Análisis estadístico** - Correlaciones y distribuciones
            • **Análisis petrofísico** - Porosidad, saturación, Vshale
            • **Interpretación avanzada** - Clusterización y facies
            • **Reportes automáticos** - Generación profesional
            • **Machine Learning** - Predicción inteligente
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 **Novedades Implementadas**
            
            **🔄 Multicarga de Archivos:**
            • Carga múltiples archivos LAS/Excel simultáneamente
            • Combinación automática de datos
            • Preservación de metadatos
            • Filtrado por archivo individual
            
            **🔧 Lasio Integrado:**
            • Procesamiento robusto de archivos LAS
            • Información completa de curvas (unidades, descripciones)
            • Metadatos estructurados del pozo
            • Manejo de valores nulos específicos
            
            **💾 Exportación Profesional:**
            • Excel con múltiples hojas
            • Configuraciones guardadas
            • Reportes estadísticos completos
            """)
        
        # Información de instalación
        st.markdown("---")
        st.subheader("🔧 Instalación y Requisitos")
        
        col_inst1, col_inst2 = st.columns(2)
        
        with col_inst1:
            st.markdown("""
            **📦 Para mejor experiencia, instala:**
            ```bash
            pip install lasio
            ```
            
            **✅ Características con Lasio:**
            • Procesamiento profesional de LAS
            • Metadatos estructurados
            • Unidades y descripciones
            • Compatibilidad con estándares
            """)
        
        with col_inst2:
            st.markdown("""
            **🛠️ Funcionalidades base:**
            • Procesamiento manual de LAS
            • Carga de Excel
            • Visualización básica
            • Análisis estadístico
            • Multicarga de archivos
            • Todos los módulos de análisis
            """)
        
    elif modulo == "📊 Visualización Básica":
        modulo_visualizacion_basica()
        
    elif modulo == "📈 Análisis Estadístico":
        modulo_analisis_estadistico()
        
    elif modulo == "🛢️ Análisis Petrofísico":
        modulo_analisis_petrofisico()
        
    elif modulo == "🔍 Interpretación Avanzada":
        modulo_interpretacion_avanzada()
        
    elif modulo == "📋 Reportes Automáticos":
        modulo_reportes_automaticos()
        
    elif modulo == "🧠 Machine Learning Avanzado":
        modulo_machine_learning()

# Ejecutar la aplicación
if __name__ == "__main__":
    main()