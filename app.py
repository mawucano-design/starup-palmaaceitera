import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from shapely.geometry import Point, Polygon, box
from datetime import datetime, timedelta
import os
import requests
import json
import hashlib
import mercadopago
from branca.colormap import LinearColormap
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import io
import base64

# ========== SUPABASE (persistencia de usuarios) ==========
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Faltan credenciales de Supabase. Configúralas en secrets.")
    st.stop()

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(email, password):
    password_hash = hash_password(password)
    payload = {
        "email": email,
        "password_hash": password_hash,
        "subscription_expires": None
    }
    response = requests.post(f"{SUPABASE_URL}/rest/v1/users", headers=headers, json=payload)
    return response.status_code == 201

def login_user(email, password):
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/users",
        headers=headers,
        params={"email": f"eq.{email}"}
    )
    if response.status_code == 200 and response.json():
        user = response.json()[0]
        if user["password_hash"] == hash_password(password):
            return {
                "id": user["id"],
                "email": user["email"],
                "subscription_expires": user["subscription_expires"]
            }
    return None

def update_subscription(email, days=30):
    new_expiry = (datetime.now() + timedelta(days=days)).isoformat()
    response = requests.patch(
        f"{SUPABASE_URL}/rest/v1/users",
        headers=headers,
        params={"email": f"eq.{email}"},
        json={"subscription_expires": new_expiry}
    )
    return response.status_code == 204

# ========== MERCADO PAGO ==========
MERCADOPAGO_ACCESS_TOKEN = os.environ.get("MERCADOPAGO_ACCESS_TOKEN")
if not MERCADOPAGO_ACCESS_TOKEN:
    st.error("❌ No se encontró MERCADOPAGO_ACCESS_TOKEN")
    st.stop()

sdk = mercadopago.SDK(MERCADOPAGO_ACCESS_TOKEN)

def create_preference(email, amount=150.0, description="Suscripción mensual"):
    base_url = os.environ.get("APP_BASE_URL", "https://tuapp.streamlit.app")
    preference_data = {
        "items": [{
            "title": description,
            "quantity": 1,
            "currency_id": "USD",
            "unit_price": amount
        }],
        "payer": {"email": email},
        "back_urls": {
            "success": f"{base_url}?payment=success",
            "failure": f"{base_url}?payment=failure",
            "pending": f"{base_url}?payment=pending"
        },
        "auto_return": "approved",
        "external_reference": email,
    }
    preference_response = sdk.preference().create(preference_data)
    if preference_response["status"] in [200, 201]:
        return preference_response["response"]["init_point"]
    return None

def check_payment_status(payment_id):
    payment_info = sdk.payment().get(payment_id)
    if payment_info["status"] == 200:
        payment = payment_info["response"]
        if payment["status"] == "approved":
            email = payment.get("external_reference")
            if email:
                update_subscription(email)
                return True
    return False

# ========== FUNCIONES DE AUTENTICACIÓN EN LA APP ==========
def show_login_signup():
    with st.sidebar:
        st.markdown("## 🔐 Acceso")
        menu = st.radio("", ["Iniciar sesión", "Registrarse"])
        email = st.text_input("Email")
        password = st.text_input("Contraseña", type="password")
        if menu == "Registrarse":
            if st.button("Registrar"):
                if register_user(email, password):
                    st.success("Registro exitoso. Ahora inicia sesión.")
                else:
                    st.error("El email ya está registrado.")
        else:
            if st.button("Ingresar"):
                user = login_user(email, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Email o contraseña incorrectos")

def logout():
    if st.sidebar.button("Cerrar sesión"):
        del st.session_state.user
        st.rerun()

def check_subscription():
    if 'user' not in st.session_state:
        show_login_signup()
        st.stop()
    with st.sidebar:
        st.markdown(f"👤 Usuario: {st.session_state.user['email']}")
        logout()
    user = st.session_state.user
    expiry = user.get('subscription_expires')
    if expiry:
        try:
            expiry_date = datetime.fromisoformat(expiry)
            if expiry_date > datetime.now():
                dias = (expiry_date - datetime.now()).days
                st.sidebar.info(f"✅ Suscripción activa ({dias} días)")
                return True
        except:
            pass
    # Si no hay suscripción activa, mostrar opciones de pago
    st.warning("🔒 Tu suscripción ha expirado o no tienes una activa.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💵 Pagar ahora 150 USD"):
            init_point = create_preference(user['email'])
            if init_point:
                st.markdown(f"[Haz clic aquí para pagar]({init_point})")
            else:
                st.error("Error al generar link de pago.")
    with col2:
        if st.button("🎮 Continuar con DEMO"):
            st.session_state.demo_mode = True
            st.rerun()
    query_params = st.query_params
    if 'payment' in query_params and query_params['payment'] == 'success' and 'collection_id' in query_params:
        if check_payment_status(query_params['collection_id']):
            st.success("✅ Pago aprobado. Suscripción activada.")
            st.session_state.user['subscription_expires'] = (datetime.now() + timedelta(days=30)).isoformat()
            st.rerun()
    st.stop()

# ========== FUNCIONES DE EJEMPLO (datos simulados) ==========
def cargar_ejemplo_demo():
    minx, miny, maxx, maxy = -67.5, 8.5, -67.3, 8.7
    polygon = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs='EPSG:4326')
    gdf['id_bloque'] = 1
    return gdf

def dividir_plantacion_en_bloques(gdf, n_bloques):
    if gdf is None: return gdf
    plantacion = gdf.iloc[0].geometry
    bounds = plantacion.bounds
    minx, miny, maxx, maxy = bounds
    n_cols = int(np.ceil(np.sqrt(n_bloques)))
    n_rows = int(np.ceil(n_bloques / n_cols))
    width = (maxx - minx) / n_cols
    height = (maxy - miny) / n_rows
    sub_poligonos = []
    for i in range(n_rows):
        for j in range(n_cols):
            cell = box(minx + j*width, miny + i*height, minx + (j+1)*width, miny + (i+1)*height)
            inter = plantacion.intersection(cell)
            if not inter.is_empty and inter.area > 0:
                sub_poligonos.append(inter)
    if sub_poligonos:
        return gpd.GeoDataFrame({'id_bloque': range(1, len(sub_poligonos)+1), 'geometry': sub_poligonos}, crs='EPSG:4326')
    return gdf

def calcular_superficie(gdf):
    if gdf is None: return 0
    return gdf.to_crs('EPSG:3857').geometry.area.sum() / 10000

def generar_datos_simulados(gdf, n_divisiones):
    gdf_dividido = dividir_plantacion_en_bloques(gdf, n_divisiones)
    areas = []
    for _, row in gdf_dividido.iterrows():
        areas.append(calcular_superficie(gpd.GeoDataFrame({'geometry': [row.geometry]}, crs='EPSG:4326')))
    gdf_dividido['area_ha'] = areas
    np.random.seed(42)
    n = len(gdf_dividido)
    ndvi = 0.5 + 0.2 * np.random.randn(n)
    ndvi = np.clip(ndvi, 0.2, 0.9)
    gdf_dividido['ndvi'] = np.round(ndvi, 3)
    ndwi = 0.3 + 0.15 * np.random.randn(n)
    ndwi = np.clip(ndwi, 0.1, 0.7)
    gdf_dividido['ndwi'] = np.round(ndwi, 3)
    gdf_dividido['edad'] = np.round(5 + 10 * np.random.rand(n), 1)
    def salud(ndvi):
        if ndvi < 0.4: return 'Crítica'
        if ndvi < 0.6: return 'Baja'
        if ndvi < 0.75: return 'Moderada'
        return 'Buena'
    gdf_dividido['salud'] = gdf_dividido['ndvi'].apply(salud)
    return gdf_dividido

def generar_clima_simulado():
    return {
        'precipitacion': {'total': 120.5, 'dias_con_lluvia': 12, 'diaria': [2.3]*60},
        'temperatura': {'promedio': 26.5, 'diaria': [26]*60},
        'radiacion': {'promedio': 18.2, 'diaria': [18]*60},
        'viento': {'promedio': 3.4, 'diaria': [3.4]*60},
        'periodo': 'Últimos 60 días (simulado)',
        'fuente': 'Simulado'
    }

# ========== VISUALIZACIÓN ==========
def crear_mapa_base(gdf, columna_color=None, colormap=None):
    centroide = gdf.geometry.unary_union.centroid
    m = folium.Map(location=[centroide.y, centroide.x], zoom_start=16, control_scale=True)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     attr='Esri', name='Satélite').add_to(m)
    if columna_color and colormap:
        def style_func(feature):
            val = feature['properties'].get(columna_color, 0)
            return {'fillColor': colormap(val), 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.7}
    else:
        style_func = lambda x: {'fillColor': '#3388ff', 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.4}
    folium.GeoJson(gdf.to_json(), name='Polígonos', style_function=style_func,
                   tooltip=folium.GeoJsonTooltip(fields=['id_bloque', 'ndvi', 'salud'],
                                                 aliases=['Bloque', 'NDVI', 'Salud'])).add_to(m)
    folium.LayerControl().add_to(m)
    return m

# ========== CONFIGURACIÓN PÁGINA ==========
st.set_page_config(page_title="AgroAI Platform", page_icon="🌴", layout="wide")
st.markdown("""
<style>
#MainMenu {display: none;}
footer {display: none;}
header {display: none;}
[data-testid="stToolbar"] {display: none;}
.app-title {font-size: 32px; font-weight: 700; color: white; padding: 10px 0;}
.app-subtitle {font-size: 14px; color: #aaa;}
</style>
<div class="app-title">🌴 AgroAI Platform</div>
<div class="app-subtitle">Satellite Intelligence for Agriculture</div>
<hr>
""", unsafe_allow_html=True)

# ========== INICIALIZACIÓN ==========
if 'user' not in st.session_state:
    st.session_state.user = None
if 'gdf_original' not in st.session_state:
    st.session_state.gdf_original = None
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False
if 'analisis_completado' not in st.session_state:
    st.session_state.analisis_completado = False
if 'resultados' not in st.session_state:
    st.session_state.resultados = None
if 'clima' not in st.session_state:
    st.session_state.clima = None

check_subscription()

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## 🌴 Configuración")
    uploaded_file = st.file_uploader("Subir plantación (GeoJSON, KML, ZIP)", type=['geojson','kml','kmz','zip'])
    n_divisiones = st.slider("Número de bloques", 8, 32, 16)

if uploaded_file and st.session_state.gdf_original is None:
    # Simulación: cargamos el ejemplo (en una versión real habría que parsear el archivo)
    st.session_state.gdf_original = cargar_ejemplo_demo()
    st.success("✅ Plantación cargada (ejemplo)")

if st.session_state.gdf_original is not None:
    gdf = st.session_state.gdf_original
    area_total = calcular_superficie(gdf)
    st.markdown(f"### 📊 Área total: {area_total:.1f} ha")
    if st.button("🚀 Ejecutar análisis"):
        with st.spinner("Analizando..."):
            gdf_resultado = generar_datos_simulados(gdf, n_divisiones)
            st.session_state.resultados = gdf_resultado
            st.session_state.clima = generar_clima_simulado()
            st.session_state.analisis_completado = True
        st.rerun()

# ========== RESULTADOS ==========
if st.session_state.analisis_completado and st.session_state.resultados is not None:
    gdf_res = st.session_state.resultados
    clima = st.session_state.clima

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen", "🗺️ Mapa", "📈 Índices", "🌤️ Clima"])

    with tab1:
        st.subheader("Resumen de la plantación")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Área total", f"{area_total:.1f} ha")
        col2.metric("Bloques", len(gdf_res))
        col3.metric("NDVI promedio", f"{gdf_res['ndvi'].mean():.3f}")
        col4.metric("Salud buena", f"{(gdf_res['salud']=='Buena').sum()} bloques")
        st.dataframe(gdf_res[['id_bloque','area_ha','ndvi','ndwi','edad','salud']])

    with tab2:
        st.subheader("Mapa interactivo")
        colormap = LinearColormap(colors=['red','yellow','green'], vmin=0.2, vmax=0.9)
        mapa = crear_mapa_base(gdf_res, columna_color='ndvi', colormap=colormap)
        folium_static(mapa, width=1000, height=600)

    with tab3:
        st.subheader("Índices de vegetación")
        fig = px.scatter(gdf_res, x='ndvi', y='ndwi', color='salud', size='area_ha',
                         hover_data=['id_bloque'], title='NDVI vs NDWI')
        st.plotly_chart(fig, use_container_width=True)

        csv = gdf_res[['id_bloque','ndvi','ndwi','salud']].to_csv(index=False)
        st.download_button("📥 Exportar CSV", csv, "indices.csv", "text/csv")

    with tab4:
        st.subheader("Datos climáticos (simulados)")
        st.json(clima)
else:
    st.info("👆 Sube un archivo de plantación y ejecuta el análisis para ver resultados.")

# ========== PIE DE PÁGINA ==========
st.markdown("---")
st.markdown("© 2026 AgroAI Platform - Contacto: mawucano@gmail.com")
