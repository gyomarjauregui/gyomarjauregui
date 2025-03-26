# Código completo de la aplicación Streamlit con animaciones y rutas integradas

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Verificación de Prophet
try:
    from prophet import Prophet
    prophet_ok = True
except ImportError:
    prophet_ok = False

st.set_page_config(layout="wide")
st.title("✈️ Análisis Global del Tráfico Aéreo: Salidas, Pasajeros, Proyecciones y Rutas")

# Cargar y transformar datos
@st.cache_data
def load_data():
    df = pd.read_csv("air.csv")
    df.columns = df.columns.str.strip()

    def transformar(df, serie_nombre):
        df_serie = df[df['Series Name'] == serie_nombre].copy()
        df_serie = df_serie.melt(
            id_vars=["Country Name", "Country Code", "Series Name", "Series Code"],
            var_name="Year",
            value_name="Value"
        )
        df_serie['Year'] = df_serie['Year'].str.extract(r'(\d{4})')
        df_serie = df_serie.dropna(subset=['Year', 'Value'])
        df_serie['Year'] = pd.to_datetime(df_serie['Year'], format='%Y')
        df_serie['Value'] = pd.to_numeric(df_serie['Value'], errors='coerce').astype(float)
        df_serie = df_serie.dropna(subset=['Value'])
        return df_serie

    df_departures = transformar(df, 'Air transport, registered carrier departures worldwide')
    df_passengers = transformar(df, 'Air transport, passengers carried')

    return df_departures, df_passengers

df_dep, df_psg = load_data()

# Sidebar
st.sidebar.header("Filtros")
paises = sorted(df_dep['Country Name'].dropna().unique())
pais = st.sidebar.selectbox("🌍 Selecciona un país", paises)

min_year = int(df_dep['Year'].min().year)
max_year = int(df_dep['Year'].max().year)
year_range = st.sidebar.slider("Selecciona el rango de años", min_year, max_year, (min_year, max_year))

# ---- Módulo 1: SALIDAS DE VUELOS ----
st.subheader(f"📈 Salidas de vuelos - {pais}")
df_pais = df_dep[df_dep['Country Name'] == pais].copy()
df_pais = df_pais[df_pais['Year'].dt.year.between(*year_range)]
if df_pais.empty:
    st.warning("No hay datos de salidas de vuelos para este país en el rango seleccionado.")
else:
    df_pais = df_pais.sort_values('Year').reset_index(drop=True)
    df_pais['Year_Num'] = np.arange(len(df_pais))
    df_pais['Year'] = df_pais['Year'].dt.year

    fig = px.line(df_pais, x='Year', y='Value', title="Evolución de salidas de vuelos")
    fig.update_layout(yaxis_tickformat=',')
    st.plotly_chart(fig, use_container_width=True)

    modelo = LinearRegression()
    modelo.fit(df_pais[['Year_Num']], df_pais['Value'])
    años_futuros = 5
    pred_x = pd.DataFrame({'Year_Num': np.arange(len(df_pais), len(df_pais) + años_futuros)})
    pred_y = modelo.predict(pred_x)
    fechas_futuras = np.arange(df_pais['Year'].max() + 1, df_pais['Year'].max() + 1 + años_futuros)

    df_pred = pd.DataFrame({'Year': fechas_futuras, 'Value': pred_y, 'Tipo': 'Proyección'})
    df_real = df_pais[['Year', 'Value']].copy()
    df_real['Tipo'] = 'Real'
    df_comb = pd.concat([df_real, df_pred], ignore_index=True)
    fig2 = px.line(df_comb, x='Year', y='Value', color='Tipo', markers=True,
                   title=f"Proyección Lineal de salidas de vuelos en {pais}")
    fig2.update_layout(yaxis_tickformat=',')
    st.plotly_chart(fig2, use_container_width=True)

    valor_inicial = df_pais['Value'].iloc[0]
    valor_final = df_pais['Value'].iloc[-1]
    años = len(df_pais) - 1
    if valor_inicial > 0 and años > 0:
        cagr = (valor_final / valor_inicial) ** (1 / años) - 1
        st.metric(label="📊 CAGR (Salidas de vuelos)", value=f"{cagr*100:.2f}%")

    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(df_pais[['Year_Num']], df_pais['Value'])
    pred_y_rf = modelo_rf.predict(pred_x)
    df_pred_rf = pd.DataFrame({'Year': fechas_futuras, 'Value': pred_y_rf, 'Tipo': 'Proyección RF'})
    df_comb_rf = pd.concat([df_real, df_pred_rf], ignore_index=True)
    fig_rf = px.line(df_comb_rf, x='Year', y='Value', color='Tipo', markers=True,
                     title=f"Proyección RF de salidas de vuelos en {pais}")
    fig_rf.update_layout(yaxis_tickformat=',')
    st.plotly_chart(fig_rf, use_container_width=True)

    if prophet_ok:
        df_prophet = df_pais[['Year', 'Value']].rename(columns={"Year": "ds", "Value": "y"})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
        modelo_prophet = Prophet()
        modelo_prophet.fit(df_prophet)
        future = modelo_prophet.make_future_dataframe(periods=años_futuros, freq='Y')
        forecast = modelo_prophet.predict(future)
        fig5 = px.line(forecast, x='ds', y='yhat', title=f"Proyección con Prophet - Salidas de vuelos en {pais}")
        fig5.add_scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='markers', name='Datos reales')
        fig5.update_layout(yaxis_tickformat=',')
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("⚠️ El módulo 'prophet' no está instalado. Ejecuta `pip install prophet` para habilitar esta proyección avanzada.")

# ---- Módulo 2: PASAJEROS ----
st.subheader(f"🧍 Pasajeros transportados - {pais}")
df_psg_pais = df_psg[df_psg['Country Name'] == pais].copy()
df_psg_pais = df_psg_pais[df_psg_pais['Year'].dt.year.between(*year_range)]
if df_psg_pais.empty:
    st.warning("No hay datos de pasajeros para este país.")
else:
    df_psg_pais = df_psg_pais.sort_values('Year').reset_index(drop=True)
    df_psg_pais['Year_Num'] = np.arange(len(df_psg_pais))
    df_psg_pais['Year'] = df_psg_pais['Year'].dt.year

    fig3 = px.line(df_psg_pais, x='Year', y='Value', title="Evolución de pasajeros transportados")
    fig3.update_layout(yaxis_tickformat=',')
    st.plotly_chart(fig3, use_container_width=True)

    valor_inicial_psg = df_psg_pais['Value'].iloc[0]
    valor_final_psg = df_psg_pais['Value'].iloc[-1]
    años_psg = len(df_psg_pais) - 1
    if valor_inicial_psg > 0 and años_psg > 0:
        cagr_psg = (valor_final_psg / valor_inicial_psg) ** (1 / años_psg) - 1
        st.metric(label="📊 CAGR (Pasajeros)", value=f"{cagr_psg*100:.2f}%")

# ---- Módulo 3: MAPA INTERACTIVO ----
st.subheader("🗺️ Mapa mundial: pasajeros transportados (último año disponible)")
latest_year = df_psg['Year'].max()
map_data = df_psg[df_psg['Year'] == latest_year]
map_data['Year'] = map_data['Year'].dt.year
map_fig = px.choropleth(
    map_data,
    locations="Country Name",
    locationmode="country names",
    color="Value",
    hover_name="Country Name",
    title=f"Pasajeros transportados por país - {latest_year.year}",
    color_continuous_scale="Blues"
)
st.plotly_chart(map_fig, use_container_width=True)

# ---- Módulo 4: Comparación entre países ----
st.subheader("🌐 Comparación entre países: Salidas de vuelos")
paises_multi = st.multiselect("Selecciona países para comparar", paises, default=[pais])
df_comp = df_dep[df_dep['Country Name'].isin(paises_multi)].copy()
if not df_comp.empty:
    df_comp['Year'] = df_comp['Year'].dt.year
    fig_comp = px.line(df_comp, x='Year', y='Value', color='Country Name',
                       title="Comparación de salidas de vuelos entre países")
    fig_comp.update_layout(yaxis_tickformat=',')
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("Selecciona al menos un país con datos.")

# ---- Módulo 5: Animación temporal de pasajeros ----
st.subheader("🌀 Animación: Pasajeros transportados por país y año")
df_anim = df_psg.copy()
df_anim['Year'] = df_anim['Year'].dt.year
fig_anim = px.scatter_geo(
    df_anim,
    locations="Country Name",
    locationmode="country names",
    color="Value",
    size="Value",
    hover_name="Country Name",
    animation_frame="Year",
    title="✈️ Animación: Pasajeros transportados por país a lo largo del tiempo",
    projection="natural earth",
    color_continuous_scale="Plasma"
)
fig_anim.update_layout(height=600)
st.plotly_chart(fig_anim, use_container_width=True)

# ---- Módulo 6: Rutas aéreas simuladas ----
st.subheader("🛫 Simulación de rutas aéreas entre países (rutas ficticias)")
capitales = pd.DataFrame({
    'Country': ['United States', 'Brazil', 'Germany', 'India', 'Australia'],
    'City': ['Washington', 'Brasília', 'Berlin', 'New Delhi', 'Canberra'],
    'Lat': [38.89511, -15.793889, 52.516667, 28.613889, -35.282],
    'Lon': [-77.03637, -47.882778, 13.383333, 77.208889, 149.128684]
})
routes = []
for i in range(len(capitales)):
    for j in range(i + 1, len(capitales)):
        origin = capitales.iloc[i]
        dest = capitales.iloc[j]
        routes.append(dict(
            origin_country=origin['Country'],
            dest_country=dest['Country'],
            lon=[origin['Lon'], dest['Lon']],
            lat=[origin['Lat'], dest['Lat']]
        ))
fig_routes = go.Figure()
for route in routes:
    fig_routes.add_trace(go.Scattergeo(
        locationmode='country names',
        lon=route['lon'],
        lat=route['lat'],
        mode='lines',
        line=dict(width=2, color='blue'),
        opacity=0.5,
        hoverinfo='text',
        text=f"{route['origin_country']} → {route['dest_country']}"
    ))
fig_routes.update_layout(
    title_text="🌍 Rutas aéreas simuladas entre países (capitales)",
    showlegend=False,
    geo=dict(
        projection_type='natural earth',
        showland=True,
        landcolor='rgb(243, 243, 243)',
        countrycolor='rgb(204, 204, 204)',
    ),
    height=600
)
st.plotly_chart(fig_routes, use_container_width=True)