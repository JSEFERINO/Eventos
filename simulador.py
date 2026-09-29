import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="🎲 Simulador de Experimentos Aleatorios",
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 Simulador de Experimentos Aleatorios")
st.markdown("*Laboratorio interactivo de probabilidad y estadística*")
st.markdown("---")

# ============================================================
# SESSION STATE
# ============================================================
if "historial" not in st.session_state:
    st.session_state.historial = {}

# ============================================================
# UTILIDADES
# ============================================================

def simulador_monedas(n_monedas=1, n_lanzamientos=100, prob_cara=0.5):
    """Simula el lanzamiento de n monedas."""
    resultados = np.random.choice(["CARA", "SELLO"], size=(n_lanzamientos, n_monedas),
                                   p=[prob_cara, 1 - prob_cara])
    return resultados


def simulador_dados(n_dados=1, n_lanzamientos=100, caras=6):
    """Simula el lanzamiento de n dados."""
    return np.random.randint(1, caras + 1, size=(n_lanzamientos, n_dados))


def simulador_moneda_dado(n_lanzamientos=100):
    """Simula lanzar una moneda y un dado simultáneamente."""
    monedas = np.random.choice(["CARA", "SELLO"], size=n_lanzamientos)
    dados = np.random.randint(1, 7, size=n_lanzamientos)
    return monedas, dados


def simulador_cartas(n_cartas=1, n_lanzamientos=100):
    """Simula extracción de cartas (con reemplazo)."""
    palos = ["♠", "♥", "♦", "♣"]
    valores = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    baraja = [f"{v}{p}" for v in valores for p in palos]
    return np.random.choice(baraja, size=(n_lanzamientos, n_cartas))


def simulador_balotas(n_balotas=6, rango=(1, 43), n_lanzamientos=10):
    """Simula sorteo tipo baloto/lotería."""
    resultados = []
    for _ in range(n_lanzamientos):
        numeros = np.random.choice(range(rango[0], rango[1] + 1),
                                    size=n_balotas + 1, replace=False)
        resultados.append((sorted(numeros[:-1].tolist()), int(numeros[-1])))
    return resultados


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🪙 Monedas", "🎲 Dados", "🪙🎲 Moneda + Dado",
    "🃏 Cartas", "🎰 Baloto/Lotería", "📊 Análisis Comparativo"
])

# ============================================================
# TAB 1: MONEDAS
# ============================================================
with tab1:
    st.header("🪙 Simulación con Monedas")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Configuración")
        n_monedas = st.slider("Número de monedas por lanzamiento:", 1, 10, 1)
        n_lanzamientos = st.number_input("Número total de lanzamientos:",
                                          min_value=10, max_value=100000, value=500, step=50)
        prob_cara = st.slider("Probabilidad de CARA:", 0.0, 1.0, 0.5, 0.01)
        semilla = st.number_input("Semilla (0 = aleatorio):", 0, 99999, 42)
        if semilla > 0:
            np.random.seed(int(semilla))

        if st.button("🎲 Simular", key="btn_monedas"):
            st.session_state.historial["monedas"] = simulador_monedas(
                n_monedas, n_lanzamientos, prob_cara
            )

    with col2:
        if "monedas" in st.session_state.historial:
            res = st.session_state.historial["monedas"]

            # Resumen
            if n_monedas == 1:
                caras = np.sum(res == "CARA")
                sellos = np.sum(res == "SELLO")
                total = len(res)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total lanzamientos", total)
                col_b.metric("Caras", caras, f"{caras/total*100:.2f}%")
                col_c.metric("Sellos", sellos, f"{sellos/total*100:.2f}%")

                # Evolución de la frecuencia relativa
                caras_acum = np.cumsum(res.flatten() == "CARA") / np.arange(1, len(res) + 1)
                fig_evol = go.Figure()
                fig_evol.add_trace(go.Scatter(y=caras_acum, mode="lines",
                                              name="Frec. relativa CARA",
                                              line=dict(color="#3498db")))
                fig_evol.add_hline(y=prob_cara, line_dash="dash",
                                   line_color="red",
                                   annotation_text=f"P(CARA) = {prob_cara}")
                fig_evol.update_layout(title="Ley de los Grandes Números - Evolución",
                                       xaxis_title="Lanzamiento",
                                       yaxis_title="Frecuencia relativa",
                                       height=380)
                st.plotly_chart(fig_evol, use_container_width=True)

                # Barras finales
                df_res = pd.DataFrame({"Resultado": ["CARA", "SELLO"],
                                       "Frecuencia": [caras, sellos]})
                fig_bar = px.bar(df_res, x="Resultado", y="Frecuencia",
                                 color="Resultado", text_auto=True,
                                 color_discrete_map={"CARA": "#3498db", "SELLO": "#e74c3c"})
                fig_bar.update_layout(showlegend=False, height=350,
                                      title="Frecuencias finales")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                # Distribución del número de caras
                num_caras = np.sum(res == "CARA", axis=1)
                st.metric("Lanzamientos", len(res))
                st.metric("Monedas por lanzamiento", n_monedas)

                dist = Counter(num_caras)
                df_dist = pd.DataFrame({
                    "Num. Caras": sorted(dist.keys()),
                    "Frecuencia": [dist[k] for k in sorted(dist.keys())]
                })
                fig_dist = px.bar(df_dist, x="Num. Caras", y="Frecuencia",
                                  text_auto=True, color="Frecuencia",
                                  color_continuous_scale="Blues")
                fig_dist.update_layout(title=f"Distribución del número de caras ({n_monedas} monedas)",
                                       height=400, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)

                # Distribución teórica binomial superpuesta
                from scipy.stats import binom
                x_teorico = np.arange(0, n_monedas + 1)
                y_teorico = binom.pmf(x_teorico, n_monedas, prob_cara) * len(res)
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(x=df_dist["Num. Caras"],
                                          y=df_dist["Frecuencia"],
                                          name="Simulado",
                                          marker_color="#3498db"))
                fig_comp.add_trace(go.Scatter(x=x_teorico, y=y_teorico,
                                              mode="lines+markers",
                                              name="Teórico (Binomial)",
                                              line=dict(color="red", width=3)))
                fig_comp.update_layout(title="Simulado vs Teórico",
                                       xaxis_title="Número de caras",
                                       yaxis_title="Frecuencia",
                                       height=400)
                st.plotly_chart(fig_comp, use_container_width=True)

            st.dataframe(pd.DataFrame(res[:20],
                                       columns=[f"Moneda {i+1}" for i in range(n_monedas)]),
                          use_container_width=True)

# ============================================================
# TAB 2: DADOS
# ============================================================
with tab2:
    st.header("🎲 Simulación con Dados")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Configuración")
        n_dados = st.slider("Número de dados por lanzamiento:", 1, 10, 1, key="nd_dados")
        n_lanz_dados = st.number_input("Número de lanzamientos:",
                                        min_value=10, max_value=100000, value=500, step=50,
                                        key="nl_dados")
        caras_dado = st.selectbox("Número de caras del dado:", [4, 6, 8, 10, 12, 20], index=1)
        semilla_d = st.number_input("Semilla (0 = aleatorio):", 0, 99999, 42, key="sem_dados")
        if semilla_d > 0:
            np.random.seed(int(semilla_d))

        if st.button("🎲 Simular", key="btn_dados"):
            st.session_state.historial["dados"] = simulador_dados(
                n_dados, n_lanz_dados, caras_dado
            )

    with col2:
        if "dados" in st.session_state.historial:
            res = st.session_state.historial["dados"]

            if n_dados == 1:
                valores = res.flatten()
                frec = Counter(valores)
                df_frec = pd.DataFrame({
                    "Cara": list(range(1, caras_dado + 1)),
                    "Frecuencia": [frec.get(i, 0) for i in range(1, caras_dado + 1)]
                })
                df_frec["Prob. Simulada"] = df_frec["Frecuencia"] / len(valores)
                df_frec["Prob. Teórica"] = 1 / caras_dado

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total lanzamientos", len(valores))
                col_b.metric("Media", f"{valores.mean():.4f}",
                             f"Teórica: {(caras_dado+1)/2:.2f}")
                col_c.metric("Desv. Est.", f"{valores.std():.4f}",
                             f"Teórica: {np.sqrt((caras_dado**2-1)/12):.2f}")

                fig_bar = px.bar(df_frec, x="Cara", y="Frecuencia",
                                 text_auto=True, color="Frecuencia",
                                 color_continuous_scale="Viridis")
                fig_bar.add_hline(y=len(valores)/caras_dado, line_dash="dash",
                                  line_color="red",
                                  annotation_text="Frecuencia esperada")
                fig_bar.update_layout(title=f"Frecuencias del dado de {caras_dado} caras",
                                       height=400, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Evolución de la media
                medias_acum = np.cumsum(valores) / np.arange(1, len(valores) + 1)
                fig_media = go.Figure()
                fig_media.add_trace(go.Scatter(y=medias_acum, mode="lines",
                                                name="Media acumulada",
                                                line=dict(color="#9b59b6")))
                fig_media.add_hline(y=(caras_dado+1)/2, line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Media teórica = {(caras_dado+1)/2}")
                fig_media.update_layout(title="Convergencia de la media",
                                        xaxis_title="Lanzamiento",
                                        yaxis_title="Media acumulada",
                                        height=380)
                st.plotly_chart(fig_media, use_container_width=True)

                st.dataframe(df_frec, use_container_width=True)
            else:
                # Suma de n dados
                sumas = np.sum(res, axis=1)
                media_teo = n_dados * (caras_dado + 1) / 2
                var_teo = n_dados * (caras_dado**2 - 1) / 12

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total lanzamientos", len(sumas))
                col_b.metric("Media", f"{sumas.mean():.3f}",
                             f"Teórica: {media_teo:.2f}")
                col_c.metric("Desv. Est.", f"{sumas.std():.3f}",
                             f"Teórica: {np.sqrt(var_teo):.2f}")

                fig_hist = px.histogram(x=sumas, nbins=caras_dado*n_dados//2 + 1,
                                        title=f"Distribución de la suma de {n_dados} dados",
                                        color_discrete_sequence=["#e67e22"])
                fig_hist.update_layout(xaxis_title="Suma", yaxis_title="Frecuencia",
                                       height=400, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)

                # Teorema Central del Límite
                medias = np.mean(res, axis=1)
                fig_tcl = px.histogram(x=medias, nbins=40,
                                        title="Teorema Central del Límite - Medias",
                                        color_discrete_sequence=["#16a085"])
                fig_tcl.update_layout(xaxis_title="Media por lanzamiento",
                                       yaxis_title="Frecuencia",
                                       height=400, showlegend=False)
                st.plotly_chart(fig_tcl, use_container_width=True)

# ============================================================
# TAB 3: MONEDA + DADO
# ============================================================
with tab3:
    st.header("🪙🎲 Lanzamiento simultáneo: Moneda + Dado")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Configuración")
        n_lanz_md = st.number_input("Número de lanzamientos:",
                                     min_value=10, max_value=100000, value=500, step=50,
                                     key="nl_md")
        semilla_md = st.number_input("Semilla (0 = aleatorio):", 0, 99999, 42, key="sem_md")
        if semilla_md > 0:
            np.random.seed(int(semilla_md))

        if st.button("🎲 Simular", key="btn_md"):
            monedas, dados = simulador_moneda_dado(n_lanz_md)
            st.session_state.historial["moneda_dado"] = (monedas, dados)

    with col2:
        if "moneda_dado" in st.session_state.historial:
            monedas, dados = st.session_state.historial["moneda_dado"]

            # Tabla de contingencia
            df_md = pd.DataFrame({"Moneda": monedas, "Dado": dados})
            contingencia = pd.crosstab(df_md["Dado"], df_md["Moneda"])
            st.subheader("📋 Tabla de Contingencia (Dado × Moneda)")
            st.dataframe(contingencia, use_container_width=True)

            # Heatmap
            fig_heat = px.imshow(contingencia, text_auto=True, aspect="auto",
                                  color_continuous_scale="Blues",
                                  title="Frecuencias por combinación")
            fig_heat.update_layout(height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

            # Probabilidad conjunta
            st.subheader("📊 Probabilidades conjuntas")
            total = len(df_md)
            df_prob = contingencia / total
            st.dataframe(df_prob.style.format("{:.4f}"), use_container_width=True)

            # Test chi-cuadrado de independencia
            from scipy.stats import chi2_contingency
            chi2, p, dof, expected = chi2_contingency(contingencia)
            st.subheader("🔬 Test de Independencia (Chi-cuadrado)")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Chi²", f"{chi2:.4f}")
            col_b.metric("Grados de libertad", dof)
            col_c.metric("p-valor", f"{p:.4f}")
            if p > 0.05:
                st.success("✅ No se rechaza la independencia (p > 0.05). Los eventos son independientes.")
            else:
                st.warning("⚠️ Se rechaza la independencia (p ≤ 0.05).")

            # Gráfico de barras agrupadas
            df_group = df_md.groupby(["Moneda", "Dado"]).size().reset_index(name="Frecuencia")
            fig_group = px.bar(df_group, x="Dado", y="Frecuencia", color="Moneda",
                               barmode="group", text_auto=True,
                               color_discrete_map={"CARA": "#3498db", "SELLO": "#e74c3c"})
            fig_group.update_layout(title="Frecuencias por combinación",
                                     height=400)
            st.plotly_chart(fig_group, use_container_width=True)

# ============================================================
# TAB 4: CARTAS
# ============================================================
with tab4:
    st.header("🃏 Simulación con Cartas")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Configuración")
        n_cartas = st.slider("Cartas por extracción:", 1, 10, 5, key="nc_cartas")
        n_extr = st.number_input("Número de extracciones:",
                                  min_value=10, max_value=10000, value=200, step=10,
                                  key="ne_cartas")
        semilla_c = st.number_input("Semilla (0 = aleatorio):", 0, 99999, 42, key="sem_cartas")
        if semilla_c > 0:
            np.random.seed(int(semilla_c))

        if st.button("🎲 Simular", key="btn_cartas"):
            st.session_state.historial["cartas"] = simulador_cartas(n_cartas, n_extr)

    with col2:
        if "cartas" in st.session_state.historial:
            res = st.session_state.historial["cartas"]

            if n_cartas == 1:
                valores = res.flatten()
                palos = [c[-1] for c in valores]
                conteo_palos = Counter(palos)
                df_palos = pd.DataFrame({
                    "Palo": list(conteo_palos.keys()),
                    "Frecuencia": list(conteo_palos.values())
                })
                color_map = {"♠": "#2c3e50", "♥": "#e74c3c",
                             "♦": "#e67e22", "♣": "#27ae60"}
                fig_p = px.pie(df_palos, names="Palo", values="Frecuencia",
                               title="Distribución por palo",
                               color="Palo", color_discrete_map=color_map)
                st.plotly_chart(fig_p, use_container_width=True)

                # Valores
                valores_num = [c[:-1] for c in valores]
                conteo_val = Counter(valores_num)
                orden = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
                df_val = pd.DataFrame({
                    "Valor": orden,
                    "Frecuencia": [conteo_val.get(v, 0) for v in orden]
                })
                fig_v = px.bar(df_val, x="Valor", y="Frecuencia",
                               text_auto=True, color="Frecuencia",
                               color_continuous_scale="Viridis")
                fig_v.update_layout(title="Frecuencia por valor", height=380,
                                    showlegend=False)
                st.plotly_chart(fig_v, use_container_width=True)
            else:
                st.write("**Primeras 10 extracciones:**")
                st.dataframe(pd.DataFrame(res[:10],
                                           columns=[f"Carta {i+1}" for i in range(n_cartas)]),
                              use_container_width=True)

                # ¿Hay pares?
                pares = sum(len(set(row)) < len(row) for row in res)
                st.metric("Extracciones con al menos un par", f"{pares}/{len(res)}",
                          f"{pares/len(res)*100:.2f}%")

# ============================================================
# TAB 5: BALOTO / LOTERÍA
# ============================================================
with tab5:
    st.header("🎰 Simulación tipo Baloto / Lotería")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Configuración")
        n_balotas = st.slider("Balotas principales:", 4, 6, 5)
        rango_max = st.number_input("Número máximo:", 10, 100, 43)
        n_sorteos = st.number_input("Número de sorteos:", 1, 1000, 10)
        semilla_b = st.number_input("Semilla (0 = aleatorio):", 0, 99999, 42, key="sem_bal")
        if semilla_b > 0:
            np.random.seed(int(semilla_b))

        if st.button("🎲 Simular", key="btn_baloto"):
            st.session_state.historial["baloto"] = simulador_balotas(
                n_balotas, (1, rango_max), n_sorteos
            )

    with col2:
        if "baloto" in st.session_state.historial:
            res = st.session_state.historial["baloto"]
            st.subheader(f"🎯 {len(res)} sorteos simulados")

            filas = []
            for i, (nums, extra) in enumerate(res, 1):
                filas.append({"Sorteo": i, "Números": ", ".join(map(str, nums)),
                              "Balota Extra": extra})
            st.dataframe(pd.DataFrame(filas), use_container_width=True)

            # Frecuencia de cada número
            todos = [n for nums, _ in res for n in nums]
            frec = Counter(todos)
            df_frec = pd.DataFrame({
                "Número": list(range(1, rango_max + 1)),
                "Frecuencia": [frec.get(i, 0) for i in range(1, rango_max + 1)]
            })
            fig_b = px.bar(df_frec, x="Número", y="Frecuencia",
                           title="Frecuencia de aparición por número",
                           color="Frecuencia", color_continuous_scale="Turbo")
            fig_b.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_b, use_container_width=True)

            # Números calientes
            st.subheader("🔥 Números más frecuentes")
            top = df_frec.sort_values("Frecuencia", ascending=False).head(10)
            st.dataframe(top, use_container_width=True)

# ============================================================
# TAB 6: ANÁLISIS COMPARATIVO
# ============================================================
with tab6:
    st.header("📊 Análisis Comparativo - Ley de los Grandes Números")
    st.markdown("Compara cómo convergen las frecuencias relativas al aumentar el número de ensayos.")

    col1, col2 = st.columns(2)
    with col1:
        n_max = st.number_input("Número máximo de ensayos:", 100, 50000, 5000, 100)
    with col2:
        tipo = st.selectbox("Experimento:", ["Moneda (CARA)", "Dado (sacar 6)",
                                              "Dado (sacar par)"])

    if st.button("🚀 Ejecutar análisis"):
        if tipo == "Moneda (CARA)":
            resultados = np.random.choice([1, 0], size=n_max, p=[0.5, 0.5])
            prob_teo = 0.5
            etiqueta = "P(CARA) = 0.5"
        elif tipo == "Dado (sacar 6)":
            resultados = (np.random.randint(1, 7, n_max) == 6).astype(int)
            prob_teo = 1/6
            etiqueta = "P(6) = 1/6 ≈ 0.1667"
        else:
            resultados = (np.random.randint(1, 7, n_max) % 2 == 0).astype(int)
            prob_teo = 0.5
            etiqueta = "P(par) = 0.5"

        acumulada = np.cumsum(resultados) / np.arange(1, n_max + 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=acumulada, mode="lines",
                                  name="Frecuencia relativa",
                                  line=dict(color="#3498db", width=2)))
        fig.add_hline(y=prob_teo, line_dash="dash", line_color="red",
                      annotation_text=etiqueta)
        fig.update_layout(title=f"Convergencia - {tipo}",
                          xaxis_title="Número de ensayos",
                          yaxis_title="Frecuencia relativa",
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Valor final", f"{acumulada[-1]:.5f}")
        col_b.metric("Valor teórico", f"{prob_teo:.5f}")
        col_c.metric("Error absoluto", f"{abs(acumulada[-1] - prob_teo):.5f}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🎲 Simulador de Experimentos Aleatorios - v1.0<br>
    Desarrollado con Streamlit, Plotly y NumPy
</div>
""", unsafe_allow_html=True)
