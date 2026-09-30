import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binom, chi2_contingency
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🎬 Simulador v5.5.0", page_icon="🎬", layout="wide")

st.title("🎬 Simulador de Experimentos Aleatorios con Animaciones")
st.markdown("*Con distribuciones teóricas y operaciones con eventos*")
st.markdown("---")

if "historial" not in st.session_state:
    st.session_state.historial = {}

# ============================================================
# UTILIDADES PÓKER
# ============================================================
VALORES = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
           "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
PALOS = {"♠": "picas", "♥": "corazones", "♦": "diamantes", "♣": "tréboles"}

def crear_baraja():
    return [f"{v}{p}" for v in VALORES for p in PALOS]

def evaluar_mano(mano):
    valores = sorted([VALORES[c[:-1]] for c in mano], reverse=True)
    palos = [c[-1] for c in mano]
    cuenta = Counter(valores)
    frec = sorted(cuenta.values(), reverse=True)
    es_color = len(set(palos)) == 1
    es_escalera = False
    if valores == [14, 5, 4, 3, 2]:
        es_escalera = True
    elif len(set(valores)) == 5 and valores[0] - valores[-1] == 4:
        es_escalera = True
    if es_color and es_escalera and valores[0] == 14: return "Escalera Real"
    if es_color and es_escalera: return "Escalera de Color"
    if frec == [4, 1]: return "Póker"
    if frec == [3, 2]: return "Full House"
    if es_color: return "Color"
    if es_escalera: return "Escalera"
    if frec == [3, 1, 1]: return "Trío"
    if frec == [2, 2, 1]: return "Doble Pareja"
    if frec == [2, 1, 1, 1]: return "Pareja"
    return "Carta Alta"

PROB_TEORICAS_POKER = {
    "Escalera Real": 1/649740, "Escalera de Color": 10/649740,
    "Póker": 624/2598960, "Full House": 3744/2598960,
    "Color": 5108/2598960, "Escalera": 10200/2598960,
    "Trío": 54912/2598960, "Doble Pareja": 123552/2598960,
    "Pareja": 1098240/2598960, "Carta Alta": 1302540/2598960,
}
ORDEN_MANOS = ["Escalera Real", "Escalera de Color", "Póker", "Full House",
                "Color", "Escalera", "Trío", "Doble Pareja", "Pareja", "Carta Alta"]
COLORES_MANOS = {
    "Escalera Real": "#8e44ad", "Escalera de Color": "#9b59b6",
    "Póker": "#c0392b", "Full House": "#e74c3c",
    "Color": "#e67e22", "Escalera": "#f39c12",
    "Trío": "#27ae60", "Doble Pareja": "#16a085",
    "Pareja": "#3498db", "Carta Alta": "#95a5a6",
}

# ============================================================
# UTILIDAD: gráfico de barras doble
# ============================================================
def grafico_barras_doble(categorias, frecuencias, titulo="Frecuencias",
                          color_abs="#3498db", color_rel="#e67e22",
                          colores_custom=None, etiqueta_x="Categoría"):
    total = sum(frecuencias)
    frec_rel = [f/total*100 if total > 0 else 0 for f in frecuencias]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("📊 Frecuencias absolutas (histograma)",
                        "📈 Frecuencias relativas (%)"))
    if colores_custom:
        fig.add_trace(go.Bar(x=categorias, y=frecuencias, marker_color=colores_custom,
            text=frecuencias, textposition="outside", showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Bar(x=categorias, y=frec_rel, marker_color=colores_custom,
            text=[f"{v:.2f}%" for v in frec_rel], textposition="outside",
            showlegend=False, hoverinfo="skip"), row=1, col=2)
    else:
        fig.add_trace(go.Bar(x=categorias, y=frecuencias, marker_color=color_abs,
            text=frecuencias, textposition="outside", showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Bar(x=categorias, y=frec_rel, marker_color=color_rel,
            text=[f"{v:.2f}%" for v in frec_rel], textposition="outside",
            showlegend=False, hoverinfo="skip"), row=1, col=2)
    fig.update_xaxes(title_text=etiqueta_x, row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
    fig.update_xaxes(title_text=etiqueta_x, row=1, col=2)
    fig.update_yaxes(title_text="Porcentaje (%)", row=1, col=2)
    fig.update_layout(title_text=titulo, height=450, showlegend=False,
                      margin=dict(t=60, b=40))
    return fig


# ============================================================
# UTILIDAD: comparación observada vs teórica
# ============================================================
def grafico_comparacion_teorica(categorias, frec_obs, prob_teor, n_total,
                                  titulo="Comparación observada vs teórica"):
    """Compara frecuencias observadas con las esperadas por teoría."""
    frec_esp = [p * n_total for p in prob_teor]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("📊 Frec. observadas vs esperadas",
                        "📈 Probabilidades observadas vs teóricas (%)"))

    fig.add_trace(go.Bar(x=categorias, y=frec_obs,
        marker_color="#3498db", name="Observado",
        text=frec_obs, textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=categorias, y=frec_esp,
        marker_color="#e67e22", name="Esperado",
        text=[f"{v:.1f}" for v in frec_esp], textposition="outside"), row=1, col=1)

    total_obs = sum(frec_obs) if sum(frec_obs) > 0 else 1
    prob_obs = [f / total_obs * 100 for f in frec_obs]
    prob_teo_pct = [p * 100 for p in prob_teor]

    fig.add_trace(go.Bar(x=categorias, y=prob_obs,
        marker_color="#3498db", name="Observado %",
        text=[f"{v:.2f}%" for v in prob_obs], textposition="outside",
        showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=categorias, y=prob_teo_pct,
        marker_color="#e67e22", name="Teórico %",
        text=[f"{v:.2f}%" for v in prob_teo_pct], textposition="outside",
        showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text="Categoría", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
    fig.update_xaxes(title_text="Categoría", row=1, col=2)
    fig.update_yaxes(title_text="Porcentaje (%)", row=1, col=2)
    fig.update_layout(title_text=titulo, height=480, barmode="group",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="center", x=0.5))
    return fig


# ============================================================
# ANIMACIONES
# ============================================================
def anim_moneda(n=30, prob_cara=0.5, n_monedas=1):
    resultados = np.random.choice(["CARA", "SELLO"], size=(n, n_monedas),
                                   p=[prob_cara, 1 - prob_cara])
    frames = []
    for i in range(n):
        caras = int(np.sum(resultados[:i+1] == "CARA"))
        sellos = int(np.sum(resultados[:i+1] == "SELLO"))
        total = caras + sellos
        pct_caras = caras/total*100 if total > 0 else 0
        pct_sellos = sellos/total*100 if total > 0 else 0
        texto_actual = " ".join(resultados[i])
        n_caras_actual = int(np.sum(resultados[i] == "CARA"))
        color = "#3498db" if n_caras_actual == n_monedas else ("#e74c3c" if n_caras_actual == 0 else "#f39c12")
        trace_actual = go.Scatter(x=[0.5], y=[0.5], mode="markers+text",
            marker=dict(size=100, color=color, line=dict(color="black", width=2)),
            text=[texto_actual], textposition="middle center",
            textfont=dict(color="white", size=min(20, 40 // n_monedas) if n_monedas > 1 else 16,
                          family="Arial Black"), showlegend=False, hoverinfo="skip")
        trace_abs = go.Bar(x=["CARA", "SELLO"], y=[caras, sellos],
            marker_color=["#3498db", "#e74c3c"], text=[caras, sellos],
            textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2")
        trace_rel = go.Bar(x=["CARA", "SELLO"], y=[pct_caras, pct_sellos],
            marker_color=["#3498db", "#e74c3c"],
            text=[f"{pct_caras:.2f}%", f"{pct_sellos:.2f}%"],
            textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3")
        if n_monedas > 1:
            dist_caras = [int(np.sum(np.sum(resultados[:i+1] == "CARA", axis=1) == k))
                          for k in range(n_monedas+1)]
            total_l = i + 1
            dist_pct = [d/total_l*100 for d in dist_caras]
            trace_d_abs = go.Bar(x=list(range(n_monedas+1)), y=dist_caras,
                marker_color="#9b59b6", text=dist_caras, textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x4", yaxis="y4")
            trace_d_rel = go.Bar(x=list(range(n_monedas+1)), y=dist_pct,
                marker_color="#8e44ad", text=[f"{v:.1f}%" for v in dist_pct],
                textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x5", yaxis="y5")
            frames.append(go.Frame(data=[trace_actual, trace_abs, trace_rel, trace_d_abs, trace_d_rel],
                name=f"f{i}", layout=go.Layout(
                    title_text=f"🪙 Lanzamiento {i+1}/{n}: {texto_actual} | "
                               f"{caras} CARA ({pct_caras:.1f}%), {sellos} SELLO ({pct_sellos:.1f}%)")))
        else:
            frames.append(go.Frame(data=[trace_actual, trace_abs, trace_rel], name=f"f{i}",
                layout=go.Layout(title_text=f"🪙 Lanzamiento {i+1}/{n}: {texto_actual} | "
                                            f"CARA: {pct_caras:.1f}%, SELLO: {pct_sellos:.1f}%")))
    if n_monedas > 1:
        fig = make_subplots(rows=2, cols=3, column_widths=[0.33, 0.33, 0.34], row_heights=[0.5, 0.5],
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}, None]],
            subplot_titles=("Lanzamiento actual", "Frec. absolutas", "Frec. relativas (%)",
                            f"Distribución de caras ({n_monedas} monedas)", "Distribución (%)"))
        for idx, (r, c) in enumerate([(1,1),(1,2),(1,3),(2,1),(2,2)]):
            fig.add_trace(frames[0].data[idx], row=r, col=c)
        fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
        fig.update_xaxes(title_text="Resultado", row=1, col=2); fig.update_yaxes(title_text="Frec.", row=1, col=2)
        fig.update_xaxes(title_text="Resultado", row=1, col=3); fig.update_yaxes(title_text="%", row=1, col=3)
        fig.update_xaxes(title_text="Núm. caras", row=2, col=1); fig.update_yaxes(title_text="Frec.", row=2, col=1)
        fig.update_xaxes(title_text="Núm. caras", row=2, col=2); fig.update_yaxes(title_text="%", row=2, col=2)
        height = 700
    else:
        fig = make_subplots(rows=1, cols=3, column_widths=[0.25, 0.375, 0.375],
            subplot_titles=("Lanzamiento actual", "Frec. absolutas", "Frec. relativas (%)"))
        for idx, c in enumerate([1, 2, 3]):
            fig.add_trace(frames[0].data[idx], row=1, col=c)
        fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
        fig.update_xaxes(title_text="Resultado", row=1, col=2); fig.update_yaxes(title_text="Frec.", row=1, col=2)
        fig.update_xaxes(title_text="Resultado", row=1, col=3); fig.update_yaxes(title_text="%", range=[0, 100], row=1, col=3)
        height = 450
    fig.frames = frames
    fig.update_layout(title_text=f"🪙 Animación: {n_monedas} moneda(s)",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.13,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 400, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=height, showlegend=False)
    return fig


def anim_dado(n=30, caras=6, n_dados=1):
    resultados = np.random.randint(1, caras + 1, size=(n, n_dados))
    frames = []
    for i in range(n):
        todos = resultados[:i+1].flatten()
        frec = [int(np.sum(todos == k)) for k in range(1, caras + 1)]
        total = len(todos)
        frec_pct = [f/total*100 for f in frec]
        texto_actual = " ".join(map(str, resultados[i]))
        suma_actual = int(np.sum(resultados[i]))
        trace_actual = go.Scatter(x=[0.5], y=[0.5], mode="markers+text",
            marker=dict(size=100, color="#e67e22", line=dict(color="black", width=2), symbol="square"),
            text=[texto_actual], textposition="middle center",
            textfont=dict(color="white", size=min(28, 40 // n_dados) if n_dados > 1 else 28,
                          family="Arial Black"), showlegend=False, hoverinfo="skip")
        trace_abs = go.Bar(x=list(range(1, caras + 1)), y=frec,
            marker_color="skyblue", marker_line_color="black", marker_line_width=1.5,
            text=frec, textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2")
        trace_rel = go.Bar(x=list(range(1, caras + 1)), y=frec_pct,
            marker_color="#e67e22", text=[f"{v:.2f}%" for v in frec_pct],
            textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3")
        if n_dados > 1:
            sumas = np.sum(resultados[:i+1], axis=1)
            rango = list(range(n_dados, n_dados * caras + 1))
            dist_s = [int(np.sum(sumas == s)) for s in rango]
            dist_pct = [d/total*100 for d in dist_s]
            trace_d_abs = go.Bar(x=rango, y=dist_s, marker_color="#9b59b6",
                text=dist_s, textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x4", yaxis="y4")
            trace_d_rel = go.Bar(x=rango, y=dist_pct, marker_color="#8e44ad",
                text=[f"{v:.1f}%" for v in dist_pct], textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x5", yaxis="y5")
            frames.append(go.Frame(data=[trace_actual, trace_abs, trace_rel, trace_d_abs, trace_d_rel],
                name=f"f{i}", layout=go.Layout(title_text=f"🎲 {i+1}/{n}: {texto_actual} (suma={suma_actual})")))
        else:
            frames.append(go.Frame(data=[trace_actual, trace_abs, trace_rel], name=f"f{i}",
                layout=go.Layout(title_text=f"🎲 {i+1}/{n}: {texto_actual}")))
    if n_dados > 1:
        fig = make_subplots(rows=2, cols=3, column_widths=[0.33, 0.33, 0.34], row_heights=[0.5, 0.5],
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}, None]],
            subplot_titles=("Lanzamiento actual", "Frec. absolutas", "Frec. relativas (%)",
                            "Distribución de la suma", "Distribución (%)"))
        for idx, (r, c) in enumerate([(1,1),(1,2),(1,3),(2,1),(2,2)]):
            fig.add_trace(frames[0].data[idx], row=r, col=c)
        fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
        fig.update_xaxes(title_text="Cara", row=1, col=2); fig.update_yaxes(title_text="Frec.", row=1, col=2)
        fig.update_xaxes(title_text="Cara", row=1, col=3); fig.update_yaxes(title_text="%", row=1, col=3)
        fig.update_xaxes(title_text="Suma", row=2, col=1); fig.update_yaxes(title_text="Frec.", row=2, col=1)
        fig.update_xaxes(title_text="Suma", row=2, col=2); fig.update_yaxes(title_text="%", row=2, col=2)
        height = 700
    else:
        fig = make_subplots(rows=1, cols=3, column_widths=[0.25, 0.375, 0.375],
            subplot_titles=("Lanzamiento actual", "Frec. absolutas", "Frec. relativas (%)"))
        for idx, c in enumerate([1, 2, 3]):
            fig.add_trace(frames[0].data[idx], row=1, col=c)
        fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
        fig.update_xaxes(title_text="Cara", row=1, col=2); fig.update_yaxes(title_text="Frec.", row=1, col=2)
        fig.update_xaxes(title_text="Cara", row=1, col=3); fig.update_yaxes(title_text="%", row=1, col=3)
        height = 450
    fig.frames = frames
    fig.update_layout(title_text=f"🎲 Animación: {n_dados} dado(s) de {caras} caras",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.13,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 400, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=height, showlegend=False)
    return fig


def anim_moneda_dado(n=20):
    monedas = np.random.choice(["CARA", "SELLO"], size=n)
    dados = np.random.randint(1, 7, size=n)
    frames = []
    for i in range(n):
        color = "#3498db" if monedas[i] == "CARA" else "#e74c3c"
        caras = int(np.sum(monedas[:i+1] == "CARA"))
        sellos = int(np.sum(monedas[:i+1] == "SELLO"))
        total = i + 1
        dados_frec = [int(np.sum(dados[:i+1] == k)) for k in range(1, 7)]
        dados_pct = [f/total*100 for f in dados_frec]
        frames.append(go.Frame(data=[
            go.Scatter(x=[0.5], y=[0.5], mode="markers+text",
                marker=dict(size=100, color=color, line=dict(color="black", width=2)),
                text=[monedas[i]], textposition="middle center",
                textfont=dict(color="white", size=14, family="Arial Black"),
                showlegend=False, hoverinfo="skip"),
            go.Scatter(x=[0.5], y=[0.5], mode="markers+text",
                marker=dict(size=100, color="#e67e22", line=dict(color="black", width=2), symbol="square"),
                text=[str(dados[i])], textposition="middle center",
                textfont=dict(color="white", size=28, family="Arial Black"),
                showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2"),
            go.Bar(x=["CARA", "SELLO"], y=[caras, sellos],
                marker_color=["#3498db", "#e74c3c"],
                text=[f"{caras/total*100:.1f}%", f"{sellos/total*100:.1f}%"],
                textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3"),
            go.Bar(x=list(range(1, 7)), y=dados_frec, marker_color="#e67e22",
                text=[f"{v:.1f}%" for v in dados_pct], textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x4", yaxis="y4")
        ], name=f"f{i}", layout=go.Layout(title_text=f"🪙🎲 {i+1}: {monedas[i]} + {dados[i]}")))
    fig = make_subplots(rows=2, cols=2, column_widths=[0.5, 0.5], row_heights=[0.5, 0.5],
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Moneda", "Dado", "Frec. relativas moneda (%)", "Frec. relativas dado (%)"))
    for idx, (r, c) in enumerate([(1,1),(1,2),(2,1),(2,2)]):
        fig.add_trace(frames[0].data[idx], row=r, col=c)
    fig.frames = frames
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=2); fig.update_yaxes(visible=False, row=1, col=2)
    fig.update_xaxes(title_text="Resultado", row=2, col=1); fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_xaxes(title_text="Cara", row=2, col=2); fig.update_yaxes(title_text="%", row=2, col=2)
    fig.update_layout(title_text=f"🪙🎲 Animación {n} lanzamientos",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.13,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 400, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=650, showlegend=False)
    return fig


def anim_cartas(n=15):
    baraja = crear_baraja()
    extracciones = np.random.choice(baraja, size=n)
    frames = []
    for i in range(n):
        hist = extracciones[:i+1]
        n_mostrar = min(len(hist), 8)
        cartas_vis = hist[-n_mostrar:]
        xs = list(range(n_mostrar))
        colores = ["#e74c3c" if c[-1] in ["♥", "♦"] else "#2c3e50" for c in cartas_vis]
        palos_hist = [c[-1] for c in hist]
        cp = Counter(palos_hist)
        palos_lista = ["♠", "♥", "♦", "♣"]
        frec_palos = [cp.get(p, 0) for p in palos_lista]
        frec_pct = [f/len(hist)*100 for f in frec_palos]
        colores_palos = ["#2c3e50", "#e74c3c", "#e67e22", "#27ae60"]
        frames.append(go.Frame(data=[
            go.Scatter(x=xs, y=[0.5]*n_mostrar, mode="markers+text",
                marker=dict(size=55, color="white", line=dict(color=colores, width=3), symbol="square"),
                text=cartas_vis, textposition="middle center",
                textfont=dict(color=colores, size=13, family="Arial Black"),
                showlegend=False, hoverinfo="skip"),
            go.Bar(x=palos_lista, y=frec_palos, marker_color=colores_palos,
                text=frec_palos, textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2"),
            go.Bar(x=palos_lista, y=frec_pct, marker_color=colores_palos,
                text=[f"{v:.1f}%" for v in frec_pct], textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3")
        ], name=f"f{i}", layout=go.Layout(title_text=f"🃏 Extracción {i+1}/{n}: {extracciones[i]}")))
    fig = make_subplots(rows=1, cols=3, column_widths=[0.5, 0.25, 0.25],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Cartas actuales", "Frec. absolutas por palo", "Frec. relativas (%)"))
    for idx, c in enumerate([1, 2, 3]):
        fig.add_trace(frames[0].data[idx], row=1, col=c)
    fig.frames = frames
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Palo", row=1, col=2); fig.update_yaxes(title_text="Frec.", row=1, col=2)
    fig.update_xaxes(title_text="Palo", row=1, col=3); fig.update_yaxes(title_text="%", row=1, col=3)
    fig.update_layout(title_text=f"🃏 Cartas - {n} extracciones",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.13,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=500, showlegend=False)
    return fig


def anim_baloto(n_sorteos=8, n_balotas=5, rango_max=43):
    sorteos = []
    for _ in range(n_sorteos):
        nums = sorted(np.random.choice(range(1, rango_max + 1), size=n_balotas, replace=False).tolist())
        extra = int(np.random.choice(range(1, rango_max + 1)))
        sorteos.append((nums, extra))
    frames = []
    for i, (nums, extra) in enumerate(sorteos):
        xs = list(range(len(nums) + 1))
        textos = [str(x) for x in nums] + [str(extra)]
        colores = ["#3498db"] * len(nums) + ["#e74c3c"]
        todos = [n for ns, _ in sorteos[:i+1] for n in ns]
        frec = Counter(todos)
        top_nums = [n for n, _ in frec.most_common(15)]
        frec_top = [frec[n] for n in top_nums]
        total_ap = sum(frec_top)
        frec_pct = [f/total_ap*100 if total_ap > 0 else 0 for f in frec_top]
        frames.append(go.Frame(data=[
            go.Scatter(x=xs, y=[0.5]*(len(nums)+1), mode="markers+text",
                marker=dict(size=55, color=colores, line=dict(color="black", width=2)),
                text=textos, textposition="middle center",
                textfont=dict(color="white", size=13, family="Arial Black"),
                showlegend=False, hoverinfo="skip"),
            go.Bar(x=top_nums, y=frec_top, marker_color="#3498db",
                text=frec_top, textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2"),
            go.Bar(x=top_nums, y=frec_pct, marker_color="#e67e22",
                text=[f"{v:.1f}%" for v in frec_pct], textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3")
        ], name=f"f{i}", layout=go.Layout(title_text=f"🎰 Sorteo {i+1}/{n_sorteos}")))
    fig = make_subplots(rows=1, cols=3, column_widths=[0.5, 0.25, 0.25],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Sorteo actual", "Frec. absolutas (top 15)", "Frec. relativas (%)"))
    for idx, c in enumerate([1, 2, 3]):
        fig.add_trace(frames[0].data[idx], row=1, col=c)
    fig.frames = frames
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Número", row=1, col=2); fig.update_yaxes(title_text="Frec.", row=1, col=2)
    fig.update_xaxes(title_text="Número", row=1, col=3); fig.update_yaxes(title_text="%", row=1, col=3)
    fig.update_layout(title_text=f"🎰 Baloto - {n_sorteos} sorteos",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.13,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 700, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=500, showlegend=False)
    return fig


def anim_ruleta(n_giros=15, tipo="europea"):
    if tipo == "europea":
        casillas = list(range(0, 37))
    else:
        casillas = list(range(0, 37)) + ["00"]
    rojos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
    tiradas = np.random.choice(casillas, size=n_giros)
    def color_num(t):
        if t == 0 or t == "00": return "#27ae60"
        return "#e74c3c" if t in rojos else "#2c3e50"
    saldos = [0]
    for t in tiradas:
        saldos.append(saldos[-1] + (1 if t in rojos else -1))
    frames = []
    for i in range(n_giros):
        hist = tiradas[:i+1]
        n_r = sum(1 for x in hist if x in rojos)
        n_n = sum(1 for x in hist if x not in rojos and x != 0 and x != "00")
        n_v = sum(1 for x in hist if x == 0 or x == "00")
        total = len(hist)
        pct = [n_r/total*100, n_n/total*100, n_v/total*100]
        frames.append(go.Frame(data=[
            go.Scatter(x=[0.5], y=[0.5], mode="markers+text",
                marker=dict(size=100, color=color_num(tiradas[i]), line=dict(color="black", width=3)),
                text=[str(tiradas[i])], textposition="middle center",
                textfont=dict(color="white", size=20, family="Arial Black"),
                showlegend=False, hoverinfo="skip"),
            go.Scatter(x=list(range(i+2)), y=saldos[:i+2],
                mode="lines+markers", line=dict(color="#e67e22", width=3),
                marker=dict(size=8), showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2"),
            go.Bar(x=["Rojo", "Negro", "Verde"], y=[n_r, n_n, n_v],
                marker_color=["#e74c3c", "#2c3e50", "#27ae60"],
                text=[n_r, n_n, n_v], textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3"),
            go.Bar(x=["Rojo", "Negro", "Verde"], y=pct,
                marker_color=["#e74c3c", "#2c3e50", "#27ae60"],
                text=[f"{v:.1f}%" for v in pct], textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x4", yaxis="y4")
        ], name=f"f{i}", layout=go.Layout(title_text=f"🎡 Giro {i+1}/{n_giros} → {tiradas[i]}")))
    fig = make_subplots(rows=2, cols=2, column_widths=[0.3, 0.7], row_heights=[0.5, 0.5],
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Giro actual", "Saldo acumulado",
                        "Frec. absolutas por color", "Frec. relativas (%)"))
    for idx, (r, c) in enumerate([(1,1),(1,2),(2,1),(2,2)]):
        fig.add_trace(frames[0].data[idx], row=r, col=c)
    fig.frames = frames
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Giro", row=1, col=2); fig.update_yaxes(title_text="Saldo", row=1, col=2)
    fig.update_xaxes(title_text="Color", row=2, col=1); fig.update_yaxes(title_text="Frec.", row=2, col=1)
    fig.update_xaxes(title_text="Color", row=2, col=2); fig.update_yaxes(title_text="%", row=2, col=2)
    fig.update_layout(title_text=f"🎡 Ruleta - {n_giros} giros",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.13,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=650, showlegend=False)
    return fig


def anim_powerball(n_sorteos=6):
    sorteos = []
    for _ in range(n_sorteos):
        nums = sorted(np.random.choice(range(1, 70), size=5, replace=False).tolist())
        pb = int(np.random.randint(1, 27))
        sorteos.append((nums, pb))
    frames = []
    for i, (nums, pb) in enumerate(sorteos):
        xs = list(range(6))
        textos = [str(x) for x in nums] + [str(pb)]
        colores = ["#3498db"] * 5 + ["#e74c3c"]
        todos = [n for ns, _ in sorteos[:i+1] for n in ns]
        frec = Counter(todos)
        top_nums = [n for n, _ in frec.most_common(15)]
        frec_top = [frec[n] for n in top_nums]
        total_ap = sum(frec_top)
        frec_top_pct = [f/total_ap*100 if total_ap > 0 else 0 for f in frec_top]
        frames.append(go.Frame(data=[
            go.Scatter(x=xs, y=[0.5]*6, mode="markers+text",
                marker=dict(size=55, color=colores, line=dict(color="black", width=2)),
                text=textos, textposition="middle center",
                textfont=dict(color="white", size=12, family="Arial Black"),
                showlegend=False, hoverinfo="skip"),
            go.Bar(x=top_nums, y=frec_top, marker_color="#3498db",
                text=frec_top, textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2"),
            go.Bar(x=top_nums, y=frec_top_pct, marker_color="#e67e22",
                text=[f"{v:.1f}%" for v in frec_top_pct],
                textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3")
        ], name=f"f{i}", layout=go.Layout(title_text=f"💥 Sorteo {i+1}/{n_sorteos}")))
    fig = make_subplots(rows=1, cols=3, column_widths=[0.5, 0.25, 0.25],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Sorteo actual", "Frec. absolutas (top 15)", "Frec. relativas (%)"))
    for idx, c in enumerate([1, 2, 3]):
        fig.add_trace(frames[0].data[idx], row=1, col=c)
    fig.frames = frames
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Número", row=1, col=2); fig.update_yaxes(title_text="Frec.", row=1, col=2)
    fig.update_xaxes(title_text="Número", row=1, col=3); fig.update_yaxes(title_text="%", row=1, col=3)
    fig.update_layout(title_text=f"💥 Powerball - {n_sorteos} sorteos",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.13,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 700, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=500, showlegend=False)
    return fig


def anim_monte_carlo(n_puntos=1500, n_frames=60):
    x = np.random.uniform(-1, 1, n_puntos)
    y = np.random.uniform(-1, 1, n_puntos)
    dentro = x**2 + y**2 <= 1
    theta = np.linspace(0, 2*np.pi, 200)
    frames = []
    for f in range(n_frames):
        k = int((f + 1) * n_puntos / n_frames)
        xa, ya, da = x[:k], y[:k], dentro[:k]
        pi_est = 4 * np.sum(da) / k
        n_in = int(np.sum(da))
        n_out = k - n_in
        pct_in = n_in/k*100
        pct_out = n_out/k*100
        frames.append(go.Frame(data=[
            go.Scatter(x=xa[da], y=ya[da], mode="markers",
                marker=dict(color="#27ae60", size=5, opacity=0.6), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=xa[~da], y=ya[~da], mode="markers",
                marker=dict(color="#e74c3c", size=5, opacity=0.6), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines",
                line=dict(color="blue", width=3), showlegend=False, hoverinfo="skip"),
            go.Bar(x=["Dentro", "Fuera"], y=[n_in, n_out],
                marker_color=["#27ae60", "#e74c3c"], text=[n_in, n_out],
                textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2"),
            go.Bar(x=["Dentro", "Fuera"], y=[pct_in, pct_out],
                marker_color=["#27ae60", "#e74c3c"],
                text=[f"{pct_in:.1f}%", f"{pct_out:.1f}%"],
                textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3")
        ], name=f"f{f}", layout=go.Layout(title_text=f"🥧 {k} puntos | π ≈ {pi_est:.4f}")))
    fig = make_subplots(rows=1, cols=3, column_widths=[0.5, 0.25, 0.25],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Dispersión", "Frec. absolutas", "Frec. relativas (%)"))
    fig.add_trace(frames[0].data[0], row=1, col=1)
    fig.add_trace(frames[0].data[1], row=1, col=1)
    fig.add_trace(frames[0].data[2], row=1, col=1)
    fig.add_trace(frames[0].data[3], row=1, col=2)
    fig.add_trace(frames[0].data[4], row=1, col=3)
    fig.frames = frames
    fig.update_xaxes(range=[-1.1, 1.1], scaleanchor="y", scaleratio=1,
                     showgrid=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[-1.1, 1.1], showgrid=False, visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Zona", row=1, col=2); fig.update_yaxes(title_text="Frec.", row=1, col=2)
    fig.update_xaxes(title_text="Zona", row=1, col=3); fig.update_yaxes(title_text="%", row=1, col=3)
    fig.update_layout(title_text="🥧 Monte Carlo π",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.13,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 80, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=500, showlegend=False)
    return fig


def anim_cumpleanos(n_personas=23, n_simulaciones=100):
    dias = np.random.randint(1, 366, size=(n_simulaciones, n_personas))
    coincidencias = [len(set(row)) < len(row) for row in dias]
    coincidencias_acum = np.cumsum(coincidencias)
    frames = []
    for i in range(n_simulaciones):
        row = dias[i]
        vistos = set()
        colores = []
        meses = []
        for d in row:
            mes = (d - 1) // 30 + 1
            meses.append(mes)
            colores.append("#27ae60" if d in vistos else "#3498db")
            vistos.add(d)
        xs = [j % 12 for j in range(n_personas)]
        ys = [-(j // 12) for j in range(n_personas)]
        prob = coincidencias_acum[i] / (i + 1)
        n_si = int(coincidencias_acum[i])
        n_no = (i + 1) - n_si
        frames.append(go.Frame(data=[
            go.Scatter(x=xs, y=ys, mode="markers+text",
                marker=dict(size=30, color=colores, line=dict(color="black", width=1.5)),
                text=[str(m) for m in meses], textposition="middle center",
                textfont=dict(color="white", size=9), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=list(range(1, i + 2)),
                y=list(coincidencias_acum[:i+1] / np.arange(1, i + 2)),
                mode="lines", line=dict(color="#9b59b6", width=3),
                showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2"),
            go.Bar(x=["Con coincidencia", "Sin coincidencia"], y=[n_si, n_no],
                marker_color=["#27ae60", "#e74c3c"], text=[n_si, n_no],
                textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3")
        ], name=f"f{i}", layout=go.Layout(
            title_text=f"🎂 Sim {i+1}/{n_simulaciones} - {'✅' if coincidencias[i] else '❌'} | Prob: {prob:.3f}")))
    fig = make_subplots(rows=1, cols=3, column_widths=[0.4, 0.3, 0.3],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Personas", "Convergencia", "Frec. absolutas"))
    for idx, c in enumerate([1, 2, 3]):
        fig.add_trace(frames[0].data[idx], row=1, col=c)
    fig.frames = frames
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Simulación", row=1, col=2); fig.update_yaxes(title_text="Prob.", range=[0, 1], row=1, col=2)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_xaxes(title_text="Resultado", row=1, col=3); fig.update_yaxes(title_text="Frec.", row=1, col=3)
    fig.update_layout(title_text="🎂 Problema del Cumpleaños",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.13,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 120, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=500, showlegend=False)
    return fig


def anim_poker(n_manos=15):
    baraja = crear_baraja()
    manos, tipos = [], []
    for _ in range(n_manos):
        mano = list(np.random.choice(baraja, size=5, replace=False))
        tipos.append(evaluar_mano(mano))
        manos.append(mano)
    frames = []
    conteo_acum = Counter()
    for i in range(n_manos):
        conteo_acum[tipos[i]] += 1
        mano = manos[i]
        xs = list(range(5))
        colores_cartas = ["#e74c3c" if c[-1] in ["♥", "♦"] else "#2c3e50" for c in mano]
        tipos_orden = [t for t in ORDEN_MANOS if conteo_acum.get(t, 0) > 0]
        valores = [conteo_acum[t] for t in tipos_orden]
        total = i + 1
        valores_pct = [v/total*100 for v in valores]
        colores_barras = [COLORES_MANOS[t] for t in tipos_orden]
        frames.append(go.Frame(data=[
            go.Scatter(x=xs, y=[0.7]*5, mode="markers+text",
                marker=dict(size=70, color="white", line=dict(color=colores_cartas, width=4), symbol="square"),
                text=mano, textposition="middle center",
                textfont=dict(color=colores_cartas, size=15, family="Arial Black"),
                showlegend=False, hoverinfo="skip"),
            go.Bar(x=tipos_orden, y=valores, marker_color=colores_barras,
                text=valores, textposition="outside", showlegend=False, hoverinfo="skip",
                xaxis="x2", yaxis="y2"),
            go.Bar(x=tipos_orden, y=valores_pct, marker_color=colores_barras,
                text=[f"{v:.1f}%" for v in valores_pct], textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3")
        ], name=f"f{i}", layout=go.Layout(title_text=f"🃏 Mano {i+1}/{n_manos}: {tipos[i]}")))
    fig = make_subplots(rows=1, cols=3, column_widths=[0.4, 0.3, 0.3],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Mano actual", "Frec. absolutas", "Frec. relativas (%)"))
    for idx, c in enumerate([1, 2, 3]):
        fig.add_trace(frames[0].data[idx], row=1, col=c)
    fig.frames = frames
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Tipo", tickangle=-45, row=1, col=2); fig.update_yaxes(title_text="Frec.", row=1, col=2)
    fig.update_xaxes(title_text="Tipo", tickangle=-45, row=1, col=3); fig.update_yaxes(title_text="%", row=1, col=3)
    fig.update_layout(title_text=f"🃏 Póker - {n_manos} manos",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.13,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 600, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=500, showlegend=False)
    return fig


def anim_monty_hall(n_partidas=15, cambiar=True):
    resultados = []
    for _ in range(n_partidas):
        premio = np.random.randint(0, 3)
        eleccion = np.random.randint(0, 3)
        opciones = [i for i in range(3) if i != eleccion and i != premio]
        if len(opciones) == 0:
            opciones = [i for i in range(3) if i != eleccion]
        abierta = np.random.choice(opciones)
        if cambiar:
            opciones_finales = [i for i in range(3) if i != eleccion and i != abierta]
            eleccion_final = opciones_finales[0] if opciones_finales else eleccion
        else:
            eleccion_final = eleccion
        gano = (eleccion_final == premio)
        resultados.append((premio, eleccion, abierta, eleccion_final, gano, cambiar))
    frames = []
    ganadas = 0
    for i in range(n_partidas):
        premio, eleccion, abierta, ef, gano, cambio = resultados[i]
        if gano:
            ganadas += 1
        prob = ganadas / (i + 1)
        pct_g = prob*100
        pct_p = (1-prob)*100
        colores = ["#f39c12"] * 3
        colores[abierta] = "#7f8c8d"
        simbolos = ["🚪"] * 3
        simbolos[abierta] = "🐐"
        simbolos[ef] = "🏆" if ef == premio else "🐐"
        frames.append(go.Frame(data=[
            go.Scatter(x=[0, 1, 2], y=[0.5, 0.5, 0.5], mode="markers+text",
                marker=dict(size=100, color=colores, line=dict(color="black", width=3), symbol="square"),
                text=simbolos, textposition="middle center", textfont=dict(size=30),
                showlegend=False, hoverinfo="skip"),
            go.Bar(x=["Ganadas", "Perdidas"], y=[ganadas, (i+1) - ganadas],
                marker_color=["#27ae60", "#e74c3c"], text=[ganadas, (i+1) - ganadas],
                textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2"),
            go.Bar(x=["Ganadas", "Perdidas"], y=[pct_g, pct_p],
                marker_color=["#27ae60", "#e74c3c"],
                text=[f"{pct_g:.1f}%", f"{pct_p:.1f}%"], textposition="outside",
                showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3"),
            go.Scatter(x=[0, 1], y=[prob, prob], mode="lines",
                line=dict(color="#9b59b6", width=4), showlegend=False, hoverinfo="skip",
                xaxis="x4", yaxis="y4")
        ], name=f"f{i}", layout=go.Layout(
            title_text=f"🚪 Partida {i+1}/{n_partidas}: {'✅' if gano else '❌'} | Prob: {prob:.3f}")))
    fig = make_subplots(rows=2, cols=3, column_widths=[0.3, 0.35, 0.35], row_heights=[0.5, 0.5],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 3}, None, None]],
        subplot_titles=("Puertas", "Frec. absolutas", "Frec. relativas (%)", "Convergencia"))
    for idx, (r, c) in enumerate([(1,1),(1,2),(1,3),(2,1)]):
        fig.add_trace(frames[0].data[idx], row=r, col=c)
    fig.frames = frames
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Resultado", row=1, col=2); fig.update_yaxes(title_text="Frec.", row=1, col=2)
    fig.update_xaxes(title_text="Resultado", row=1, col=3); fig.update_yaxes(title_text="%", row=1, col=3)
    fig.update_xaxes(range=[-0.1, 1.1], showticklabels=False, row=2, col=1)
    fig.update_yaxes(range=[0, 1], title_text="P(ganar)", row=2, col=1)
    fig.add_hline(y=2/3 if cambiar else 1/3, line_dash="dash", line_color="red",
                  row=2, col=1, annotation_text=f"Teórica = {2/3 if cambiar else 1/3:.3f}")
    fig.update_layout(title_text=f"🚪 Monty Hall - {'CAMBIAR' if cambiar else 'NO CAMBIAR'}",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.12,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 700, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=750, showlegend=False)
    return fig


def anim_ley_grandes_numeros(n_max=1000, tipo="Moneda (CARA)", n_frames=50):
    if tipo == "Moneda (CARA)":
        resultados = np.random.choice([1, 0], size=n_max, p=[0.5, 0.5])
        prob_teo = 0.5; etiqueta = "P(CARA) = 0.5"; color = "#3498db"
    elif tipo == "Dado (6)":
        resultados = (np.random.randint(1, 7, n_max) == 6).astype(int)
        prob_teo = 1/6; etiqueta = "P(6) = 1/6"; color = "#e67e22"
    elif tipo == "Dado (par)":
        resultados = (np.random.randint(1, 7, n_max) % 2 == 0).astype(int)
        prob_teo = 0.5; etiqueta = "P(par) = 0.5"; color = "#27ae60"
    else:
        baraja = crear_baraja()
        rojas = [c for c in baraja if c[-1] in ["♥", "♦"]]
        resultados = np.array([1 if c in rojas else 0
                                for c in np.random.choice(baraja, n_max)])
        prob_teo = 0.5; etiqueta = "P(roja) = 0.5"; color = "#e74c3c"
    frames = []
    for f in range(n_frames):
        k = int((f + 1) * n_max / n_frames)
        acumulada = np.cumsum(resultados[:k]) / np.arange(1, k + 1)
        exitos = int(np.sum(resultados[:k]))
        fracasos = k - exitos
        pct_e = exitos/k*100
        pct_f = fracasos/k*100
        frames.append(go.Frame(data=[
            go.Scatter(x=np.arange(1, k + 1), y=acumulada, mode="lines",
                line=dict(color=color, width=3), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=[1, k], y=[prob_teo, prob_teo], mode="lines",
                line=dict(color="red", width=3, dash="dash"), showlegend=False, hoverinfo="skip"),
            go.Bar(x=["Éxitos", "Fallos"], y=[exitos, fracasos],
                marker_color=[color, "#95a5a6"], text=[exitos, fracasos],
                textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2"),
            go.Bar(x=["Éxitos", "Fallos"], y=[pct_e, pct_f],
                marker_color=[color, "#95a5a6"], text=[f"{pct_e:.1f}%", f"{pct_f:.1f}%"],
                textposition="outside", showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3")
        ], name=f"f{f}", layout=go.Layout(
            title_text=f"📊 {k} ensayos | Frec = {acumulada[-1]:.4f} | Teórica = {prob_teo:.4f}")))
    fig = make_subplots(rows=1, cols=3, column_widths=[0.5, 0.25, 0.25],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Convergencia", "Frec. absolutas", "Frec. relativas (%)"))
    for idx, c in enumerate([1, 1, 2, 3]):
        fig.add_trace(frames[0].data[idx], row=1, col=c)
    fig.frames = frames
    fig.update_xaxes(title_text="Número de ensayos", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia relativa", row=1, col=1)
    fig.update_xaxes(title_text="Resultado", row=1, col=2); fig.update_yaxes(title_text="Frec.", row=1, col=2)
    fig.update_xaxes(title_text="Resultado", row=1, col=3); fig.update_yaxes(title_text="%", row=1, col=3)
    fig.update_layout(title_text=f"📊 Ley de los Grandes Números - {tipo}",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.13,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[dict(label="▶️ Play", method="animate",
                          args=[None, {"frame": {"duration": 100, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}]),
                     dict(label="⏸️ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])])],
        height=500, showlegend=False)
    fig.add_hline(y=prob_teo, line_dash="dash", line_color="red",
                  row=1, col=1, annotation_text=etiqueta)
    return fig


# ============================================================
# SIMULADORES NUMÉRICOS
# ============================================================
def simulador_monedas(n_monedas=1, n_lanzamientos=100, prob_cara=0.5):
    return np.random.choice(["CARA", "SELLO"], size=(n_lanzamientos, n_monedas),
                            p=[prob_cara, 1 - prob_cara])

def simulador_dados(n_dados=1, n_lanzamientos=100, caras=6):
    return np.random.randint(1, caras + 1, size=(n_lanzamientos, n_dados))

def simulador_moneda_dado(n_lanzamientos=100):
    return (np.random.choice(["CARA", "SELLO"], size=n_lanzamientos),
            np.random.randint(1, 7, size=n_lanzamientos))

def simulador_cartas(n_cartas=1, n_lanzamientos=100):
    baraja = crear_baraja()
    return np.random.choice(baraja, size=(n_lanzamientos, n_cartas))

def simulador_balotas(n_balotas=6, rango=(1, 43), n_lanzamientos=10):
    res = []
    for _ in range(n_lanzamientos):
        nums = np.random.choice(range(rango[0], rango[1]+1), size=n_balotas+1, replace=False)
        res.append((sorted(nums[:-1].tolist()), int(nums[-1])))
    return res

def simulador_ruleta(tipo="europea", n_tiradas=100):
    if tipo == "europea":
        casillas = list(range(0, 37))
    else:
        casillas = list(range(0, 37)) + ["00"]
    rojos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
    negros = set(range(1, 37)) - rojos
    return np.random.choice(casillas, size=n_tiradas), rojos, negros

def simulador_powerball(n_sorteos=10):
    res = []
    for _ in range(n_sorteos):
        nums = sorted(np.random.choice(range(1, 70), size=5, replace=False).tolist())
        pb = int(np.random.randint(1, 27))
        res.append((nums, pb))
    return res

def monte_carlo_pi(n_puntos=1000):
    x = np.random.uniform(-1, 1, n_puntos)
    y = np.random.uniform(-1, 1, n_puntos)
    dentro = x**2 + y**2 <= 1
    return x, y, dentro, 4*np.sum(dentro)/n_puntos

def problema_cumpleanos(n_personas=23, n_simulaciones=1000):
    c = 0
    for _ in range(n_simulaciones):
        if len(set(np.random.randint(1, 366, size=n_personas))) < n_personas:
            c += 1
    return c / n_simulaciones

def prob_cumpleanos_teorica(n):
    if n > 365: return 1.0
    p = 1.0
    for i in range(n):
        p *= (365 - i) / 365
    return 1 - p

def simular_poker(n_manos=1000):
    baraja = crear_baraja()
    tipos = []
    for _ in range(n_manos):
        mano = np.random.choice(baraja, size=5, replace=False)
        tipos.append(evaluar_mano(list(mano)))
    return Counter(tipos), tipos

def simular_monty_hall(n_partidas=1000, cambiar=True):
    victorias = 0
    historial = []
    for _ in range(n_partidas):
        premio = np.random.randint(0, 3)
        eleccion = np.random.randint(0, 3)
        opciones = [i for i in range(3) if i != eleccion and i != premio]
        if len(opciones) == 0:
            opciones = [i for i in range(3) if i != eleccion]
        abierta = np.random.choice(opciones)
        if cambiar:
            opf = [i for i in range(3) if i != eleccion and i != abierta]
            ef = opf[0] if opf else eleccion
        else:
            ef = eleccion
        gano = (ef == premio)
        if gano: victorias += 1
        historial.append(gano)
    return victorias / n_partidas, historial


# ============================================================
# INTERFAZ
# ============================================================
tabs = st.tabs([
    "🪙 Monedas", "🎲 Dados", "🪙🎲 Moneda + Dado",
    "🃏 Cartas", "🎰 Baloto", "📊 Ley Grandes Números",
    "🎡 Ruleta", "💥 Powerball", "🥧 Monte Carlo π", "🎂 Cumpleaños",
    "🃏 Póker", "🚪 Monty Hall", "🎯 Operaciones con Eventos"
])

# ---------------- TAB 1: MONEDAS ----------------
with tabs[0]:
    st.header("🪙 Simulación con Monedas")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_monedas = st.slider("Monedas por lanzamiento:", 1, 10, 1)
        n_lanzamientos = st.number_input("Total lanzamientos:", 10, 100000, 500, 50)
        prob_cara = st.slider("P(CARA):", 0.0, 1.0, 0.5, 0.01)
        semilla = st.number_input("Semilla:", 0, 99999, 42)
        if semilla > 0: np.random.seed(int(semilla))
        st.markdown("---")
        n_anim = st.slider("Lanzamientos animados:", 5, 60, 30, key="gif_mon")
        if st.button("🎬 Generar Animación", key="btn_anim_mon"):
            st.session_state.historial["anim_moneda"] = anim_moneda(n_anim, prob_cara, n_monedas)
        if st.button("🎲 Simular (datos)", key="btn_monedas"):
            st.session_state.historial["monedas"] = simulador_monedas(
                n_monedas, n_lanzamientos, prob_cara)
    with col2:
        if "anim_moneda" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_moneda"], use_container_width=True)
        if "monedas" in st.session_state.historial:
            res = st.session_state.historial["monedas"]
            if n_monedas == 1:
                caras = int(np.sum(res == "CARA")); sellos = int(np.sum(res == "SELLO"))
                t = len(res)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", t)
                c2.metric("Caras", caras, f"{caras/t*100:.2f}%")
                c3.metric("Sellos", sellos, f"{sellos/t*100:.2f}%")
                fig = grafico_barras_doble(["CARA", "SELLO"], [caras, sellos],
                    titulo="Monedas - Frec. absolutas y relativas",
                    colores_custom=["#3498db", "#e74c3c"], etiqueta_x="Resultado")
                st.plotly_chart(fig, use_container_width=True)
                fig_teo = grafico_comparacion_teorica(
                    ["CARA", "SELLO"], [caras, sellos],
                    [prob_cara, 1 - prob_cara], caras + sellos,
                    titulo=f"🎯 Observado vs Teórico (Bernoulli, p={prob_cara})")
                st.plotly_chart(fig_teo, use_container_width=True)
            else:
                num_caras = np.sum(res == "CARA", axis=1)
                dist = Counter(num_caras)
                cats = list(range(0, n_monedas + 1))
                frecs = [dist.get(k, 0) for k in cats]
                fig = grafico_barras_doble(cats, frecs,
                    titulo=f"Distribución del número de caras ({n_monedas} monedas)",
                    color_abs="#3498db", color_rel="#e67e22", etiqueta_x="Número de caras")
                st.plotly_chart(fig, use_container_width=True)
                probs_teo = [binom.pmf(k, n_monedas, prob_cara) for k in cats]
                fig_teo = grafico_comparacion_teorica(
                    [str(c) for c in cats], frecs, probs_teo, sum(frecs),
                    titulo=f"🎯 Binomial({n_monedas}, {prob_cara}): Observado vs Teórico")
                st.plotly_chart(fig_teo, use_container_width=True)

# ---------------- TAB 2: DADOS ----------------
with tabs[1]:
    st.header("🎲 Simulación con Dados")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_dados = st.slider("Dados por lanzamiento:", 1, 10, 1)
        n_lanz = st.number_input("Lanzamientos:", 10, 100000, 500, 50, key="nl_dados")
        caras_d = st.selectbox("Caras:", [4, 6, 8, 10, 12, 20], 1)
        sem_d = st.number_input("Semilla:", 0, 99999, 42, key="sem_dados")
        if sem_d > 0: np.random.seed(int(sem_d))
        st.markdown("---")
        n_anim_d = st.slider("Lanzamientos animados:", 5, 60, 30, key="gif_dado")
        if st.button("🎬 Generar Animación", key="btn_anim_dado"):
            st.session_state.historial["anim_dado"] = anim_dado(n_anim_d, caras_d, n_dados)
        if st.button("🎲 Simular (datos)", key="btn_dados"):
            st.session_state.historial["dados"] = simulador_dados(n_dados, n_lanz, caras_d)
    with col2:
        if "anim_dado" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_dado"], use_container_width=True)
        if "dados" in st.session_state.historial:
            res = st.session_state.historial["dados"]
            if n_dados == 1:
                valores = res.flatten()
                frec = Counter(valores)
                cats = list(range(1, caras_d + 1))
                frecs = [frec.get(i, 0) for i in range(1, caras_d + 1)]
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(valores))
                c2.metric("Media", f"{valores.mean():.3f}", f"Teo: {(caras_d+1)/2:.2f}")
                c3.metric("Desv.Est.", f"{valores.std():.3f}")
                fig = grafico_barras_doble(cats, frecs,
                    titulo=f"Dado de {caras_d} caras - Frec. absolutas y %",
                    color_abs="skyblue", color_rel="#e67e22", etiqueta_x="Cara")
                st.plotly_chart(fig, use_container_width=True)
                fig_teo = grafico_comparacion_teorica(
                    [str(c) for c in cats], frecs, [1/caras_d]*caras_d, sum(frecs),
                    titulo=f"🎯 Uniforme discreta(1,{caras_d}): Observado vs Teórico")
                st.plotly_chart(fig_teo, use_container_width=True)
            else:
                sumas = np.sum(res, axis=1)
                dist_s = Counter(sumas)
                cats = list(range(n_dados, n_dados*caras_d + 1))
                frecs = [dist_s.get(s, 0) for s in cats]
                fig = grafico_barras_doble(cats, frecs,
                    titulo=f"Distribución de la suma de {n_dados} dados",
                    color_abs="#9b59b6", color_rel="#8e44ad", etiqueta_x="Suma")
                st.plotly_chart(fig, use_container_width=True)
                todos = res.flatten()
                frec_t = Counter(todos)
                cats2 = list(range(1, caras_d + 1))
                frecs2 = [frec_t.get(i, 0) for i in range(1, caras_d + 1)]
                fig2 = grafico_barras_doble(cats2, frecs2,
                    titulo=f"Frecuencia por cara (todos los {n_dados} dados)",
                    color_abs="skyblue", color_rel="#e67e22", etiqueta_x="Cara")
                st.plotly_chart(fig2, use_container_width=True)

# ---------------- TAB 3: MONEDA + DADO ----------------
with tabs[2]:
    st.header("🪙🎲 Moneda + Dado")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_lanz_md = st.number_input("Lanzamientos:", 10, 100000, 500, 50, key="nl_md")
        sem_md = st.number_input("Semilla:", 0, 99999, 42, key="sem_md")
        if sem_md > 0: np.random.seed(int(sem_md))
        st.markdown("---")
        n_anim_md = st.slider("Lanzamientos animados:", 5, 40, 20, key="gif_md")
        if st.button("🎬 Generar Animación", key="btn_anim_md"):
            st.session_state.historial["anim_md"] = anim_moneda_dado(n_anim_md)
        if st.button("🎲 Simular (datos)", key="btn_md"):
            m, d = simulador_moneda_dado(n_lanz_md)
            st.session_state.historial["moneda_dado"] = (m, d)
    with col2:
        if "anim_md" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_md"], use_container_width=True)
        if "moneda_dado" in st.session_state.historial:
            m, d = st.session_state.historial["moneda_dado"]
            df = pd.DataFrame({"Moneda": m, "Dado": d})
            cont = pd.crosstab(df["Dado"], df["Moneda"])
            st.dataframe(cont, use_container_width=True)
            fig = px.bar(df.groupby(["Dado", "Moneda"]).size().reset_index(name="Frec"),
                         x="Dado", y="Frec", color="Moneda", barmode="group",
                         color_discrete_map={"CARA": "#3498db", "SELLO": "#e74c3c"},
                         text_auto=True, title="Gráfico de barras - distribución conjunta")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 4: CARTAS ----------------
with tabs[3]:
    st.header("🃏 Cartas")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_cartas = st.slider("Cartas por extracción:", 1, 10, 1)
        n_extr = st.number_input("Extracciones:", 10, 10000, 200, 10, key="ne_cartas")
        sem_c = st.number_input("Semilla:", 0, 99999, 42, key="sem_cartas")
        if sem_c > 0: np.random.seed(int(sem_c))
        st.markdown("---")
        n_anim_c = st.slider("Cartas animadas:", 5, 25, 15, key="gif_c")
        if st.button("🎬 Generar Animación", key="btn_anim_c"):
            st.session_state.historial["anim_cartas"] = anim_cartas(n_anim_c)
        if st.button("🎲 Simular (datos)", key="btn_cartas"):
            st.session_state.historial["cartas"] = simulador_cartas(n_cartas, n_extr)
    with col2:
        if "anim_cartas" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_cartas"], use_container_width=True)
        if "cartas" in st.session_state.historial:
            res = st.session_state.historial["cartas"]
            valores = res.flatten()
            palos = [c[-1] for c in valores]
            cp = Counter(palos)
            cats = ["♠", "♥", "♦", "♣"]
            frecs = [cp.get(p, 0) for p in cats]
            fig = grafico_barras_doble(cats, frecs,
                titulo="Cartas por palo - Frec. absolutas y %",
                colores_custom=["#2c3e50", "#e74c3c", "#e67e22", "#27ae60"],
                etiqueta_x="Palo")
            st.plotly_chart(fig, use_container_width=True)
            fig_teo = grafico_comparacion_teorica(
                cats, frecs, [0.25, 0.25, 0.25, 0.25], sum(frecs),
                titulo="🎯 4 palos equiprobables (p=0.25): Observado vs Teórico")
            st.plotly_chart(fig_teo, use_container_width=True)

# ---------------- TAB 5: BALOTO ----------------
with tabs[4]:
    st.header("🎰 Baloto")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_bal = st.slider("Balotas:", 4, 6, 5)
        r_max = st.number_input("Máx. número:", 10, 100, 43)
        n_sort = st.number_input("Sorteos:", 1, 1000, 10)
        sem_b = st.number_input("Semilla:", 0, 99999, 42, key="sem_bal")
        if sem_b > 0: np.random.seed(int(sem_b))
        st.markdown("---")
        n_anim_b = st.slider("Sorteos animados:", 3, 20, 8, key="gif_b")
        if st.button("🎬 Generar Animación", key="btn_anim_b"):
            st.session_state.historial["anim_baloto"] = anim_baloto(n_anim_b, n_bal, r_max)
        if st.button("🎲 Simular (datos)", key="btn_baloto"):
            st.session_state.historial["baloto"] = simulador_balotas(n_bal, (1, r_max), n_sort)
    with col2:
        if "anim_baloto" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_baloto"], use_container_width=True)
        if "baloto" in st.session_state.historial:
            res = st.session_state.historial["baloto"]
            todos = [n for nums, _ in res for n in nums]
            frec = Counter(todos)
            cats = list(range(1, r_max + 1))
            frecs = [frec.get(i, 0) for i in range(1, r_max + 1)]
            fig = grafico_barras_doble(cats, frecs,
                titulo="Frecuencia por número - Absoluta y relativa",
                color_abs="#3498db", color_rel="#e67e22", etiqueta_x="Número")
            st.plotly_chart(fig, use_container_width=True)
            prob_teor_bal = [n_bal / r_max] * min(15, len(cats))
            fig_teo = grafico_comparacion_teorica(
                [str(c) for c in cats[:15]], frecs[:15],
                prob_teor_bal, sum(frecs),
                titulo="🎯 Baloto (primeros 15 números): Observado vs Teórico")
            st.plotly_chart(fig_teo, use_container_width=True)

# ---------------- TAB 6: LEY GRANDES NÚMEROS ----------------
with tabs[5]:
    st.header("📊 Ley de los Grandes Números")
    col1, col2 = st.columns([1, 2])
    with col1:
        tipo_lgn = st.selectbox("Experimento:",
            ["Moneda (CARA)", "Dado (6)", "Dado (par)", "Carta roja (♥♦)"], key="tipo_lgn")
        n_max_lgn = st.number_input("Ensayos:", 100, 50000, 5000, 100, key="n_lgn")
        sem_lgn = st.number_input("Semilla:", 0, 99999, 42, key="sem_lgn")
        if sem_lgn > 0: np.random.seed(int(sem_lgn))
        st.markdown("---")
        n_anim_lgn = st.slider("Frames animados:", 20, 100, 60, key="gif_lgn")
        if st.button("🎬 Generar Animación", key="btn_anim_lgn"):
            st.session_state.historial["anim_lgn"] = anim_ley_grandes_numeros(n_max_lgn, tipo_lgn, n_anim_lgn)
        if st.button("🚀 Ejecutar simulación", key="btn_lgn"):
            if tipo_lgn == "Moneda (CARA)":
                r = np.random.choice([1, 0], n_max_lgn); pt = 0.5; et = "P(CARA) = 0.5"
            elif tipo_lgn == "Dado (6)":
                r = (np.random.randint(1, 7, n_max_lgn) == 6).astype(int); pt = 1/6; et = "P(6) = 1/6"
            elif tipo_lgn == "Dado (par)":
                r = (np.random.randint(1, 7, n_max_lgn) % 2 == 0).astype(int); pt = 0.5; et = "P(par) = 0.5"
            else:
                baraja = crear_baraja()
                rojas = [c for c in baraja if c[-1] in ["♥", "♦"]]
                r = np.array([1 if c in rojas else 0 for c in np.random.choice(baraja, n_max_lgn)])
                pt = 0.5; et = "P(roja) = 0.5"
            st.session_state.historial["lgn"] = (r, pt, et, tipo_lgn)
    with col2:
        if "anim_lgn" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_lgn"], use_container_width=True)
        if "lgn" in st.session_state.historial:
            r, pt, et, tipo_lgn = st.session_state.historial["lgn"]
            acumulada = np.cumsum(r) / np.arange(1, len(r) + 1)
            c1, c2, c3 = st.columns(3)
            c1.metric("Frec. final", f"{acumulada[-1]:.5f}")
            c2.metric("Prob. teórica", f"{pt:.5f}")
            c3.metric("Error", f"{abs(acumulada[-1] - pt):.5f}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=acumulada, mode="lines",
                line=dict(color="#3498db", width=2), name="Frec. relativa"))
            fig.add_hline(y=pt, line_dash="dash", line_color="red", annotation_text=et)
            fig.update_layout(title=f"Convergencia - {tipo_lgn}",
                xaxis_title="Ensayos", yaxis_title="Frec. relativa", height=400)
            st.plotly_chart(fig, use_container_width=True)
            exitos = int(np.sum(r)); fallos = len(r) - exitos
            fig_bar = grafico_barras_doble(["Éxitos", "Fallos"], [exitos, fallos],
                titulo="Ley G.N. - Frec. absolutas y relativas",
                colores_custom=["#27ae60", "#e74c3c"], etiqueta_x="Resultado")
            st.plotly_chart(fig_bar, use_container_width=True)
            fig_teo = grafico_comparacion_teorica(
                ["Éxitos", "Fallos"], [exitos, fallos],
                [pt, 1 - pt], exitos + fallos,
                titulo="🎯 Ley G.N.: Observado vs Teórico")
            st.plotly_chart(fig_teo, use_container_width=True)

# ---------------- TAB 7: RULETA ----------------
with tabs[6]:
    st.header("🎡 Ruleta")
    col1, col2 = st.columns([1, 2])
    with col1:
        tipo_r = st.selectbox("Tipo:", ["Europea (0-36)", "Americana (0, 00, 1-36)"])
        n_tir = st.number_input("Tiradas:", 10, 100000, 1000, 100)
        sem_r = st.number_input("Semilla:", 0, 99999, 42, key="sem_r")
        if sem_r > 0: np.random.seed(int(sem_r))
        st.markdown("---")
        n_anim_r = st.slider("Giros animados:", 5, 30, 15, key="gif_r")
        if st.button("🎬 Generar Animación", key="btn_anim_r"):
            tipo_str = "europea" if "Europea" in tipo_r else "americana"
            st.session_state.historial["anim_ruleta"] = anim_ruleta(n_anim_r, tipo_str)
        if st.button("🎡 Simular (datos)", key="btn_ruleta"):
            tipo_str = "europea" if "Europea" in tipo_r else "americana"
            t, roj, neg = simulador_ruleta(tipo_str, n_tir)
            st.session_state.historial["ruleta"] = (t, roj, neg)
    with col2:
        if "anim_ruleta" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_ruleta"], use_container_width=True)
        if "ruleta" in st.session_state.historial:
            t, roj, neg = st.session_state.historial["ruleta"]
            nr = sum(1 for x in t if x in roj)
            nn = sum(1 for x in t if x in neg)
            nc = sum(1 for x in t if x == 0 or x == "00")
            fig = grafico_barras_doble(["Rojo", "Negro", "Verde"], [nr, nn, nc],
                titulo="Ruleta - Frec. absolutas y relativas",
                colores_custom=["#e74c3c", "#2c3e50", "#27ae60"], etiqueta_x="Color")
            st.plotly_chart(fig, use_container_width=True)
            prob_teo_r = [18/37, 18/37, 1/37] if "Europea" in tipo_r else [18/38, 18/38, 2/38]
            fig_teo = grafico_comparacion_teorica(
                ["Rojo", "Negro", "Verde"], [nr, nn, nc], prob_teo_r, nr + nn + nc,
                titulo="🎯 Ruleta: Observado vs Teórico")
            st.plotly_chart(fig_teo, use_container_width=True)

# ---------------- TAB 8: POWERBALL ----------------
with tabs[7]:
    st.header("💥 Powerball")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_pb = st.number_input("Sorteos:", 1, 10000, 100)
        sem_pb = st.number_input("Semilla:", 0, 99999, 42, key="sem_pb")
        if sem_pb > 0: np.random.seed(int(sem_pb))
        n_anim_pb = st.slider("Sorteos animados:", 3, 15, 6, key="gif_pb")
        if st.button("🎬 Generar Animación", key="btn_anim_pb"):
            st.session_state.historial["anim_powerball"] = anim_powerball(n_anim_pb)
        if st.button("💥 Simular (datos)", key="btn_pb"):
            st.session_state.historial["powerball"] = simulador_powerball(n_pb)
    with col2:
        if "anim_powerball" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_powerball"], use_container_width=True)
        if "powerball" in st.session_state.historial:
            res = st.session_state.historial["powerball"]
            todos = [n for nums, _ in res for n in nums]
            frec = Counter(todos)
            cats = list(range(1, 70))
            frecs = [frec.get(i, 0) for i in range(1, 70)]
            fig = grafico_barras_doble(cats, frecs, titulo="Powerball - Frecuencia",
                color_abs="#3498db", color_rel="#e67e22", etiqueta_x="Número")
            st.plotly_chart(fig, use_container_width=True)
            prob_teor_pb = [5/69] * 15
            fig_teo = grafico_comparacion_teorica(
                [str(c) for c in cats[:15]], frecs[:15],
                prob_teor_pb, sum(frecs),
                titulo="🎯 Powerball (primeros 15 números): Observado vs Teórico")
            st.plotly_chart(fig_teo, use_container_width=True)

# ---------------- TAB 9: MONTE CARLO π ----------------
with tabs[8]:
    st.header("🥧 Monte Carlo π")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_pt = st.slider("Puntos:", 100, 100000, 5000, 100)
        sem_pi = st.number_input("Semilla:", 0, 99999, 42, key="sem_pi")
        if sem_pi > 0: np.random.seed(int(sem_pi))
        n_anim_pi = st.slider("Puntos animados:", 200, 5000, 1500, 100, key="gif_pi")
        if st.button("🎬 Generar Animación", key="btn_anim_pi"):
            st.session_state.historial["anim_pi"] = anim_monte_carlo(n_anim_pi)
        if st.button("🎯 Calcular (datos)", key="btn_pi"):
            x, y, d, pe = monte_carlo_pi(n_pt)
            st.session_state.historial["pi"] = (x, y, d, pe, n_pt)
    with col2:
        if "anim_pi" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_pi"], use_container_width=True)
        if "pi" in st.session_state.historial:
            x, y, d, pe, np_ = st.session_state.historial["pi"]
            c1, c2, c3 = st.columns(3)
            c1.metric("π est.", f"{pe:.6f}")
            c2.metric("π real", f"{np.pi:.6f}")
            c3.metric("Error", f"{abs(pe-np.pi):.6f}")
            dentro_count = int(np.sum(d))
            fuera_count = int(len(d) - dentro_count)
            fig = grafico_barras_doble(["Dentro", "Fuera"], [dentro_count, fuera_count],
                titulo="Monte Carlo - Distribución de puntos",
                colores_custom=["#27ae60", "#e74c3c"], etiqueta_x="Zona")
            st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 10: CUMPLEAÑOS ----------------
with tabs[9]:
    st.header("🎂 Problema del Cumpleaños")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_per = st.slider("Personas:", 2, 100, 23)
        n_sim = st.number_input("Simulaciones:", 100, 100000, 2000, 100)
        sem_c = st.number_input("Semilla:", 0, 99999, 42, key="sem_cum")
        if sem_c > 0: np.random.seed(int(sem_c))
        n_anim_cum = st.slider("Simulaciones animadas:", 20, 200, 100, 10, key="gif_cum")
        if st.button("🎬 Generar Animación", key="btn_anim_cum"):
            st.session_state.historial["anim_cumple"] = anim_cumpleanos(n_per, n_anim_cum)
        if st.button("🎂 Simular (datos)", key="btn_cum"):
            ps = problema_cumpleanos(n_per, n_sim)
            pt = prob_cumpleanos_teorica(n_per)
            st.session_state.historial["cumple"] = (ps, pt, n_per, n_sim)
    with col2:
        if "anim_cumple" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_cumple"], use_container_width=True)
        if "cumple" in st.session_state.historial:
            ps, pt, n, n_sim = st.session_state.historial["cumple"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Simulada", f"{ps*100:.2f}%")
            c2.metric("Teórica", f"{pt*100:.2f}%")
            c3.metric("Error", f"{abs(ps-pt)*100:.3f}%")
            exitos = int(ps * n_sim)
            fallos = n_sim - exitos
            fig = grafico_barras_doble(["Con coincidencia", "Sin coincidencia"], [exitos, fallos],
                titulo="Cumpleaños - Distribución de simulaciones",
                colores_custom=["#27ae60", "#e74c3c"], etiqueta_x="Resultado")
            st.plotly_chart(fig, use_container_width=True)
            fig_teo = grafico_comparacion_teorica(
                ["Con coincidencia", "Sin coincidencia"], [exitos, fallos],
                [pt, 1 - pt], exitos + fallos,
                titulo=f"🎯 Cumpleaños ({n} personas): Observado vs Teórico")
            st.plotly_chart(fig_teo, use_container_width=True)

# ---------------- TAB 11: PÓKER ----------------
with tabs[10]:
    st.header("🃏 Simulación de Póker - 5 Cartas")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_manos = st.number_input("Manos a simular:", 10, 100000, 5000, 100, key="n_poker")
        sem_p = st.number_input("Semilla:", 0, 99999, 42, key="sem_poker")
        if sem_p > 0: np.random.seed(int(sem_p))
        n_anim_p = st.slider("Manos animadas:", 5, 30, 15, key="gif_poker")
        if st.button("🎬 Generar Animación", key="btn_anim_p"):
            st.session_state.historial["anim_poker"] = anim_poker(n_anim_p)
        if st.button("🃏 Simular (datos)", key="btn_poker"):
            conteo, _ = simular_poker(n_manos)
            st.session_state.historial["poker"] = (conteo, n_manos)
    with col2:
        if "anim_poker" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_poker"], use_container_width=True)
        if "poker" in st.session_state.historial:
            conteo, n_manos = st.session_state.historial["poker"]
            filas = []
            for tipo in ORDEN_MANOS:
                sim = conteo.get(tipo, 0)
                prob_sim = sim / n_manos
                prob_teo = PROB_TEORICAS_POKER[tipo]
                filas.append({"Mano": tipo, "Frecuencia": sim,
                              "Prob. Simulada": f"{prob_sim*100:.4f}%",
                              "Prob. Teórica": f"{prob_teo*100:.4f}%",
                              "Diferencia": f"{abs(prob_sim - prob_teo)*100:.4f}%"})
            st.dataframe(pd.DataFrame(filas), use_container_width=True)
            frecs = [conteo.get(t, 0) for t in ORDEN_MANOS]
            colores = [COLORES_MANOS[t] for t in ORDEN_MANOS]
            fig = grafico_barras_doble(ORDEN_MANOS, frecs,
                titulo="Póker - Frec. absolutas y relativas",
                colores_custom=colores, etiqueta_x="Tipo de mano")
            st.plotly_chart(fig, use_container_width=True)
            probs_teo = [PROB_TEORICAS_POKER[t] for t in ORDEN_MANOS]
            fig_teo = grafico_comparacion_teorica(
                ORDEN_MANOS, frecs, probs_teo, sum(frecs),
                titulo="🎯 Póker: Observado vs Probabilidades Teóricas")
            st.plotly_chart(fig_teo, use_container_width=True)

# ---------------- TAB 12: MONTY HALL ----------------
with tabs[11]:
    st.header("🚪 Problema de Monty Hall")
    col1, col2 = st.columns([1, 2])
    with col1:
        estrategia = st.radio("Estrategia:", ["Cambiar siempre", "Nunca cambiar"],
                               horizontal=True, key="estrategia_mh")
        cambiar_bool = (estrategia == "Cambiar siempre")
        n_part = st.number_input("Partidas:", 10, 100000, 5000, 100, key="n_mh")
        sem_mh = st.number_input("Semilla:", 0, 99999, 42, key="sem_mh")
        if sem_mh > 0: np.random.seed(int(sem_mh))
        n_anim_mh = st.slider("Partidas animadas:", 5, 40, 15, key="gif_mh")
        if st.button("🎬 Generar Animación", key="btn_anim_mh"):
            st.session_state.historial["anim_mh"] = anim_monty_hall(n_anim_mh, cambiar_bool)
        if st.button("🚪 Simular (datos)", key="btn_mh"):
            prob, historial = simular_monty_hall(n_part, cambiar_bool)
            st.session_state.historial["monty_hall"] = (prob, historial, cambiar_bool, n_part)
    with col2:
        if "anim_mh" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_mh"], use_container_width=True)
        if "monty_hall" in st.session_state.historial:
            prob, historial, cambio, n_part = st.session_state.historial["monty_hall"]
            prob_teo = 2/3 if cambio else 1/3
            c1, c2, c3 = st.columns(3)
            c1.metric("Prob. Simulada", f"{prob*100:.2f}%")
            c2.metric("Prob. Teórica", f"{prob_teo*100:.2f}%")
            c3.metric("Error", f"{abs(prob - prob_teo)*100:.3f}%")
            ganadas = int(prob * n_part); perdidas = n_part - ganadas
            fig_bar = grafico_barras_doble(["Ganadas 🏆", "Perdidas 🐐"], [ganadas, perdidas],
                titulo=f"Monty Hall - {'CAMBIAR' if cambio else 'NO CAMBIAR'}",
                colores_custom=["#27ae60", "#e74c3c"], etiqueta_x="Resultado")
            st.plotly_chart(fig_bar, use_container_width=True)
            prob_teo_mh = [2/3, 1/3] if cambio else [1/3, 2/3]
            fig_teo = grafico_comparacion_teorica(
                ["Ganadas", "Perdidas"], [ganadas, perdidas],
                prob_teo_mh, ganadas + perdidas,
                titulo=f"🎯 Monty Hall ({'CAMBIAR' if cambio else 'NO CAMBIAR'}): Observado vs Teórico")
            st.plotly_chart(fig_teo, use_container_width=True)

# ---------------- TAB 13: OPERACIONES CON EVENTOS ----------------
with tabs[12]:
    st.header("🎯 Operaciones con Eventos y Diagramas de Venn")
    st.markdown("""
    **Define eventos** como conjuntos de elementos y realiza **operaciones entre ellos**.
    El sistema dibujará el **diagrama de Venn** correspondiente.
    """)

    st.subheader("🌌 1. Define el Universo (U)")
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        universo_tipo = st.selectbox(
            "Tipo de universo:",
            ["Dígitos (0-9)", "Números del 1 al 20", "Números del 1 al 30",
             "Números del 1 al 50", "Personalizado"],
            key="univ_tipo_evt")

    if universo_tipo == "Dígitos (0-9)":
        universo = list(range(0, 10))
    elif universo_tipo == "Números del 1 al 20":
        universo = list(range(1, 21))
    elif universo_tipo == "Números del 1 al 30":
        universo = list(range(1, 31))
    elif universo_tipo == "Números del 1 al 50":
        universo = list(range(1, 51))
    else:
        texto_u = st.text_input("Escribe los elementos de U separados por coma:",
            value="1, 2, 3, 4, 5, 6, 7, 8, 9, 10", key="univ_custom_evt")
        try:
            universo = [int(x.strip()) for x in texto_u.split(",") if x.strip()]
        except:
            universo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            st.warning("⚠️ No se pudo leer el universo personalizado. Usando default.")

    U = set(universo)
    st.info(f"**U = {{{', '.join(map(str, sorted(U)))}}}**   |U| = {len(U)}")

    st.markdown("---")
    st.subheader("📝 2. Define los eventos (subconjuntos de U)")

    if universo_tipo == "Dígitos (0-9)":
        plantillas = {
            "A = {x | x es par}": lambda s: {x for x in s if x % 2 == 0},
            "B = {x | x es impar}": lambda s: {x for x in s if x % 2 == 1},
            "C = {x | x es primo}": lambda s: {x for x in s if x in {2, 3, 5, 7}},
            "D = {x | x es múltiplo de 3}": lambda s: {x for x in s if x % 3 == 0},
            "E = {x | x > 5}": lambda s: {x for x in s if x > 5},
            "F = {x | x < 3}": lambda s: {x for x in s if x < 3},
            "Personalizado": None,
        }
    else:
        def es_primo(x):
            if x < 2: return False
            for d in range(2, int(x**0.5)+1):
                if x % d == 0: return False
            return True
        plantillas = {
            "A = {x | x es par}": lambda s: {x for x in s if x % 2 == 0},
            "B = {x | x es impar}": lambda s: {x for x in s if x % 2 == 1},
            "C = {x | x es primo}": lambda s: {x for x in s if es_primo(x)},
            "D = {x | x es múltiplo de 3}": lambda s: {x for x in s if x % 3 == 0},
            "E = {x | x > mitad}": lambda s: {x for x in s if x > max(s)//2},
            "F = {x | x < mitad}": lambda s: {x for x in s if x < max(s)//2},
            "Personalizado": None,
        }

    n_eventos = st.slider("¿Cuántos eventos quieres definir?", 2, 4, 3, key="n_evt_slider")

    eventos = {}
    colores_eventos = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]

    for i in range(n_eventos):
        st.markdown(f"**Evento {chr(65+i)}**")
        col_e1, col_e2 = st.columns([1, 2])
        with col_e1:
            plantilla = st.selectbox(f"Plantilla para {chr(65+i)}:",
                list(plantillas.keys()), key=f"plantilla_{i}_evt")
        with col_e2:
            if plantilla == "Personalizado":
                texto_e = st.text_input(f"Elementos de {chr(65+i)} separados por coma:",
                    value=", ".join(map(str, sorted(list(U)[:3]))), key=f"custom_{i}_evt")
                try:
                    conjunto = {int(x.strip()) for x in texto_e.split(",") if x.strip()}
                except:
                    conjunto = set()
            else:
                conjunto = plantillas[plantilla](U)

        eventos[chr(65+i)] = {
            "conjunto": conjunto & U,
            "color": colores_eventos[i],
            "plantilla": plantilla if plantilla != "Personalizado" else f"{chr(65+i)} personalizado"
        }
        st.caption(f"**{chr(65+i)} = {{{', '.join(map(str, sorted(eventos[chr(65+i)]['conjunto'])))}}}**  (|{chr(65+i)}| = {len(eventos[chr(65+i)]['conjunto'])})")

    st.markdown("---")
    st.subheader("🔧 3. Selecciona la operación")

    nombres_eventos = list(eventos.keys())
    operaciones = [
        "Unión (A ∪ B)", "Intersección (A ∩ B)",
        "Diferencia (A \\ B)", "Diferencia simétrica (A △ B)",
        "Complemento (Aᶜ)", "Intersección triple (A ∩ B ∩ C)",
        "Unión triple (A ∪ B ∪ C)",
    ]

    col_op1, col_op2, col_op3 = st.columns([2, 1, 1])
    with col_op1:
        operacion = st.selectbox("Operación:", operaciones, key="op_evt")
    with col_op2:
        ev_a = st.selectbox("Evento A:", nombres_eventos, index=0, key="ev_a")
    with col_op3:
        ev_b = st.selectbox("Evento B:", nombres_eventos,
                             index=1 if len(nombres_eventos) > 1 else 0, key="ev_b")

    if "triple" in operacion.lower():
        ev_c = st.selectbox("Evento C:", nombres_eventos,
                             index=2 if len(nombres_eventos) > 2 else 0, key="ev_c")

    A = eventos[ev_a]["conjunto"]
    B = eventos[ev_b]["conjunto"]

    if operacion == "Unión (A ∪ B)":
        resultado = A | B; expr = f"{ev_a} ∪ {ev_b}"; color_res = "#9b59b6"
    elif operacion == "Intersección (A ∩ B)":
        resultado = A & B; expr = f"{ev_a} ∩ {ev_b}"; color_res = "#16a085"
    elif operacion == "Diferencia (A \\ B)":
        resultado = A - B; expr = f"{ev_a} \\ {ev_b}"; color_res = "#e67e22"
    elif operacion == "Diferencia simétrica (A △ B)":
        resultado = A ^ B; expr = f"{ev_a} △ {ev_b}"; color_res = "#c0392b"
    elif operacion == "Complemento (Aᶜ)":
        resultado = U - A; expr = f"{ev_a}ᶜ"; color_res = "#8e44ad"
    elif operacion == "Intersección triple (A ∩ B ∩ C)":
        C = eventos[ev_c]["conjunto"]
        resultado = A & B & C; expr = f"{ev_a} ∩ {ev_b} ∩ {ev_c}"; color_res = "#16a085"
    elif operacion == "Unión triple (A ∪ B ∪ C)":
        C = eventos[ev_c]["conjunto"]
        resultado = A | B | C; expr = f"{ev_a} ∪ {ev_b} ∪ {ev_c}"; color_res = "#9b59b6"

    st.markdown("---")
    st.subheader("📊 4. Resultados")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Resultado", expr)
    with col_r2:
        st.metric("|Resultado|", len(resultado))
    with col_r3:
        prob = len(resultado) / len(U) if len(U) > 0 else 0
        st.metric("P(resultado)", f"{prob:.4f}")

    st.markdown(f"**Conjunto resultado:** {{{', '.join(map(str, sorted(resultado)))}}}")

    st.markdown("---")
    st.subheader("📋 5. Probabilidades de cada evento")
    tabla_probs = []
    for nombre, info in eventos.items():
        cjto = info["conjunto"]
        tabla_probs.append({
            "Evento": nombre, "Definición": info["plantilla"],
            "Elementos": ", ".join(map(str, sorted(cjto))),
            "|Evento|": len(cjto),
            "P(Evento)": f"{len(cjto)/len(U):.4f}" if len(U) > 0 else "0"})
    st.dataframe(pd.DataFrame(tabla_probs), use_container_width=True)

    st.markdown("---")
    st.subheader("🎨 6. Diagrama de Venn")

    def crear_venn_2(A, B, U, nombre_a, nombre_b, operacion, resultado,
                      color_a, color_b, color_res):
        """Diagrama de Venn 2 círculos centrado y proporcionado."""
        fig = go.Figure()
        fig.add_shape(type="rect", x0=-2.0, y0=-1.3, x1=2.0, y1=1.2,
                      line=dict(color="black", width=2),
                      fillcolor="rgba(245,245,245,0.5)", layer="below")
        fig.add_shape(type="circle", x0=-1.30, y0=-0.75, x1=0.20, y1=0.75,
                      fillcolor=color_a, line=dict(color=color_a, width=3),
                      opacity=0.30, layer="below")
        fig.add_shape(type="circle", x0=-0.20, y0=-0.75, x1=1.30, y1=0.75,
                      fillcolor=color_b, line=dict(color=color_b, width=3),
                      opacity=0.30, layer="below")
        fig.add_annotation(x=-1.55, y=1.02, text=f"<b>{nombre_a}</b>",
                          showarrow=False, font=dict(size=18, color=color_a))
        fig.add_annotation(x=1.55, y=1.02, text=f"<b>{nombre_b}</b>",
                          showarrow=False, font=dict(size=18, color=color_b))
        fig.add_annotation(x=-1.85, y=1.05, text="<b>U</b>",
                          showarrow=False, font=dict(size=16, color="black"))
        solo_a = sorted(A - B)
        solo_b = sorted(B - A)
        ambos = sorted(A & B)
        solo_u = sorted(U - A - B)
        for i, elem in enumerate(solo_a):
            cols = 2
            x = -1.05 + (i % cols) * 0.22
            y = 0.45 - (i // cols) * 0.28
            en_res = elem in resultado
            fig.add_annotation(x=x, y=y, text=f"<b>{elem}</b>", showarrow=False,
                              font=dict(size=13, color="white" if en_res else "#1a1a1a"),
                              bgcolor=color_res if en_res else "white",
                              bordercolor="black", borderwidth=1.2, borderpad=4)
        for i, elem in enumerate(solo_b):
            cols = 2
            x = 0.83 + (i % cols) * 0.22
            y = 0.45 - (i // cols) * 0.28
            en_res = elem in resultado
            fig.add_annotation(x=x, y=y, text=f"<b>{elem}</b>", showarrow=False,
                              font=dict(size=13, color="white" if en_res else "#1a1a1a"),
                              bgcolor=color_res if en_res else "white",
                              bordercolor="black", borderwidth=1.2, borderpad=4)
        for i, elem in enumerate(ambos):
            y = 0.45 - i * 0.28
            en_res = elem in resultado
            fig.add_annotation(x=-0.11, y=y, text=f"<b>{elem}</b>", showarrow=False,
                              font=dict(size=13, color="white" if en_res else "#1a1a1a"),
                              bgcolor=color_res if en_res else "white",
                              bordercolor="black", borderwidth=1.2, borderpad=4)
        # Elementos SOLO EN U: distribuir uniformemente en las esquinas y laterales
        n_u = len(solo_u)
        esquinas = [
            (-1.75, 0.95), (1.75, 0.95), (-1.75, -1.05), (1.75, -1.05),
            (-1.75, 0.0), (1.75, 0.0),
            (-0.9, 0.95), (0.9, 0.95), (-0.9, -1.05), (0.9, -1.05),
        ]
        for i, elem in enumerate(solo_u):
            x, y = esquinas[i % len(esquinas)]
            fig.add_annotation(x=x, y=y, text=f"{elem}", showarrow=False,
                              font=dict(size=12, color="#555"),
                              bgcolor="white", bordercolor="#999",
                              borderwidth=1, borderpad=3)
        fig.update_layout(
            title=f"<b>Diagrama de Venn:</b> {expr} = "
                  f"{{{', '.join(map(str, sorted(resultado)))}}}",
            xaxis=dict(range=[-2.1, 2.1], showgrid=False, zeroline=False,
                       visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-1.4, 1.3], showgrid=False, zeroline=False,
                       visible=False),
            height=560, plot_bgcolor="white",
            margin=dict(l=20, r=20, t=60, b=20))
        return fig

    eventos_venn = [ev_a, ev_b]
    if "triple" in operacion.lower():
        eventos_venn.append(ev_c)

    if len(eventos_venn) == 2:
        fig_venn = crear_venn_2(
            eventos[ev_a]["conjunto"], eventos[ev_b]["conjunto"],
            U, ev_a, ev_b, operacion, resultado,
            eventos[ev_a]["color"], eventos[ev_b]["color"], color_res)
        st.plotly_chart(fig_venn, use_container_width=True)
    else:
        st.info("ℹ️ El diagrama de Venn interactivo se muestra para 2 eventos.")

    st.markdown("---")
    st.subheader("🎬 7. Animación: construcción paso a paso")

    def anim_operacion_eventos(A, B, U, nombre_a, nombre_b, operacion, resultado,
                                color_a, color_b, color_res, n_frames=24):
        """Animación que construye el resultado paso a paso (empieza vacío)."""
        frames = []
        elementos_res = sorted(resultado)
        n_elem = len(elementos_res)
        if n_elem == 0:
            n_elem = 1
        def shapes_base():
            return [
                dict(type="rect", x0=-2.0, y0=-1.3, x1=2.0, y1=1.2,
                     line=dict(color="black", width=2),
                     fillcolor="rgba(245,245,245,0.5)", layer="below"),
                dict(type="circle", x0=-1.30, y0=-0.75, x1=0.20, y1=0.75,
                     fillcolor=color_a, line=dict(color=color_a, width=3),
                     opacity=0.30, layer="below"),
                dict(type="circle", x0=-0.20, y0=-0.75, x1=1.30, y1=0.75,
                     fillcolor=color_b, line=dict(color=color_b, width=3),
                     opacity=0.30, layer="below"),
            ]
        def anotaciones_base():
            return [
                dict(x=-1.55, y=1.02, text=f"<b>{nombre_a}</b>",
                     showarrow=False, font=dict(size=18, color=color_a)),
                dict(x=1.55, y=1.02, text=f"<b>{nombre_b}</b>",
                     showarrow=False, font=dict(size=18, color=color_b)),
                dict(x=-1.85, y=1.05, text="<b>U</b>",
                     showarrow=False, font=dict(size=16, color="black")),
            ]
        solo_a = sorted(A - B)
        solo_b = sorted(B - A)
        ambos = sorted(A & B)
        solo_u = sorted(U - A - B)
        posiciones = {}
        for i, elem in enumerate(solo_a):
            cols = 2
            x = -1.05 + (i % cols) * 0.22
            y = 0.45 - (i // cols) * 0.28
            posiciones[elem] = (x, y)
        for i, elem in enumerate(solo_b):
            cols = 2
            x = 0.83 + (i % cols) * 0.22
            y = 0.45 - (i // cols) * 0.28
            posiciones[elem] = (x, y)
        for i, elem in enumerate(ambos):
            x = -0.11
            y = 0.45 - i * 0.28
            posiciones[elem] = (x, y)
        n_u = len(solo_u)
        esquinas = [
            (-1.75, 0.95), (1.75, 0.95), (-1.75, -1.05), (1.75, -1.05),
            (-1.75, 0.0), (1.75, 0.0),
            (-0.9, 0.95), (0.9, 0.95), (-0.9, -1.05), (0.9, -1.05),
        ]
        for i, elem in enumerate(solo_u):
            posiciones[elem] = esquinas[i % len(esquinas)]
        for frame_idx in range(n_elem + 1):
            elementos_visibles = set(elementos_res[:frame_idx])
            anots = anotaciones_base()
            for elem in sorted(U):
                if elem not in posiciones:
                    continue
                x, y = posiciones[elem]
                en_res = elem in elementos_visibles
                es_del_resultado = elem in resultado
                if es_del_resultado:
                    bg = color_res if en_res else "white"
                    fc = "white" if en_res else "#1a1a1a"
                else:
                    bg = "white"
                    fc = "#1a1a1a"
                anots.append(dict(
                    x=x, y=y, text=f"<b>{elem}</b>", showarrow=False,
                    font=dict(size=13, color=fc),
                    bgcolor=bg, bordercolor="black",
                    borderwidth=1.2, borderpad=4))
            frames.append(go.Frame(
                data=[],
                layout=go.Layout(
                    shapes=shapes_base(),
                    annotations=anots,
                    title_text=f"<b>Paso {frame_idx}/{n_elem}:</b> {expr} → "
                               f"{{{', '.join(map(str, sorted(elementos_visibles)))}}} "
                               f"({len(elementos_visibles)}/{n_elem})"),
                name=f"f{frame_idx}"))
        fig = go.Figure(
            data=[], frames=frames,
            layout=go.Layout(
                title_text=f"<b>Paso 0/{n_elem}:</b> Construyendo {expr}",
                xaxis=dict(range=[-2.1, 2.1], showgrid=False, zeroline=False,
                           visible=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[-1.4, 1.3], showgrid=False, zeroline=False,
                           visible=False),
                height=560, plot_bgcolor="white",
                shapes=frames[0].layout.shapes,
                annotations=frames[0].layout.annotations,
                updatemenus=[dict(
                    type="buttons", direction="left", x=0.5, y=1.10,
                    xanchor="center", yanchor="top", showactive=False,
                    buttons=[
                        dict(label="▶️ Play", method="animate",
                             args=[None, {"frame": {"duration": 450, "redraw": True},
                                           "fromcurrent": True,
                                           "transition": {"duration": 0}}]),
                        dict(label="⏸️ Pause", method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate"}])
                    ])],
                margin=dict(l=20, r=20, t=70, b=20)))
        return fig

    if st.button("🎬 Generar Animación", key="btn_anim_evt"):
        with st.spinner("Creando animación..."):
            fig_anim = anim_operacion_eventos(
                eventos[ev_a]["conjunto"], eventos[ev_b]["conjunto"],
                U, ev_a, ev_b, operacion, resultado,
                eventos[ev_a]["color"], eventos[ev_b]["color"], color_res)
            st.session_state.historial["anim_evt"] = fig_anim

    if "anim_evt" in st.session_state.historial:
        st.plotly_chart(st.session_state.historial["anim_evt"], use_container_width=True)

    st.markdown("---")
    st.subheader("📚 8. Propiedades y axiomas de probabilidad")

    with st.expander("📖 Ver propiedades y axiomas", expanded=False):
        st.markdown(r"""
        **Axiomas de Kolmogorov:**
        1. P(A) >= 0 para todo evento A.
        2. P(U) = 1.
        3. Si A ∩ B = ∅, entonces P(A ∪ B) = P(A) + P(B).

        **Propiedades derivadas:**
        - P(Aᶜ) = 1 - P(A)
        - P(∅) = 0
        - P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
        - Si A ⊆ B, entonces P(A) <= P(B)
        """)
        st.markdown("**Verificación con los eventos actuales:**")
        pa = len(A) / len(U) if len(U) > 0 else 0
        pb = len(B) / len(U) if len(U) > 0 else 0
        pab = len(A & B) / len(U) if len(U) > 0 else 0
        paub = len(A | B) / len(U) if len(U) > 0 else 0
        st.write(f"- P({ev_a}) = {pa:.4f}")
        st.write(f"- P({ev_b}) = {pb:.4f}")
        st.write(f"- P({ev_a} ∩ {ev_b}) = {pab:.4f}")
        st.write(f"- P({ev_a} ∪ {ev_b}) = {paub:.4f}")
        st.write(f"- **Verificación:** P(A) + P(B) - P(A∩B) = {pa:.4f} + {pb:.4f} - {pab:.4f} = {pa + pb - pab:.4f} ≈ P(A∪B) = {paub:.4f}")
        if abs((pa + pb - pab) - paub) < 1e-6:
            st.success("✅ Se cumple: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)")
        else:
            st.warning("⚠️ Revisar el cálculo")

    st.markdown("---")
    st.subheader("💾 9. Guardar resultados")
    df_export = pd.DataFrame({
        "Evento/Operación": [ev_a, ev_b] + ([ev_c] if "triple" in operacion.lower() else []) + [expr],
        "Elementos": [
            ", ".join(map(str, sorted(eventos[ev_a]["conjunto"]))),
            ", ".join(map(str, sorted(eventos[ev_b]["conjunto"]))),
        ] + ([", ".join(map(str, sorted(eventos[ev_c]["conjunto"])))] if "triple" in operacion.lower() else []) + [
            ", ".join(map(str, sorted(resultado)))
        ],
        "Cardinalidad": [
            len(eventos[ev_a]["conjunto"]), len(eventos[ev_b]["conjunto"]),
        ] + ([len(eventos[ev_c]["conjunto"])] if "triple" in operacion.lower() else []) + [
            len(resultado)
        ],
        "Probabilidad": [
            f"{len(eventos[ev_a]['conjunto'])/len(U):.4f}",
            f"{len(eventos[ev_b]['conjunto'])/len(U):.4f}",
        ] + ([f"{len(eventos[ev_c]['conjunto'])/len(U):.4f}"] if "triple" in operacion.lower() else []) + [
            f"{len(resultado)/len(U):.4f}"
        ]
    })
    st.dataframe(df_export, use_container_width=True)
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar resultados (CSV)", data=csv,
        file_name="operaciones_eventos.csv", mime="text/csv", key="dl_evt")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🎬 Simulador con Animaciones Plotly - v5.5.0<br>
    13 pestañas · Con distribuciones teóricas y operaciones con eventos
</div>
""", unsafe_allow_html=True)
