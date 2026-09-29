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

st.set_page_config(
    page_title="🎬 Simulador Animado v5.1",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Simulador de Experimentos Aleatorios con Animaciones")
st.markdown("*Animaciones interactivas con Plotly - Play, Pause y Scrub*")
st.markdown("---")

# ============================================================
# SESSION STATE
# ============================================================
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
    """Evalúa una mano de 5 cartas y devuelve la categoría."""
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
    if es_color and es_escalera and valores[0] == 14:
        return "Escalera Real"
    if es_color and es_escalera:
        return "Escalera de Color"
    if frec == [4, 1]:
        return "Póker"
    if frec == [3, 2]:
        return "Full House"
    if es_color:
        return "Color"
    if es_escalera:
        return "Escalera"
    if frec == [3, 1, 1]:
        return "Trío"
    if frec == [2, 2, 1]:
        return "Doble Pareja"
    if frec == [2, 1, 1, 1]:
        return "Pareja"
    return "Carta Alta"

PROB_TEORICAS_POKER = {
    "Escalera Real": 1 / 649740,
    "Escalera de Color": 10 / 649740,
    "Póker": 624 / 2598960,
    "Full House": 3744 / 2598960,
    "Color": 5108 / 2598960,
    "Escalera": 10200 / 2598960,
    "Trío": 54912 / 2598960,
    "Doble Pareja": 123552 / 2598960,
    "Pareja": 1098240 / 2598960,
    "Carta Alta": 1302540 / 2598960,
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
# ANIMACIONES PLOTLY
# ============================================================

def anim_moneda(n=30, prob_cara=0.5):
    resultados = np.random.choice(["CARA", "SELLO"], size=n,
                                   p=[prob_cara, 1 - prob_cara])
    frames = []
    for i in range(n):
        caras = int(np.sum(resultados[:i+1] == "CARA"))
        sellos = int(np.sum(resultados[:i+1] == "SELLO"))
        color = "#3498db" if resultados[i] == "CARA" else "#e74c3c"
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[0.5], y=[0.5], mode="markers+text",
                    marker=dict(size=120, color=color, line=dict(color="black", width=2)),
                    text=[resultados[i]], textposition="middle center",
                    textfont=dict(color="white", size=16, family="Arial Black"),
                    showlegend=False, hoverinfo="skip"),
                go.Bar(x=["CARA", "SELLO"], y=[caras, sellos],
                    marker_color=["#3498db", "#e74c3c"],
                    text=[caras, sellos], textposition="outside",
                    showlegend=False, hoverinfo="skip")
            ],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🪙 Lanzamiento {i+1}/{n}: {resultados[i]}")
        ))
    fig = go.Figure(data=frames[0].data, frames=frames,
        layout=go.Layout(title_text=f"🪙 Lanzamiento 1/{n}",
            xaxis=dict(range=[0, 1], visible=False),
            yaxis=dict(range=[0, 1], visible=False),
            xaxis2=dict(range=[-0.5, 1.5], domain=[0.6, 1.0], title=""),
            yaxis2=dict(range=[0, max(10, n)], domain=[0.1, 0.9],
                        anchor="x2", title="Frecuencia"),
            updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 300, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ])],
            height=450, showlegend=False))
    return fig


def anim_dado(n=30, caras=6):
    resultados = np.random.randint(1, caras + 1, size=n)
    frames = []
    for i in range(n):
        counts = [int(np.sum(resultados[:i+1] == k)) for k in range(1, caras + 1)]
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[0.5], y=[0.5], mode="markers+text",
                    marker=dict(size=120, color="#e67e22",
                                line=dict(color="black", width=2), symbol="square"),
                    text=[str(resultados[i])], textposition="middle center",
                    textfont=dict(color="white", size=28, family="Arial Black"),
                    showlegend=False, hoverinfo="skip"),
                go.Bar(x=list(range(1, caras + 1)), y=counts,
                    marker_color="skyblue", marker_line_color="black",
                    marker_line_width=1.5, text=counts, textposition="outside",
                    showlegend=False, hoverinfo="skip")
            ],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🎲 Lanzamiento {i+1}/{n}: {resultados[i]}")
        ))
    fig = go.Figure(data=frames[0].data, frames=frames,
        layout=go.Layout(title_text=f"🎲 Lanzamiento 1/{n}",
            xaxis=dict(range=[0, 1], visible=False),
            yaxis=dict(range=[0, 1], visible=False),
            xaxis2=dict(range=[0.5, caras + 0.5], domain=[0.6, 1.0], title="Cara"),
            yaxis2=dict(range=[0, max(10, n)], domain=[0.1, 0.9],
                        anchor="x2", title="Frecuencia"),
            updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 300, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ])],
            height=450, showlegend=False))
    return fig


def anim_moneda_dado(n=20):
    monedas = np.random.choice(["CARA", "SELLO"], size=n)
    dados = np.random.randint(1, 7, size=n)
    frames = []
    for i in range(n):
        color = "#3498db" if monedas[i] == "CARA" else "#e74c3c"
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[0.5], y=[0.5], mode="markers+text",
                    marker=dict(size=120, color=color, line=dict(color="black", width=2)),
                    text=[monedas[i]], textposition="middle center",
                    textfont=dict(color="white", size=14, family="Arial Black"),
                    showlegend=False, hoverinfo="skip"),
                go.Scatter(x=[0.5], y=[0.5], mode="markers+text",
                    marker=dict(size=120, color="#e67e22",
                                line=dict(color="black", width=2), symbol="square"),
                    text=[str(dados[i])], textposition="middle center",
                    textfont=dict(color="white", size=28, family="Arial Black"),
                    showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2")
            ],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🪙🎲 Lanzamiento {i+1}: {monedas[i]} + {dados[i]}")
        ))
    fig = go.Figure(data=frames[0].data, frames=frames,
        layout=go.Layout(title_text=f"🪙🎲 Lanzamiento 1/{n}",
            xaxis=dict(range=[0, 1], visible=False, domain=[0, 0.45]),
            yaxis=dict(range=[0, 1], visible=False),
            xaxis2=dict(range=[0, 1], visible=False, domain=[0.55, 1.0]),
            yaxis2=dict(range=[0, 1], visible=False, anchor="x2"),
            updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 400, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ])],
            height=450, showlegend=False))
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
        frames.append(go.Frame(
            data=[go.Scatter(x=xs, y=[0.5]*n_mostrar, mode="markers+text",
                marker=dict(size=60, color="white",
                            line=dict(color=colores, width=3), symbol="square"),
                text=cartas_vis, textposition="middle center",
                textfont=dict(color=colores, size=13, family="Arial Black"),
                showlegend=False, hoverinfo="skip")],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🃏 Extracción {i+1}/{n}: {extracciones[i]}")
        ))
    fig = go.Figure(data=frames[0].data, frames=frames,
        layout=go.Layout(title_text=f"🃏 Extracción 1/{n}",
            xaxis=dict(range=[-0.5, 8.5], visible=False),
            yaxis=dict(range=[0, 1], visible=False),
            updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 400, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ])],
            height=350, showlegend=False))
    return fig


def anim_baloto(n_sorteos=8, n_balotas=5, rango_max=43):
    sorteos = []
    for _ in range(n_sorteos):
        nums = sorted(np.random.choice(range(1, rango_max + 1),
                                        size=n_balotas, replace=False).tolist())
        extra = int(np.random.choice(range(1, rango_max + 1)))
        sorteos.append((nums, extra))
    frames = []
    for i, (nums, extra) in enumerate(sorteos):
        xs = list(range(len(nums) + 1))
        ys = [0.5] * (len(nums) + 1)
        textos = [str(x) for x in nums] + [str(extra)]
        colores = ["#3498db"] * len(nums) + ["#e74c3c"]
        frames.append(go.Frame(
            data=[go.Scatter(x=xs, y=ys, mode="markers+text",
                marker=dict(size=55, color=colores, line=dict(color="black", width=2)),
                text=textos, textposition="middle center",
                textfont=dict(color="white", size=13, family="Arial Black"),
                showlegend=False, hoverinfo="skip")],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🎰 Sorteo {i+1}/{n_sorteos}")
        ))
    fig = go.Figure(data=frames[0].data, frames=frames,
        layout=go.Layout(title_text=f"🎰 Sorteo 1/{n_sorteos}",
            xaxis=dict(range=[-0.5, n_balotas + 0.5], visible=False),
            yaxis=dict(range=[0, 1], visible=False),
            updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 700, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ])],
            height=350, showlegend=False))
    return fig


def anim_ruleta(n_giros=15, tipo="europea"):
    if tipo == "europea":
        casillas = list(range(0, 37))
    else:
        casillas = list(range(0, 37)) + ["00"]
    rojos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
    tiradas = np.random.choice(casillas, size=n_giros)

    def color_num(t):
        if t == 0 or t == "00":
            return "#27ae60"
        return "#e74c3c" if t in rojos else "#2c3e50"

    saldos = [0]
    for t in tiradas:
        saldos.append(saldos[-1] + (1 if t in rojos else -1))

    frames = []
    for i in range(n_giros):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[0.5], y=[0.5], mode="markers+text",
                    marker=dict(size=100, color=color_num(tiradas[i]),
                                line=dict(color="black", width=3)),
                    text=[str(tiradas[i])], textposition="middle center",
                    textfont=dict(color="white", size=20, family="Arial Black"),
                    showlegend=False, hoverinfo="skip"),
                go.Scatter(x=list(range(i+2)), y=saldos[:i+2],
                    mode="lines+markers", line=dict(color="#e67e22", width=3),
                    marker=dict(size=8), showlegend=False, hoverinfo="skip",
                    xaxis="x2", yaxis="y2")
            ],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🎡 Giro {i+1}/{n_giros} → {tiradas[i]}")
        ))
    fig = go.Figure(data=frames[0].data, frames=frames,
        layout=go.Layout(title_text=f"🎡 Giro 1/{n_giros}",
            xaxis=dict(range=[0, 1], visible=False, domain=[0, 0.45]),
            yaxis=dict(range=[0, 1], visible=False),
            xaxis2=dict(range=[0, n_giros+1], domain=[0.55, 1.0], title="Giro"),
            yaxis2=dict(domain=[0.1, 0.9], anchor="x2", title="Saldo"),
            updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 500, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ])],
            height=450, showlegend=False))
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
        ys = [0.5] * 6
        textos = [str(x) for x in nums] + [str(pb)]
        colores = ["#3498db"] * 5 + ["#e74c3c"]
        frames.append(go.Frame(
            data=[go.Scatter(x=xs, y=ys, mode="markers+text",
                marker=dict(size=55, color=colores, line=dict(color="black", width=2)),
                text=textos, textposition="middle center",
                textfont=dict(color="white", size=12, family="Arial Black"),
                showlegend=False, hoverinfo="skip")],
            name=f"f{i}",
            layout=go.Layout(title_text=f"💥 Sorteo {i+1}/{n_sorteos}")
        ))
    fig = go.Figure(data=frames[0].data, frames=frames,
        layout=go.Layout(title_text=f"💥 Sorteo 1/{n_sorteos}",
            xaxis=dict(range=[-0.5, 5.5], visible=False),
            yaxis=dict(range=[0, 1], visible=False),
            updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 700, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ])],
            height=350, showlegend=False))
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
        frames.append(go.Frame(
            data=[
                go.Scatter(x=xa[da], y=ya[da], mode="markers",
                    marker=dict(color="#27ae60", size=5, opacity=0.6),
                    showlegend=False, hoverinfo="skip"),
                go.Scatter(x=xa[~da], y=ya[~da], mode="markers",
                    marker=dict(color="#e74c3c", size=5, opacity=0.6),
                    showlegend=False, hoverinfo="skip"),
                go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines",
                    line=dict(color="blue", width=3),
                    showlegend=False, hoverinfo="skip")
            ],
            name=f"f{f}",
            layout=go.Layout(title_text=f"🥧 {k} puntos | π ≈ {pi_est:.4f}")
        ))
    fig = go.Figure(data=frames[0].data, frames=frames,
        layout=go.Layout(title_text="🥧 Monte Carlo π",
            xaxis=dict(range=[-1.1, 1.1], scaleanchor="y", scaleratio=1,
                       showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[-1.1, 1.1], showgrid=False, zeroline=False, visible=False),
            updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 80, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ])],
            height=600, showlegend=False, plot_bgcolor="white"))
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
        frames.append(go.Frame(
            data=[
                go.Scatter(x=xs, y=ys, mode="markers+text",
                    marker=dict(size=30, color=colores,
                                line=dict(color="black", width=1.5)),
                    text=[str(m) for m in meses],
                    textposition="middle center",
                    textfont=dict(color="white", size=9),
                    showlegend=False, hoverinfo="skip"),
                go.Scatter(x=list(range(1, i + 2)),
                    y=list(coincidencias_acum[:i+1] / np.arange(1, i + 2)),
                    mode="lines", line=dict(color="#9b59b6", width=3),
                    showlegend=False, hoverinfo="skip",
                    xaxis="x2", yaxis="y2")
            ],
            name=f"f{i}",
            layout=go.Layout(
                title_text=f"🎂 Simulación {i+1}/{n_simulaciones} - "
                           f"{'✅ Coincidencia' if coincidencias[i] else '❌ Sin coincidencia'} "
                           f"| Prob: {prob:.3f}")
        ))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                        specs=[[{"type": "xy"}, {"type": "xy"}]])
    for trace in frames[0].data:
        if hasattr(trace, "xaxis") and trace.xaxis == "x2":
            fig.add_trace(trace, row=1, col=2)
        else:
            fig.add_trace(trace, row=1, col=1)
    fig.frames = frames
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Simulación", row=1, col=2)
    fig.update_yaxes(title_text="Prob. estimada", range=[0, 1], row=1, col=2)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_layout(
        title_text="🎂 Problema del Cumpleaños",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.15,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 120, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ])],
        height=500, showlegend=False)
    return fig


# ============================================================
# ANIMACIÓN PÓKER
# ============================================================

def anim_poker(n_manos=15):
    baraja = crear_baraja()
    manos = []
    tipos = []
    for _ in range(n_manos):
        mano = list(np.random.choice(baraja, size=5, replace=False))
        tipo = evaluar_mano(mano)
        manos.append(mano)
        tipos.append(tipo)
    frames = []
    conteo_acum = Counter()
    for i in range(n_manos):
        conteo_acum[tipos[i]] += 1
        mano = manos[i]
        xs = list(range(5))
        ys = [0.7] * 5
        colores_cartas = ["#e74c3c" if c[-1] in ["♥", "♦"] else "#2c3e50" for c in mano]
        tipos_ordenados = [t for t in ORDEN_MANOS if conteo_acum.get(t, 0) > 0]
        valores = [conteo_acum[t] for t in tipos_ordenados]
        colores_barras = [COLORES_MANOS[t] for t in tipos_ordenados]
        frames.append(go.Frame(
            data=[
                go.Scatter(x=xs, y=ys, mode="markers+text",
                    marker=dict(size=70, color="white",
                                line=dict(color=colores_cartas, width=4),
                                symbol="square"),
                    text=mano, textposition="middle center",
                    textfont=dict(color=colores_cartas, size=15, family="Arial Black"),
                    showlegend=False, hoverinfo="skip"),
                go.Bar(x=tipos_ordenados, y=valores,
                    marker_color=colores_barras,
                    text=valores, textposition="outside",
                    showlegend=False, hoverinfo="skip",
                    xaxis="x2", yaxis="y2")
            ],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🃏 Mano {i+1}/{n_manos}: {tipos[i]}")
        ))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Mano actual", "Conteo acumulado de manos"))
    for trace in frames[0].data:
        if hasattr(trace, "xaxis") and trace.xaxis == "x2":
            fig.add_trace(trace, row=1, col=2)
        else:
            fig.add_trace(trace, row=1, col=1)
    fig.frames = frames
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Tipo de mano", tickangle=-45, row=1, col=2)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
    fig.update_layout(
        title_text=f"🃏 Póker - {n_manos} manos",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.15,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 600, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ])],
        height=500, showlegend=False)
    return fig


# ============================================================
# ANIMACIÓN MONTY HALL
# ============================================================

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
        colores = ["#f39c12"] * 3
        colores[abierta] = "#7f8c8d"
        simbolos = ["🚪"] * 3
        simbolos[abierta] = "🐐"
        if ef == premio:
            simbolos[ef] = "🏆"
        else:
            simbolos[ef] = "🐐"
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[0, 1, 2], y=[0.5, 0.5, 0.5], mode="markers+text",
                    marker=dict(size=100, color=colores,
                                line=dict(color="black", width=3), symbol="square"),
                    text=simbolos, textposition="middle center",
                    textfont=dict(size=30),
                    showlegend=False, hoverinfo="skip"),
                go.Bar(x=["Ganadas", "Perdidas"],
                    y=[ganadas, (i+1) - ganadas],
                    marker_color=["#27ae60", "#e74c3c"],
                    text=[ganadas, (i+1) - ganadas], textposition="outside",
                    showlegend=False, hoverinfo="skip",
                    xaxis="x2", yaxis="y2"),
                go.Scatter(x=[0, 1], y=[prob, prob],
                    mode="lines", line=dict(color="#9b59b6", width=4),
                    showlegend=False, hoverinfo="skip",
                    xaxis="x3", yaxis="y3")
            ],
            name=f"f{i}",
            layout=go.Layout(
                title_text=f"🚪 Partida {i+1}/{n_partidas}: "
                           f"{'✅ Ganaste 🏆' if gano else '❌ Perdiste 🐐'} "
                           f"| Prob. acum: {prob:.3f}")
        ))
    fig = make_subplots(rows=2, cols=2, column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5],
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=("Puertas", "Resultado acumulado",
                        "Probabilidad de ganar (acumulada)"))
    for trace in frames[0].data:
        if hasattr(trace, "xaxis") and trace.xaxis == "x2":
            fig.add_trace(trace, row=1, col=2)
        elif hasattr(trace, "xaxis") and trace.xaxis == "x3":
            fig.add_trace(trace, row=2, col=1)
        else:
            fig.add_trace(trace, row=1, col=1)
    fig.frames = frames
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="Partidas", row=1, col=2)
    fig.update_xaxes(range=[-0.1, 1.1], showticklabels=False, row=2, col=1)
    fig.update_yaxes(range=[0, 1], title_text="P(ganar)", row=2, col=1)
    fig.add_hline(y=2/3 if cambiar else 1/3, line_dash="dash",
                  line_color="red", row=2, col=1,
                  annotation_text=f"Teórica = {2/3 if cambiar else 1/3:.3f}")
    fig.update_layout(
        title_text=f"🚪 Monty Hall - Estrategia: {'CAMBIAR' if cambiar else 'NO CAMBIAR'}",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.12,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 700, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ])],
        height=700, showlegend=False)
    return fig


# ============================================================
# ANIMACIÓN LEY DE LOS GRANDES NÚMEROS
# ============================================================

def anim_ley_grandes_numeros(n_max=1000, tipo="Moneda (CARA)", n_frames=50):
    """Animación: convergencia de la frecuencia relativa a la probabilidad teórica."""
    if tipo == "Moneda (CARA)":
        resultados = np.random.choice([1, 0], size=n_max, p=[0.5, 0.5])
        prob_teo = 0.5
        etiqueta = "P(CARA) = 0.5"
        color = "#3498db"
    elif tipo == "Dado (6)":
        resultados = (np.random.randint(1, 7, n_max) == 6).astype(int)
        prob_teo = 1/6
        etiqueta = "P(6) = 1/6 ≈ 0.1667"
        color = "#e67e22"
    elif tipo == "Dado (par)":
        resultados = (np.random.randint(1, 7, n_max) % 2 == 0).astype(int)
        prob_teo = 0.5
        etiqueta = "P(par) = 0.5"
        color = "#27ae60"
    else:  # Carta roja
        baraja = crear_baraja()
        rojas = [c for c in baraja if c[-1] in ["♥", "♦"]]
        resultados = np.array([1 if c in rojas else 0
                                for c in np.random.choice(baraja, n_max)])
        prob_teo = 0.5
        etiqueta = "P(roja) = 0.5"
        color = "#e74c3c"

    frames = []
    for f in range(n_frames):
        k = int((f + 1) * n_max / n_frames)
        acumulada = np.cumsum(resultados[:k]) / np.arange(1, k + 1)
        # Contar victorias/derrotas
        exitos = int(np.sum(resultados[:k]))
        fracasos = k - exitos
        frames.append(go.Frame(
            data=[
                go.Scatter(x=np.arange(1, k + 1), y=acumulada,
                    mode="lines", line=dict(color=color, width=3),
                    name="Frec. relativa",
                    showlegend=False, hoverinfo="skip"),
                go.Scatter(x=[1, k], y=[prob_teo, prob_teo],
                    mode="lines", line=dict(color="red", width=3, dash="dash"),
                    name="Teórica",
                    showlegend=False, hoverinfo="skip"),
                go.Bar(x=["Éxitos", "Fallos"], y=[exitos, fracasos],
                    marker_color=[color, "#95a5a6"],
                    text=[exitos, fracasos], textposition="outside",
                    showlegend=False, hoverinfo="skip",
                    xaxis="x2", yaxis="y2")
            ],
            name=f"f{f}",
            layout=go.Layout(
                title_text=f"📊 {k} ensayos | Frec = {acumulada[-1]:.4f} | "
                           f"Teórica = {prob_teo:.4f} | Error = {abs(acumulada[-1]-prob_teo):.4f}"
            )
        ))

    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Convergencia", "Resultados acumulados"))
    for trace in frames[0].data:
        if hasattr(trace, "xaxis") and trace.xaxis == "x2":
            fig.add_trace(trace, row=1, col=2)
        else:
            fig.add_trace(trace, row=1, col=1)
    fig.frames = frames
    fig.update_xaxes(title_text="Número de ensayos", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia relativa", row=1, col=1)
    fig.update_xaxes(title_text="Resultado", row=1, col=2)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
    fig.update_layout(
        title_text=f"📊 Ley de los Grandes Números - {tipo}",
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=1.15,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 100, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ])],
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
    monedas = np.random.choice(["CARA", "SELLO"], size=n_lanzamientos)
    dados = np.random.randint(1, 7, size=n_lanzamientos)
    return monedas, dados

def simulador_cartas(n_cartas=1, n_lanzamientos=100):
    baraja = crear_baraja()
    return np.random.choice(baraja, size=(n_lanzamientos, n_cartas))

def simulador_balotas(n_balotas=6, rango=(1, 43), n_lanzamientos=10):
    res = []
    for _ in range(n_lanzamientos):
        nums = np.random.choice(range(rango[0], rango[1]+1),
                                size=n_balotas+1, replace=False)
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
    if n > 365:
        return 1.0
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
            opciones_finales = [i for i in range(3) if i != eleccion and i != abierta]
            eleccion_final = opciones_finales[0] if opciones_finales else eleccion
        else:
            eleccion_final = eleccion
        gano = (eleccion_final == premio)
        if gano:
            victorias += 1
        historial.append(gano)
    return victorias / n_partidas, historial


# ============================================================
# INTERFAZ - TABS
# ============================================================

tabs = st.tabs([
    "🪙 Monedas", "🎲 Dados", "🪙🎲 Moneda + Dado",
    "🃏 Cartas", "🎰 Baloto", "📊 Ley Grandes Números",
    "🎡 Ruleta", "💥 Powerball", "🥧 Monte Carlo π", "🎂 Cumpleaños",
    "🃏 Póker", "🚪 Monty Hall"
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
        if semilla > 0:
            np.random.seed(int(semilla))
        st.markdown("---")
        n_anim = st.slider("Lanzamientos animados:", 5, 60, 30, key="gif_mon")
        if st.button("🎬 Generar Animación", key="btn_anim_mon"):
            st.session_state.historial["anim_moneda"] = anim_moneda(n_anim, prob_cara)
        if st.button("🎲 Simular (datos)", key="btn_monedas"):
            st.session_state.historial["monedas"] = simulador_monedas(
                n_monedas, n_lanzamientos, prob_cara)
    with col2:
        if "anim_moneda" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_moneda"],
                            use_container_width=True)
        if "monedas" in st.session_state.historial:
            res = st.session_state.historial["monedas"]
            if n_monedas == 1:
                caras = np.sum(res == "CARA"); sellos = np.sum(res == "SELLO")
                t = len(res)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", t)
                c2.metric("Caras", caras, f"{caras/t*100:.2f}%")
                c3.metric("Sellos", sellos, f"{sellos/t*100:.2f}%")
                df_b = pd.DataFrame({"Resultado": ["CARA", "SELLO"],
                                     "Frecuencia": [caras, sellos],
                                     "Prob": [caras/t, sellos/t]})
                fig = px.bar(df_b, x="Resultado", y="Frecuencia", text_auto=True,
                             color="Resultado",
                             color_discrete_map={"CARA": "#3498db", "SELLO": "#e74c3c"})
                fig.update_layout(height=350, showlegend=False,
                                  title="Gráfico de barras - resultados")
                st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2: DADOS ----------------
with tabs[1]:
    st.header("🎲 Simulación con Dados")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_dados = st.slider("Dados:", 1, 10, 1)
        n_lanz = st.number_input("Lanzamientos:", 10, 100000, 500, 50, key="nl_dados")
        caras_d = st.selectbox("Caras:", [4, 6, 8, 10, 12, 20], 1)
        sem_d = st.number_input("Semilla:", 0, 99999, 42, key="sem_dados")
        if sem_d > 0:
            np.random.seed(int(sem_d))
        st.markdown("---")
        n_anim_d = st.slider("Lanzamientos animados:", 5, 60, 30, key="gif_dado")
        if st.button("🎬 Generar Animación", key="btn_anim_dado"):
            st.session_state.historial["anim_dado"] = anim_dado(n_anim_d, caras_d)
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
                df_frec = pd.DataFrame({
                    "Cara": list(range(1, caras_d + 1)),
                    "Frecuencia": [frec.get(i, 0) for i in range(1, caras_d + 1)]
                })
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(valores))
                c2.metric("Media", f"{valores.mean():.3f}")
                c3.metric("Desv.Est.", f"{valores.std():.3f}")
                fig = px.bar(df_frec, x="Cara", y="Frecuencia", text_auto=True,
                             color="Frecuencia", color_continuous_scale="Viridis")
                fig.update_layout(height=380, showlegend=False,
                                  title="Gráfico de barras - frecuencias")
                st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 3: MONEDA + DADO ----------------
with tabs[2]:
    st.header("🪙🎲 Moneda + Dado")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_lanz_md = st.number_input("Lanzamientos:", 10, 100000, 500, 50, key="nl_md")
        sem_md = st.number_input("Semilla:", 0, 99999, 42, key="sem_md")
        if sem_md > 0:
            np.random.seed(int(sem_md))
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
            fig = px.histogram(df, x="Dado", color="Moneda", barmode="group",
                                color_discrete_map={"CARA": "#3498db", "SELLO": "#e74c3c"},
                                title="Distribución conjunta")
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
        if sem_c > 0:
            np.random.seed(int(sem_c))
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
            df_p = pd.DataFrame({"Palo": list(cp.keys()), "Frecuencia": list(cp.values())})
            color_map = {"♠": "#2c3e50", "♥": "#e74c3c",
                         "♦": "#e67e22", "♣": "#27ae60"}
            fig = px.bar(df_p, x="Palo", y="Frecuencia", text_auto=True,
                         color="Palo", color_discrete_map=color_map,
                         title="Gráfico de barras - por palo")
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 5: BALOTO ----------------
with tabs[4]:
    st.header("🎰 Baloto")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_bal = st.slider("Balotas:", 4, 6, 5)
        r_max = st.number_input("Máx. número:", 10, 100, 43)
        n_sort = st.number_input("Sorteos:", 1, 1000, 10)
        sem_b = st.number_input("Semilla:", 0, 99999, 42, key="sem_bal")
        if sem_b > 0:
            np.random.seed(int(sem_b))
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
            df_frec = pd.DataFrame({
                "Número": list(range(1, r_max + 1)),
                "Frecuencia": [frec.get(i, 0) for i in range(1, r_max + 1)]
            })
            fig = px.bar(df_frec, x="Número", y="Frecuencia",
                         color="Frecuencia", color_continuous_scale="Turbo",
                         title="Frecuencia de números")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 6: LEY GRANDES NÚMEROS ----------------
with tabs[5]:
    st.header("📊 Ley de los Grandes Números")
    st.markdown("""
    **Ley de los Grandes Números:** al aumentar el número de ensayos, la **frecuencia relativa**
    de un evento se acerca a su **probabilidad teórica**.

    Ejemplo: si lanzas una moneda 10 veces podrías obtener 7 caras (70%),
    pero si la lanzas 10.000 veces, la proporción se acercará a 50%.
    """)
    col1, col2 = st.columns([1, 2])
    with col1:
        tipo_lgn = st.selectbox("Experimento:",
            ["Moneda (CARA)", "Dado (6)", "Dado (par)", "Carta roja (♥♦)"],
            key="tipo_lgn")
        n_max_lgn = st.number_input("Ensayos:", 100, 50000, 5000, 100, key="n_lgn")
        sem_lgn = st.number_input("Semilla:", 0, 99999, 42, key="sem_lgn")
        if sem_lgn > 0:
            np.random.seed(int(sem_lgn))
        st.markdown("---")
        st.subheader("🎬 Animación")
        n_anim_lgn = st.slider("Frames animados:", 20, 100, 60, key="gif_lgn")
        if st.button("🎬 Generar Animación", key="btn_anim_lgn"):
            with st.spinner("Generando animación..."):
                st.session_state.historial["anim_lgn"] = anim_ley_grandes_numeros(
                    n_max_lgn, tipo_lgn, n_anim_lgn)
        if st.button("🚀 Ejecutar simulación (datos)", key="btn_lgn"):
            if tipo_lgn == "Moneda (CARA)":
                r = np.random.choice([1, 0], n_max_lgn); pt = 0.5
                etiqueta = "P(CARA) = 0.5"
            elif tipo_lgn == "Dado (6)":
                r = (np.random.randint(1, 7, n_max_lgn) == 6).astype(int); pt = 1/6
                etiqueta = "P(6) = 1/6"
            elif tipo_lgn == "Dado (par)":
                r = (np.random.randint(1, 7, n_max_lgn) % 2 == 0).astype(int); pt = 0.5
                etiqueta = "P(par) = 0.5"
            else:
                baraja = crear_baraja()
                rojas = [c for c in baraja if c[-1] in ["♥", "♦"]]
                r = np.array([1 if c in rojas else 0
                              for c in np.random.choice(baraja, n_max_lgn)])
                pt = 0.5
                etiqueta = "P(roja) = 0.5"
            st.session_state.historial["lgn"] = (r, pt, etiqueta, tipo_lgn)

    with col2:
        if "anim_lgn" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_lgn"],
                            use_container_width=True)

        if "lgn" in st.session_state.historial:
            r, pt, etiqueta, tipo_lgn = st.session_state.historial["lgn"]
            acumulada = np.cumsum(r) / np.arange(1, len(r) + 1)
            error = abs(acumulada[-1] - pt)
            c1, c2, c3 = st.columns(3)
            c1.metric("Frecuencia final", f"{acumulada[-1]:.5f}")
            c2.metric("Prob. teórica", f"{pt:.5f}")
            c3.metric("Error absoluto", f"{error:.5f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=acumulada, mode="lines",
                                      name="Frec. relativa",
                                      line=dict(color="#3498db", width=2)))
            fig.add_hline(y=pt, line_dash="dash", line_color="red",
                          annotation_text=etiqueta)
            fig.update_layout(title=f"Convergencia - {tipo_lgn}",
                              xaxis_title="Número de ensayos",
                              yaxis_title="Frecuencia relativa",
                              height=450)
            st.plotly_chart(fig, use_container_width=True)

            # Gráfico de barras de éxitos vs fallos
            exitos = int(np.sum(r))
            fallos = len(r) - exitos
            df_bar = pd.DataFrame({
                "Resultado": ["Éxitos", "Fallos"],
                "Frecuencia": [exitos, fallos],
                "Probabilidad": [exitos/len(r), fallos/len(r)]
            })
            fig_bar = px.bar(
                df_bar, x="Resultado", y="Frecuencia",
                text="Frecuencia", color="Resultado",
                color_discrete_map={"Éxitos": "#27ae60", "Fallos": "#e74c3c"},
                title="Gráfico de barras - Resultados acumulados"
            )
            fig_bar.update_traces(texttemplate="%{text} (%{customdata:.2%})",
                                   customdata=df_bar["Probabilidad"],
                                   textposition="outside")
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Comparación frecuencias a lo largo del tiempo
            st.subheader("📈 Convergencia paso a paso")
            puntos = np.unique(np.logspace(0, np.log10(len(r)), 30).astype(int))
            puntos = puntos[puntos <= len(r)]
            valores = [np.mean(r[:k]) for k in puntos]
            df_conv = pd.DataFrame({"Ensayos": puntos, "Frecuencia": valores})
            fig_conv = px.line(df_conv, x="Ensayos", y="Frecuencia",
                                title="Convergencia (escala logarítmica)",
                                markers=True)
            fig_conv.add_hline(y=pt, line_dash="dash", line_color="red",
                                annotation_text=f"Teórica = {pt:.4f}")
            fig_conv.update_xaxes(type="log")
            fig_conv.update_layout(height=400)
            st.plotly_chart(fig_conv, use_container_width=True)

# ---------------- TAB 7: RULETA ----------------
with tabs[6]:
    st.header("🎡 Ruleta")
    col1, col2 = st.columns([1, 2])
    with col1:
        tipo_r = st.selectbox("Tipo:", ["Europea (0-36)", "Americana (0, 00, 1-36)"])
        n_tir = st.number_input("Tiradas:", 10, 100000, 1000, 100)
        sem_r = st.number_input("Semilla:", 0, 99999, 42, key="sem_r")
        if sem_r > 0:
            np.random.seed(int(sem_r))
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
            total = len(t)
            nr = sum(1 for x in t if x in roj)
            nn = sum(1 for x in t if x in neg)
            nc = sum(1 for x in t if x == 0 or x == "00")
            df_r = pd.DataFrame({"Color": ["Rojo", "Negro", "Verde"],
                                  "Frecuencia": [nr, nn, nc]})
            fig = px.bar(df_r, x="Color", y="Frecuencia", text_auto=True,
                         color="Color",
                         color_discrete_map={"Rojo": "#e74c3c",
                                              "Negro": "#2c3e50",
                                              "Verde": "#27ae60"},
                         title="Gráfico de barras - por color")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 8: POWERBALL ----------------
with tabs[7]:
    st.header("💥 Powerball")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_pb = st.number_input("Sorteos:", 1, 10000, 100)
        sem_pb = st.number_input("Semilla:", 0, 99999, 42, key="sem_pb")
        if sem_pb > 0:
            np.random.seed(int(sem_pb))
        st.markdown("---")
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
            df_frec = pd.DataFrame({
                "Número": list(range(1, 70)),
                "Frecuencia": [frec.get(i, 0) for i in range(1, 70)]
            })
            fig = px.bar(df_frec, x="Número", y="Frecuencia",
                         color="Frecuencia", color_continuous_scale="Turbo",
                         title="Frecuencia de números Powerball")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 9: MONTE CARLO π ----------------
with tabs[8]:
    st.header("🥧 Monte Carlo π")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_pt = st.slider("Puntos:", 100, 100000, 5000, 100)
        sem_pi = st.number_input("Semilla:", 0, 99999, 42, key="sem_pi")
        if sem_pi > 0:
            np.random.seed(int(sem_pi))
        st.markdown("---")
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

# ---------------- TAB 10: CUMPLEAÑOS ----------------
with tabs[9]:
    st.header("🎂 Problema del Cumpleaños")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_per = st.slider("Personas:", 2, 100, 23)
        n_sim = st.number_input("Simulaciones:", 100, 100000, 2000, 100)
        sem_c = st.number_input("Semilla:", 0, 99999, 42, key="sem_cum")
        if sem_c > 0:
            np.random.seed(int(sem_c))
        st.markdown("---")
        n_anim_cum = st.slider("Simulaciones animadas:", 20, 200, 100, 10, key="gif_cum")
        if st.button("🎬 Generar Animación", key="btn_anim_cum"):
            st.session_state.historial["anim_cumple"] = anim_cumpleanos(n_per, n_anim_cum)
        if st.button("🎂 Simular (datos)", key="btn_cum"):
            ps = problema_cumpleanos(n_per, n_sim)
            pt = prob_cumpleanos_teorica(n_per)
            st.session_state.historial["cumple"] = (ps, pt, n_per)
    with col2:
        if "anim_cumple" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_cumple"], use_container_width=True)
        if "cumple" in st.session_state.historial:
            ps, pt, n = st.session_state.historial["cumple"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Simulada", f"{ps*100:.2f}%")
            c2.metric("Teórica", f"{pt*100:.2f}%")
            c3.metric("Error", f"{abs(ps-pt)*100:.3f}%")

# ---------------- TAB 11: PÓKER ----------------
with tabs[10]:
    st.header("🃏 Simulación de Póker - 5 Cartas")
    st.markdown("Se reparten 5 cartas de una baraja estándar (52 cartas) y se evalúa la mano.")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_manos = st.number_input("Número de manos a simular:", 10, 100000, 5000, 100, key="n_poker")
        sem_p = st.number_input("Semilla:", 0, 99999, 42, key="sem_poker")
        if sem_p > 0:
            np.random.seed(int(sem_p))
        st.markdown("---")
        st.subheader("🎬 Animación")
        n_anim_p = st.slider("Manos animadas:", 5, 30, 15, key="gif_poker")
        if st.button("🎬 Generar Animación", key="btn_anim_p"):
            with st.spinner("Generando animación..."):
                st.session_state.historial["anim_poker"] = anim_poker(n_anim_p)
        if st.button("🃏 Simular (datos)", key="btn_poker"):
            with st.spinner("Simulando manos..."):
                conteo, _ = simular_poker(n_manos)
                st.session_state.historial["poker"] = (conteo, n_manos)
    with col2:
        if "anim_poker" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_poker"], use_container_width=True)
        if "poker" in st.session_state.historial:
            conteo, n_manos = st.session_state.historial["poker"]
            st.subheader(f"📊 Resultados de {n_manos:,} manos simuladas")
            filas = []
            for tipo in ORDEN_MANOS:
                sim = conteo.get(tipo, 0)
                prob_sim = sim / n_manos
                prob_teo = PROB_TEORICAS_POKER[tipo]
                filas.append({
                    "Mano": tipo, "Frecuencia": sim,
                    "Prob. Simulada": f"{prob_sim*100:.4f}%",
                    "Prob. Teórica": f"{prob_teo*100:.4f}%",
                    "Diferencia": f"{abs(prob_sim - prob_teo)*100:.4f}%"
                })
            st.dataframe(pd.DataFrame(filas), use_container_width=True)

            df_graf = pd.DataFrame({
                "Mano": ORDEN_MANOS,
                "Simulada": [conteo.get(t, 0) for t in ORDEN_MANOS],
                "Teórica": [PROB_TEORICAS_POKER[t] * n_manos for t in ORDEN_MANOS]
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_graf["Mano"], y=df_graf["Simulada"],
                name="Simulada",
                marker_color=[COLORES_MANOS[t] for t in ORDEN_MANOS],
                text=df_graf["Simulada"], textposition="outside"
            ))
            fig.add_trace(go.Scatter(
                x=df_graf["Mano"], y=df_graf["Teórica"],
                mode="markers+lines", name="Teórica",
                marker=dict(size=12, color="red", symbol="x"),
                line=dict(color="red", width=2, dash="dash")
            ))
            fig.update_layout(
                title=f"🃏 Comparación: Simulada vs Teórica ({n_manos:,} manos)",
                xaxis_title="Tipo de mano",
                yaxis_title="Frecuencia",
                xaxis_tickangle=-45,
                height=500,
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📖 Descripción de cada mano"):
                st.markdown("""
                - **Escalera Real**: A, K, Q, J, 10 del mismo palo (prob. 0.00015%)
                - **Escalera de Color**: 5 cartas consecutivas del mismo palo (0.0015%)
                - **Póker**: 4 cartas del mismo valor (0.024%)
                - **Full House**: 3 de un valor + 2 de otro (0.144%)
                - **Color**: 5 cartas del mismo palo no consecutivas (0.197%)
                - **Escalera**: 5 cartas consecutivas de palos distintos (0.392%)
                - **Trío**: 3 cartas del mismo valor (2.11%)
                - **Doble Pareja**: 2 pares distintos (4.75%)
                - **Pareja**: 2 cartas del mismo valor (42.26%)
                - **Carta Alta**: sin combinación (50.12%)
                """)

# ---------------- TAB 12: MONTY HALL ----------------
with tabs[11]:
    st.header("🚪 Problema de Monty Hall")
    st.markdown("""
    **Reglas del juego:**
    1. Hay 3 puertas: detrás de una hay un **premio** 🏆, detrás de las otras dos hay **cabras** 🐐.
    2. Eliges una puerta.
    3. El presentador (que sabe dónde está el premio) abre una puerta con cabra.
    4. Te da la opción de **cambiar** o **mantener** tu elección.
    5. ¿Conviene cambiar? 📊 La matemática dice: **SÍ, cambia siempre**.
    """)
    col1, col2 = st.columns([1, 2])
    with col1:
        estrategia = st.radio("Estrategia:", ["Cambiar siempre", "Nunca cambiar"],
                               horizontal=True, key="estrategia_mh")
        cambiar_bool = (estrategia == "Cambiar siempre")
        n_part = st.number_input("Número de partidas:", 10, 100000, 5000, 100, key="n_mh")
        sem_mh = st.number_input("Semilla:", 0, 99999, 42, key="sem_mh")
        if sem_mh > 0:
            np.random.seed(int(sem_mh))
        st.markdown("---")
        st.subheader("🎬 Animación")
        n_anim_mh = st.slider("Partidas animadas:", 5, 40, 15, key="gif_mh")
        if st.button("🎬 Generar Animación", key="btn_anim_mh"):
            with st.spinner("Generando animación..."):
                st.session_state.historial["anim_mh"] = anim_monty_hall(n_anim_mh, cambiar_bool)
        if st.button("🚪 Simular (datos)", key="btn_mh"):
            with st.spinner("Simulando partidas..."):
                prob, historial = simular_monty_hall(n_part, cambiar_bool)
                st.session_state.historial["monty_hall"] = (prob, historial, cambiar_bool, n_part)
    with col2:
        if "anim_mh" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_mh"], use_container_width=True)
        if "monty_hall" in st.session_state.historial:
            prob, historial, cambio, n_part = st.session_state.historial["monty_hall"]
            prob_teo = 2/3 if cambio else 1/3
            st.subheader(f"📊 Resultados de {n_part:,} partidas")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prob. Simulada", f"{prob*100:.2f}%")
            c2.metric("Prob. Teórica", f"{prob_teo*100:.2f}%",
                      f"Estrategia: {'CAMBIAR' if cambio else 'NO CAMBIAR'}")
            c3.metric("Error", f"{abs(prob - prob_teo)*100:.3f}%")
            ganadas = int(prob * n_part)
            perdidas = n_part - ganadas
            df_mh = pd.DataFrame({
                "Resultado": ["Ganadas 🏆", "Perdidas 🐐"],
                "Frecuencia": [ganadas, perdidas]
            })
            fig_bar = px.bar(
                df_mh, x="Resultado", y="Frecuencia", text_auto=True,
                color="Resultado",
                color_discrete_map={"Ganadas 🏆": "#27ae60", "Perdidas 🐐": "#e74c3c"},
                title=f"Resultados - Estrategia: {'CAMBIAR' if cambio else 'NO CAMBIAR'}"
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("🔬 Comparación: Cambiar vs No cambiar")
            df_comp = pd.DataFrame({
                "Estrategia": ["Cambiar", "No cambiar"],
                "Prob. Teórica": [2/3, 1/3],
                "Prob. Simulada": [prob if cambio else 1 - prob,
                                   1 - prob if cambio else prob]
            })
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=df_comp["Estrategia"], y=df_comp["Prob. Teórica"] * 100,
                name="Teórica", marker_color="#3498db",
                text=[f"{v*100:.2f}%" for v in df_comp["Prob. Teórica"]],
                textposition="outside"
            ))
            fig_comp.add_trace(go.Bar(
                x=df_comp["Estrategia"], y=df_comp["Prob. Simulada"] * 100,
                name="Simulada", marker_color="#e67e22",
                text=[f"{v*100:.2f}%" for v in df_comp["Prob. Simulada"]],
                textposition="outside"
            ))
            fig_comp.update_layout(
                title="Comparación de estrategias",
                xaxis_title="Estrategia",
                yaxis_title="Probabilidad de ganar (%)",
                barmode="group", height=400
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            hist = np.array(historial, dtype=int)
            acum = np.cumsum(hist) / np.arange(1, len(hist) + 1)
            fig_evol = go.Figure()
            fig_evol.add_trace(go.Scatter(
                y=acum, mode="lines", name="Prob. acumulada",
                line=dict(color="#9b59b6", width=2)
            ))
            fig_evol.add_hline(y=prob_teo, line_dash="dash", line_color="red",
                                annotation_text=f"Teórica = {prob_teo:.4f}")
            fig_evol.update_layout(
                title="Convergencia de la probabilidad",
                xaxis_title="Partida", yaxis_title="P(ganar)", height=400
            )
            st.plotly_chart(fig_evol, use_container_width=True)

            with st.expander("📖 ¿Por qué cambiar duplica la probabilidad?"):
                st.markdown("""
                **Explicación intuitiva:**

                - Si eliges la puerta del premio al inicio (**prob. 1/3**), cambiar te hace **perder**.
                - Si eliges una puerta con cabra al inicio (**prob. 2/3**), el presentador se ve obligado
                  a abrir la otra cabra, y **cambiar te hace ganar**.
                - Por lo tanto, **cambiar gana 2/3 de las veces** y **no cambiar gana 1/3**.

                **Formalización:**
                - P(ganar | cambiar) = 2/3
                - P(ganar | no cambiar) = 1/3
                """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🎬 Simulador con Animaciones Plotly - v5.1<br>
    12 pestañas: monedas, dados, cartas, baloto, ruleta, Powerball, π, cumpleaños, <b>póker</b>, <b>Monty Hall</b> y <b>Ley de los Grandes Números</b>
</div>
""", unsafe_allow_html=True)
