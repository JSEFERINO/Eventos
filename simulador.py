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
    page_title="🎬 Simulador Animado",
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
# ANIMACIONES PLOTLY
# ============================================================

def anim_moneda(n=30, prob_cara=0.5):
    """Animación: moneda + barras acumuladas."""
    resultados = np.random.choice(["CARA", "SELLO"], size=n,
                                   p=[prob_cara, 1 - prob_cara])
    frames = []
    for i in range(n):
        caras = int(np.sum(resultados[:i+1] == "CARA"))
        sellos = int(np.sum(resultados[:i+1] == "SELLO"))
        color = "#3498db" if resultados[i] == "CARA" else "#e74c3c"
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=[0.5], y=[0.5], mode="markers+text",
                    marker=dict(size=120, color=color, line=dict(color="black", width=2)),
                    text=[resultados[i]],
                    textposition="middle center",
                    textfont=dict(color="white", size=16, family="Arial Black"),
                    showlegend=False, hoverinfo="skip"
                ),
                go.Bar(
                    x=["CARA", "SELLO"], y=[caras, sellos],
                    marker_color=["#3498db", "#e74c3c"],
                    text=[caras, sellos], textposition="outside",
                    showlegend=False, hoverinfo="skip"
                )
            ],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🪙 Lanzamiento {i+1}/{n}: {resultados[i]}")
        ))
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title_text=f"🪙 Lanzamiento 1/{n}",
            xaxis=dict(range=[0, 1], visible=False),
            yaxis=dict(range=[0, 1], visible=False),
            xaxis2=dict(range=[-0.5, 1.5], domain=[0.6, 1.0], title=""),
            yaxis2=dict(range=[0, max(10, n)], domain=[0.1, 0.9],
                        anchor="x2", title="Frecuencia"),
            updatemenus=[dict(
                type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 300, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ]
            )],
            height=450, showlegend=False
        )
    )
    return fig


def anim_dado(n=30, caras=6):
    """Animación: dado + histograma acumulado."""
    resultados = np.random.randint(1, caras + 1, size=n)
    frames = []
    for i in range(n):
        counts = [int(np.sum(resultados[:i+1] == k)) for k in range(1, caras + 1)]
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=[0.5], y=[0.5], mode="markers+text",
                    marker=dict(size=120, color="#e67e22",
                                line=dict(color="black", width=2), symbol="square"),
                    text=[str(resultados[i])],
                    textposition="middle center",
                    textfont=dict(color="white", size=28, family="Arial Black"),
                    showlegend=False, hoverinfo="skip"
                ),
                go.Bar(
                    x=list(range(1, caras + 1)), y=counts,
                    marker_color="skyblue", marker_line_color="black",
                    marker_line_width=1.5,
                    text=counts, textposition="outside",
                    showlegend=False, hoverinfo="skip"
                )
            ],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🎲 Lanzamiento {i+1}/{n}: {resultados[i]}")
        ))
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title_text=f"🎲 Lanzamiento 1/{n}",
            xaxis=dict(range=[0, 1], visible=False),
            yaxis=dict(range=[0, 1], visible=False),
            xaxis2=dict(range=[0.5, caras + 0.5], domain=[0.6, 1.0], title="Cara"),
            yaxis2=dict(range=[0, max(10, n)], domain=[0.1, 0.9],
                        anchor="x2", title="Frecuencia"),
            updatemenus=[dict(
                type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 300, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ]
            )],
            height=450, showlegend=False
        )
    )
    return fig


def anim_moneda_dado(n=20):
    """Animación: moneda + dado en paralelo."""
    monedas = np.random.choice(["CARA", "SELLO"], size=n)
    dados = np.random.randint(1, 7, size=n)
    frames = []
    for i in range(n):
        color = "#3498db" if monedas[i] == "CARA" else "#e74c3c"
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=[0.5], y=[0.5], mode="markers+text",
                    marker=dict(size=120, color=color, line=dict(color="black", width=2)),
                    text=[monedas[i]],
                    textposition="middle center",
                    textfont=dict(color="white", size=14, family="Arial Black"),
                    showlegend=False, hoverinfo="skip"
                ),
                go.Scatter(
                    x=[0.5], y=[0.5], mode="markers+text",
                    marker=dict(size=120, color="#e67e22",
                                line=dict(color="black", width=2), symbol="square"),
                    text=[str(dados[i])],
                    textposition="middle center",
                    textfont=dict(color="white", size=28, family="Arial Black"),
                    showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2"
                )
            ],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🪙🎲 Lanzamiento {i+1}: {monedas[i]} + {dados[i]}")
        ))
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title_text=f"🪙🎲 Lanzamiento 1/{n}",
            xaxis=dict(range=[0, 1], visible=False, domain=[0, 0.45]),
            yaxis=dict(range=[0, 1], visible=False),
            xaxis2=dict(range=[0, 1], visible=False, domain=[0.55, 1.0]),
            yaxis2=dict(range=[0, 1], visible=False, anchor="x2"),
            updatemenus=[dict(
                type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 400, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ]
            )],
            height=450, showlegend=False
        )
    )
    return fig


def anim_cartas(n=15):
    """Animación: cartas apareciendo una a una."""
    palos = ["♠", "♥", "♦", "♣"]
    valores = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    baraja = [f"{v}{p}" for v in valores for p in palos]
    extracciones = np.random.choice(baraja, size=n)
    frames = []
    for i in range(n):
        hist = extracciones[:i+1]
        n_mostrar = min(len(hist), 8)
        cartas_vis = hist[-n_mostrar:]
        xs = list(range(n_mostrar))
        colores = ["#e74c3c" if c[-1] in ["♥", "♦"] else "#2c3e50" for c in cartas_vis]
        frames.append(go.Frame(
            data=[go.Scatter(
                x=xs, y=[0.5]*n_mostrar, mode="markers+text",
                marker=dict(size=60, color="white",
                            line=dict(color=colores, width=3), symbol="square"),
                text=cartas_vis,
                textposition="middle center",
                textfont=dict(color=colores, size=13, family="Arial Black"),
                showlegend=False, hoverinfo="skip"
            )],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🃏 Extracción {i+1}/{n}: {extracciones[i]}")
        ))
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title_text=f"🃏 Extracción 1/{n}",
            xaxis=dict(range=[-0.5, 8.5], visible=False),
            yaxis=dict(range=[0, 1], visible=False),
            updatemenus=[dict(
                type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 400, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ]
            )],
            height=350, showlegend=False
        )
    )
    return fig


def anim_baloto(n_sorteos=8, n_balotas=5, rango_max=43):
    """Animación: sorteos con balotas."""
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
            data=[go.Scatter(
                x=xs, y=ys, mode="markers+text",
                marker=dict(size=55, color=colores,
                            line=dict(color="black", width=2)),
                text=textos,
                textposition="middle center",
                textfont=dict(color="white", size=13, family="Arial Black"),
                showlegend=False, hoverinfo="skip"
            )],
            name=f"f{i}",
            layout=go.Layout(title_text=f"🎰 Sorteo {i+1}/{n_sorteos}")
        ))
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title_text=f"🎰 Sorteo 1/{n_sorteos}",
            xaxis=dict(range=[-0.5, n_balotas + 0.5], visible=False),
            yaxis=dict(range=[0, 1], visible=False),
            updatemenus=[dict(
                type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 700, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ]
            )],
            height=350, showlegend=False
        )
    )
    return fig


def anim_ruleta(n_giros=15, tipo="europea"):
    """Animación: ruleta + saldo."""
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
        # Rueda: pastel de los colores + número actual
        fig_data = [
            # Número actual en el centro
            go.Scatter(
                x=[0], y=[0], mode="markers+text",
                marker=dict(size=80, color="white",
                            line=dict(color="black", width=3)),
                text=[str(tiradas[i])],
                textposition="middle center",
                textfont=dict(color=color_num(tiradas[i]), size=18,
                              family="Arial Black"),
                showlegend=False, hoverinfo="skip",
                xaxis="x", yaxis="y"
            ),
            # Saldo acumulado
            go.Scatter(
                x=list(range(i+2)), y=saldos[:i+2],
                mode="lines+markers",
                line=dict(color="#e67e22", width=3),
                marker=dict(size=8),
                showlegend=False, hoverinfo="skip",
                xaxis="x2", yaxis="y2"
            )
        ]
        frames.append(go.Frame(
            data=fig_data,
            name=f"f{i}",
            layout=go.Layout(
                title_text=f"🎡 Giro {i+1}/{n_giros} → {tiradas[i]} "
                           f"({'🔴' if tiradas[i] in rojos else '⚫' if tiradas[i] != 0 and tiradas[i] != '00' else '🟢'})"
            )
        ))

    # Pastel de fondo (siempre visible)
    colores = [color_num(c) for c in casillas]
    valores = [1] * len(casillas)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5],
        specs=[[{"type": "domain"}, {"type": "xy"}]]
    )
    # Pastel de la ruleta
    fig.add_trace(go.Pie(
        labels=[str(c) for c in casillas],
        values=valores,
        marker=dict(colors=colores, line=dict(color="white", width=1)),
        hole=0.5, showlegend=False, hoverinfo="skip",
        textinfo="none"
    ), row=1, col=1)

    # Añadir los datos del primer frame
    for trace in frames[0].data:
        if trace.xaxis == "x2" or (hasattr(trace, "xaxis") and trace.xaxis == "x2"):
            fig.add_trace(trace, row=1, col=2)
        else:
            fig.add_trace(trace, row=1, col=1)

    fig.frames = frames
    fig.update_layout(
        title_text=f"🎡 Giro 1/{n_giros}",
        updatemenus=[dict(
            type="buttons", direction="left", x=0.5, y=1.15,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 500, "redraw": True},
                                   "fromcurrent": True,
                                   "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ]
        )],
        height=500, showlegend=False
    )
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    return fig


def anim_powerball(n_sorteos=6):
    """Animación: sorteos de Powerball."""
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
            data=[go.Scatter(
                x=xs, y=ys, mode="markers+text",
                marker=dict(size=55, color=colores,
                            line=dict(color="black", width=2)),
                text=textos,
                textposition="middle center",
                textfont=dict(color="white", size=12, family="Arial Black"),
                showlegend=False, hoverinfo="skip"
            )],
            name=f"f{i}",
            layout=go.Layout(title_text=f"💥 Sorteo {i+1}/{n_sorteos}")
        ))
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title_text=f"💥 Sorteo 1/{n_sorteos}",
            xaxis=dict(range=[-0.5, 5.5], visible=False),
            yaxis=dict(range=[0, 1], visible=False),
            updatemenus=[dict(
                type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 700, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ]
            )],
            height=350, showlegend=False
        )
    )
    return fig


def anim_monte_carlo(n_puntos=1500, n_frames=60):
    """Animación: puntos de Monte Carlo apareciendo."""
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
                go.Scatter(
                    x=xa[da], y=ya[da], mode="markers",
                    marker=dict(color="#27ae60", size=5, opacity=0.6),
                    showlegend=False, hoverinfo="skip"
                ),
                go.Scatter(
                    x=xa[~da], y=ya[~da], mode="markers",
                    marker=dict(color="#e74c3c", size=5, opacity=0.6),
                    showlegend=False, hoverinfo="skip"
                ),
                go.Scatter(
                    x=np.cos(theta), y=np.sin(theta), mode="lines",
                    line=dict(color="blue", width=3),
                    showlegend=False, hoverinfo="skip"
                )
            ],
            name=f"f{f}",
            layout=go.Layout(
                title_text=f"🥧 {k} puntos | π ≈ {pi_est:.4f} (real: {np.pi:.4f})"
            )
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title_text=f"🥧 Monte Carlo π",
            xaxis=dict(range=[-1.1, 1.1], scaleanchor="y", scaleratio=1,
                       showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[-1.1, 1.1], showgrid=False, zeroline=False,
                       visible=False),
            updatemenus=[dict(
                type="buttons", direction="left", x=0.5, y=1.15,
                xanchor="center", yanchor="top", showactive=False,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, {"frame": {"duration": 80, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 0}}]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ]
            )],
            height=600, showlegend=False,
            plot_bgcolor="white"
        )
    )
    return fig


def anim_cumpleanos(n_personas=23, n_simulaciones=100):
    """Animación: personas + probabilidad acumulada."""
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
                go.Scatter(
                    x=xs, y=ys, mode="markers+text",
                    marker=dict(size=30, color=colores,
                                line=dict(color="black", width=1.5)),
                    text=[str(m) for m in meses],
                    textposition="middle center",
                    textfont=dict(color="white", size=9),
                    showlegend=False, hoverinfo="skip"
                ),
                go.Scatter(
                    x=list(range(1, i + 2)),
                    y=list(coincidencias_acum[:i+1] / np.arange(1, i + 2)),
                    mode="lines",
                    line=dict(color="#9b59b6", width=3),
                    showlegend=False, hoverinfo="skip",
                    xaxis="x2", yaxis="y2"
                )
            ],
            name=f"f{i}",
            layout=go.Layout(
                title_text=f"🎂 Simulación {i+1}/{n_simulaciones} - "
                           f"{'✅ Coincidencia' if coincidencias[i] else '❌ Sin coincidencia'} "
                           f"| Prob: {prob:.3f}"
            )
        ))

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5],
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )
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
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  row=1, col=2)
    fig.update_layout(
        title_text="🎂 Problema del Cumpleaños",
        updatemenus=[dict(
            type="buttons", direction="left", x=0.5, y=1.15,
            xanchor="center", yanchor="top", showactive=False,
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 120, "redraw": True},
                                   "fromcurrent": True,
                                   "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ]
        )],
        height=500, showlegend=False
    )
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
    palos = ["♠", "♥", "♦", "♣"]
    valores = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    baraja = [f"{v}{p}" for v in valores for p in palos]
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


# ============================================================
# INTERFAZ
# ============================================================

tabs = st.tabs([
    "🪙 Monedas", "🎲 Dados", "🪙🎲 Moneda + Dado",
    "🃏 Cartas", "🎰 Baloto", "📊 Ley Grandes Números",
    "🎡 Ruleta", "💥 Powerball", "🥧 Monte Carlo π", "🎂 Cumpleaños"
])

# ---------------- TAB 1: MONEDAS ----------------
with tabs[0]:
    st.header("🪙 Simulación con Monedas")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Configuración")
        n_monedas = st.slider("Monedas por lanzamiento:", 1, 10, 1)
        n_lanzamientos = st.number_input("Total lanzamientos:", 10, 100000, 500, 50)
        prob_cara = st.slider("P(CARA):", 0.0, 1.0, 0.5, 0.01)
        semilla = st.number_input("Semilla:", 0, 99999, 42)
        if semilla > 0:
            np.random.seed(int(semilla))
        st.markdown("---")
        st.subheader("🎬 Animación")
        n_anim_mon = st.slider("Lanzamientos animados:", 5, 60, 30, key="gif_mon")
        if st.button("🎬 Generar Animación", key="btn_anim_mon"):
            st.session_state.historial["anim_moneda"] = anim_moneda(n_anim_mon, prob_cara)
        if st.button("🎲 Simular (datos)", key="btn_monedas"):
            st.session_state.historial["monedas"] = simulador_monedas(
                n_monedas, n_lanzamientos, prob_cara
            )
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
                caras_acum = np.cumsum(res.flatten() == "CARA") / np.arange(1, t+1)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=caras_acum, mode="lines",
                                          name="Frec. CARA", line=dict(color="#3498db")))
                fig.add_hline(y=prob_cara, line_dash="dash", line_color="red")
                fig.update_layout(title="Evolución P(CARA)",
                                  xaxis_title="Lanzamiento",
                                  yaxis_title="Frecuencia relativa", height=350)
                st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2: DADOS ----------------
with tabs[1]:
    st.header("🎲 Simulación con Dados")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_dados = st.slider("Dados por lanzamiento:", 1, 10, 1)
        n_lanz_dados = st.number_input("Total lanzamientos:", 10, 100000, 500, 50, key="nl_dados")
        caras_dado = st.selectbox("Caras:", [4, 6, 8, 10, 12, 20], 1)
        semilla_d = st.number_input("Semilla:", 0, 99999, 42, key="sem_dados")
        if semilla_d > 0:
            np.random.seed(int(semilla_d))
        st.markdown("---")
        n_anim_d = st.slider("Lanzamientos animados:", 5, 60, 30, key="gif_dado")
        if st.button("🎬 Generar Animación", key="btn_anim_dado"):
            st.session_state.historial["anim_dado"] = anim_dado(n_anim_d, caras_dado)
        if st.button("🎲 Simular (datos)", key="btn_dados"):
            st.session_state.historial["dados"] = simulador_dados(
                n_dados, n_lanz_dados, caras_dado
            )
    with col2:
        if "anim_dado" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_dado"],
                            use_container_width=True)
        if "dados" in st.session_state.historial:
            res = st.session_state.historial["dados"]
            if n_dados == 1:
                valores = res.flatten()
                frec = Counter(valores)
                df_frec = pd.DataFrame({
                    "Cara": list(range(1, caras_dado + 1)),
                    "Frecuencia": [frec.get(i, 0) for i in range(1, caras_dado + 1)]
                })
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(valores))
                c2.metric("Media", f"{valores.mean():.3f}", f"Teo: {(caras_dado+1)/2:.2f}")
                c3.metric("Desv.Est.", f"{valores.std():.3f}",
                          f"Teo: {np.sqrt((caras_dado**2-1)/12):.2f}")
                fig = px.bar(df_frec, x="Cara", y="Frecuencia", text_auto=True,
                             color="Frecuencia", color_continuous_scale="Viridis")
                fig.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 3: MONEDA + DADO ----------------
with tabs[2]:
    st.header("🪙🎲 Moneda + Dado")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_lanz_md = st.number_input("Lanzamientos:", 10, 100000, 500, 50, key="nl_md")
        semilla_md = st.number_input("Semilla:", 0, 99999, 42, key="sem_md")
        if semilla_md > 0:
            np.random.seed(int(semilla_md))
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
            st.plotly_chart(st.session_state.historial["anim_md"],
                            use_container_width=True)
        if "moneda_dado" in st.session_state.historial:
            m, d = st.session_state.historial["moneda_dado"]
            df = pd.DataFrame({"Moneda": m, "Dado": d})
            cont = pd.crosstab(df["Dado"], df["Moneda"])
            st.dataframe(cont, use_container_width=True)
            fig = px.imshow(cont, text_auto=True, aspect="auto",
                            color_continuous_scale="Blues")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 4: CARTAS ----------------
with tabs[3]:
    st.header("🃏 Cartas")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_cartas = st.slider("Cartas por extracción:", 1, 10, 1)
        n_extr = st.number_input("Extracciones:", 10, 10000, 200, 10, key="ne_cartas")
        semilla_c = st.number_input("Semilla:", 0, 99999, 42, key="sem_cartas")
        if semilla_c > 0:
            np.random.seed(int(semilla_c))
        st.markdown("---")
        n_anim_c = st.slider("Cartas animadas:", 5, 25, 15, key="gif_c")
        if st.button("🎬 Generar Animación", key="btn_anim_c"):
            st.session_state.historial["anim_cartas"] = anim_cartas(n_anim_c)
        if st.button("🎲 Simular (datos)", key="btn_cartas"):
            st.session_state.historial["cartas"] = simulador_cartas(n_cartas, n_extr)
    with col2:
        if "anim_cartas" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_cartas"],
                            use_container_width=True)
        if "cartas" in st.session_state.historial:
            res = st.session_state.historial["cartas"]
            valores = res.flatten()
            palos = [c[-1] for c in valores]
            cp = Counter(palos)
            df_p = pd.DataFrame({"Palo": list(cp.keys()), "Frec": list(cp.values())})
            color_map = {"♠": "#2c3e50", "♥": "#e74c3c",
                         "♦": "#e67e22", "♣": "#27ae60"}
            fig = px.pie(df_p, names="Palo", values="Frec", color="Palo",
                         color_discrete_map=color_map)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 5: BALOTO ----------------
with tabs[4]:
    st.header("🎰 Baloto")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_bal = st.slider("Balotas:", 4, 6, 5)
        r_max = st.number_input("Máx. número:", 10, 100, 43)
        n_sort = st.number_input("Sorteos:", 1, 1000, 10)
        semilla_b = st.number_input("Semilla:", 0, 99999, 42, key="sem_bal")
        if semilla_b > 0:
            np.random.seed(int(semilla_b))
        st.markdown("---")
        n_anim_b = st.slider("Sorteos animados:", 3, 20, 8, key="gif_b")
        if st.button("🎬 Generar Animación", key="btn_anim_b"):
            st.session_state.historial["anim_baloto"] = anim_baloto(n_anim_b, n_bal, r_max)
        if st.button("🎲 Simular (datos)", key="btn_baloto"):
            st.session_state.historial["baloto"] = simulador_balotas(
                n_bal, (1, r_max), n_sort
            )
    with col2:
        if "anim_baloto" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_baloto"],
                            use_container_width=True)
        if "baloto" in st.session_state.historial:
            res = st.session_state.historial["baloto"]
            filas = [{"Sorteo": i+1, "Números": ", ".join(map(str, n)),
                      "Extra": e} for i, (n, e) in enumerate(res)]
            st.dataframe(pd.DataFrame(filas), use_container_width=True)

# ---------------- TAB 6: LEY GRANDES NÚMEROS ----------------
with tabs[5]:
    st.header("📊 Ley de los Grandes Números")
    n_max = st.number_input("Ensayos:", 100, 50000, 5000, 100)
    tipo = st.selectbox("Experimento:", ["Moneda (CARA)", "Dado (6)", "Dado (par)"])
    if st.button("🚀 Ejecutar"):
        if tipo == "Moneda (CARA)":
            r = np.random.choice([1, 0], n_max); pt = 0.5
        elif tipo == "Dado (6)":
            r = (np.random.randint(1, 7, n_max) == 6).astype(int); pt = 1/6
        else:
            r = (np.random.randint(1, 7, n_max) % 2 == 0).astype(int); pt = 0.5
        acum = np.cumsum(r) / np.arange(1, n_max + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=acum, mode="lines", name="Frec. rel.",
                                  line=dict(color="#3498db", width=2)))
        fig.add_hline(y=pt, line_dash="dash", line_color="red")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 7: RULETA ----------------
with tabs[6]:
    st.header("🎡 Ruleta")
    col1, col2 = st.columns([1, 2])
    with col1:
        tipo_r = st.selectbox("Tipo:", ["Europea (0-36)", "Americana (0, 00, 1-36)"])
        n_tir = st.number_input("Tiradas:", 10, 100000, 1000, 100)
        semilla_r = st.number_input("Semilla:", 0, 99999, 42, key="sem_r")
        if semilla_r > 0:
            np.random.seed(int(semilla_r))
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
            st.plotly_chart(st.session_state.historial["anim_ruleta"],
                            use_container_width=True)
        if "ruleta" in st.session_state.historial:
            t, roj, neg = st.session_state.historial["ruleta"]
            total = len(t)
            nr = sum(1 for x in t if x in roj)
            nn = sum(1 for x in t if x in neg)
            nc = sum(1 for x in t if x == 0 or x == "00")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tiradas", total)
            c2.metric("🔴", nr, f"{nr/total*100:.1f}%")
            c3.metric("⚫", nn, f"{nn/total*100:.1f}%")
            c4.metric("🟢", nc, f"{nc/total*100:.1f}%")

# ---------------- TAB 8: POWERBALL ----------------
with tabs[7]:
    st.header("💥 Powerball")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_pb = st.number_input("Sorteos:", 1, 10000, 100)
        semilla_pb = st.number_input("Semilla:", 0, 99999, 42, key="sem_pb")
        if semilla_pb > 0:
            np.random.seed(int(semilla_pb))
        st.markdown("---")
        n_anim_pb = st.slider("Sorteos animados:", 3, 15, 6, key="gif_pb")
        if st.button("🎬 Generar Animación", key="btn_anim_pb"):
            st.session_state.historial["anim_powerball"] = anim_powerball(n_anim_pb)
        if st.button("💥 Simular (datos)", key="btn_pb"):
            st.session_state.historial["powerball"] = simulador_powerball(n_pb)
    with col2:
        if "anim_powerball" in st.session_state.historial:
            st.subheader("🎬 Animación interactiva")
            st.plotly_chart(st.session_state.historial["anim_powerball"],
                            use_container_width=True)
        if "powerball" in st.session_state.historial:
            res = st.session_state.historial["powerball"]
            filas = [{"Sorteo": i+1, "Números": ", ".join(map(str, n)),
                      "PB": p} for i, (n, p) in enumerate(res)]
            st.dataframe(pd.DataFrame(filas), use_container_width=True)

# ---------------- TAB 9: MONTE CARLO π ----------------
with tabs[8]:
    st.header("🥧 Monte Carlo π")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_pt = st.slider("Puntos:", 100, 100000, 5000, 100)
        semilla_pi = st.number_input("Semilla:", 0, 99999, 42, key="sem_pi")
        if semilla_pi > 0:
            np.random.seed(int(semilla_pi))
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
            st.plotly_chart(st.session_state.historial["anim_pi"],
                            use_container_width=True)
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
        semilla_c = st.number_input("Semilla:", 0, 99999, 42, key="sem_cum")
        if semilla_c > 0:
            np.random.seed(int(semilla_c))
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
            st.plotly_chart(st.session_state.historial["anim_cumple"],
                            use_container_width=True)
        if "cumple" in st.session_state.historial:
            ps, pt, n = st.session_state.historial["cumple"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Simulada", f"{ps*100:.2f}%")
            c2.metric("Teórica", f"{pt*100:.2f}%")
            c3.metric("Error", f"{abs(ps-pt)*100:.3f}%")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🎬 Simulador con Animaciones Plotly - v4.0<br>
    Sin dependencias externas · 100% interactivo
</div>
""", unsafe_allow_html=True)
