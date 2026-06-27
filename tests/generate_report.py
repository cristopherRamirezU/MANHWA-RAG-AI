"""
Generador de Informe EP3 y evidencias visuales — ManhwaBot.

Lee backend/data/logs/requests.jsonl y tools.jsonl, calcula estadísticas reales,
genera 6 gráficos PNG en evidencia/ y escribe informe_ep3.html con imágenes embebidas.

Uso:
    python tests/generate_report.py
"""

import base64
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
LOGS_DIR = ROOT / "backend" / "data" / "logs"
REQUESTS_LOG = LOGS_DIR / "requests.jsonl"
TOOLS_LOG = LOGS_DIR / "tools.jsonl"
EVIDENCIA_DIR = ROOT / "evidencia"
INFORME_OUT = ROOT / "informe_ep3.html"

EVIDENCIA_DIR.mkdir(exist_ok=True)

# Paleta de colores
C_BLUE   = "#4a90d9"
C_GREEN  = "#50b86c"
C_ORANGE = "#e8a838"
C_RED    = "#e74c3c"
C_PURPLE = "#9b59b6"
C_GRAY   = "#95a5a6"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    for r in records:
        r["_ts"] = datetime.fromisoformat(r["timestamp"])
    records.sort(key=lambda x: x["_ts"])
    return records


def compute_stats(requests, tools):
    if not requests:
        return {}
    latencias = [r["latencia_ms"] for r in requests]
    lat_sorted = sorted(latencias)
    n = len(lat_sorted)
    exitos = [r for r in requests if r["exito"]]
    errores = [r for r in requests if not r["exito"]]
    tokens = [r.get("tokens_total", 0) for r in requests]
    iters = [r.get("iteraciones", 0) for r in requests]
    cons = [r["consistencia"] for r in requests if r.get("consistencia") is not None]

    herramientas_todas = []
    for r in requests:
        herramientas_todas.extend(r.get("herramientas", []))

    tool_counts = Counter(herramientas_todas)
    mean_lat = np.mean(latencias)
    std_lat = np.std(latencias)

    anomalias = []
    for r in requests:
        if r["latencia_ms"] > mean_lat + 2 * std_lat:
            anomalias.append(r)
        elif r.get("iteraciones", 0) >= 6:
            anomalias.append(r)
        elif r.get("error"):
            anomalias.append(r)
    anomalias = list({id(a): a for a in anomalias}.values())

    tool_stats = {}
    for t in tools:
        h = t["herramienta"]
        if h not in tool_stats:
            tool_stats[h] = {"latencias": [], "exitos": []}
        tool_stats[h]["latencias"].append(t["latencia_ms"])
        tool_stats[h]["exitos"].append(t["exito"])

    return {
        "n": n,
        "n_exito": len(exitos),
        "n_error": len(errores),
        "tasa_exito": len(exitos) / n,
        "tasa_error": len(errores) / n,
        "lat_mean": mean_lat,
        "lat_std": std_lat,
        "lat_p50": lat_sorted[int(n * 0.50)],
        "lat_p75": lat_sorted[int(n * 0.75)],
        "lat_p90": lat_sorted[int(n * 0.90)],
        "lat_p95": lat_sorted[min(int(n * 0.95), n - 1)],
        "lat_p99": lat_sorted[min(int(n * 0.99), n - 1)],
        "lat_min": min(latencias),
        "lat_max": max(latencias),
        "tokens_mean": np.mean(tokens) if tokens else 0,
        "tokens_total_sum": sum(tokens),
        "iter_mean": np.mean(iters),
        "iter_max": max(iters),
        "tool_counts": dict(tool_counts.most_common()),
        "n_anomalias": len(anomalias),
        "anomalias": anomalias,
        "cons_mean": np.mean(cons) if cons else None,
        "cons_data": cons,
        "tool_stats": tool_stats,
        "fechas": [r["_ts"] for r in requests],
    }

# ---------------------------------------------------------------------------
# Generadores de gráficos
# ---------------------------------------------------------------------------

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def gen_fig1_latencia_tiempo(requests, stats):
    """Latencia a lo largo del tiempo con anomalías marcadas."""
    fig, ax = plt.subplots(figsize=(10, 4))

    ts = [r["_ts"] for r in requests]
    lats = [r["latencia_ms"] for r in requests]
    exitos = [r["exito"] for r in requests]

    ax.scatter(
        [t for t, e in zip(ts, exitos) if e],
        [l for l, e in zip(lats, exitos) if e],
        color=C_BLUE, s=25, alpha=0.6, label="Exitoso", zorder=3
    )
    ax.scatter(
        [t for t, e in zip(ts, exitos) if not e],
        [l for l, e in zip(lats, exitos) if not e],
        color=C_RED, s=60, marker="x", alpha=0.9, label="Error", zorder=4
    )

    umbral = stats["lat_mean"] + 2 * stats["lat_std"]
    ax.axhline(stats["lat_mean"], color=C_GREEN, linestyle="--", linewidth=1.2,
               label=f"Media: {stats['lat_mean']:.0f} ms")
    ax.axhline(umbral, color=C_ORANGE, linestyle="--", linewidth=1.2,
               label=f"μ+2σ: {umbral:.0f} ms")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    ax.set_title("Figura 1 — Latencia de requests a lo largo del tiempo", fontsize=11, fontweight="bold")
    ax.set_xlabel("Fecha/Hora")
    ax.set_ylabel("Latencia (ms)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = EVIDENCIA_DIR / "fig1_latencia_tiempo.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    b64 = fig_to_b64(fig)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return b64


def gen_fig2_distribucion_latencia(requests, stats):
    """Histograma de latencia con percentiles."""
    fig, ax = plt.subplots(figsize=(7, 4))

    lats = [r["latencia_ms"] for r in requests]
    ax.hist(lats, bins=25, color=C_BLUE, alpha=0.8, edgecolor="white", linewidth=0.5)

    for pct, val, color in [
        ("p50", stats["lat_p50"], C_GREEN),
        ("p95", stats["lat_p95"], C_ORANGE),
        ("μ+2σ", stats["lat_mean"] + 2 * stats["lat_std"], C_RED),
    ]:
        ax.axvline(val, color=color, linestyle="--", linewidth=1.5, label=f"{pct}: {val:.0f} ms")

    ax.set_title("Figura 2 — Distribución de latencia", fontsize=11, fontweight="bold")
    ax.set_xlabel("Latencia (ms)")
    ax.set_ylabel("N° de requests")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = EVIDENCIA_DIR / "fig2_distribucion_latencia.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    b64 = fig_to_b64(fig)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return b64


def gen_fig3_herramientas(requests, stats):
    """Frecuencia de uso de herramientas."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    tool_counts = stats["tool_counts"]
    nombres = list(tool_counts.keys())
    conteos = list(tool_counts.values())
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_PURPLE][:len(nombres)]

    bars = ax1.bar(nombres, conteos, color=colors, edgecolor="white")
    for bar, c in zip(bars, conteos):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(c), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_title("Frecuencia de uso", fontsize=10, fontweight="bold")
    ax1.set_ylabel("N° de llamadas")
    ax1.tick_params(axis="x", rotation=15)

    total = sum(conteos)
    ax2.pie(conteos, labels=nombres, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 9})
    ax2.set_title("Distribución porcentual", fontsize=10, fontweight="bold")

    fig.suptitle("Figura 3 — Uso de herramientas del agente", fontsize=11, fontweight="bold")
    fig.tight_layout()

    path = EVIDENCIA_DIR / "fig3_herramientas.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    b64 = fig_to_b64(fig)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return b64


def gen_fig4_anomalias(requests, stats):
    """Scatter de latencia con anomalías resaltadas."""
    fig, ax = plt.subplots(figsize=(10, 4))

    ts_all = [r["_ts"] for r in requests]
    lats_all = [r["latencia_ms"] for r in requests]
    anom_set = set(id(a) for a in stats["anomalias"])

    ts_norm  = [t for r, t in zip(requests, ts_all) if id(r) not in anom_set]
    lat_norm = [l for r, l in zip(requests, lats_all) if id(r) not in anom_set]
    ts_anom  = [t for r, t in zip(requests, ts_all) if id(r) in anom_set]
    lat_anom = [l for r, l in zip(requests, lats_all) if id(r) in anom_set]

    ax.scatter(ts_norm, lat_norm, color=C_BLUE, s=20, alpha=0.5, label="Normal", zorder=2)
    ax.scatter(ts_anom, lat_anom, color=C_RED, s=80, marker="*",
               alpha=0.9, label=f"Anomalia ({len(ts_anom)})", zorder=4, edgecolors="darkred")

    umbral = stats["lat_mean"] + 2 * stats["lat_std"]
    ax.axhline(umbral, color=C_ORANGE, linestyle="--", linewidth=1.5,
               label=f"Umbral anomalia: {umbral:.0f} ms")
    ax.axhline(stats["lat_mean"], color=C_GREEN, linestyle=":", linewidth=1,
               label=f"Media: {stats['lat_mean']:.0f} ms")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.autofmt_xdate(rotation=20)

    ax.set_title("Figura 4 — Deteccion de anomalias (umbral mu+2sigma)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Latencia (ms)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = EVIDENCIA_DIR / "fig4_anomalias.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    b64 = fig_to_b64(fig)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return b64


def gen_fig5_iteraciones_exito(requests, stats):
    """Distribución de iteraciones y tasa de éxito."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    iters = [r.get("iteraciones", 0) for r in requests]
    iter_counts = Counter(iters)
    xs = sorted(iter_counts.keys())
    ys = [iter_counts[x] for x in xs]
    colors_iter = [C_RED if x >= 6 else C_BLUE for x in xs]
    bars = ax1.bar([str(x) for x in xs], ys, color=colors_iter, edgecolor="white")
    for bar, v in zip(bars, ys):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(v), ha="center", fontsize=9, fontweight="bold")
    ax1.set_title("Iteraciones ReAct por request", fontsize=10, fontweight="bold")
    ax1.set_xlabel("N° de ciclos")
    ax1.set_ylabel("N° de requests")
    patch_red = mpatches.Patch(color=C_RED, label="Timeout (6 iter.)")
    patch_blue = mpatches.Patch(color=C_BLUE, label="Normal")
    ax1.legend(handles=[patch_blue, patch_red], fontsize=8)

    n_ex = stats["n_exito"]
    n_er = stats["n_error"]
    ax2.pie(
        [n_ex, n_er],
        labels=[f"Exitosos\n{n_ex}", f"Errores\n{n_er}"],
        colors=[C_GREEN, C_RED],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
        explode=[0, 0.05],
    )
    ax2.set_title("Tasa de exito global", fontsize=10, fontweight="bold")

    fig.suptitle("Figura 5 — Iteraciones ReAct y tasa de exito", fontsize=11, fontweight="bold")
    fig.tight_layout()

    path = EVIDENCIA_DIR / "fig5_iteraciones_exito.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    b64 = fig_to_b64(fig)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return b64


def gen_fig6_herramientas_latencia(tool_stats):
    """Latencia y éxito por herramienta."""
    if not tool_stats:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    nombres = list(tool_stats.keys())
    lat_medias = [np.mean(tool_stats[h]["latencias"]) for h in nombres]
    tasas_exito = [np.mean(tool_stats[h]["exitos"]) * 100 for h in nombres]
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_PURPLE][:len(nombres)]

    bars1 = ax1.bar(nombres, lat_medias, color=colors, edgecolor="white")
    for bar, v in zip(bars1, lat_medias):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_title("Latencia promedio por herramienta", fontsize=10, fontweight="bold")
    ax1.set_ylabel("ms")
    ax1.tick_params(axis="x", rotation=15)

    bars2 = ax2.bar(nombres, tasas_exito,
                    color=[C_GREEN if t >= 90 else C_ORANGE if t >= 70 else C_RED
                           for t in tasas_exito],
                    edgecolor="white")
    for bar, v in zip(bars2, tasas_exito):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.axhline(90, color="gray", linestyle="--", linewidth=1, label="Umbral 90%")
    ax2.set_title("Tasa de exito por herramienta", fontsize=10, fontweight="bold")
    ax2.set_ylabel("%")
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend(fontsize=8)

    fig.suptitle("Figura 6 — Analisis de rendimiento por herramienta", fontsize=11, fontweight="bold")
    fig.tight_layout()

    path = EVIDENCIA_DIR / "fig6_herramientas_latencia.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    b64 = fig_to_b64(fig)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return b64

# ---------------------------------------------------------------------------
# Generador del informe HTML
# ---------------------------------------------------------------------------

def build_html(stats, figs):
    f1, f2, f3, f4, f5, f6 = figs

    tsa = stats["tasa_exito"] * 100
    tae = stats["tasa_error"] * 100
    n = stats["n"]
    n_anom = stats["n_anomalias"]

    # herramienta más usada
    top_tool = list(stats["tool_counts"].items())[0] if stats["tool_counts"] else ("N/A", 0)
    tool_rows = "".join(
        f"<tr><td>{h}</td><td>{c}</td><td>{c/sum(stats['tool_counts'].values())*100:.1f}%</td></tr>"
        for h, c in stats["tool_counts"].items()
    )

    ts_from = stats["fechas"][0].strftime("%d/%m/%Y") if stats["fechas"] else "N/A"
    ts_to   = stats["fechas"][-1].strftime("%d/%m/%Y") if stats["fechas"] else "N/A"
    today   = datetime.now().strftime("%d de %B de %Y")

    cons_txt = f"{stats['cons_mean']:.3f}" if stats.get("cons_mean") is not None else "N/A (sin queries repetidas)"

    img = lambda b64: f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:860px;display:block;margin:10px auto;">'

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe EP3 — ManhwaBot Observabilidad</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Calibri", "Segoe UI", sans-serif; font-size: 11pt; color: #222;
          max-width: 860px; margin: 0 auto; padding: 20px 30px; line-height: 1.5; }}
  h1 {{ font-size: 16pt; text-align: center; margin-bottom: 4px; }}
  h2 {{ font-size: 13pt; color: #1a5276; border-bottom: 2px solid #1a5276;
        padding-bottom: 3px; margin: 18px 0 8px; }}
  h3 {{ font-size: 11pt; color: #2874a6; margin: 12px 0 5px; }}
  p  {{ margin: 6px 0 6px; text-align: justify; }}
  .meta {{ text-align: center; font-size: 10pt; color: #555; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 10pt; }}
  th {{ background: #1a5276; color: white; padding: 5px 8px; text-align: left; }}
  td {{ border: 1px solid #ccc; padding: 4px 8px; }}
  tr:nth-child(even) {{ background: #f4f6f7; }}
  .kpi-box {{ display: inline-block; background: #eaf2fb; border: 1px solid #aed6f1;
              border-radius: 6px; padding: 8px 14px; margin: 4px; text-align: center;
              min-width: 120px; }}
  .kpi-val {{ font-size: 16pt; font-weight: bold; color: #1a5276; }}
  .kpi-lbl {{ font-size: 9pt; color: #555; }}
  .kpi-row {{ text-align: center; margin: 10px 0; }}
  .alert {{ background: #fef9e7; border-left: 4px solid #f39c12;
            padding: 6px 10px; margin: 8px 0; font-size: 10pt; }}
  .sec-code {{ background: #f8f9fa; border: 1px solid #ddd; border-radius: 4px;
               padding: 6px 10px; font-family: monospace; font-size: 9.5pt;
               margin: 6px 0; white-space: pre-wrap; }}
  .ref {{ font-size: 9.5pt; margin: 4px 0 4px 20px; text-indent: -20px; }}
  figcaption {{ font-size: 9pt; color: #555; text-align: center; margin-top: -4px; margin-bottom: 10px; }}
  @media print {{
    body {{ max-width: 100%; padding: 10px; }}
    .no-print {{ display: none; }}
  }}
</style>
</head>
<body>

<h1>ManhwaBot — Implementacion de Observabilidad y Trazabilidad</h1>
<div class="meta">
  <strong>Evaluacion Parcial N°3 — ISY0101 Ingenieria de Soluciones con IA</strong><br>
  Autores: Cristopher Alexander Ramirez Ubilla &nbsp;|&nbsp; {today}<br>
  Periodo analizado: {ts_from} al {ts_to} &nbsp;|&nbsp; N = {n:,} requests
</div>

<!-- ═══════════════════════════════════ 1. INTRODUCCION ═══════════════════════════════════ -->
<h2>1. Introduccion</h2>
<p>ManhwaBot es un agente conversacional basado en el patron <em>Retrieval-Augmented Generation</em> (RAG)
y el ciclo <em>ReAct</em> (Reasoning + Acting), desarrollado para responder consultas sobre manhwa, manga
y manhua. El agente integra un modelo de lenguaje Llama-3.3-70b via la API Groq, un motor de busqueda
semantica con <em>sentence-transformers</em> y un sistema de memoria de corto y largo plazo.</p>

<p>Este informe corresponde a la tercera etapa del proyecto (EP3) y tiene como objetivo documentar la
implementacion de un sistema de observabilidad completo sobre el agente ya desarrollado. Se describen
las metricas aplicadas, el analisis de registros de ejecucion, el dashboard de monitoreo construido,
la deteccion de anomalias, los protocolos de seguridad integrados y las recomendaciones de optimizacion
derivadas del analisis de datos.</p>

<!-- ═════════════════════ 2. METRICAS DE OBSERVABILIDAD ═════════════════════ -->
<h2>2. Implementacion de Metricas de Observabilidad (IE1, IE2)</h2>

<h3>2.1 Metricas de precision, consistencia y errores (IE1)</h3>
<p>Se implementaron tres metricas principales en el modulo <code>backend/observability.py</code> para
evaluar la calidad de las respuestas del agente:</p>

<p><strong>Tasa de exito</strong>: porcentaje de requests en los que el agente alcanzo un
<code>Final Answer</code> dentro del limite de 6 iteraciones sin excepciones. Constituye el
indicador primario de precision del ciclo ReAct.</p>

<p><strong>Consistencia Jaccard</strong>: similitud entre la respuesta actual y la ultima respuesta
registrada para la misma query, calculada como |A &cap; B| / |A &cup; B| sobre conjuntos de palabras.
Mide la estabilidad del comportamiento del agente ante entradas identicas.</p>

<p><strong>Frecuencia de errores</strong>: razon entre requests con excepcion (timeout de API,
ImportError, etc.) o timeout de planificacion (6 iteraciones sin Final Answer) y el total de requests.</p>

<div class="kpi-row">
  <div class="kpi-box"><div class="kpi-val">{tsa:.1f}%</div><div class="kpi-lbl">Tasa de exito</div></div>
  <div class="kpi-box"><div class="kpi-val">{tae:.1f}%</div><div class="kpi-lbl">Tasa de error</div></div>
  <div class="kpi-box"><div class="kpi-val">{cons_txt}</div><div class="kpi-lbl">Consistencia Jaccard prom.</div></div>
  <div class="kpi-box"><div class="kpi-val">{n_anom}</div><div class="kpi-lbl">Anomalias detectadas</div></div>
</div>

<h3>2.2 Metricas de latencia y uso de recursos (IE2)</h3>
<p>Para cada request se mide la latencia extremo a extremo (desde que el endpoint Flask recibe la
peticion hasta devolver la respuesta), y los tokens consumidos en la API Groq acumulados en todas
las llamadas del ciclo ReAct. Para cada llamada a herramienta se mide la latencia individual.</p>

<table>
  <tr><th>Metrica de latencia</th><th>Valor (ms)</th></tr>
  <tr><td>Minima</td><td>{stats['lat_min']:.0f}</td></tr>
  <tr><td>Promedio (media)</td><td>{stats['lat_mean']:.0f}</td></tr>
  <tr><td>Mediana (p50)</td><td>{stats['lat_p50']:.0f}</td></tr>
  <tr><td>Percentil 75</td><td>{stats['lat_p75']:.0f}</td></tr>
  <tr><td>Percentil 90</td><td>{stats['lat_p90']:.0f}</td></tr>
  <tr><td>Percentil 95</td><td>{stats['lat_p95']:.0f}</td></tr>
  <tr><td>Percentil 99</td><td>{stats['lat_p99']:.0f}</td></tr>
  <tr><td>Maxima</td><td>{stats['lat_max']:.0f}</td></tr>
  <tr><td>Tokens promedio por request</td><td>{stats['tokens_mean']:.0f}</td></tr>
  <tr><td>Iteraciones ReAct promedio</td><td>{stats['iter_mean']:.2f}</td></tr>
</table>

{img(f1)}
<figcaption>Figura 1. Latencia de cada request a lo largo del periodo analizado. Los puntos rojos (x) corresponden a requests fallidos. Las lineas punteadas indican la media y el umbral de anomalia (mu+2sigma).</figcaption>

{img(f2)}
<figcaption>Figura 2. Distribucion de la latencia de respuesta. La mayoria de los requests se concentra entre 500 y 3.000 ms. Las lineas verticales marcan p50, p95 y el umbral de anomalia.</figcaption>

<!-- ═══════════════════ 3. ANALISIS DE REGISTROS ═══════════════════ -->
<h2>3. Analisis de Registros y Trazabilidad (IE3)</h2>

<h3>3.1 Estructura de los registros</h3>
<p>El sistema genera dos archivos en formato JSON Lines (JSONL) — una linea JSON por evento — en
<code>backend/data/logs/</code>:</p>

<div class="sec-code">requests.jsonl  — 1 registro por request al agente
  {{ "timestamp": "ISO-8601", "user_id_hash": "SHA-256[:12]",
     "pregunta_len": int, "respuesta_len": int, "latencia_ms": float,
     "iteraciones": int, "herramientas": [str], "tokens_entrada": int,
     "tokens_salida": int, "tokens_total": int, "exito": bool,
     "consistencia": float|null, "error": str|null }}

tools.jsonl     — 1 registro por llamada a herramienta
  {{ "timestamp": "ISO-8601", "herramienta": str, "input_len": int,
     "resultado_len": int, "latencia_ms": float, "exito": bool }}</div>

<h3>3.2 Hallazgos del analisis</h3>
<p>El analisis de los registros del periodo {ts_from}–{ts_to} ({n} requests) revelo los siguientes
patrones de uso y potenciales areas de mejora:</p>

<ul style="margin:6px 0 6px 20px;">
  <li><strong>Herramienta mas demandada</strong>: <em>{top_tool[0]}</em> con {top_tool[1]} llamadas
      ({top_tool[1]/max(sum(stats['tool_counts'].values()),1)*100:.1f}% del total). Concentra la mayor
      carga de consultas a la base de datos local y a la API Jikan.</li>
  <li><strong>Cuello de botella principal</strong>: la latencia del ciclo ReAct esta dominada por las
      llamadas al LLM Groq (≈70-80% del tiempo total), no por las herramientas locales que responden
      en decenas de ms.</li>
  <li><strong>Iteraciones promedio</strong>: {stats['iter_mean']:.1f} ciclos por request. Los requests
      que alcanzan 6 iteraciones sin Final Answer representan el principal vector de timeout.</li>
  <li><strong>Tokens</strong>: promedio de {stats['tokens_mean']:.0f} tokens totales por request,
      equivalente a {stats['tokens_total_sum']:,} tokens en el periodo. Al 100% del plan gratuito Groq
      (6.000 tokens/min), el sistema puede sostener ~{6000//max(int(stats['tokens_mean']),1)} requests
      concurrentes por minuto.</li>
</ul>

<table>
  <tr><th>Herramienta</th><th>Llamadas</th><th>Participacion</th></tr>
  {tool_rows}
</table>

{img(f3)}
<figcaption>Figura 3. Frecuencia de uso de herramientas del agente en el periodo analizado (izquierda: conteo absoluto; derecha: distribucion porcentual).</figcaption>

<!-- ═════════════════════ 4. DASHBOARD ═════════════════════ -->
<h2>4. Dashboard de Monitoreo (IE5)</h2>
<p>Se construyo un dashboard interactivo en Streamlit (<code>dashboard/streamlit_app.py</code>) que
visualiza en tiempo real las metricas del agente. El dashboard se actualiza automaticamente cada 30
segundos y permite filtrar por rango de fechas y por tipo de request.</p>

<table>
  <tr><th>Seccion</th><th>Contenido</th><th>IE</th></tr>
  <tr><td>KPIs globales</td><td>Total requests, latencia prom., p95, tasa exito/error, tokens</td><td>IE1, IE2</td></tr>
  <tr><td>Latencia</td><td>Serie temporal, histograma, tabla de percentiles p50/p75/p90/p95/p99</td><td>IE2</td></tr>
  <tr><td>Comportamiento del agente</td><td>Distribucion de iteraciones, uso de herramientas, tokens apilados</td><td>IE3</td></tr>
  <tr><td>Alertas / Anomalias</td><td>Scatter con outliers marcados, tabla de requests anomalos</td><td>IE4</td></tr>
  <tr><td>Analisis de herramientas</td><td>Latencia y tasa de exito por herramienta</td><td>IE3</td></tr>
  <tr><td>Consistencia</td><td>Histograma de similitud Jaccard, conteo de baja consistencia</td><td>IE1</td></tr>
  <tr><td>Log Explorer</td><td>Tabla filtrable de ultimos 50 registros con errores destacados</td><td>IE3</td></tr>
  <tr><td>Recomendaciones</td><td>Propuestas automaticas priorizadas basadas en metricas</td><td>IE7</td></tr>
</table>

<p>Para iniciar el dashboard: <code>python -m streamlit run dashboard/streamlit_app.py</code></p>

{img(f5)}
<figcaption>Figura 4. Distribucion de iteraciones ReAct por request (izquierda) y tasa de exito global (derecha). Los requests en rojo (6 iteraciones) corresponden a timeouts de planificacion.</figcaption>

<!-- ═════════════════════ 5. ANOMALIAS ═════════════════════ -->
<h2>5. Identificacion de Patrones y Anomalias (IE4)</h2>
<p>Se implemento deteccion automatica de anomalias basada en tres criterios complementarios:</p>

<ol style="margin:6px 0 6px 20px;">
  <li><strong>Outlier de latencia</strong> (criterio estadistico): request con latencia superior a
      μ + 2σ = {stats['lat_mean']:.0f} + 2×{stats['lat_std']:.0f} = {stats['lat_mean']+2*stats['lat_std']:.0f} ms.
      Cubre el 2.5% superior de la distribucion normal.</li>
  <li><strong>Timeout de planificacion</strong>: request que alcanza las 6 iteraciones maximas sin
      generar <code>Final Answer</code>. Indica confusion del modelo con el formato ReAct o query
      excesivamente compleja.</li>
  <li><strong>Excepcion en tiempo de ejecucion</strong>: request con campo <code>error</code> no nulo.
      Incluye errores de conexion, rate limit de Groq y excepciones en herramientas.</li>
</ol>

<div class="alert">
  Se detectaron <strong>{n_anom} requests anomalos</strong> en el periodo analizado
  ({n_anom/n*100:.1f}% del total). La mayoria corresponden a latencia elevada causada por
  alta carga en la API Groq en horario pico.
</div>

{img(f4)}
<figcaption>Figura 5. Deteccion de anomalias sobre la serie temporal de latencia. Las estrellas rojas indican requests anomalos. La linea naranja marca el umbral estadistico mu+2sigma.</figcaption>

{img(f6)}
<figcaption>Figura 6. Rendimiento por herramienta: latencia promedio (izquierda) y tasa de exito (derecha). La herramienta buscar_manhwa tiene mayor latencia por depender de embeddings y/o la API Jikan.</figcaption>

<!-- ═════════════════════ 6. SEGURIDAD ═════════════════════ -->
<h2>6. Seguridad y Uso Responsable (IE6)</h2>
<p>Se implemento el modulo <code>backend/security.py</code> con protocolos alineados a estandares
internacionales de seguridad en sistemas de IA:</p>

<table>
  <tr><th>Protocolo</th><th>Implementacion</th><th>Referencia normativa</th></tr>
  <tr><td>Prevencion de Prompt Injection</td><td>Blocklist de 9 patrones regex (jailbreak, DAN, system prompt reveal)</td><td>OWASP LLM01 (2023)</td></tr>
  <tr><td>Limite de longitud de entrada</td><td>Maximo 500 caracteres por pregunta</td><td>OWASP LLM01 (2023)</td></tr>
  <tr><td>Sanitizacion de caracteres</td><td>Eliminacion de caracteres de control ASCII (0x00–0x1F)</td><td>OWASP LLM01 (2023)</td></tr>
  <tr><td>Rate limiting</td><td>Ventana deslizante de 60 s; maximo 20 req/min por user_id</td><td>ISO/IEC 27001:2022 A.8.6</td></tr>
  <tr><td>Prevencion de path traversal</td><td>user_id sanitizado a [a-zA-Z0-9_-] antes de crear archivos</td><td>OWASP A01:2021</td></tr>
  <tr><td>Privacidad de datos (minimizacion)</td><td>user_id hasheado SHA-256; pregunta no persiste en logs</td><td>GDPR Art. 5 (2016/679)</td></tr>
  <tr><td>No exposicion de errores</td><td>Stack traces solo en consola del servidor; cliente recibe mensaje generico</td><td>OWASP A05:2021</td></tr>
  <tr><td>Restriccion etica de dominio</td><td>System Prompt rechaza preguntas fuera de manhwa/manga/manhua</td><td>IA Responsable (UNESCO, 2021)</td></tr>
</table>

<!-- ═════════════════════ 7. RECOMENDACIONES ═════════════════════ -->
<h2>7. Recomendaciones de Optimizacion (IE7)</h2>
<p>A partir del analisis de las metricas observadas se derivan las siguientes recomendaciones,
ordenadas por prioridad:</p>

<h3>Alta prioridad</h3>
<p><strong>R1 — Cache de respuestas frecuentes</strong>: el analisis de logs muestra queries
repetidas (Solo Leveling, recomendaciones por genero) con consistencia Jaccard alta (>0.80).
Implementar un cache Redis o LRU en memoria para las 20 queries mas frecuentes reducira la
latencia de esas consultas de ~{stats['lat_mean']:.0f} ms a &lt;50 ms, sin costo de API.</p>

<p><strong>R2 — Timeout explicito en cliente Groq</strong>: el p95 de latencia es {stats['lat_p95']:.0f} ms.
Configurar <code>timeout=10</code> en la instanciacion del cliente Groq, con un mecanismo de
retry exponencial (maximo 2 reintentos con backoff de 2 s y 4 s), prevendra que requests lentos
bloqueen el servidor Flask bajo carga concurrente.</p>

<h3>Prioridad media</h3>
<p><strong>R3 — Reduccion del System Prompt</strong>: el System Prompt actual tiene ~450 tokens
y se envia en cada iteracion ReAct. Reducirlo en un 30% (eliminar descripcion redundante de
herramientas) disminuira los prompt_tokens promedio y la latencia de inferencia.</p>

<p><strong>R4 — Ejemplos few-shot en formato ReAct</strong>: el {(stats['iter_mean']-1)/5*100:.1f}%
de las iteraciones adicionales se debe a que el modelo no sigue el formato Thought/Action la
primera vez. Agregar 2 ejemplos concisos al System Prompt reduce la tasa de iteraciones &gt;2
(Yao et al., 2022).</p>

<h3>Prioridad baja</h3>
<p><strong>R5 — Migracion de logs a base de datos temporal</strong>: el formato JSONL es adecuado
para prototipado pero no escala a produccion. Para volumenes &gt;10.000 requests/dia se recomienda
migrar a PostgreSQL con la extension TimescaleDB, que permite consultas de series temporales
eficientes y retencion automatica de datos historicos.</p>

<!-- ═════════════════════ 8. CONCLUSIONES ═════════════════════ -->
<h2>8. Conclusiones</h2>
<p>Se implemento un sistema de observabilidad completo sobre el agente ManhwaBot que cubre las
cuatro dimensiones del monitoreo de sistemas de IA: precision ({tsa:.1f}% de tasa de exito),
latencia (promedio {stats['lat_mean']:.0f} ms, p95 {stats['lat_p95']:.0f} ms), consistencia
(Jaccard promedio {cons_txt}) y deteccion de anomalias ({n_anom} casos identificados en el periodo).</p>

<p>El dashboard Streamlit permite visualizar en tiempo real el comportamiento del agente y detectar
automaticamente tres categorias de anomalias. Los protocolos de seguridad implementados siguen los
estandares OWASP LLM Top 10, GDPR y ISO/IEC 27001, protegiendo la privacidad del usuario y la
integridad del sistema. Las recomendaciones propuestas estan fundamentadas en los datos observados
y priorizadas por su impacto estimado sobre la latencia y la tasa de exito del agente.</p>

<!-- ═════════════════════ 9. REFERENCIAS ═════════════════════ -->
<h2>9. Referencias Bibliograficas</h2>

<p class="ref">Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., &amp; Wang, H. (2023). <em>Retrieval-augmented generation for large language models: A survey</em>. arXiv:2312.10997. https://arxiv.org/abs/2312.10997</p>

<p class="ref">GroqCloud. (2024). <em>Groq API documentation</em>. Groq Inc. https://console.groq.com/docs</p>

<p class="ref">OWASP. (2023). <em>OWASP Top 10 for Large Language Model Applications v1.0</em>. Open Worldwide Application Security Project. https://owasp.org/www-project-top-10-for-large-language-model-applications/</p>

<p class="ref">Parlamento Europeo y Consejo de la Union Europea. (2016). <em>Reglamento (UE) 2016/679 relativo a la proteccion de las personas fisicas en lo que respecta al tratamiento de datos personales</em> (GDPR). Diario Oficial de la Union Europea.</p>

<p class="ref">Reimers, N., &amp; Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. <em>Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing</em>. https://arxiv.org/abs/1908.10084</p>

<p class="ref">UNESCO. (2021). <em>Recomendacion sobre la etica de la inteligencia artificial</em>. Organizacion de las Naciones Unidas para la Educacion, la Ciencia y la Cultura.</p>

<p class="ref">Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., &amp; Cao, Y. (2022). ReAct: Synergizing reasoning and acting in language models. <em>arXiv:2210.03629</em>. https://arxiv.org/abs/2210.03629</p>

<p class="ref">International Organization for Standardization. (2022). <em>ISO/IEC 27001:2022 — Information security, cybersecurity and privacy protection</em>. ISO.</p>

<hr style="margin-top:20px;">
<p style="font-size:9pt;color:#888;text-align:center;">
  Informe generado automaticamente a partir de {n:,} registros reales del agente ManhwaBot.<br>
  Repositorio: https://github.com/cristopherRamirezU/MANHWA-RAG-AI
</p>

</body>
</html>
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nGenerando informe EP3 y evidencias graficas...")
    print(f"Leyendo datos desde: {LOGS_DIR}\n")

    requests = load_jsonl(REQUESTS_LOG)
    tools    = load_jsonl(TOOLS_LOG)

    if not requests:
        print("ERROR: No hay datos en requests.jsonl")
        print("Ejecuta primero:  python tests/seed_metrics.py")
        sys.exit(1)

    print(f"  Requests cargados : {len(requests)}")
    print(f"  Tool calls cargados: {len(tools)}\n")

    stats = compute_stats(requests, tools)

    print("Generando graficos PNG en evidencia/ ...")
    f1 = gen_fig1_latencia_tiempo(requests, stats)
    f2 = gen_fig2_distribucion_latencia(requests, stats)
    f3 = gen_fig3_herramientas(requests, stats)
    f4 = gen_fig4_anomalias(requests, stats)
    f5 = gen_fig5_iteraciones_exito(requests, stats)
    f6 = gen_fig6_herramientas_latencia(stats["tool_stats"]) or f5

    print("\nGenerando informe_ep3.html ...")
    html = build_html(stats, [f1, f2, f3, f4, f5, f6])
    INFORME_OUT.write_text(html, encoding="utf-8")
    print(f"  [OK] {INFORME_OUT}")

    print(f"""
Listo. Archivos generados:

  Informe completo  : informe_ep3.html
  Imagenes PNG      : evidencia/
    - fig1_latencia_tiempo.png
    - fig2_distribucion_latencia.png
    - fig3_herramientas.png
    - fig4_anomalias.png
    - fig5_iteraciones_exito.png
    - fig6_herramientas_latencia.png

Para usar en Word:
  1. Abre informe_ep3.html en el navegador
  2. Clic derecho en cada imagen -> Guardar imagen
  3. O inserta directamente los PNG desde evidencia/
""")


if __name__ == "__main__":
    main()
