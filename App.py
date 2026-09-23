"""Streamlit app for English comment sentiment analysis.

Run with: python -m streamlit run App.py
"""

from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


EXAMPLES = [
    "I love this product! It works perfectly and the quality is excellent.",
    "Terrible service. My order arrived broken and nobody helped me.",
    "The package arrived on Tuesday.",
    "Absolutely fantastic experience. I highly recommend it!",
    "I am very disappointed. This is a complete waste of money.",
    "The item is available in three colors.",
    "Fast delivery and friendly staff. Thank you!",
    "The app crashes constantly. I hate using it.",
    "It is not bad at all. I really like it.",
    "The store opens at nine in the morning.",
    "Amazing value for the price. I am really happy with my purchase.",
    "I would never buy this again. Awful quality.",
]
LABELS = ["Positivo", "Negativo", "Neutral"]
COLORS = {"Positivo": "#16856b", "Negativo": "#d1495b", "Neutral": "#65758b"}
MAX_BYTES = 5 * 1024 * 1024
MAX_COMMENTS = 10000
MAX_LENGTH = 10000


@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()


def analyze_comments(comments):
    analyzer = get_analyzer()
    rows = []
    for comment in comments:
        score = analyzer.polarity_scores(comment)["compound"]
        label = "Positivo" if score >= 0.05 else "Negativo" if score <= -0.05 else "Neutral"
        rows.append({"Comentario": comment, "Sentimiento": label, "Puntuacion": score})
    return pd.DataFrame(rows, columns=["Comentario", "Sentimiento", "Puntuacion"])


def decode_file(data):
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252")


def main():
    st.set_page_config(page_title="Comment Pulse | Sentimiento", page_icon=":bar_chart:", layout="wide")
    st.title("Comment Pulse")
    st.markdown("### De comentarios a resultados")
    st.write("Analiza comentarios **en inglés** y descubre su sentimiento en segundos.")

    with st.sidebar:
        st.header("Acerca del análisis")
        st.write("VADER analiza palabras, negaciones, intensificadores y puntuación. No necesita una API ni descargar modelos.")
        st.markdown("**Positivo:** puntuación ≥ 0.05\n\n**Negativo:** puntuación ≤ -0.05\n\n**Neutral:** entre ambos límites.")
        st.caption("La puntuación va de -1 a +1; no representa una probabilidad ni un porcentaje de confianza.")
        st.warning("Introduce texto en inglés. El sarcasmo, el contexto y otros idiomas pueden producir resultados incorrectos.")
        st.caption("Los comentarios se procesan en el servidor de esta app y no se guardan en archivos. Evita subir información personal o confidencial.")

    source = st.radio("Origen de los comentarios", ["Ejemplos internos", "Escribir comentarios", "Subir archivo"], horizontal=True)
    comments = []
    if source == "Ejemplos internos":
        comments = EXAMPLES
    elif source == "Escribir comentarios":
        text = st.text_area("Un comentario en inglés por línea", height=200, placeholder="I love this product!\nThe service was terrible.")
        comments = text.splitlines()
    else:
        uploaded = st.file_uploader("Archivo CSV o TXT (máximo 5 MB)", type=["csv", "txt"])
        st.caption("TXT: un comentario por línea. CSV: primera fila con encabezados; elige la columna de comentarios. Codificación UTF-8 o Windows-1252.")
        if uploaded is None:
            st.info("Sube un archivo para comenzar o selecciona los ejemplos internos.")
            return
        if uploaded.size > MAX_BYTES:
            st.error("El archivo supera el límite de 5 MB.")
            return
        try:
            text = decode_file(uploaded.getvalue())
            if uploaded.name.lower().endswith(".csv"):
                separator = st.selectbox("Separador CSV", ["Coma (,)", "Punto y coma (;)", "Tabulación"])
                delimiter = {"Coma (,)": ",", "Punto y coma (;)": ";", "Tabulación": "\t"}[separator]
                frame = pd.read_csv(StringIO(text), sep=delimiter, dtype=str, keep_default_na=False)
                column = st.selectbox("Columna de comentarios", frame.columns)
                comments = frame[column].tolist()
            else:
                comments = text.splitlines()
        except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, OverflowError):
            st.error("No se pudo leer el archivo. Revisa la codificación, los encabezados y el formato CSV/TXT.")
            return

    comments = [str(comment).strip() for comment in comments if str(comment).strip()]
    if not comments:
        st.info("No hay comentarios para analizar. Escribe texto o carga un archivo con contenido.")
        return
    if len(comments) > MAX_COMMENTS:
        st.error(f"El límite es de {MAX_COMMENTS:,} comentarios. Divide el conjunto en archivos más pequeños.")
        return
    if any(len(comment) > MAX_LENGTH for comment in comments):
        st.error(f"Cada comentario debe tener como máximo {MAX_LENGTH:,} caracteres.")
        return

    with st.spinner("Analizando comentarios..."):
        results = analyze_comments(comments)
    counts = results["Sentimiento"].value_counts().reindex(LABELS, fill_value=0)
    metrics = st.columns(4)
    metrics[0].metric("Comentarios", len(results))
    for slot, label in zip(metrics[1:], LABELS):
        slot.metric(label, int(counts[label]))

    summary = counts.rename_axis("Sentimiento").reset_index(name="Cantidad")
    left, right = st.columns(2)
    with left:
        st.subheader("Cantidad por sentimiento")
        bar = px.bar(summary, x="Sentimiento", y="Cantidad", color="Sentimiento", text="Cantidad", color_discrete_map=COLORS)
        bar.update_layout(showlegend=False, yaxis={"dtick": max(1, int(counts.max()) // 5)})
        st.plotly_chart(bar, width="stretch")
    with right:
        st.subheader("Distribución porcentual")
        pie = px.pie(summary[summary["Cantidad"] > 0], names="Sentimiento", values="Cantidad", color="Sentimiento", color_discrete_map=COLORS, hole=0.55)
        pie.update_traces(textinfo="percent+label")
        st.plotly_chart(pie, width="stretch")

    st.subheader("Detalle de los comentarios")
    selected = st.multiselect("Filtrar resultados", LABELS, default=LABELS)
    filtered = results[results["Sentimiento"].isin(selected)]
    st.dataframe(filtered, hide_index=True, width="stretch", column_config={"Puntuacion": st.column_config.NumberColumn("Puntuación VADER", format="%.4f")})
    st.caption(f"Se muestran {len(filtered)} de {len(results)} comentarios. Los gráficos incluyen todo el conjunto. Los duplicados se conservan.")

    # Neutralize spreadsheet formulas in user-supplied text before CSV export.
    exported = results.copy()
    exported["Comentario"] = exported["Comentario"].map(lambda value: "'" + value if value.startswith(("=", "+", "-", "@")) else value)
    st.download_button("Descargar todos los resultados (CSV)", exported.to_csv(index=False).encode("utf-8-sig"), file_name="resultados_sentimiento.csv", mime="text/csv")


if __name__ == "__main__":
    main()
