import streamlit as st
import numpy as np
import plotly.express as px


def plot_result(top_languages):
    top_k = len(top_languages)
    languages = [f'{r["language"]} ({r["code"]})' for r in top_languages]
    scores = np.array([r["score"] for r in top_languages])
    scores *= 100
    fig = px.bar(
        x=scores,
        y=languages,
        orientation='h',
        labels={'x': 'Confidence', 'y': 'Language'},
        text=scores,
        range_x=(0, 115),
        title=f'Top {top_k} Detections',
        color=np.linspace(0, 1, len(scores)),
        color_continuous_scale='Viridis'
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
