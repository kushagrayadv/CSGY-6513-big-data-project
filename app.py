# realtime_dashboard.py  – listen to wisdm_predictions and visualise live
import streamlit as st, pandas as pd, numpy as np, altair as alt, asyncio, json, time
from aiokafka import AIOKafkaConsumer

BROKER       = "localhost:9092"
PRED_TOPIC   = "wisdm_predictions"
STREAM_HZ    = 50

st.set_page_config(page_title="Real-Time Activity Dashboard", layout="wide")

# ─── session state containers ───────────────────────────────────────────
if "preds" not in st.session_state:
    st.session_state.preds = []        # list of latest prediction dicts

# ─── Kafka listener runs once per browser session ───────────────────────
async def kafka_listener():
    consumer = AIOKafkaConsumer(
        PRED_TOPIC,
        bootstrap_servers=BROKER,
        value_deserializer=lambda m: json.loads(m.decode()),
        auto_offset_reset="latest"
    )
    await consumer.start()
    try:
        async for msg in consumer:
            st.session_state.preds.append(msg.value)
            # keep only latest 300 predictions
            st.session_state.preds = st.session_state.preds[-300:]
            st.experimental_rerun()
    finally:
        await consumer.stop()

if "listener_task" not in st.session_state:
    st.session_state.listener_task = asyncio.create_task(kafka_listener())

# ─── UI when no data yet ────────────────────────────────────────────────
if not st.session_state.preds:
    st.info("Waiting for predictions on Kafka topic “wisdm_predictions”…")
    st.stop()

# latest prediction drives badge
latest = st.session_state.preds[-1]
badge_colour = "#28a745" if latest["confidence"] >= .8 else \
               "#ff9800" if latest["confidence"] >= .6 else "#d9534f"

st.markdown(
    f"<h2>Real-Time Activity Recognition from Smartphone & Smartwatch Sensors</h2>"
    f"<div style='display:inline-block;padding:12px 24px;border-radius:10px;"
    f"background:{badge_colour};color:white;font-size:28px;font-weight:600;'>"
    f"{latest['activity']} ({latest['confidence']:.2f})</div>",
    unsafe_allow_html=True,
)
st.metric("Latency (ms)", f"{(time.time()-latest['ts'])*1000:,.0f}")
st.divider()

# ─── Precision / recall / F1 running table ──────────────────────────────
df = pd.DataFrame(st.session_state.preds)
classes = sorted(df["activity"].unique())
y_pred  = df["activity"]
# If ground-truth isn’t available yet we just fake y_true = y_pred
y_true  = df.get("true_label", y_pred)

cm = pd.crosstab(y_true, y_pred).reindex(index=classes, columns=classes, fill_value=0)
tp = np.diag(cm)
prec = tp / np.where(cm.sum(axis=0)==0, 1, cm.sum(axis=0))
rec  = tp / np.where(cm.sum(axis=1)==0, 1, cm.sum(axis=1))
f1   = 2*prec*rec / np.where(prec+rec==0, 1, prec+rec)

prf = (
    pd.DataFrame({"precision":prec, "recall":rec, "f1":f1}, index=classes)
    .round(3)
    .reset_index()
    .melt(id_vars="index", var_name="metric", value_name="score")
)

st.altair_chart(
    alt.Chart(prf).mark_rect().encode(
        x="metric:N",
        y=alt.Y("index:N", title=None),
        color=alt.Color("score:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["index","metric","score"]
    ).properties(height=350, width=280)
    +
    alt.Chart(prf).mark_text(color="white", size=11).encode(
        x="metric:N", y="index:N", text=alt.Text("score:Q", format=".3f")
    ),
    use_container_width=False
)

# You can add the sensor-wave charts here once your consumer
# includes the window’s sensor readings in the Kafka message.
