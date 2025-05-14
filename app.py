# realtime_dashboard.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
import asyncio, threading, json, time
from aiokafka import AIOKafkaConsumer

# ───────── CONFIG ─────────────────────────────────────────────────────
BROKER      = "localhost:9092"
PRED_TOPIC  = "wisdm_predictions"
STREAM_HZ   = 50                    # used only if you later plot sensor windows

ACTIVITIES = [                      # used for palettes; UI adapts to whatever arrives
    "Walking","Jogging","Stairs","Sitting","Standing","Typing",
    "Brush Teeth","Eat Soup","Eat Chips","Eat Pasta","Drinking",
    "Eat Sandwich","Kicking","Catch","Dribbling","Writing",
    "Clapping","Fold Clothes",
]
SENSORS = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]

# ───────── Streamlit page setup ───────────────────────────────────────
st.set_page_config(
    page_title="Real-Time Activity Recognition Dashboard",
    layout="wide"
)

# ───────── Session-state containers ───────────────────────────────────
if "preds" not in st.session_state:
    st.session_state.preds = []          # list of prediction dicts
if "listener_started" not in st.session_state:
    st.session_state.listener_started = False

# ───────── Async Kafka listener in its own thread ─────────────────────
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
            st.session_state.preds = st.session_state.preds[-300:]  # keep latest 300
            st.experimental_rerun()                                 # trigger UI refresh
    finally:
        await consumer.stop()

def start_listener_thread():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    asyncio.run_coroutine_threadsafe(kafka_listener(), loop)

if not st.session_state.listener_started:
    start_listener_thread()
    st.session_state.listener_started = True

# ───────── TOP-LEVEL HEADING ──────────────────────────────────────────
st.markdown(
    """
    <h2 style='margin-bottom:0.2rem'>
        Real-Time Activity Recognition from Smartphone and Smartwatch Sensors
    </h2>
    <p style='font-style:italic;margin-top:0'>
        Interactive visual dashboard (live Kafka feed)
    </p>
    """,
    unsafe_allow_html=True,
)

# ───────── If no data yet, show placeholder ───────────────────────────
if not st.session_state.preds:
    st.info("Waiting for predictions on Kafka topic “wisdm_predictions”…")
    st.stop()

# quick filters + KPI panel
kpi_col, filter_col = st.columns([1, 1.6])

with filter_col:
    st.markdown("#### Quick Filters")
    # derive unique activities from data in case your model adds new classes
    all_acts = sorted({p["activity"] for p in st.session_state.preds})
    selected_activities = st.multiselect(
        "Activities", all_acts, default=all_acts
    )
    selected_sensors = st.multiselect(
        "Sensors", SENSORS, default=SENSORS
    )

# latest prediction → badge
latest = st.session_state.preds[-1]
badge_colour = "#28a745" if latest["confidence"] >= .8 else \
               "#ff9800" if latest["confidence"] >= .6 else "#d9534f"

with kpi_col:
    st.markdown(
        f"<div style='display:inline-block;padding:12px 24px;border-radius:10px;"
        f"background:{badge_colour};color:white;font-size:28px;font-weight:600;'>"
        f"{latest['activity']} ({latest['confidence']:.2f})</div>",
        unsafe_allow_html=True,
    )
    st.metric("Latency (ms)", f"{(time.time()-latest['ts'])*1000:,.0f}")
    st.metric("Total predictions", f"{len(st.session_state.preds):,}")

st.divider()

# ───────── Build DataFrame from predictions ───────────────────────────
df = pd.DataFrame(st.session_state.preds)
df = df[df["activity"].isin(selected_activities)]

# ───────── Sensor line charts (optional) ───────────────────────────────
# If your producer includes sensor arrays in each message, unpack & plot here.

# ───────── Distribution bar chart ──────────────────────────────────────
dist_df = df["activity"].value_counts().rename_axis("activity").reset_index(name="total")
st.altair_chart(
    alt.Chart(dist_df).mark_bar(cornerRadius=3).encode(
        x=alt.X("activity:N", sort="-y"), y="total:Q",
        tooltip=["activity","total"]
    ).properties(height=300),
    use_container_width=True,
)
st.divider()

# ───────── Confusion matrix & PRF metrics ──────────────────────────────
# If ground-truth not included, we mock y_true = y_pred
y_pred = df["activity"]
y_true = df.get("true_label", y_pred)

cm = pd.crosstab(y_true, y_pred).reindex(index=selected_activities,
                                         columns=selected_activities,
                                         fill_value=0)

st.subheader("Confusion matrix (live)")

cm_long = cm.reset_index().melt("True", var_name="Pred", value_name="Count")
st.altair_chart(
    alt.Chart(cm_long).mark_rect().encode(
        x="Pred:N", y="True:N",
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="greens")),
        tooltip=["True","Pred","Count"]
    ).properties(height=400,width=400),
    use_container_width=False,
)

tp = np.diag(cm)
fp = cm.sum(axis=0).values - tp
fn = cm.sum(axis=1).values - tp
prec = tp / np.where(tp+fp==0, 1, tp+fp)
rec  = tp / np.where(tp+fn==0, 1, tp+fn)
f1   = 2*prec*rec / np.where(prec+rec==0, 1, prec+rec)

prf_df = pd.DataFrame(
    {"precision":prec,"recall":rec,"f1":f1},
    index=selected_activities
).round(2)

st.subheader("Per-class precision · recall · F1")
prf_long = prf_df.reset_index().melt(id_vars="index", var_name="metric", value_name="score")

heat = alt.Chart(prf_long).mark_rect().encode(
    x="metric:N",
    y=alt.Y("index:N", sort=selected_activities),
    color=alt.Color("score:Q", scale=alt.Scale(scheme="blues")),
    tooltip=["index","metric","score"]
).properties(height=350,width=300)

text = alt.Chart(prf_long).mark_text(size=11,color="white").encode(
    x="metric:N",
    y=alt.Y("index:N", sort=selected_activities),
    text=alt.Text("score:Q", format=".2f")
)

st.altair_chart(heat + text, use_container_width=False)

st.caption("Live dashboard – replace dummy model with real LSTM for production.")
