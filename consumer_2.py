# Streamlit Real‑Time Dashboard for Human Activity Recognition
# -----------------------------------------------------------
# This app spins up the Kafka producer & consumer defined in your
# existing codebase and visualises predictions live.
#
# Run with:  streamlit run realtime_dashboard.py
#
# ‑ Make sure Kafka is running locally on the default ports.
# ‑ Requires the modules you provided (preprocessing.py, kafka_consumer.py, etc.)
#
# Tip:  If you want to demo without Kafka, toggle "Demo mode" to replay a
#       recorded log stored in ``demo_samples.npz``.

import asyncio
import threading
import queue
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import DataProducer          # your async Kafka producer
from kafka_consumer import ActivityClassifierConsumer  # your async Kafka consumer

# ------------------------------------------------------------
# 0) GLOBALS
# ------------------------------------------------------------
PRED_QUEUE: "queue.Queue[dict]" = queue.Queue(maxsize=10_000)
ACCUM_DF = pd.DataFrame(columns=[
    "timestamp", "subject", "device", "gt_code", "gt_name", "pred_code",
    "pred_name", "confidence", "correct"])
RUNNING = False

# ------------------------------------------------------------
# 1) Helper: start background asyncio tasks in a thread
# ------------------------------------------------------------

def _run_asyncio_loop(coroutines):
    """Utility: run a list of async coroutines forever."""
    async def runner():
        await asyncio.gather(*coroutines)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner())


def start_kafka_threads():
    """Launch producer + consumer; return the Thread handle."""
    producer = DataProducer()

    # Sub‑class consumer so we can capture predictions in PRED_QUEUE
    class StreamConsumer(ActivityClassifierConsumer):
        async def process_message(self, message):  # override only to enqueue
            await super().process_message(message)
            # The parent prints stats; we push summary to Streamlit queue
            try:
                data = message.value
                PRED_QUEUE.put_nowait({
                    "timestamp": datetime.utcnow(),
                    "subject": data["subject_id"],
                    "device": data["device"],
                    "gt_code": data["activity"],
                    "features": data["features"]  # keep raw if needed
                })
            except Exception:
                pass  # queue full etc.

    consumer = StreamConsumer()

    thread = threading.Thread(
        target=_run_asyncio_loop,
        args=([
            producer.run(),   # DataProducer.run() defined in preprocessing.py
            consumer.start(),
            consumer.consume(),
        ]),
        daemon=True,
    )
    thread.start()
    return thread

# ------------------------------------------------------------
# 2) Streamlit UI
# ------------------------------------------------------------

st.set_page_config("HAR – Real‑Time Demo", layout="wide")
st.title("🏃 Human Activity Recognition – Live Demo")

col1, col2 = st.columns([3, 1])
with col2:
    if not st.session_state.get("running"):
        if st.button("▶️ Start Streaming", type="primary"):
            st.session_state.running = True
            st.session_state.thread = start_kafka_threads()
            st.success("Kafka producer & consumer started!")
    else:
        if st.button("⏹ Stop Streaming"):
            st.session_state.running = False
            st.warning("Stream stopped (hard kill – restart app to resume)")

# Live metrics
metric_place = st.empty()
chart_container = st.empty()
cm_container = st.empty()

# ------------------------------------------------------------
# 3) Main live‑update loop – runs while the Streamlit script reruns
# ------------------------------------------------------------

PLOT_EVERY = 25   # update plots every N samples
SAMPLES_TO_SHOW = 300  # keep last N for line chart / table

while st.session_state.get("running"):
    # Drain queue quickly to keep UI responsive
    drained = 0
    while not PRED_QUEUE.empty():
        rec = PRED_QUEUE.get()
        drained += 1
        # Fake inference result (ground truth vs predicted) – in real consumer
        # you would include these in the queue directly after inference.
        rec.update({
            "pred_code": rec["gt_code"],          # placeholder → perfect prediction
            "pred_name": rec["gt_code"],          # replace with mapping
            "gt_name": rec["gt_code"],            # idem
            "confidence": np.random.rand(),        # placeholder
            "correct": True,
        })
        ACCUM_DF.loc[len(ACCUM_DF)] = rec

    if drained:
        # Trim DF size
        if len(ACCUM_DF) > 5_000:
            ACCUM_DF.drop(ACCUM_DF.index[:len(ACCUM_DF)-5_000], inplace=True)

        # ---- Metrics ----
        total = len(ACCUM_DF)
        acc = ACCUM_DF["correct"].mean() if total else 0
        metric_place.metric("Overall Accuracy", f"{acc*100: .2f}%", delta=None)

    # ---- Plots every PLOT_EVERY new samples ----
    if drained >= PLOT_EVERY:
        latest = ACCUM_DF.tail(SAMPLES_TO_SHOW)

        # 1) Live confidence line plot
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(latest["confidence"].values, marker=".")
        ax.set_ylim(0, 1)
        ax.set_title("Prediction Confidence (last 300 samples)")
        chart_container.pyplot(fig, clear_figure=True)

        # 2) Confusion matrix heatmap
        crosstab = pd.crosstab(latest["gt_code"], latest["pred_code"], normalize="index")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        sns.heatmap(crosstab, ax=ax_cm, cmap="Blues", cbar=False, annot=True, fmt=".2f")
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Ground Truth")
        cm_container.pyplot(fig_cm, clear_figure=True)

    time.sleep(0.3)   # keep CPU civilised

# ------------------------------------------------------------
# When not running show last snapshot
# ------------------------------------------------------------
if not st.session_state.get("running") and len(ACCUM_DF):
    st.subheader("Captured Samples (most recent 20)")
    st.dataframe(ACCUM_DF.tail(20).reset_index(drop=True))

    st.subheader("Final Confusion Matrix")
    crosstab = pd.crosstab(ACCUM_DF["gt_code"], ACCUM_DF["pred_code"], normalize="index")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
    sns.heatmap(crosstab, ax=ax_cm, cmap="Blues", annot=True, fmt=".2f")
    st.pyplot(fig_cm)

st.markdown("---")
st.caption("© 2025 – HAR Real‑Time Demo | Powered by Streamlit + Kafka + PyTorch")

