# Streamlit Real‑Time Dashboard for Human Activity Recognition – **FIXED**
# -------------------------------------------------------------------------
# This version addresses two runtime issues that surfaced on Python 3.13:
#   1. Streamlit's file‑watcher crashing when it inspects `torch.classes`.
#      → We disable the watcher via an env‑var **before** importing Streamlit.
#   2. A `TypeError` caused by mis‑shaped `args` when launching the background
#      asyncio loop in a thread.  A trailing comma turns the list into the
#      single tuple argument the helper expects.
# -------------------------------------------------------------------------
# Usage:
#     streamlit run realtime_dashboard_fixed.py
# -------------------------------------------------------------------------

# --- 0)  Disable Streamlit hot‑reload watcher (buggy with PyTorch + Python 3.13)
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# --- 1)  Standard libs ------------------------------------------------------
import asyncio
import threading
import queue
import time
from datetime import datetime
from pathlib import Path

# --- 2)  Third‑party libs ----------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --- 3)  Project‑specific imports ------------------------------------------
from preprocessing import DataProducer                    # async Kafka producer
from kafka_consumer import ActivityClassifierConsumer      # async Kafka consumer

# ---------------------------------------------------------------------------
# 4)  Globals
# ---------------------------------------------------------------------------
PRED_QUEUE: "queue.Queue[dict]" = queue.Queue(maxsize=10_000)
ACCUM_DF = pd.DataFrame(columns=[
    "timestamp", "subject", "device", "gt_code", "gt_name", "pred_code",
    "pred_name", "confidence", "correct"
])

# ---------------------------------------------------------------------------
# 5)  Helper – run multiple coroutines in their own event‑loop thread
# ---------------------------------------------------------------------------

def _run_asyncio_loop(coros: list[asyncio.coroutines]):
    """Spin up a fresh event loop and drive *coros* until they return."""

    async def runner():
        await asyncio.gather(*coros)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner())


def start_kafka_threads() -> threading.Thread:
    """Launch DataProducer + ActivityConsumer in a background thread."""

    producer = DataProducer()

    # ---- Wrap the consumer so we can push predictions into Streamlit ----------
    class StreamConsumer(ActivityClassifierConsumer):
        async def process_message(self, message):  # noqa: D401 – simple verb
            await super().process_message(message)
            try:
                data = message.value
                PRED_QUEUE.put_nowait({
                    "timestamp": datetime.utcnow(),
                    "subject": data["subject_id"],
                    "device": data["device"],
                    "gt_code": data["activity"],
                    "features": data["features"],  # raw window – handy for debug
                })
            except queue.Full:
                pass  # Drop if UI falls behind.

    consumer = StreamConsumer()

    # ---- Launch thread (***note the trailing comma inside *args*) -------------
    thread = threading.Thread(
        target=_run_asyncio_loop,
        args=([
            producer.run(),     # coroutine
            consumer.start(),   # coroutine
            consumer.consume(), # coroutine
        ],),                    # <-- trailing comma turns list into 1‑tuple
        daemon=True,
    )
    thread.start()
    return thread

# ---------------------------------------------------------------------------
# 6)  Streamlit UI setup
# ---------------------------------------------------------------------------

st.set_page_config("HAR – Real‑Time Demo", layout="wide")
st.title("🏃 Human Activity Recognition – Live Demo")

col_main, col_controls = st.columns([3, 1])

with col_controls:
    if not st.session_state.get("running"):
        if st.button("▶️ Start Streaming", type="primary"):
            st.session_state.running = True
            st.session_state.thread = start_kafka_threads()
            st.success("Kafka producer & consumer started!")
    else:
        if st.button("⏹ Stop Streaming"):
            st.session_state.running = False
            st.warning("Stream stopped (hard kill – restart app to resume)")

metric_place = st.empty()
chart_container = st.empty()
cm_container = st.empty()

# ---------------------------------------------------------------------------
# 7)  Live‑update loop (runs on every Streamlit rerun)
# ---------------------------------------------------------------------------

PLOT_EVERY = 25       # refresh charts after this many samples arrive
SAMPLES_TO_SHOW = 300 # keep this many for the line chart + table

while st.session_state.get("running"):

    # ---- 7a) Drain queue fast -------------------------------------------------
    drained = 0
    while not PRED_QUEUE.empty():
        rec = PRED_QUEUE.get()
        drained += 1

        # --- Placeholder inference logic (replace with real consumer output) ---
        rec.update({
            "pred_code": rec["gt_code"],  # echo ground‑truth – replace!
            "pred_name": rec["gt_code"],
            "gt_name":   rec["gt_code"],
            "confidence": np.random.rand(),
            "correct": True,
        })
        ACCUM_DF.loc[len(ACCUM_DF)] = rec

    # ---- 7b) Update metric ----------------------------------------------------
    if drained:
        if len(ACCUM_DF) > 5_000:
            ACCUM_DF.drop(ACCUM_DF.index[:-5_000], inplace=True)

        total = len(ACCUM_DF)
        acc = ACCUM_DF["correct"].mean() if total else 0.0
        metric_place.metric("Overall Accuracy", f"{acc*100: .2f}%")

    # ---- 7c) Plot every *PLOT_EVERY* samples ---------------------------------
    if drained >= PLOT_EVERY:
        latest = ACCUM_DF.tail(SAMPLES_TO_SHOW)

        # Confidence trace ------------------------------------------------------
        fig_conf, ax_conf = plt.subplots(figsize=(10, 2.5))
        ax_conf.plot(latest["confidence"].values, marker=".")
        ax_conf.set_ylim(0, 1)
        ax_conf.set_title("Prediction Confidence (last 300 samples)")
        chart_container.pyplot(fig_conf, clear_figure=True)

        # Confusion matrix ------------------------------------------------------
        crosstab = pd.crosstab(latest["gt_code"], latest["pred_code"],
                               normalize="index")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        sns.heatmap(crosstab, ax=ax_cm, cmap="Blues", cbar=False,
                    annot=True, fmt=".2f")
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Ground Truth")
        cm_container.pyplot(fig_cm, clear_figure=True)

    time.sleep(0.3)  # keep CPU usage reasonable

# ---------------------------------------------------------------------------
# 8)  Static view when stopped ------------------------------------------------
# ---------------------------------------------------------------------------

if not st.session_state.get("running") and len(ACCUM_DF):
    st.subheader("Captured Samples (most recent 20)")
    st.dataframe(ACCUM_DF.tail(20).reset_index(drop=True))

    st.subheader("Final Confusion Matrix")
    crosstab = pd.crosstab(ACCUM_DF["gt_code"], ACCUM_DF["pred_code"],
                           normalize="index")
    fig_stop, ax_stop = plt.subplots(figsize=(5, 5))
    sns.heatmap(crosstab, ax=ax_stop, cmap="Blues", annot=True, fmt=".2f")
    st.pyplot(fig_stop)

st.markdown("---")
st.caption("© 2025 – HAR Real‑Time Demo | Powered by Streamlit, Kafka & PyTorch")
