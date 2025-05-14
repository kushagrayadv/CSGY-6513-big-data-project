# Streamlit Real-Time Dashboard for Human Activity Recognition – **FIX v2**
# -------------------------------------------------------------------------
# 🔄  What’s new in this version (2025-05-14)
# -------------------------------------------------------------------------
# 1.  **Streamlit watcher crash (torch.classes)**
#     * Added the environment variable **`STREAMLIT_WATCHER_TYPE=none`** which
#       fully disables the file-watcher in newer Streamlit versions, in addition
#       to the older `STREAMLIT_SERVER_ENABLE_FILE_WATCHER=false` flag.
# 2.  **Kafka consumer race-condition**
#     * `aiokafka.AIOKafkaConsumer` must be **started before** it is consumed.
#       A wrapper coroutine now calls `await consumer.start()` and only then
#       begins `consumer.consume()`, preventing the `NoneType.check_errors`
#       crash.
# 3.  **Graceful shutdown**
#     * Ensures `consumer.stop()` executes on exit to close the coordinator.
# -------------------------------------------------------------------------
# Usage:
#     streamlit run realtime_dashboard_fixed.py
# -------------------------------------------------------------------------

# --- 0)  Disable Streamlit hot-reload watcher (buggy with PyTorch) ------------
import os
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_FILE_WATCHER", "false")
os.environ.setdefault("STREAMLIT_WATCHER_TYPE", "none")  # new flag (≥1.32)

# --- 1)  Standard libs -------------------------------------------------------
import asyncio
import threading
import queue
import time
from datetime import datetime

# --- 2)  Third-party libs ----------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --- 3)  Project-specific modules ------------------------------------------
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
# 5)  Helper – run coroutines in a background thread with a fresh event-loop
# ---------------------------------------------------------------------------

def _run_asyncio_loop(coros: list[asyncio.coroutines]):
    """Run *coros* in a new asyncio event-loop inside this thread."""

    async def runner():
        await asyncio.gather(*coros)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner())


def start_kafka_threads() -> threading.Thread:
    """Launch DataProducer + ActivityConsumer in a background daemon thread."""

    producer = DataProducer()

    # ---- Consumer subclass to feed Streamlit queue -------------------------
    class StreamConsumer(ActivityClassifierConsumer):
        async def process_message(self, message):  # noqa: D401
            await super().process_message(message)
            try:
                data = message.value
                PRED_QUEUE.put_nowait({
                    "timestamp": datetime.utcnow(),
                    "subject": data.get("subject_id"),
                    "device": data.get("device"),
                    "gt_code": data.get("activity"),
                    "features": data.get("features"),
                })
            except queue.Full:
                pass  # UI fell behind – drop sample.

    consumer = StreamConsumer()

    # ---- Wrapper to ensure proper consumer lifecycle -----------------------
    async def consumer_pipeline():
        await consumer.start()
        try:
            await consumer.consume()  # runs forever
        finally:
            await consumer.stop()

    # ---- Launch thread -----------------------------------------------------
    thread = threading.Thread(
        target=_run_asyncio_loop,
        args=([
            producer.run(),
            consumer_pipeline(),
        ],),  # <- 1-tuple, as required
        daemon=True,
    )
    thread.start()
    return thread

# ---------------------------------------------------------------------------
# 6)  Streamlit UI setup ------------------------------------------------------

st.set_page_config("HAR – Real-Time Demo", layout="wide")
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
            st.warning("Stream stopped – restart app to resume")

metric_place = st.empty()
chart_container = st.empty()
cm_container = st.empty()

# ---------------------------------------------------------------------------
# 7)  Live-update logic -------------------------------------------------------
# ---------------------------------------------------------------------------

PLOT_EVERY = 25        # refresh charts after this many samples
SAMPLES_TO_SHOW = 300  # keep this many rows for rolling charts

while st.session_state.get("running"):

    drained = 0
    while not PRED_QUEUE.empty():
        rec = PRED_QUEUE.get()
        drained += 1

        # --- Placeholder inference logic (replace with real predictions) -----
        rec.update({
            "pred_code": rec["gt_code"],
            "pred_name": rec["gt_code"],
            "gt_name":   rec["gt_code"],
            "confidence": np.random.rand(),
            "correct": True,
        })
        ACCUM_DF.loc[len(ACCUM_DF)] = rec

    # ---- Metric card --------------------------------------------------------
    if drained:
        if len(ACCUM_DF) > 5_000:
            ACCUM_DF.drop(ACCUM_DF.index[:-5_000], inplace=True)

        acc = ACCUM_DF["correct"].mean() if len(ACCUM_DF) else 0.0
        metric_place.metric("Overall Accuracy", f"{acc*100: .2f}%")

    # ---- Charts -------------------------------------------------------------
    if drained >= PLOT_EVERY:
        latest = ACCUM_DF.tail(SAMPLES_TO_SHOW)

        # Confidence trace ----------------------------------------------------
        fig_conf, ax_conf = plt.subplots(figsize=(10, 2.5))
        ax_conf.plot(latest["confidence"].values, marker=".")
        ax_conf.set_ylim(0, 1)
        ax_conf.set_title("Prediction Confidence (last 300 samples)")
        chart_container.pyplot(fig_conf, clear_figure=True)

        # Confusion matrix ----------------------------------------------------
        crosstab = pd.crosstab(latest["gt_code"], latest["pred_code"],
                               normalize="index")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        sns.heatmap(crosstab, ax=ax_cm, cmap="Blues", cbar=False,
                    annot=True, fmt=".2f")
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Ground Truth")
        cm_container.pyplot(fig_cm, clear_figure=True)

    time.sleep(0.3)

# ---------------------------------------------------------------------------
# 8)  Static summary when stopped -------------------------------------------
# ---------------------------------------------------------------------------

if not st.session_state.get("running") and len(ACCUM_DF):
    st.subheader("Captured Samples – Most Recent 20")
    st.dataframe(ACCUM_DF.tail(20).reset_index(drop=True))

    st.subheader("Final Confusion Matrix")
    crosstab = pd.crosstab(ACCUM_DF["gt_code"], ACCUM_DF["pred_code"],
                           normalize="index")
    fig_stop, ax_stop = plt.subplots(figsize=(5, 5))
    sns.heatmap(crosstab, ax=ax_stop, cmap="Blues", annot=True, fmt=".2f")
    st.pyplot(fig_stop)

st.markdown("---")
st.caption("© 2025 – HAR Real-Time Demo | Powered by Streamlit, Kafka & PyTorch")
