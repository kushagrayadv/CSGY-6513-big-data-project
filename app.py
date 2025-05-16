import streamlit as st
import asyncio
import threading
import queue
import time
import numpy as np
import pandas as pd
import altair as alt

from kafka_consumer import ActivityClassifierConsumer, WINDOW_SIZE

# ---------------- shared queue -----------------
data_q = queue.Queue(maxsize=1)   # holds (features (100x6), pred_name, confidence)

# Extend the consumer so it pushes into our queue
class StreamlitConsumer(ActivityClassifierConsumer):
    async def process_message(self, message):
        await super().process_message(message)    # keeps console logs / accuracy
        # after superclass prints results we build payload for UI
        features = np.array(message.value["features"])  # (100,6)

        pred_name = self.predicted_name
        conf = self.confidence  # (you can pass real confidence if desired)
        if data_q.full():
            _ = data_q.get_nowait()
        data_q.put((features, pred_name, conf))

# ---------- start the consumer in a background thread ------------
def start_consumer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    consumer = StreamlitConsumer(topic_name="wisdm_predictions")
    loop.run_until_complete(consumer.start())
    loop.run_until_complete(consumer.consume())

threading.Thread(target=start_consumer, daemon=True).start()

# ---------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="Real‑Time Activity Dashboard", layout="wide")
st.title("Real‑Time Activity Recognition")

chart_placeholder = st.empty()
pred_placeholder  = st.empty()

# Colormap for 6 channels
cols = ['ax','ay','az','gx','gy','gz']

while True:
    try:
        features, pred_name, conf = data_q.get(timeout=1)   # wait up to 1s
        # plot
        df = pd.DataFrame(features, columns=cols).reset_index()
        chart = (
            alt.Chart(df)
            .transform_fold(cols, as_=['axis', 'value'])  # melt to long form
            .mark_line(interpolate='monotone', size=2)  # ‘basis’ or ‘monotone’
            .encode(
                x='index:Q',
                y='value:Q',
                color='axis:N'
            )
        )
        chart_placeholder.altair_chart(chart, use_container_width=True)
        # text
        pred_placeholder.markdown(
            f"### Predicted Activity: **{pred_name}**  &nbsp;&nbsp;"
            f"Confidence: `{conf:.2f}`"
        )
    except queue.Empty:
        pred_placeholder.markdown("*Waiting for data…*")
    except Exception as e:
        st.error(f"Error: {e}")
    time.sleep(0.5)      # refresh rate
