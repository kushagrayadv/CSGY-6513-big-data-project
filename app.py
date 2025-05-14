# Streamlit dashboard for Human Activity Recognition project
# ---------------------------------------------------------
# Run with:  streamlit run app.py
# Make sure the file `val_classification_stats.npz` (generated during model validation)
# is present in the same directory, or update the path in `load_stats()`.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Human Activity Recognition Dashboard", layout="wide")

st.title("🏃‍♂️📱 Human Activity Recognition – Project Showcase")

st.markdown(
    """
    This interactive dashboard summarizes the performance of the **CNN‑BiLSTM classifier** trained on the
    WISDM activity‑recognition dataset.  It is designed as a quick, visually engaging overview that you can
    share with stakeholders while demoing the real‑time Kafka + Spark pipeline.
    """
)

# ---------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_stats(path: str = "val_classification_stats.npz"):
    """Load validation statistics saved by the training notebook."""
    stats = np.load(path)

    y_true = stats["y_true"].astype(int)
    y_pred = stats["y_pred"].astype(int)
    probs   = stats.get("probabilities")  # Optional – may be absent
    class_names = stats.get("class_names")

    # Fallback if class names were not stored
    if class_names is None:
        n_classes = int(max(y_true.max(), y_pred.max()) + 1)
        class_names = np.array([f"Class‑{i}" for i in range(n_classes)])

    return y_true, y_pred, probs, class_names.astype(str)


try:
    y_true, y_pred, probabilities, class_names = load_stats()
except FileNotFoundError:
    st.error(
        "Couldn't find **val_classification_stats.npz**.  Make sure you've run the validation notebook "
        "or point `load_stats()` to the correct file.")
    st.stop()

# ---------------------------------------------------------
# Quick metrics banner
# ---------------------------------------------------------
overall_acc = accuracy_score(y_true, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("📈 Overall Accuracy", f"{overall_acc*100:.2f}%")
col2.metric("🗂️ Total Samples", f"{len(y_true):,}")
col3.metric("🏷️ Classes", f"{len(class_names)}")

st.divider()

# ---------------------------------------------------------
# Confusion matrix & per‑class accuracy
# ---------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

left, right = st.columns([0.65, 0.35])

with left:
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    st.pyplot(fig_cm)

with right:
    st.subheader("Per‑Class Accuracy")
    fig_acc, ax_acc = plt.subplots(figsize=(4, 6))
    sns.barplot(x=per_class_acc, y=class_names, orient="h", palette="Blues_r", ax=ax_acc)
    ax_acc.set_xlim(0, 1)
    ax_acc.set_xlabel("Accuracy")
    ax_acc.set_ylabel("")
    for i, v in enumerate(per_class_acc):
        ax_acc.text(v + 0.02, i, f"{v*100:.1f}%", va="center")
    st.pyplot(fig_acc)

st.divider()

# ---------------------------------------------------------
# Detailed classification report
# ---------------------------------------------------------
report_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=class_names, output_dict=True)).T
st.subheader("Classification Report (precision / recall / F1)")

# Highlight the support column separately for readability
styler = (
    report_df.round(3)
    .style
    .background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])
    .format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:,.0f}"})
)

st.dataframe(styler, use_container_width=True)

# ---------------------------------------------------------
# Confidence distribution (optional)
# ---------------------------------------------------------
if probabilities is not None:
    st.divider()
    st.subheader("Prediction Confidence Distribution")
    max_conf = np.max(probabilities, axis=1)
    fig_conf, ax_conf = plt.subplots(figsize=(8, 3))
    ax_conf.hist(max_conf, bins=30, edgecolor="white")
    ax_conf.set_xlabel("Max Softmax Confidence")
    ax_conf.set_ylabel("Count")
    ax_conf.grid(True, linestyle=":", linewidth=0.5)
    st.pyplot(fig_conf)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.caption("Built with Streamlit | Powered by PyTorch, Spark & Kafka | © 2025 Your Name")
