"""Streamlit app for Emotion Propagation in YouTube transcripts.
Improved version with proper state handling, caching and a cleaner UI.
"""
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import matplotlib.pyplot as plt
import preprocess as pre
import propagation as prop
import visualization as visual

# -------------------------------------------------
# Caching helpers
# -------------------------------------------------

@st.cache_resource(show_spinner="Loading emotion classifier …")
def get_classifier():
    """Load the Hugging Face emotion classifier exactly once."""
    return pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")


@st.cache_data(show_spinner="Fetching transcript …")
def fetch_transcript_lines(video_id: str):
    """Download, normalize and return the transcript as a list of lines."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = pre.normalize_yt(transcript=transcript)
    df = pre.normalize_text(text)
    return df.line.values.tolist()


@st.cache_data(show_spinner="Predicting emotions …")
def predict_emotions(lines):
    """Return one emotion label per line using the cached classifier."""
    clf = get_classifier()
    return prop.predict(lines, clf)


# -------------------------------------------------
# Propagation wrapper
# -------------------------------------------------

def run_propagation(algo: str, lines: list[str], params: dict):
    clf = get_classifier()

    if algo == "Leaky Competing Accumulators":
        return prop.leaky_competing_accumulators(
            lines,
            clf,
            max_iters=params["max_iters"],
            decay=params["decay"],
            inhibition=params["inhibition"],
            influence=params["influence"],
        )

    # default → Loopy Belief Propagation
    return prop.loopy_belief_propagation(
        lines,
        clf,
        window_size=params["window_size"],
        max_iterations=params["max_iters"],
    )


# -------------------------------------------------
# Utility
# -------------------------------------------------

def extract_video_id(url_or_id: str) -> str:
    """Robustly extract the video ID from a full URL or short‑code."""
    if "youtube.com/watch?v=" in url_or_id:
        return url_or_id.split("watch?v=")[1].split("&")[0]
    if "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    return url_or_id.strip()


# -------------------------------------------------
# Streamlit layout
# -------------------------------------------------

def sidebar():
    """Sidebar widgets return a dict with the current UI state."""
    with st.sidebar:
        st.header("🎛️ Controls")

        # 1️⃣ YouTube input  --------------------------------------------------
        raw = st.text_input(
            "YouTube URL or ID",
            value="https://www.youtube.com/watch?v=MRtg6A1f2Ko",
            help="Paste a full YouTube link or just a video ID",
        )
        video_id = extract_video_id(raw)

        fetch_clicked = st.button("Fetch transcript", key="fetch")

        # Show a success badge when transcript already loaded
        if st.session_state.get("lines"):
            st.success("Transcript loaded ✅")

        st.divider()

        # 2️⃣ Algorithm choice  ----------------------------------------------
        algorithm = st.radio(
            "Propagation algorithm",
            ("Leaky Competing Accumulators", "Loopy Belief Propagation"),
            key="algorithm",
        )

        params = {}
        if algorithm == "Leaky Competing Accumulators":
            params["max_iters"] = st.slider("Max iterations", 1, 20, 10)
            params["decay"] = st.slider("Decay", 0.0, 1.0, 0.2, 0.05)
            params["inhibition"] = st.slider("Inhibition", 0.0, 1.0, 0.1, 0.05)
            params["influence"] = st.slider("Influence", 0.0, 1.0, 0.3, 0.05)
        else:
            params["window_size"] = st.slider("Window size", 1, 5, 2)
            params["max_iters"] = st.slider("Max iterations", 1, 20, 10)

        run_clicked = st.button("Run propagation", key="run", disabled="lines" not in st.session_state)

    return {
        "video_id": video_id,
        "fetch_clicked": fetch_clicked,
        "algorithm": algorithm,
        "params": params,
        "run_clicked": run_clicked,
    }


# -------------------------------------------------
# Main view
# -------------------------------------------------

def main():
    st.set_page_config(page_title="Emotion Propagation", layout="wide")
    st.title("🧠 Emotion Propagation in YouTube Transcripts")

    ui_state = sidebar()

    # 1️⃣ Fetch transcript ----------------------------------------------
    if ui_state["fetch_clicked"]:
        try:
            lines = fetch_transcript_lines(ui_state["video_id"])
            st.session_state["lines"] = lines
            st.session_state["orig"] = predict_emotions(lines)
            st.session_state.pop("prop", None)  # clear old results
            st.experimental_rerun()
        except Exception as err:
            st.error(f"Failed to fetch transcript: {err}")

    # 2️⃣ Run propagation ---------------------------------------------
    if ui_state["run_clicked"]:
        if "lines" in st.session_state:
            st.session_state["prop"] = run_propagation(
                ui_state["algorithm"],
                st.session_state["lines"],
                ui_state["params"],
            )
            st.experimental_rerun()

    # 3️⃣ Display results ----------------------------------------------
    if "prop" in st.session_state:
        orig = st.session_state["orig"]
        prop_emos = st.session_state["prop"]
        lines = st.session_state["lines"]

        # Metrics
        colm1, colm2 = st.columns(2)
        bleu = prop.get_bleu(lines, get_classifier, prop.loopy_belief_propagation)
        drift = prop.emotion_drift_score(lines, get_classifier, prop.loopy_belief_propagation)
        colm1.metric("BLEU", f"{bleu:.3f}")
        colm2.metric("Emotion Drift", f"{drift:.3f}")

        # Visualisations
        st.subheader("📈 Emotion timeline")
        fig_line, ax_line = plt.subplots(figsize=(10, 3))
        visual.line_chart(orig, prop_emos, ax=ax_line)
        st.pyplot(fig_line)

        st.subheader("📊 Emotion distribution")
        fig_dist = visual.plot_emotion_distribution(orig, prop_emos)
        st.pyplot(fig_dist)

        # Transcript with colours
        with st.expander("💬 Transcript with propagated emotions", expanded=False):
            for txt, emo in zip(lines, prop_emos):
                st.markdown(f"**{emo.upper()}**: {txt}")


if __name__ == "__main__":
    main()
