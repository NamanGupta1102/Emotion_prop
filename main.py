import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import preprocess as pre
import propagation as prop
import visualization as visual
# import eval as evaluation
from transformers import pipeline
# Title of the App
if __name__ == '__main__':
    # print("This ran: ffffffffffffff")
    classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
    # print("This Finished rinningkkkkkkkkkkkkkkkkk")

# classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

st.title("Emotion Propagation in YouTube Transcripts")
st.sidebar.header("Settings")

# Initialize session state for input parameters and results
if "transcript_fetched" not in st.session_state:
    st.session_state.transcript_fetched = False
if "final_input" not in st.session_state:
    st.session_state.final_input = None
if "original_preds" not in st.session_state:
    st.session_state.original_preds = None
if "propagated_emos" not in st.session_state:
    st.session_state.propagated_emos = None

# Sidebar to accept YouTube Video ID
st.session_state.video_id = st.sidebar.text_input("YouTube Video link", value="https://www.youtube.com/watch?v=MRtg6A1f2Ko", help="Enter the video ID from the YouTube URL")
st.session_state.video_id = st.session_state.video_id.split('=')[-1]
fetch_transcript = st.sidebar.button("Fetch Transcript")

if fetch_transcript:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(st.session_state.video_id)
        text = pre.normalize_yt(transcript=transcript)
        norm = pre.normalize_text(text)
        st.session_state.final_input = norm.line.values
        st.session_state.transcript_fetched = True
        st.session_state.original_preds = prop.predict(st.session_state.final_input,classifier)
        st.write("Transcript successfully fetched and preprocessed!")
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")

if st.session_state.transcript_fetched:
    # Sidebar for Propagation Algorithm Parameters
    st.sidebar.subheader("Propagation Parameters")
    max_iters = st.sidebar.slider("Max Iterations", 1, 20, 10)
    decay = st.sidebar.slider("Decay Rate", 0.0, 1.0, 0.2, step=0.05)
    inhibition = st.sidebar.slider("Inhibition", 0.0, 1.0, 0.1, step=0.05)
    influence = st.sidebar.slider("Influence", 0.0, 1.0, 0.3, step=0.05)
    window_size = st.sidebar.slider("Window Size (Loopy)", 1, 5, 2)

    st.sidebar.subheader("Choose Propagation Algorithm")
    algorithm = st.sidebar.radio("Algorithm", ["Leaky Competing Accumulators", "Loopy Belief Propagation"])

    # Apply Propagation Algorithm
    if algorithm == "Leaky Competing Accumulators":
        st.session_state.propagated_emos = prop.leaky_competing_accumulators(
            st.session_state.final_input, classifier, max_iters, decay, inhibition, influence
        )
    else:
        st.session_state.propagated_emos = prop.loopy_belief_propagation(
            st.session_state.final_input, classifier, window_size, max_iters
        )

    # Evaluate Performance
    st.subheader("Evaluation Metrics")
    bleu_score = prop.get_bleu(st.session_state.final_input, classifier, prop.loopy_belief_propagation)
    drift_score = prop.emotion_drift_score(st.session_state.final_input, classifier, prop.loopy_belief_propagation)

    st.metric(label="BLEU Score", value=f"{bleu_score:.3f}")
    st.metric(label="Emotion Drift Score", value=f"{drift_score:.3f}")

    # Visualization
    st.subheader("Visualization")
    import matplotlib.pyplot as plt

    st.subheader("Visualization")
    # fig, ax = plt.subplots()  # Create a figure and axes
    # visual.line_chart(st.session_state.original_preds, st.session_state.propagated_emos, ax=ax)
    # st.pyplot(fig)  # Pass the figure to st.pyplot

    # Line chart for predictions
    fig_line, ax_line = plt.subplots(figsize=(8, 6))  # Create figure and axes
    visual.line_chart(st.session_state.original_preds, st.session_state.propagated_emos, ax=ax_line)  # Pass ax explicitly
    st.pyplot(fig_line)

    # Emotion distribution plot
    st.subheader("Emotion Distribution")
    fig_distribution = visual.plot_emotion_distribution(st.session_state.original_preds, st.session_state.propagated_emos)
    st.pyplot(fig_distribution)

