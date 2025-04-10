import streamlit as st
import time
import os
from youtube_transcript_api import YouTubeTranscriptApi
import preprocess as pre
import propagation as prop
import visualization as visual
from transformers import pipeline
import matplotlib.pyplot as plt

# Set page configuration and theme
st.set_page_config(
    page_title="Emotion Propagation in YouTube Transcripts",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This app analyzes emotions in YouTube video transcripts and applies propagation algorithms."
    }
)

# Apply dark theme
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .stMetric {
        background-color: #1E2129;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .css-1v3fvcr {
        background-color: #0E1117;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .css-1kyxreq {
        justify-content: center;
        align-items: center;
    }
    .stAlert {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "transcript_fetched" not in st.session_state:
    st.session_state.transcript_fetched = False
if "final_input" not in st.session_state:
    st.session_state.final_input = None
if "original_preds" not in st.session_state:
    st.session_state.original_preds = None
if "propagated_emos" not in st.session_state:
    st.session_state.propagated_emos = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "classifier" not in st.session_state:
    st.session_state.classifier = None
if "video_id" not in st.session_state:
    st.session_state.video_id = ""
if "algorithm" not in st.session_state:
    st.session_state.algorithm = "Leaky Competing Accumulators"
if "propagation_params" not in st.session_state:
    st.session_state.propagation_params = {
        "max_iters": 10,
        "decay": 0.2,
        "inhibition": 0.1,
        "influence": 0.3,
        "window_size": 2
    }

# Function to load model with caching
@st.cache_resource
def load_emotion_classifier():
    return pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    if "youtube.com/watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        return url  # Assume it's already a video ID

# Function to fetch and process transcript
def fetch_and_process_transcript(video_id):
    try:
        with st.spinner("Fetching transcript from YouTube..."):
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        with st.spinner("Processing transcript..."):
            text = pre.normalize_yt(transcript=transcript)
            norm = pre.normalize_text(text)
            st.session_state.final_input = norm.line.values
            st.session_state.transcript_fetched = True
        
        return True
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return False

# Function to predict emotions with caching
@st.cache_data
def predict_emotions(input_text, _classifier):
    with st.spinner("Predicting emotions..."):
        return prop.predict(input_text, _classifier)

# Function to apply propagation algorithm
def apply_propagation(algorithm, input_text, classifier, params):
    with st.spinner(f"Applying {algorithm} algorithm..."):
        if algorithm == "Leaky Competing Accumulators":
            return prop.leaky_competing_accumulators(
                input_text, classifier, 
                max_iters=params["max_iters"], 
                decay=params["decay"], 
                inhibition=params["inhibition"], 
                influence=params["influence"]
            )
        else:
            return prop.loopy_belief_propagation(
                input_text, classifier, 
                window_size=params["window_size"], 
                max_iterations=params["max_iters"]
            )

# Function to calculate metrics with caching
@st.cache_data
def calculate_metrics(input_text, classifier, algorithm_func):
    bleu_score = prop.get_bleu(input_text, classifier, algorithm_func)
    drift_score = prop.emotion_drift_score(input_text, classifier, algorithm_func)
    return bleu_score, drift_score

# Main app layout
def main():
    # App header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://em-content.zobj.net/thumbs/120/apple/325/face-with-monocle_1f9d0.png", width=80)
    with col2:
        st.title("Emotion Propagation in YouTube Transcripts")
        st.markdown("Analyze and visualize emotions in YouTube video transcripts using propagation algorithms.")
    
    # Sidebar for inputs and parameters
    with st.sidebar:
        st.header("📊 Controls")
        
        # YouTube video input
        st.subheader("1️⃣ YouTube Video")
        video_url = st.text_input(
            "Enter YouTube Video URL or ID",
            value="https://www.youtube.com/watch?v=MRtg6A1f2Ko",
            help="Paste a YouTube URL or video ID"
        )
        
        # Extract video ID
        video_id = extract_video_id(video_url)
        st.session_state.video_id = video_id
        
        # Fetch transcript button
        fetch_col1, fetch_col2 = st.columns([3, 1])
        with fetch_col1:
            fetch_transcript = st.button("Fetch Transcript", use_container_width=True)
        with fetch_col2:
            if st.session_state.transcript_fetched:
                st.success("✓")
        
        # Only show algorithm settings if transcript is fetched
        if st.session_state.transcript_fetched:
            st.divider()
            
            # Algorithm selection
            st.subheader("2️⃣ Algorithm Selection")
            algorithm = st.radio(
                "Choose Propagation Algorithm",
                ["Leaky Competing Accumulators", "Loopy Belief Propagation"],
                index=0 if st.session_state.algorithm == "Leaky Competing Accumulators" else 1
            )
            st.session_state.algorithm = algorithm
            
            # Algorithm parameters
            st.subheader("3️⃣ Algorithm Parameters")
            
            # Different parameters based on selected algorithm
            if algorithm == "Leaky Competing Accumulators":
                st.session_state.propagation_params["max_iters"] = st.slider(
                    "Max Iterations", 1, 20, st.session_state.propagation_params["max_iters"]
                )
                st.session_state.propagation_params["decay"] = st.slider(
                    "Decay Rate", 0.0, 1.0, st.session_state.propagation_params["decay"], step=0.05
                )
                st.session_state.propagation_params["inhibition"] = st.slider(
                    "Inhibition", 0.0, 1.0, st.session_state.propagation_params["inhibition"], step=0.05
                )
                st.session_state.propagation_params["influence"] = st.slider(
                    "Influence", 0.0, 1.0, st.session_state.propagation_params["influence"], step=0.05
                )
            else:
                st.session_state.propagation_params["window_size"] = st.slider(
                    "Window Size", 1, 5, st.session_state.propagation_params["window_size"]
                )
                st.session_state.propagation_params["max_iters"] = st.slider(
                    "Max Iterations", 1, 20, st.session_state.propagation_params["max_iters"]
                )
            
            # Apply algorithm button
            apply_algorithm = st.button("Apply Algorithm", type="primary", use_container_width=True)
            
            # Information about the algorithms
            with st.expander("ℹ️ About the Algorithms"):
                st.markdown("""
                **Leaky Competing Accumulators**: This algorithm models emotions as competing accumulators that leak over time, with neighboring sentences influencing each other.
                
                **Loopy Belief Propagation**: This algorithm propagates emotion beliefs between neighboring sentences using message passing in a probabilistic graphical model.
                """)
    
    # Main content area
    if not st.session_state.transcript_fetched:
        # Welcome screen when no transcript is loaded
        st.info("👈 Enter a YouTube URL in the sidebar and click 'Fetch Transcript' to begin analysis.")
        
        # Example video thumbnails
        st.subheader("Try these example videos:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://img.youtube.com/vi/MRtg6A1f2Ko/0.jpg", use_container_width=True)
            st.caption("TED Talk")
        with col2:
            st.image("https://img.youtube.com/vi/8S0FDjFBj8o/0.jpg", use_container_width=True)
            st.caption("Motivational Speech")
        with col3:
            st.image("https://img.youtube.com/vi/UF8uR6Z6KLc/0.jpg", use_container_width=True)
            st.caption("Steve Jobs Speech")
    else:
        # Load model if not already loaded
        if not st.session_state.model_loaded:
            with st.spinner("Loading emotion classification model..."):
                st.session_state.classifier = load_emotion_classifier()
                st.session_state.model_loaded = True
        
        # Process fetch transcript button click
# Process fetch transcript button click
        if fetch_transcript:
            # Load model if not already loaded
            if not st.session_state.model_loaded:
                with st.spinner("Loading emotion classification model..."):
                    st.session_state.classifier = load_emotion_classifier()
                    st.session_state.model_loaded = True
                    
            success = fetch_and_process_transcript(st.session_state.video_id)
            if success:
                # Predict original emotions
                st.session_state.original_preds = predict_emotions(
                    st.session_state.final_input, 
                    st.session_state.classifier
                )
                st.experimental_rerun()  # Force a rerun to update the UI
        
        # Process apply algorithm button click
        if "apply_algorithm" in locals() and apply_algorithm:
            if st.session_state.original_preds is not None:
                # Apply selected propagation algorithm
                st.session_state.propagated_emos = apply_propagation(
                    st.session_state.algorithm,
                    st.session_state.final_input,
                    st.session_state.classifier,
                    st.session_state.propagation_params
                )
        
        # Display results if propagation has been applied
        if st.session_state.propagated_emos is not None:
            # YouTube video embed
            video_embed_col1, video_embed_col2 = st.columns([2, 1])
            with video_embed_col1:
                st.subheader("📺 YouTube Video")
                st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
            
            with video_embed_col2:
                st.subheader("📊 Evaluation Metrics")
                
                # Calculate metrics
                if st.session_state.algorithm == "Leaky Competing Accumulators":
                    algorithm_func = prop.leaky_competing_accumulators
                else:
                    algorithm_func = prop.loopy_belief_propagation
                
                bleu_score, drift_score = calculate_metrics(
                    st.session_state.final_input,
                    st.session_state.classifier,
                    algorithm_func
                )
                
                # Display metrics in a nice format
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("BLEU Score", f"{bleu_score:.3f}")
                with metric_col2:
                    st.metric("Emotion Drift Score", f"{drift_score:.3f}")
                
                # Explanation of metrics
                with st.expander("ℹ️ What do these metrics mean?"):
                    st.markdown("""
                    **BLEU Score**: Measures the similarity between the original and propagated emotion sequences.
                    
                    **Emotion Drift Score**: Measures how well the propagation algorithm preserves emotion transitions and stability.
                    """)
            
            # Visualizations
            st.subheader("📈 Emotion Analysis Visualizations")
            
            # Line chart for predictions
            st.markdown("### Emotion Progression Over Time")
            fig_line, ax_line = plt.subplots(figsize=(10, 6))
            visual.line_chart(st.session_state.original_preds, st.session_state.propagated_emos, ax=ax_line)
            st.pyplot(fig_line)
            
            # Emotion distribution plot
            st.markdown("### Emotion Distribution")
            fig_distribution = visual.plot_emotion_distribution(st.session_state.original_preds, st.session_state.propagated_emos)
            st.pyplot(fig_distribution)
            
            # Emotion breakdown
            st.subheader("🔍 Emotion Breakdown")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Original vs Propagated", "Transcript with Emotions"])
            
            with tab1:
                # Display side-by-side comparison of original and propagated emotions
                comparison_df = {
                    "Minute": list(range(len(st.session_state.original_preds))),
                    "Original Emotion": st.session_state.original_preds,
                    "Propagated Emotion": st.session_state.propagated_emos
                }
                
                # Convert to DataFrame for display
                import pandas as pd
                comparison_df = pd.DataFrame(comparison_df)
                st.dataframe(comparison_df, use_container_width=True)
            
            with tab2:
                # Display transcript text with corresponding emotions
                if len(st.session_state.final_input) == len(st.session_state.propagated_emos):
                    for i, (text, emotion) in enumerate(zip(st.session_state.final_input, st.session_state.propagated_emos)):
                        # Create color-coded emotion labels
                        emotion_colors = {
                            "anger": "#FF5252",
                            "disgust": "#8BC34A",
                            "fear": "#9C27B0",
                            "joy": "#FFD600",
                            "neutral": "#78909C",
                            "sadness": "#2196F3",
                            "surprise": "#FF9800"
                        }
                        
                        color = emotion_colors.get(emotion, "#78909C")
                        
                        # Display text with emotion badge
                        st.markdown(f"""
                        <div style="margin-bottom: 10px; padding: 10px; border-radius: 5px; background-color: #1E2129;">
                            <span style="background-color: {color}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8em; margin-right: 10px;">
                                {emotion.upper()}
                            </span>
                            <span>{text}</span>
                        </div>
                        """, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()
