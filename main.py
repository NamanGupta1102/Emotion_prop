"""Streamlit app for Emotion Propagation in YouTube transcripts.
Improved version with proper state handling, caching and a cleaner UI.
"""
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from youtube_transcript_api.formatters import TextFormatter
from transformers import pipeline
import matplotlib.pyplot as plt
import preprocess as pre
import propagation as prop
import visualization as visual
import re
from typing import Optional, List
import requests
import time
import random

# -------------------------------------------------
# State management
# -------------------------------------------------

def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "lines" not in st.session_state:
        st.session_state.lines = None
    if "orig" not in st.session_state:
        st.session_state.orig = None
    if "prop" not in st.session_state:
        st.session_state.prop = None
    if "error" not in st.session_state:
        st.session_state.error = None
    if "loading" not in st.session_state:
        st.session_state.loading = False
    if "video_title" not in st.session_state:
        st.session_state.video_title = None
    if "show_instructions" not in st.session_state:
        st.session_state.show_instructions = True
    if "propagating" not in st.session_state:
        st.session_state.propagating = False

# -------------------------------------------------
# Caching helpers
# -------------------------------------------------

@st.cache_resource(show_spinner="Loading emotion classifier …")
def get_classifier():
    """Load the Hugging Face emotion classifier exactly once."""
    return pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")


@st.cache_data(show_spinner="Fetching transcript …", ttl=3600)  # Cache for 1 hour
def fetch_transcript_lines(video_id: str) -> List[str]:
    """Download, normalize and return the transcript as a list of lines."""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Try different methods to get the transcript
            try:
                # First try: Get transcript in default language
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
            except NoTranscriptFound:
                # Second try: Get transcript in English
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            except Exception:
                # Third try: Get any available transcript
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_transcript(['en']).fetch()
            
            # Validate transcript data
            if not transcript:
                raise Exception("Empty transcript received")
            if not isinstance(transcript, list):
                raise Exception(f"Expected list but got {type(transcript)}")
            if len(transcript) == 0:
                raise Exception("Transcript is empty")
                
            # Validate transcript entries
            for i, entry in enumerate(transcript):
                if not isinstance(entry, dict):
                    raise Exception(f"Entry {i} is not a dictionary")
                if 'text' not in entry:
                    raise Exception(f"Entry {i} missing 'text' field")
                if 'start' not in entry:
                    raise Exception(f"Entry {i} missing 'start' field")
                if not isinstance(entry['text'], str):
                    raise Exception(f"Entry {i} 'text' field is not a string")
                if not isinstance(entry['start'], (int, float)):
                    raise Exception(f"Entry {i} 'start' field is not a number")
                
            # Process the transcript
            try:
                text = pre.normalize_yt(transcript=transcript)
                if text is None:
                    raise Exception("Normalization returned None")
                if text.empty:
                    raise Exception("Normalization returned empty DataFrame")
                    
                df = pre.normalize_text(text)
                if df is None:
                    raise Exception("Text normalization returned None")
                if df.empty:
                    raise Exception("Text normalization returned empty DataFrame")
                if 'line' not in df.columns:
                    raise Exception("DataFrame missing 'line' column")
                if df.line.empty:
                    raise Exception("No valid text content found in transcript")
                    
                return df.line.values.tolist()
            except ValueError as ve:
                raise Exception(f"Error processing transcript: {str(ve)}")
            
        except TranscriptsDisabled:
            raise Exception("This video has transcripts disabled")
        except NoTranscriptFound:
            raise Exception("No transcript found for this video. Try a different video.")
        except CouldNotRetrieveTranscript:
            if attempt < max_retries - 1:
                # Add some random delay to avoid rate limiting
                time.sleep(retry_delay + random.random())
                continue
            raise Exception("Could not retrieve transcript. The video might be private, restricted, or temporarily unavailable.")
        except Exception as e:
            error_msg = str(e)
            if "no element found" in error_msg.lower():
                raise Exception("Invalid transcript format received. Try a different video.")
            if attempt < max_retries - 1:
                # Add some random delay to avoid rate limiting
                time.sleep(retry_delay + random.random())
                continue
            raise Exception(f"Error fetching transcript: {error_msg}")
    
    raise Exception("Failed to fetch transcript after multiple attempts. Please try again later.")


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
    """Robustly extract the video ID from various YouTube URL formats."""
    if not url_or_id:
        raise Exception("Please enter a YouTube URL or video ID")
        
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/watch\?.*&v=)([^&\n?#]+)',
        r'youtube\.com\/shorts\/([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            video_id = match.group(1)
            if not video_id or len(video_id) < 10:  # YouTube IDs are typically 11 characters
                raise Exception("Invalid YouTube video ID extracted from URL")
            return video_id
    
    # If no pattern matches, assume it's a direct video ID
    video_id = url_or_id.strip()
    if not video_id or len(video_id) < 10:
        raise Exception("Invalid YouTube video ID format")
    return video_id


def get_video_title(video_id: str) -> str:
    """Get the title of a YouTube video using its ID."""
    try:
        # Using oEmbed API which doesn't require an API key
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['title']
        return "Unknown Title"
    except Exception:
        return "Unknown Title"


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
            help="Paste any YouTube link (watch, shorts, embed) or just a video ID",
        )
        
        try:
            video_id = extract_video_id(raw)
        except Exception as e:
            st.error(str(e))
            video_id = None

        fetch_clicked = st.button("Fetch transcript", key="fetch", disabled=not video_id)

        # Show status messages
        if st.session_state.error:
            st.error(st.session_state.error)
            st.session_state.error = None
        
        if st.session_state.loading:
            st.info("Loading transcript...")
        elif st.session_state.lines:
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
    
    # Initialize session state first
    init_session_state()
    
    st.title("🧠 Emotion Propagation in YouTube Transcripts")
    
    # Add instructions toggle and content
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("ℹ️ Show Instructions"):
            st.session_state.show_instructions = not st.session_state.show_instructions
            st.rerun()
    
    if st.session_state.show_instructions:
        with st.expander("ℹ️ How to use this app", expanded=True):
            st.markdown("""
            ### Quick Start Guide
            
            1. **Enter a YouTube URL**
               - Paste any YouTube video URL (regular, shorts, or embed)
               - Or just paste the video ID
            
            2. **Fetch Transcript**
               - Click the "Fetch transcript" button
               - Wait for the transcript to load
            
            3. **Choose Algorithm**
               - Select between two propagation algorithms:
                 - **Leaky Competing Accumulators**: Better for short-term emotion changes
                 - **Loopy Belief Propagation**: Better for long-term emotion patterns
            
            4. **Adjust Parameters**
               - Tune the algorithm parameters to get desired results
               - Higher influence = stronger emotion propagation
               - Higher decay = faster emotion fading
            
            5. **Run Analysis**
               - Click "Run propagation" to see the results
               - View emotion timeline and distribution
               - Check the transcript with propagated emotions
            
            ### Tips
            - Use videos with clear speech and good audio quality
            - Longer videos may take more time to process
            - Try different parameters to see how they affect the results
            """)

    ui_state = sidebar()

    # 1️⃣ Fetch transcript ----------------------------------------------
    if ui_state["fetch_clicked"]:
        st.session_state.loading = True
        st.session_state.error = None
        try:
            lines = fetch_transcript_lines(ui_state["video_id"])
            st.session_state.lines = lines
            st.session_state.orig = predict_emotions(lines)
            st.session_state.prop = None  # clear old results
            st.session_state.video_title = get_video_title(ui_state["video_id"])
            st.rerun()
        except Exception as err:
            st.session_state.error = str(err)
            st.session_state.lines = None
            st.session_state.orig = None
            st.session_state.prop = None
            st.session_state.video_title = None
            st.rerun()
        finally:
            st.session_state.loading = False

    # 2️⃣ Run propagation ---------------------------------------------
    if ui_state["run_clicked"] and st.session_state.lines:
        st.session_state.propagating = True
        try:
            # Create a placeholder for the loading animation
            loading_placeholder = st.empty()
            
            # Show loading animation
            with loading_placeholder.container():
                st.markdown("""
                <div style='text-align: center; padding: 20px;'>
                    <h3>🔄 Running Emotion Propagation</h3>
                    <p>Processing transcript and analyzing emotions...</p>
                    <p style='color: #666; font-size: 0.9em;'>This may take a few moments depending on the video length</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show a spinner
                st.spinner("Analyzing emotions and propagating them through the transcript...")
            
            # Run the actual propagation
            st.session_state.prop = run_propagation(
                ui_state["algorithm"],
                st.session_state.lines,
                ui_state["params"],
            )
            
            # Clear loading animation
            loading_placeholder.empty()
            st.rerun()
        except Exception as err:
            st.session_state.error = f"Error during propagation: {str(err)}"
            st.rerun()
        finally:
            st.session_state.propagating = False

    # 3️⃣ Display results ----------------------------------------------
    if st.session_state.prop:
        orig = st.session_state.orig
        prop_emos = st.session_state.prop
        lines = st.session_state.lines

        # Display video title
        if st.session_state.video_title:
            st.subheader(f"📺 {st.session_state.video_title}")

        # Get classifier instance
        classifier = get_classifier()

        # Metrics
        colm1, colm2 = st.columns(2)
        bleu = prop.get_bleu(lines, classifier, prop.loopy_belief_propagation)
        drift = prop.emotion_drift_score(lines, classifier, prop.loopy_belief_propagation)
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
