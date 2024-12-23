import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import preprocess as pre
import propagation as prop
import visualization as visual

video_id = "zHoFoSH2HYY"
transcript = YouTubeTranscriptApi.get_transcript(video_id)
# print(transcript)

text = pre.normalize_yt(transcript=transcript)
norm = pre.normalize_text(text)
print(norm)
propagated_emos = prop.loopy_belief_propagation(norm,None)
print(propagated_emos)
predicted_emos = prop.predict(norm,None)

plot = visual.line_chart(predicted_emos,propagated_emos)
print(plot)