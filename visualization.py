import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import propagation

def line_chart(true_vals, propagated_vals, ax=None):
    emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    
    emotion_to_index = {emotion: i for i, emotion in enumerate(emotions)}
    
    # Mapping true and propagated emotions to indices
    true_indices = [emotion_to_index[emotion] for emotion in true_vals if emotion in emotion_to_index]
    prop_indices = [emotion_to_index.get(emotion, -1) for emotion in propagated_vals if emotion in emotion_to_index]
    
    # Filter out invalid indices
    true_indices = [index for index in true_indices if index != -1]
    prop_indices = [index for index in prop_indices if index != -1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 8))
    else:
        fig = ax.figure
    
    ax.plot(range(len(true_indices)), true_indices, label="True Emotions", marker='o', linestyle='-', color='blue', markersize=6)
    ax.plot(range(len(prop_indices)), prop_indices, label="Propagated Emotions", marker='x', linestyle='--', color='orange', markersize=6)
    ax.set_title("True vs Propagated Emotions (Line Chart)", fontsize=14)
    ax.set_xlabel("Minute No.", fontsize=12)
    ax.set_ylabel("Emotion", fontsize=12)
    ax.set_xticks(range(0, len(true_vals), 5))
    ax.set_xticklabels(range(0, len(true_vals), 5), rotation=45)
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions)
    ax.legend()
    ax.grid(True, alpha=0.5)

    plt.savefig("line_chart.png")

    return fig  # Return the figure if needed





# import matplotlib.pyplot as plt

def plot_emotion_distribution(true_vals, propagated_vals):
    emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    emotion_counts = [true_vals.count(emotion) for emotion in emotions]
    value_counts = [propagated_vals.count(emotion) for emotion in emotions]

    fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axes
    ax.plot(emotions, emotion_counts, marker='o', color='b', linestyle='-', markersize=8, label='True Emotion Counts')
    ax.plot(emotions, value_counts, marker='o', color='orange', linestyle='-', markersize=8, label='Propagated Emotion Counts')

    ax.set_title('Emotions Distribution', fontsize=14)
    ax.set_xlabel('Emotions', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks(range(len(emotions)))
    ax.set_xticklabels(emotions, rotation=45)
    ax.grid(True)
    ax.legend()

    return fig
