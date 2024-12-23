import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def line_chart(true_vals,propagated_vals):

  emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']



  emotion_to_index = {emotion: i for i, emotion in enumerate(emotions)}
  true_indices = [emotion_to_index[emotion] for emotion in true_vals if emotion in emotion_to_index]
  prop_indices = [emotion_to_index.get(emotion, -1) for emotion in propagated_vals if emotion in emotion_to_index]


  plt.figure(figsize=(18, 8))



  plt.plot(range(len(true_vals)), true_indices, label="True Emotions", marker='o', linestyle='-', color='blue', markersize=6)
  plt.plot(range(len(prop_indices)), prop_indices, label="Propagated Emotions", marker='x', linestyle='--', color='orange', markersize=6)
  plt.title("True vs Propagated Emotions (Line Chart)", fontsize=14)
  plt.xlabel("Index", fontsize=12)
  plt.ylabel("Emotion", fontsize=12)
  plt.xticks(range(0, len(true_vals), 5), rotation=45)
  plt.yticks(range(len(emotions)), emotions)
  plt.legend()
  plt.grid(True, alpha=0.5)


  plt.show()

def plot_emotion_distribution(true_vals, propagated_vals):
    emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']



    emotion_counts = [true_vals.count(emotion) for emotion in emotions]
    value_counts = [propagated_vals.count(emotion) for emotion in emotions]




    plt.plot(emotions, emotion_counts, marker='o', color='b', linestyle='-', markersize=8, label='TrueEmotion Counts')
    plt.title('Emotions Distribution', fontsize=14)
    plt.plot(emotions, value_counts, marker='o', color='orange', linestyle='-', markersize=8, label='Propagated Emotion Counts')

    plt.xlabel('Emotions', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    plt.show()