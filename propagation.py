from transformers import pipeline
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from transformers import pipeline
import pickle

# with open('model.pkl', 'rb') as f:
#     classifier = pickle.load(f)
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

# from math import prod

def predict(x,classifier=classifier):
    y_pred = []
    # c = 0
    for i in x:
        # c+=1
        # print(c)
        y_pred.append(classifier(i)[0]['label'])
    return y_pred

def leaky_competing_accumulators(x, model, max_iters=10, decay=0.2, inhibition=0.1, influence=0.3,
                                emotions={'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'}):


    n = len(x)
    predictions = predict(x)
    neutral_penalty = 0.955  # Scale down 'neutral' importance
    accumulators = [{emotion: 0.0 for emotion in emotions} for _ in range(n)]

    # Initialize accumulators with bias against 'neutral'
    for i in range(n):
        if predictions[i] == 'neutral':
            accumulators[i][predictions[i]] = 1.0 * neutral_penalty
        else:
            accumulators[i][predictions[i]] = 1.0

    beliefs = [None] * n

    for iteration in range(max_iters):
        changes = 0
        new_accumulators = [acc.copy() for acc in accumulators]

        for i in range(n):
            total_inhibition = sum(accumulators[i].values())

            for emotion in emotions:
                evidence_strength = 1.0 if predictions[i] == emotion else 0.0

                # Penalize 'neutral'
                if emotion == 'neutral':
                    evidence_strength *= neutral_penalty

                # Apply updates
                new_value = accumulators[i][emotion]
                new_value += evidence_strength
                new_value -= decay * accumulators[i][emotion]
                new_value -= inhibition * (total_inhibition - accumulators[i][emotion])

                # Add influence from neighbors
                if emotion != 'neutral':  # Skip propagating 'neutral'
                    if i > 0:
                        new_value += influence * accumulators[i - 1][emotion]
                    if i < n - 1:
                        new_value += influence * accumulators[i + 1][emotion]

                # Track changes
                if abs(new_value - accumulators[i][emotion]) > 1e-6:
                    changes += 1

                new_accumulators[i][emotion] = new_value

        # Normalize accumulators with 'neutral' penalty
        for i in range(n):
            total = sum(new_accumulators[i].values())
            for emotion in emotions:
                if emotion == 'neutral':
                    new_accumulators[i][emotion] *= neutral_penalty
            total = sum(new_accumulators[i].values())
            for emotion in emotions:
                new_accumulators[i][emotion] /= total

        accumulators = new_accumulators

        # Update beliefs
        for i in range(n):
            beliefs[i] = max(accumulators[i], key=accumulators[i].get)

        if changes == 0:
            print(f"Converged in {iteration + 1} iterations.")
            break

    return beliefs

def loopy_belief_propagation(
    x_test, model_predict, window_size=2, max_iterations=10, convergence_threshold=1e-3
):
    """
    Robust Loopy Belief Propagation for emotion propagation with log-space calculations.
    """
    import numpy as np
    import random
    import math

    emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    num_sentences = len(x_test)

    # Step 1: Initialize beliefs from the model
    initial_predictions = predict(x_test)


    beliefs = []
    for pred in initial_predictions:
        # Use log probabilities to prevent underflow
        belief = {emotion: math.log(0.05) for emotion in emotions}
        belief[pred] = math.log(0.6)  # Higher log probability for predicted emotion
        # belief['neutral'] = math.log(0.11)

        # Convert to probability space for final normalization
        beliefs.append({emotion: math.exp(belief[emotion]) for emotion in emotions})





    # Step 2: Message passing with additive influence
    for iteration in range(max_iterations):


        # Create a copy of beliefs to update
        new_beliefs = [belief.copy() for belief in beliefs]

        # Propagate beliefs between neighboring sentences
        for i in range(num_sentences):
            # Find neighboring sentences
            neighbors = [
                j for j in range(max(0, i - window_size), min(num_sentences, i + window_size + 1))
                if i != j
            ]

            # Compute influence from neighbors
            if neighbors:
                for emotion in emotions:
                    # Compute weighted influence from neighbors
                    neighbor_influence = sum(
                        beliefs[j][emotion] * 0.3 for j in neighbors
                    )

                    # Update belief with neighbor influence
                    new_beliefs[i][emotion] += neighbor_influence

        # Normalize beliefs
        for i in range(num_sentences):
            total = sum(new_beliefs[i].values())
            new_beliefs[i] = {emotion: val/total for emotion, val in new_beliefs[i].items()}

        # Replace old beliefs
        beliefs = new_beliefs

        # Print beliefs after iteration


    # Final predictions
    y_pred = [max(belief, key=belief.get) for belief in beliefs]



    return y_pred

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_bleu(x, the_model,propagate):
  smoothing_function = SmoothingFunction().method1

  true_vals = predict(x)
  print("True vals:", true_vals)
  propagated_vals = propagate(x,the_model)
  print("Propagated vals:", propagated_vals)
  ans = sentence_bleu( [true_vals], propagated_vals ,weights=(0.5, 0.5, 0, 0) , smoothing_function=smoothing_function)

  return ans

def emotion_drift_score(x, the_model, propagate ,n=3):

    true_emotions = predict(x)
    # true_emotions = [i for i in true_emotions ]
    predicted_emotions = propagate(x ,the_model)
    # predicted_emotions = [i for i in predicted_emotions]
    correct_stability = 0
    correct_transitions = 0
    total_transitions = len(true_emotions) - n -1

    for i in range(total_transitions):

        if true_emotions[i] == true_emotions[i + 1]:
            if predicted_emotions[i] == predicted_emotions[i + 1]:
                correct_stability += 1

        else:
            if predicted_emotions[i + 1] == true_emotions[i + 1]:
                correct_transitions += 1

    # Drift score: Proportion of correct stability and transitions
    # print("Total transitons =  ", total_transitions)
    # print("Correct_stability =  ", correct_stability)
    # print("Correct_transions Transitions = ", correct_transitions)
    drift_score = (correct_stability + correct_transitions) / total_transitions
    return drift_score

