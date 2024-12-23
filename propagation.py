from transformers import pipeline
# classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier",from_pt=True)
# classifier("I love this!")
classifier = pipeline(
    "sentiment-analysis",
    model="michellejieli/emotion_text_classifier",
    from_pt=True,
    device=1
)
import numpy as np
import random
import math
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



    emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    num_sentences = len(x_test)


    initial_predictions = predict(x_test)


    beliefs = []
    for pred in initial_predictions:

        belief = {emotion: math.log(0.05) for emotion in emotions}
        belief[pred] = math.log(0.6)
        belief['neutral'] = math.log(0.11)


        beliefs.append({emotion: math.exp(belief[emotion]) for emotion in emotions})




    for iteration in range(max_iterations):



        new_beliefs = [belief.copy() for belief in beliefs]

        for i in range(num_sentences):

            neighbors = [
                j for j in range(max(0, i - window_size), min(num_sentences, i + window_size + 1))
                if i != j
            ]


            if neighbors:
                for emotion in emotions:

                    neighbor_influence = sum(
                        beliefs[j][emotion] * 0.3 for j in neighbors
                    )


                    new_beliefs[i][emotion] += neighbor_influence


        for i in range(num_sentences):
            total = sum(new_beliefs[i].values())
            new_beliefs[i] = {emotion: val/total for emotion, val in new_beliefs[i].items()}


        beliefs = new_beliefs




    # Final predictions
    y_pred = [max(belief, key=belief.get) for belief in beliefs]



    return y_pred

