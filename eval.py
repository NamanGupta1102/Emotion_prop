
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_bleu(x, the_model,propagate):
  smoothing_function = SmoothingFunction().method1

  true_vals = predict(x)
  print("True vals:", true_vals)
  propagated_vals = propagate(x,the_model)
  print("Propagated vals:", propagated_vals)
  ans = sentence_bleu( [true_vals], propagated_vals ,weights=(0.5, 0.5, 0, 0) , smoothing_function=smoothing_function)

  return ans

def emotion_drift_score(x, the_model, propagate = linear_propagate ,n=3):

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

