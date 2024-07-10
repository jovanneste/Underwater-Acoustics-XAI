import tensorflow as tf
import pickle
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import lime
from lime import lime_image
import shap
from skimage.segmentation import mark_boundaries


def plt_spectrogram(segment):
    Xdb_normalized = segment[:,:,0]
    Xdb_normalized = np.flipud(Xdb_normalized)
    plt.figure(figsize=(6, 8))
    librosa.display.specshow(Xdb_normalized, sr=52137, hop_length=1024, x_axis='time', y_axis='mel', cmap='plasma')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def predict_fn(spectrograms):
    interpreter.resize_tensor_input(input_details[0]['index'], spectrograms.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], spectrograms.astype(np.float32))
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions

def plot_explanation_side_by_side(flipped_spectrogram, explanation, num_classes=4):
    fig, axs = plt.subplots(1, num_classes + 1, figsize=(20, 5))  # Adjust the figure size as needed

    # Plot original spectrogram
    axs[0].imshow(flipped_spectrogram[:, :, 0], aspect='auto', cmap='plasma')
    axs[0].set_title('Original Spectrogram')
    axs[0].axis('off')

    # Plot explanations for each class
    for i in range(num_classes):
        temp, mask = explanation.get_image_and_mask(
            i, positive_only=True, num_features=10, hide_rest=True
        )
        axs[i + 1].imshow(mark_boundaries(flipped_spectrogram, mask))
        axs[i + 1].set_title(f'Explanation for class {i}')
        axs[i + 1].axis('off')

    plt.tight_layout()
    plt.show()
    

def show_lime_x(x):
    explainer = lime_image.LimeImageExplainer()
    spectrogram_input = np.repeat(x[0].astype('double'), 3, axis=-1)
    
    explanation = explainer.explain_instance(
        spectrogram_input,
        predict_fn,
        top_labels=4,
        hide_color=0,
        num_samples=1000
    )
    
    plot_explanation_side_by_side(spectrogram_input, explanation)
    
def get_specs(i):
    ls, spectrograms = [], []
    features = all_features[i].reshape(60, 51, 3)
    ls.append(all_labels[i])
    spectrograms.append(np.array(features[:, :, 0]).reshape(60, 51, 1))

    all_spectrograms = np.array(spectrograms).reshape(1, 60, 51, 1)
    labels = [[1 if isinstance(d[key], list) else d[key] for key in [14,5,4,0]] for d in ls]

    array = predict_fn(all_spectrograms)
    max_indices = np.argmax(array, axis=1)
    onehot_preds = np.zeros_like(array)
    onehot_preds[np.arange(array.shape[0]), max_indices] = 1
    
    return all_spectrograms, labels, array, onehot_preds
    
def explain(i):
    spectrogram, label, predictions, onehot_preds = get_specs(i)
    print(label)
    print(predictions, onehot_preds)
    show_lime_x(spectrogram) 

    
interpreter = tf.lite.Interpreter(model_path="../HTS01-AI-pipeline/ID-VesNet_models/M1_4.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load data
with open('/mnt/Data1/Acoustics/ready_for_AI/features/allnew.pkl', 'rb') as f:
    all_features = pickle.load(f)
with open('/mnt/Data1/Acoustics/ready_for_AI/labels/allnew.pkl', 'rb') as f:
    all_labels = pickle.load(f)






# =============================================================================
# 
# for label in range(4):  # Iterate over each class index directly
#     temp, mask = explanation.get_image_and_mask(
#         label, positive_only=True, num_features=10, hide_rest=False
#     )
#     plt.imshow(mark_boundaries(spectrogram_input.reshape(60, 51, 3), mask))
#     plt.title(f'Explanation for class {label}')
#     plt.show()
# =============================================================================

# =============================================================================
# # SHAP Explanation
# explainer = shap.KernelExplainer(predict_fn, spectrogram)
# shap_values = explainer.shap_values(spectrogram[0:1], nsamples=100)
# shap.image_plot(shap_values, spectrogram[0:1])
# 
# 
# 
# =============================================================================
