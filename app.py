### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

from class_names import class_names
from model import Load_model
from timeit import default_timer as timer
from typing import Tuple, Dict


### 1. Model and transforms preparation ###

# Create model and transform
model, transforms = Load_model()

# Load saved weights
def load_checkpoint(checkpoint_file, model, device='cpu'):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
load_checkpoint('model_checkpoint.pt', model)

### 2. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(model(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


### 3. Gradio app ###

# Create title, description and article strings
def main():
    title = "BirdVision 500 🦅🦆🐦🕊🦤🦢🦜"
    description = "A model based on YoLov8 classification 500 birds."
    article = "Created on [GITHUB](https://github.com/vvduc1803/Yolov8_cls)."

    # Create examples list from "examples/" directory
    example_list = [["examples/" + example] for example in os.listdir("examples")]

    # Create the Gradio demo
    demo = gr.Interface(fn=predict,  # mapping function from input to output
                        inputs=gr.Image(type="pil"),  # what are the inputs?
                        outputs=[gr.Label(num_top_classes=10, label="Predictions"),  # what are the outputs?
                                 gr.Number(label="Prediction time (s)")],
                        # our fn has two outputs, therefore we have two outputs
                        # Create examples list from "examples/" directory
                        examples=example_list,
                        title=title,
                        description=description,
                        article=article)

    # Launch the demo!
    demo.launch(server_name="127.0.0.1", server_port=1234, share=True)

if __name__ == '__main__':
    main()