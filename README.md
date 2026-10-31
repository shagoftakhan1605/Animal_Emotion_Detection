# Animal Emotion Detection

A transfer-learning image classifier that labels animal face images as **Angry**, **Happy** or **Sad**, with a Tkinter GUI for webcam or uploaded-image prediction.

* `animal_emotions.py`: trains and evaluates the model
* `gui.py`: loads the trained model and runs predictions from a webcam or an image file
* `animal_emotions.zip`: the image dataset

---

## Dataset

The zip file contains 813 JPG images in three classes:

| Split | Angry | Happy | Sad |
| --- | --- | --- | --- |
| train | 250 | 250 | 250 |
| valid | 9 | 11 | 10 |
| test | 10 | 11 | 11 |

Source: *to be added* (please credit the original dataset and check its licence before reuse).

---

## Model

* **Base:** VGG16 pre-trained on ImageNet, all convolutional layers frozen
* **Head:** Flatten, Dense (512, ReLU), Dropout (0.5), Dense (3, softmax)
* **Input:** 224 x 224 RGB, pixel values scaled to [0, 1]
* **Augmentation (train only):** rotation, width and height shift, shear, zoom, horizontal flip
* **Training:** Adam (learning rate 1e-4), categorical cross-entropy, batch size 32, up to 50 epochs, early stopping on validation loss (patience 10), best model saved as `best_model.keras`
* **Evaluation:** test accuracy and a per-class classification report

---

## Results

No results are recorded in this repository yet. Running `animal_emotions.py` prints the test accuracy and classification report. Note that the validation and test sets are very small (about 30 images each), so accuracy will vary a lot between runs.

---

## How to Run

```bash
pip install -r requirements.txt
unzip animal_emotions.zip          # creates animal_emotions/train, valid, test
python animal_emotions.py          # trains and saves animal_emotion_detection_model.keras
python gui.py                      # opens the GUI (needs the saved model)
```

In the GUI, choose **Open Camera** (press `q` to close the camera) or **Upload Image**.

---

## Limitations

* Very small validation and test sets, so results are not stable
* Only the classifier head is trained; fine-tuning upper VGG16 layers may help
* The model classifies the whole image and does not first locate the animal's face, so webcam results depend heavily on framing

---

## Tools

Python, TensorFlow/Keras, OpenCV, Pillow, Tkinter, scikit-learn
