# X-Ray Threat Detector

AI-powered baggage screening application for aviation security. Upload an X-ray scan image and get instant threat classification with confidence scores.

Built as part of the DLOR Project at Temasek Polytechnic.

## Features

- **Threat Detection** -- Classifies X-ray baggage scans into 7 threat categories (Battery, Pliers, Wrench, Explosive, Bullet, Knife, Lighter)
- **Confidence Scoring** -- Displays prediction confidence with a visual bar and probability breakdown for all classes
- **Low-Confidence Handling** -- Shows "Not Known" when confidence is below 30%, with the closest guess
- **Image Format Converter** -- Convert images between PNG, JPEG, BMP, GIF, TIFF, WEBP, and ICO formats
- **Modern UI** -- Glassmorphic design with animated background, scan-line effect, and responsive layout

## Model

- **Architecture:** EfficientNetB0 (transfer learning, fine-tuned with Optuna hyperparameter optimization)
- **Format:** TensorFlow Lite (.tflite)
- **Test Accuracy:** 94.35%
- **Input Size:** 224 x 224 px
- **Training Data:** STCray X-ray dataset

## Project Structure

```
project(new)/
├── app_code/
│   ├── app.py              # Streamlit application
│   └── requirements.txt    # Python dependencies
├── saved_models/
│   ├── best_model_part2.tflite      # TFLite model
│   └── class_names_7classes.npy     # Class label mapping
├── 3rdairport.jpg           # Background image
└── README.md
```

## Run Locally

1. Install dependencies:

```
pip install -r app_code/requirements.txt
```

2. Run the app:

```
streamlit run app_code/app.py
```

## Deploy to Streamlit Cloud

1. Push the files listed above to a GitHub repository (keep the folder structure).
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your repo.
3. Set the main file path to `app_code/app.py`.

No database or external services required -- the model runs entirely client-side via TensorFlow Lite.

## Tech Stack

- Streamlit
- TensorFlow / TensorFlow Lite
- EfficientNetB0
- Pillow
- NumPy
