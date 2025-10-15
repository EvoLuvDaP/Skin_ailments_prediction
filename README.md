# Skin Ailment Prediction

This project is a web application that utilizes a machine learning model to predict skin ailments from user-uploaded images. The model provides a list of potential conditions with corresponding probabilities, and a visual "Prediction Highlight" to show which part of the image was most influential in the decision.

## 🚀 Features

*   **Image Upload:** Easily upload an image of a skin condition.
*   **AI-Powered Prediction:** A deep learning model analyzes the image to predict potential skin ailments.
*   **Prediction Highlight:** A visual heatmap indicates the area of the image that the model focused on for its prediction.
*   **Detailed Results:** The top predictions are displayed with their confidence scores and brief descriptions.

## 🛠️ Technologies Used

*   **Backend:** Python with a web framework like Flask or Django.
*   **Machine Learning:** A deep learning framework such as TensorFlow, Keras, or PyTorch.
*   **Frontend:** HTML, CSS, and JavaScript for the user interface.

## <caption> UI Screenshot

![Skin Ailment Prediction Screenshot](https://i.imgur.com/example.png)  <!-- You should replace this with a direct link to your screenshot -->

### Example from Screenshot

*   **Vitiligo:** 96.47%
*   **Unknown\_Normal:** 2.19%
*   **Nail\_psoriasis:** 0.61%

##  Installation and Setup

To get this project up and running on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EvoLuvDaP/Skin_ailments_prediction.git
    cd Skin_ailments_prediction
    ```

2.  **Run the application:**
    ```bash
    python app.py  # Or your main Python script
    ```

3.  Open your web browser and go to `http://127.0.0.1:5000`.

## ⚠️ Disclaimer

This application is intended for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified health provider for any questions you may have regarding a medical condition.
