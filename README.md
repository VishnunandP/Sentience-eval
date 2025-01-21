# **Emotion-Driven Response Generation and Sentience Evaluation System**

## **Overview**

This project combines various cognitive models with deep learning techniques to create an emotion-driven response generation system. By leveraging face emotion recognition and empathetic dialogue generation, the system interprets both facial expressions and text input to generate contextually aware responses. This aims to simulate an intelligent system capable of adaptive emotional responses based on human emotional states.

### **Technologies Used**
- **Deep Learning Models**: Transformers, CNNs, and Pretrained Models from Hugging Face
- **Emotion Recognition**: OpenCV-based facial emotion recognition
- **Empathetic Dialogue**: Using pretrained models to generate empathetic responses
- **Frameworks**: Gradio for interactive user interfaces, OpenCV for computer vision tasks, and PyTorch/TensorFlow for model building and inference.
- **Datasets**: 
    - **FER-2013**: A large-scale dataset for facial emotion recognition.
    - **Empathetic Dialogues**: A dataset focused on generating empathetic responses.

### **What We Aim to Achieve**
The goal of this project is to bridge human-like emotional intelligence with AI systems, allowing the system to analyze human emotions through facial expressions and text input, and generate contextually appropriate and empathetic responses. This system could serve as a foundational step in creating AI-driven systems that understand and react to human emotions in a more nuanced and human-like manner.

### **Features of the Project**
1. **Emotion Recognition**: The system uses facial emotion recognition to detect human emotions from live camera input.
2. **Empathetic Dialogue Generation**: The system generates emotionally aware responses using pretrained models trained on the Empathetic Dialogues dataset.
3. **User Interface**: An interactive Gradio interface allows the user to input text and interact with the emotion detection model.
4. **Cross-Modal Interaction**: The project allows text and visual cues (via webcam) to influence the AI’s response.

### **Data Sources and Acknowledgments**
- **FER-2013 Dataset**: A facial emotion dataset used for detecting human emotions from images. The dataset can be found [here](https://www.kaggle.com/datasets/msambare/fer2013).
- **Empathetic Dialogues Dataset**: This dataset contains pairs of dialogue utterances where the AI's goal is to generate empathetic responses. The dataset can be accessed [here](https://huggingface.co/datasets/facebook/empathetic_dialogues).

We would like to thank the contributors to these datasets and the researchers for making them publicly available.

### **How to Run the Project**

1. **Clone this Repository**:
    ```bash
    git clone https://github.com/your-username/Emotion-Driven-Response-System.git
    ```

2. **Install Dependencies**:
    Install the required libraries using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the System**:
    Launch the Gradio interface by running the following command in the project directory:
    ```bash
    python app.py
    ```

    The Gradio interface will be available locally at `http://127.0.0.1:7860/`.

### **Acknowledgements**

- This project uses the FER-2013 dataset, which was provided by [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
- The Empathetic Dialogues dataset was provided by [Facebook AI](https://huggingface.co/datasets/facebook/empathetic_dialogues).

### **Contact**

For any queries or collaboration, feel free to reach out to [Your Contact Information].
