# Mask Detection Using Deep Learning

This repository contains a Jupyter Notebook for implementing a deep learning-based mask detection system. The project aims to classify whether individuals in an image are wearing a mask or not.

---

## Features
- **Dataset Handling:** Load, preprocess, and visualize data.
- **Model Building:** Create and train a deep learning model using TensorFlow/Keras.
- **Evaluation:** Evaluate the model's performance with metrics like accuracy, precision, and recall.
- **Inference:** Perform predictions on new images to detect the presence of masks.

---

## Prerequisites

### Hardware Requirements
- A system with a GPU is recommended for training.

### Software Requirements
- Python 3.8 or higher
- Jupyter Notebook
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn

You can install the required libraries using the following command:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

## Installation
1. Clone the repository:
2. Navigate to the project directory:
   ```bash
   cd mask_detection
   ```
3. Install the required libraries as mentioned in the prerequisites section.

---

## Usage

### Running the Notebook
1. Open the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
2. Navigate to `DEEP_Learning_mask_detection.ipynb` and open it.
3. Follow the steps in the notebook to:
   - Load and preprocess the dataset.
   - Train the mask detection model.
   - Evaluate the model's performance.
   - Perform predictions.

### Dataset
The notebook assumes the availability of a dataset containing images categorized into two classes: `mask` and `no_mask`. 

Dataset: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

Ensure the dataset is organized as follows:
```
/dataset
  /train
    /mask
    /no_mask
  /test
    /mask
    /no_mask
```

---

## Project Workflow
1. **Data Preparation:**
   - Load and inspect the dataset.
   - Perform data augmentation and preprocessing.
2. **Model Creation:**
   - Define a convolutional neural network (CNN).
   - Train the model using the training dataset.
3. **Evaluation:**
   - Test the model on unseen data.
   - Calculate metrics and visualize results.
4. **Inference:**
   - Use the trained model to detect masks in new images.

---

## Results
- Provide details about the accuracy and other metrics achieved by your model.
- Include sample visualizations showing predictions.

---

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and make changes via pull requests.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements
- Kaggle for datasets and inspiration.
- TensorFlow/Keras for the deep learning framework.

---

## Contact
For questions or suggestions, please contact:
**Omar Elhorbity**
Email: omarhusseinelhobity@gmail.com

**Ali Ibrahim**
Email: aliibrahimatia@gmail.com
