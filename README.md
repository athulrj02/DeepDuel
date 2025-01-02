# **DeepDuel**

### **Emphasizes the Duel Between CNN and Transfer Learning**

---

## **Project Overview**

This project investigates and compares the performance of **Convolutional Neural Networks (CNNs)** and **Transfer Learning** using the **VGG16** model for image classification tasks. The dataset used in this project consists of multiple product categories along with a background class, focusing on classifying products effectively with high accuracy.

---

## **Table of Contents**
1. [Dataset and Preprocessing](#dataset-and-preprocessing)  
2. [Model Architectures](#model-architectures)  
3. [Results and Evaluation](#results-and-evaluation)  
4. [Fine-Tuning for Enhanced Performance](#fine-tuning-for-enhanced-performance)  
5. [How to Run the Code](#how-to-run-the-code)  
6. [Dependencies and Requirements](#dependencies-and-requirements)  
7. [Acknowledgments](#acknowledgments)  

---

## **Dataset and Preprocessing**

The dataset consists of six categories:
- **Product_1, Product_2, Product_3, Product_4, Product_5, and Background**  
- Class distributions were visualized to identify imbalances, and data augmentation was applied to underrepresented classes.  
- Images were resized to **128x128** for uniformity, normalized, and labels were one-hot encoded.

---

## **Model Architectures**

1. **CNN Model**  
   - Consists of 3 convolutional blocks followed by MaxPooling layers.  
   - Dense layers include dropout for regularization and softmax for classification into 6 categories.  
   - Early stopping was implemented to avoid overfitting.  

2. **Transfer Learning using VGG16**  
   - Pre-trained **VGG16** was used without the top layers.  
   - Custom Dense and Dropout layers were added for classification.  
   - The base model layers were frozen initially to preserve pre-trained weights.  

3. **Fine-Tuning**  
   - Later layers of VGG16 were unfrozen for fine-tuning.  
   - A lower learning rate was used to adapt the model to the specific dataset.

---

## **Results and Evaluation**

1. **CNN Model:**
   - Achieved **96% accuracy** on test data.  
   - Confusion matrices and classification reports were generated for performance analysis.  

2. **Transfer Learning:**
   - Achieved **96% accuracy** before fine-tuning.  
   - Improved to **97% accuracy** after fine-tuning.  

3. **Visualizations:**
   - Predictions were visualized for both models to analyze classification behavior.  
   - Confusion matrices highlighted strengths and areas requiring improvement.  

---

## **Fine-Tuning for Enhanced Performance**

- Fine-tuning adapted pre-trained features of **VGG16** to focus on task-specific patterns.  
- It resolved issues with misclassification for underrepresented classes like **Product_2** and **Background**.  
- The process ensured better alignment with dataset requirements while leveraging transfer learning benefits.  

---

## **How to Run the Code**

1. Clone the repository:
   ```
   git clone https://github.com/athulrj02/DeepDuel.git
   ```
2. Navigate to the project folder:
   ```
   cd DeepDuel
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Open and run the Jupyter notebook:
   ```
   jupyter notebook ML|CA_Two.ipynb
   ```

---

## **Dependencies and Requirements**

- Python 3.x  
- TensorFlow  
- Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  
- OpenCV  

---

## **Acknowledgments**

- The dataset used for this project was sourced for experimental purposes.  
- The VGG16 pre-trained model was leveraged via TensorFlow/Keras.  
- Special thanks to open-source contributors for supporting frameworks and libraries used.  

---
