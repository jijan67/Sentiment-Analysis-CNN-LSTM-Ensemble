# Sentiment Analysis using CNN and LSTM Ensemble Model
This repository implements a sentiment analysis system that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks in an ensemble architecture to predict sentiment labels. The model is trained on a Twitter sentiment dataset and achieves 96% accuracy.
---
## Project Overview
The system processes raw Twitter text data for sentiment classification. It utilizes both CNN and LSTM models to extract different features from the text. The CNN model is responsible for capturing local patterns in the text, while the LSTM model captures sequential dependencies. The outputs of these models are then combined in an ensemble architecture to enhance the overall performance.
---
## Key Features
- **Data Preprocessing:** The raw text data is cleaned by removing URLs, special characters, numbers, and unnecessary white spaces. It is then tokenized and padded to ensure that the input data is suitable for neural network models.

- **Model Architecture:** The model leverages both CNN and LSTM layers. The CNN layer captures local features and patterns in the text, while the LSTM layer handles long-range dependencies. Both models' outputs are concatenated to form a stronger representation of the text.

- **Training:** The model is trained using the Adam optimizer and sparse categorical cross-entropy loss function. It was trained over 10 epochs, with both training and validation accuracy monitored.

- **Performance Metrics:** The model achieved an impressive accuracy of 96%. A classification report and confusion matrix are generated for a more detailed evaluation of the model's performance.

- **Prediction:** The trained model can predict sentiment labels for new, unseen text. A sample text input can be classified as either positive, negative, or neutral sentiment.

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- re
- scikit-learn
- tensorflow

## Dataset
The dataset used for this project consists of Twitter data with labeled sentiments (positive, negative, neutral). The training and testing datasets are in CSV format, with the primary text column being the tweet content.

## Steps Involved in the Implementation
**1. Data Loading and Preprocessing:**

- Two datasets (twitter_training.csv and twitter_validation.csv) are loaded using pandas. These datasets contain Twitter text and sentiment labels (positive, negative, neutral).
The columns of the dataset are renamed, and unnecessary columns (Header1, company) are removed.
- A cleaning function (clean_tweet) is defined to remove URLs, special characters, numbers, and whitespace, and the text is then converted to lowercase.
  
**2. Text Tokenization and Padding:**

- The cleaned text data is tokenized using Keras' Tokenizer class. This process converts the text into numerical sequences where each unique word corresponds to a unique integer index.
- To ensure uniformity in input length, the sequences are padded to the maximum length of the tokenized texts.
  
**3. Label Encoding:**

- Sentiment labels are encoded using LabelEncoder, transforming categorical labels (positive, negative, neutral) into numerical values for model training.
  
**4. Model Architecture:**

- **CNN Model:** The CNN model is initialized with an embedding layer that converts words into dense vectors. A 1D convolutional layer is applied to capture local features followed by max-pooling to reduce dimensionality.
- **LSTM Model:** The LSTM model starts with an embedding layer, followed by an LSTM layer to capture sequential dependencies. The output is passed through a global max-pooling layer to reduce the output to a fixed-size vector.
- Both models' outputs are combined using concatenation, and the combined features pass through dense layers with dropout to prevent overfitting.
- The final output layer uses softmax activation to classify the text into one of the sentiment categories.
  
**5. Model Compilation:**

- The ensemble model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function, appropriate for multi-class classification problems.
  
**6. Model Training:**

- The model is trained using the training data (train_padded) and labels (train_labels_encoded) for 10 epochs. The validation data (test_padded, test_labels_encoded) is used to evaluate the model during training.
  
**7. Model Evaluation:**

- After training, the model's performance is evaluated using the test data. The test loss and accuracy are printed.
- Predictions are made on the test dataset, and performance metrics such as precision, recall, F1 score, and accuracy are generated through the classification report.
- A confusion matrix is also plotted to visually assess the model’s performance.
  
**8. Prediction on New Text:**

- A function (predict_text_ensemble) is defined to predict the sentiment of new text inputs. The function cleans, tokenizes, and pads the input text before feeding it to the trained ensemble model for sentiment prediction.

## Results
The model achieved an impressive 96% test accuracy on the sentiment analysis task. The performance was evaluated through the following metrics:

- Precision, Recall, F1 Score: Detailed classification report generated.
- Confusion Matrix: Visual representation of true vs. predicted sentiment labels.

## Conclusion
This ensemble model, combining CNN and LSTM, shows strong performance in sentiment classification tasks. The combined architecture benefits from both CNN's ability to detect local patterns and LSTM’s capability to capture long-term dependencies in text data.

## Future Improvements
- Experiment with different hyperparameters such as the number of filters in the CNN and the number of LSTM units.
- Fine-tune the model with more training epochs and optimize the learning rate.
- Extend the dataset for better model generalization.





