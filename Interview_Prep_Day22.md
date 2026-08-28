# Day 22 - Mock Interview Preparation

## Project Walkthrough Questions

### Q1: Walk me through your Student Grade Predictor project

I built a Student Grade Predictor to predict students' grades from their study habits and academic-support factors such as weekly study time, absences, tutoring, and parental support. The main challenge was improving the model's ability to capture meaningful relationships in the data, so I performed EDA and created features such as Study Efficiency, Support Score, Risk Score, High Absence, and High Study. I compared Logistic Regression, Decision Tree, and Random Forest using accuracy, weighted F1-score, and 5-fold cross-validation, and selected the best-performing model based on F1-score. The final model achieved 75.4% accuracy and 70.19% F1-score on unseen test data. Finally, I built a prediction function that takes a new student's details and returns the predicted grade, confidence, student profile, and personalized advice.

### Q2: Why did you choose Logistic Regression over Random Forest?

I chose Logistic Regression because I wanted a strong and interpretable baseline rather than assuming that a more complex model would automatically perform better. The dataset has structured numerical features, and after feature engineering, the relationships were reasonably useful for a linear classification model. I also compared it with Decision Tree and Random Forest using accuracy, weighted F1-score, and cross-validation instead of choosing a model based only on complexity. So my decision was based on the model's actual performance and interpretability on this dataset, not on the assumption that Random Forest is always better.

### Q3: What was the biggest challenge in your NLP Sentiment Analysis project?

The biggest challenge was getting the sentiment model to generalize to completely new sentences. I initially saw very high training accuracy, but when I tested the model on unseen reviews, some predictions were incorrect, which indicated overfitting. I debugged the tokenizer and vocabulary, checked words such as "amazing," "terrible," "best," and "worst," and compared the training and testing performance. I found that the main issue was the very small dataset of only 30 reviews, which was too small for a Bidirectional LSTM, so I learned that increasing the training data, simplifying the model, and using stronger regularization would improve generalization.

## Technical + Behavioral Questions

### Q4: If given a dataset with 50% missing values in one column, what would you do?

First, I would understand why that column has 50% missing values and whether the column is important for prediction. If the column is not important, I may consider dropping it, but if it contains useful information, I would try to impute the missing values using an appropriate method such as median for numerical data or mode for categorical data. I would also check whether the missing values have a pattern because missingness itself can sometimes contain useful information. Finally, I would compare the model performance before and after handling the missing values and choose the approach that gives the most reliable result.

### Q5: Explain bias-variance tradeoff in simple terms

Bias means the model is too simple and cannot capture the patterns in the data, which leads to underfitting. Variance means the model is too sensitive to the training data, so it performs very well on training data but poorly on unseen data, which is overfitting. The bias-variance tradeoff is about finding the right balance between these two. In simple terms, I want a model that is complex enough to learn the important patterns, but not so complex that it memorizes the training data.

### Q6: Tell me about a time you had to learn something completely new quickly

During my 30-day AI and ML learning program, I had to quickly learn several concepts that were new to me, including machine learning algorithms, deep learning, CNNs, LSTMs, and NLP. I broke the learning into daily goals and followed a practical approach where I learned a concept, implemented it in Python, tested the model, and then debugged the problems I encountered. For example, while working on sentiment analysis, I initially focused on getting high training accuracy, but after testing unseen sentences, I discovered an overfitting problem and investigated the root cause. This approach helped me move from simply understanding concepts theoretically to actually building, debugging, and explaining ML models, and it made me more confident in learning new technologies independently.