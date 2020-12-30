Fake News Detection:-

Fake news detection using kaggle dataset.
In this project, we have used  NLP techniques ( text classification techniques) and eight different
machine learning algorithm to check which algorithm is well suited for classifying fake news from the real news by comparing their accuracy.


About Dataset:-

The dataset used in our approach is available at https://www.kaggle.com/c/fake-news/data.

train.csv: A full training dataset with the following attributes:- 

id: unique id for a news article
title: the title of a news article
author: author of the news article
text: the text of the article; could be incomplete (We have used this as in input)
label: a label that marks the article as potentially unreliable
1: FAKE
0: TRUE
test.csv: A testing training dataset with all the same attributes at train.csv without the label.


Model Used:-

1) Passisve Aggressive Classifier
2) Logistic Regression
3) Descision Tress Classifier
4) Support Vector Classifier
5) KNeighborsClassifier
6) Naive Bayes
7) Random Forest CLassifier
8) LSTM

Instructions to run the code: -

1)Exract the zip file. (Inside which you will find two ipynb files named RNN_TRAIN_TEST.ipynb and OTHER_ML_TRAIN_TEST.ipynb)
2) Run the requirements.txt file
3) To train and test the LSTM model follow thw following steps:-
	3.1) Open the RNN_TRAIN_TEST.ipynb in jupyter notebook.
	3.2) To train the model on the given dataset excute all the cells in a sequence starting form the top cell.
	(If you want to train the model on the different dataset make sure it follows the same format as given in the train.csv file.
	3.3) To test the model you can put the articles in test.csv file under the "text" column.
4) To train and test the other models apart from LSTM, follow the following steps:-
	4.1) Open the OTHER_ML_TRAIN_TEST.ipynb in jupyter notebook.
	4.2) To train the model on the given dataset excute all the cells in a sequence starting form the top cell.
	(If you want to train the model on the different dataset make sure it follows the same format as given in the train.csv file.
	4.3) To test the model you can put the articles in test.csv file under the "text" column.
5) If you don't want to re-train your model , you can save the models for different ML alogrithms and later re-use it.

Approach :-

1) we took the dataset from kaggle 
2) Cleaned the dataset (Removed stop words, special characters, etc.)
3) Applied Natural laguagnge proc (NLP) alogirthms on our dataset to make it suitable to train on ML models.
4) Trained different ML models on our dataset
5) Compared the results

Results - Comparaing accuracis

Models 					Training Accuracy 	Testing Accuracy 
TFIDF – Logistic Regression 		0.97 			0.93 
TFIDF – Passive Aggressive Classifier 	1.00 			0.94 
TFIDF – Decision Tree Classifier 	1.00 			0.86 
TFIDF – Support Vector Classifier 	0.99 			0.93 
TFIDF – Multinomial Naïve Bayes 	0.97 			0.79 
TFIDF – KNeighbours Classifier 		0.52 			0.53 
TFIDF – Random Forest Classifier 	1.00 			0.92 
Tokenizer – LSTM 			0.95 			0.83 

Future Scope: -

Our model functioned with greater efficiency with the combination of TF-IDF Vectorizer and Passive Aggressive Classifier. 
We achieved an efficiency of approximately 93 % with this combination. We further plan to carry on our work by analyzing 
and perform similar procedures on different social media datasets mostly Facebook and Twitter. We plan on improving the 
diversity of our dataset by including articles from more sources beyond reddit as it will help expand the spectrum and improving accuracy. 

References: -

❖	Manning, Christopher D., Christopher D. Manning, and Hinrich Schütze. Foundations of     statistical natural language processing. MIT press, 1999. 
❖	M. Granik and V. Mesyura, "Fake news detection using naive Bayes classifier," 2017 IEEE First Ukraine Conference on Electrical and Computer Engineering (UKRCON), Kiev, 2017, pp. 900-903. 
❖	Kulkarni A., Shivananda A. (2019) Converting Text to Features. In: Natural Language Processing Recipes. Apress, Berkeley, CA. 
❖	Thota, Aswini; Tilak, Priyanka; Ahluwalia, Simrat; and Lohia, Nibrat (2018) "Fake News Detection: : A Deep Learning Approach," ," SMU Data Science Review: Vol. 1: No. 3, Article 10. 
❖	Ramos, Juan. "Using tf-idf to determine word relevance in document queries." In Proceedings of the first instructional conference on machine learning, vol. 242, pp. 133142. 2003. 
❖	Tripathy, Abinash, Ankit Agrawal, and Santanu Kumar Rath. "Classification of Sentimental Reviews Using Machine Learning Techniques." Procedia Computer Science 57 (2015): 821-829 
 

