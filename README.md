# Sentiment-Analysis
Sentiment analysis is the technique aiming to gauge the attitudes of customers in relation to topics, products and services of interests. It is a pivotal technology for providing insights to enhance the business bottom line in campaign tracking, customer-centric marketing strategy and brand awareness. Sentiment analytics approaches are used to produce sentiment categories such as ‘positive’, ‘negative’ and ‘neutral’. More specific human emotions are also the topic of interest. There are two major streams of methods to develop sentiment analytics engine: the dictionary-based and machine learning-based approaches. In this project, I 
performed sentiment analytics based on both approaches.

As data scientist for the multinational technology company Apple Inc, I developed a sentiment analytics engine for Twitter, which is used to predict consumers’ review sentiments. The aim is to develop both dictionary based and machine learning-based sentiment analytics scripts using a number of R libraries and SAS Sentiment Analysis Studio. I used the developed engine to predict Apple reviewers’ sentiments and benchmark various algorithms and analytics tools. 

I developed both dictionary-based and machine-learning sentiment analytics engines using R programming language and applied it to predict the sentiments of Apple product
review tweets from a sample of data. I used the SAS Sentiment Analysis Studio to compare the results. To achieve the above, carried out the following data analytics tasks:

1. Developed a dictionary-based sentiment analytics engine based on the R library 
‘syuzhet’ and ‘tidytext’ to analyse the different emotions from Apple review 
tweets.
• Analysed and aggregated the eight emotions (anger, anticipation, disgust, fear, joy, 
sadness, surprise and trust) from the Apple review tweets file ‘apple_review.csv’ 
using the function ‘get_nrc_sentiment’. 
• Found the top 5 most frequent words in all the Apple product reviews for each 
of the eight emotions (anger, anticipation, disgust, fear, joy, sadness, surprise 
and trust). I analysed the results.
2. Developed a machine learning-based model using the R libraries ‘tm’ and ‘e1071’ as 
well as evaluated the predictive accuracies of SVM classifier.
• Developed R scripts and imported the data set ‘apple_review.csv’ for training and 
testing.
• Used the first 120 negative tweets and the first 120 positive tweets as the training 
dataset; and used the rest of the 23 negative tweets and 23 positive tweets as the 
testing dataset.
• Developed a machine learning-based sentiment analytics engine and predicted 
sentiment categories (only ‘positive’ and ‘negative’) using ‘tm’ and ‘e1071’ with 
the SVM classifier.
• Evaluated the testing accuracies and report the predicted results.

3. Developed a statistical model using SAS Sentiment Analysis studio and evaluated the 
accuracies.
• Used the data folder: ‘apple_review’ which contain ‘negative’ and ‘positive’ tweets 
for training and testing.
• Built a statistical model using SAS Sentiment Analysis (simple and 
advanced), 
• Evaluated and compared the testing accuracies for different models and reported the
results.
• Compared this result with the previous predictive results using R and discuss.

