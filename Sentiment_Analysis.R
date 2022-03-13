# 1. Develop a dictionary-based sentiment analytics engine based on the R library 
# 'syuzhet' and 'tidytext' to analyse the different emotions from Apple review 
# tweets (8%).

#loading libraries
# install.packages('syuzhet')
# install.packages('ggplot2')
# install.packages('tidytext')
# install.packages('textdata')
# install.packages('dplyr')

library(syuzhet)
library(tidytext)
library(dplyr)
library(textdata)
library(ggplot2)
library(tm)
library(caret)
library(e1071)

# Configuring working directory
setwd("C:/Users/Sazee/Desktop/BUS5CA/Assignment 1/Case 2")

# Loading the dataset.
df <- read.csv(file="apple_review_new.csv", header=TRUE, sep=",")
df <- c(df$Positive, df$Negative)

sentences <- get_sentences(df)
sentiment <- get_sentiment(sentences)

plot(sentiment,
     type = "l",
     main = "Sentiment change",
     xlab = "Narrative time",
     ylab = "Sentiment")

# Sentiment plot with smoothing methods applied
simple_plot(sentiment)


# ****************
# tidytext package
# ****************
nrc_lexicon <- get_sentiments("nrc")

# ***************
# syuzhet package
# ***************
nrc_data <- get_nrc_sentiment(sentences)
angry_sentences <- which(nrc_data$anger > 0)
sentences[angry_sentences]
fear_sentences <- which(nrc_data$fear > 0)
sentences[fear_sentences]

td <- data.frame(t(nrc_data))
td_sum <- data.frame(rowSums(td))
names(td_sum)[1] = "count"
td_sum <- cbind("sentiment" = rownames(td_sum), td_sum)
rownames(td_sum) = NULL
td_sum <- td_sum[1:8, ]

ggplot(data = td_sum, aes(x = sentiment)) + 
  geom_bar(aes(weight = count, fill = sentiment)) +
  ggtitle("Speech Sentiments") + guides(fill = FALSE)

# ****************
# tidytext package
# ****************
sentences_df <- tibble(sentence = 1:length(sentences), content = sentences)
sentences_df

# unnest sentence content into individual words
tidy_df <- unnest_tokens(sentences_df, output = word, input = content)
tidy_df

# extract the lexicon (words) associated with the emotion of "joy"
nrc_joy <- get_sentiments("nrc") %>% filter(sentiment == "joy")
nrc_joy

# obtain the frequency of words with the emotion of "joy" in the speech
joy_words <- tidy_df %>% inner_join(nrc_joy, by = "word") %>% count(word, sort = TRUE)
joy_words

# obtain the frequency of words with the emotion of "joy" for a particular sentence
tidy_df %>% filter(sentence == 1) %>% inner_join(nrc_joy, by = "word") %>% count(word, sort = TRUE)

# plot the top-5 words for the emotion of "joy" in the speech
joy_words %>% head (5) %>% 
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_bar(alpha = 0.8, fill = "orange", stat="identity") +
  geom_text(aes(label = n), hjust = -0.3, size = 3.5) +
  labs(y = "Contribution of sentiments", x = NULL) +
  coord_flip() + ggtitle("Joy Emotion")

# extract the lexicon (words) associated with the emotion of "anger"
nrc_anger <- get_sentiments("nrc") %>% filter(sentiment == "anger")
nrc_anger

# obtain the frequency of words with the emotion of "anger" in the speech
anger_words <- tidy_df %>% inner_join(nrc_anger, by = "word") %>% count(word, sort = TRUE)
anger_words

# obtain the frequency of words with the emotion of "anger" for a particular sentence
tidy_df %>% filter(sentence == 1) %>% inner_join(nrc_anger, by = "word") %>% count(word, sort = TRUE)

# plot the top-5 words for the emotion of "anger" in the speech
anger_words %>% head (5) %>% 
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_bar(alpha = 0.8, fill = "orange", stat="identity") +
  geom_text(aes(label = n), hjust = -0.3, size = 3.5) +
  labs(y = "Contribution of sentiments", x = NULL) +
  coord_flip() + ggtitle("Anger Emotion")

# extract the lexicon (words) associated with the emotion of "anticipation"
nrc_anticipation <- get_sentiments("nrc") %>% filter(sentiment == "anticipation")
nrc_anticipation

# obtain the frequency of words with the emotion of "anticipation" in the speech
anticipation_words <- tidy_df %>% inner_join(nrc_anticipation, by = "word") %>% count(word, sort = TRUE)
anticipation_words

# obtain the frequency of words with the emotion of "anticipation" for a particular sentence
tidy_df %>% filter(sentence == 1) %>% inner_join(nrc_anticipation, by = "word") %>% count(word, sort = TRUE)

# plot the top-5 words for the emotion of "anticipation" in the speech
anticipation_words %>% head (5) %>% 
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_bar(alpha = 0.8, fill = "orange", stat="identity") +
  geom_text(aes(label = n), hjust = -0.3, size = 3.5) +
  labs(y = "Contribution of sentiments", x = NULL) +
  coord_flip() + ggtitle("Anticipation Emotion")

# extract the lexicon (words) associated with the emotion of "disgust"
nrc_disgust <- get_sentiments("nrc") %>% filter(sentiment == "disgust")
nrc_disgust

# obtain the frequency of words with the emotion of "disgust" in the speech
disgust_words <- tidy_df %>% inner_join(nrc_disgust, by = "word") %>% count(word, sort = TRUE)
disgust_words

# obtain the frequency of words with the emotion of "disgust" for a particular sentence
tidy_df %>% filter(sentence == 1) %>% inner_join(nrc_disgust, by = "word") %>% count(word, sort = TRUE)

# plot the top-5 words for the emotion of "disgust" in the speech
disgust_words %>% head (5) %>% 
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_bar(alpha = 0.8, fill = "orange", stat="identity") +
  geom_text(aes(label = n), hjust = -0.3, size = 3.5) +
  labs(y = "Contribution of sentiments", x = NULL) +
  coord_flip() + ggtitle("Disgust Emotion")

# extract the lexicon (words) associated with the emotion of "fear"
nrc_fear <- get_sentiments("nrc") %>% filter(sentiment == "fear")
nrc_fear

# obtain the frequency of words with the emotion of "fear" in the speech
fear_words <- tidy_df %>% inner_join(nrc_fear, by = "word") %>% count(word, sort = TRUE)
fear_words

# obtain the frequency of words with the emotion of "fear" for a particular sentence
tidy_df %>% filter(sentence == 1) %>% inner_join(nrc_fear, by = "word") %>% count(word, sort = TRUE)

# plot the top-5 words for the emotion of "fear" in the speech
fear_words %>% head (5) %>% 
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_bar(alpha = 0.8, fill = "orange", stat="identity") +
  geom_text(aes(label = n), hjust = -0.3, size = 3.5) +
  labs(y = "Contribution of sentiments", x = NULL) +
  coord_flip() + ggtitle("Fear Emotion")

# extract the lexicon (words) associated with the emotion of "sadness"
nrc_sadness <- get_sentiments("nrc") %>% filter(sentiment == "sadness")
nrc_sadness

# obtain the frequency of words with the emotion of "sadness" in the speech
sadness_words <- tidy_df %>% inner_join(nrc_sadness, by = "word") %>% count(word, sort = TRUE)
sadness_words

# obtain the frequency of words with the emotion of "sadness" for a particular sentence
tidy_df %>% filter(sentence == 1) %>% inner_join(nrc_sadness, by = "word") %>% count(word, sort = TRUE)

# plot the top-5 words for the emotion of "sadness" in the speech
sadness_words %>% head (5) %>% 
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_bar(alpha = 0.8, fill = "orange", stat="identity") +
  geom_text(aes(label = n), hjust = -0.3, size = 3.5) +
  labs(y = "Contribution of sentiments", x = NULL) +
  coord_flip() + ggtitle("Sadness Emotion")

# extract the lexicon (words) associated with the emotion of "surprise"
nrc_surprise <- get_sentiments("nrc") %>% filter(sentiment == "surprise")
nrc_surprise

# obtain the frequency of words with the emotion of "surprise" in the speech
surprise_words <- tidy_df %>% inner_join(nrc_surprise, by = "word") %>% count(word, sort = TRUE)
surprise_words

# obtain the frequency of words with the emotion of "surprise" for a particular sentence
tidy_df %>% filter(sentence == 1) %>% inner_join(nrc_surprise, by = "word") %>% count(word, sort = TRUE)

# plot the top-5 words for the emotion of "surprise" in the speech
surprise_words %>% head (5) %>% 
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_bar(alpha = 0.8, fill = "orange", stat="identity") +
  geom_text(aes(label = n), hjust = -0.3, size = 3.5) +
  labs(y = "Contribution of sentiments", x = NULL) +
  coord_flip() + ggtitle("Surprise Emotion")

# extract the lexicon (words) associated with the emotion of "trust"
nrc_trust <- get_sentiments("nrc") %>% filter(sentiment == "trust")
nrc_trust

# obtain the frequency of words with the emotion of "trust" in the speech
trust_words <- tidy_df %>% inner_join(nrc_trust, by = "word") %>% count(word, sort = TRUE)
trust_words

# obtain the frequency of words with the emotion of "trust" for a particular sentence
tidy_df %>% filter(sentence == 1) %>% inner_join(nrc_trust, by = "word") %>% count(word, sort = TRUE)

# plot the top-5 words for the emotion of "trust" in the speech
trust_words %>% head (5) %>% 
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_bar(alpha = 0.8, fill = "orange", stat="identity") +
  geom_text(aes(label = n), hjust = -0.3, size = 3.5) +
  labs(y = "Contribution of sentiments", x = NULL) +
  coord_flip() + ggtitle("Trust Emotion")



#######################################################################################
# 2. Develop a machine learning-based model using the R libraries 'tm' and 'e1071' as 
# well as evaluate the predictive accuracies of SVM classifier (5%).

df <- read.csv(file="apple_review_new.csv", header=TRUE, sep=",")

PositiveReviewTrain <- df$Positive[0:120]
NegativeReviewTrain <- df$Negative[0:120]
PositiveReviewTest <- df$Positive[120:143]
NegativeReviewTest <- df$Negative[120:143]

# 120 for training; 23 for testing
ReviewsTrain <- c(PositiveReviewTrain, NegativeReviewTrain)
length(ReviewsTrain)
ReviewsTest <- c(PositiveReviewTest, NegativeReviewTest)
length(ReviewsTest)
ReviewsAll <- c(ReviewsTrain, ReviewsTest) # 240 for training; 48 for testing
length(ReviewsAll)

# create labels for both positive and negative reviews
sentimentTrain <- c(rep("positive", length(PositiveReviewTrain)), rep("negative", length(NegativeReviewTrain)))
sentimentTest <- c(rep("positive", length(PositiveReviewTest)), rep("negative", length(NegativeReviewTest)))
sentimentAll <- as.factor(c(sentimentTrain, sentimentTest))
sentimentAll <- as.character(sentimentAll)

# based on tm package (convert to a corpus format)
ReviwesCorpus <- Corpus(VectorSource(ReviewsAll))

# preprocess the corpus before analysis
ReviwesCorpus <- tm_map(ReviwesCorpus, removeNumbers)
ReviwesCorpus <- tm_map(ReviwesCorpus, stripWhitespace)
ReviwesCorpus <- tm_map(ReviwesCorpus, content_transformer(tolower))
ReviwesCorpus <- tm_map(ReviwesCorpus, removeWords, stopwords("english"))

# create the document-term-matrix for the corpus
ReviewsDTM <- DocumentTermMatrix(ReviwesCorpus)
ReviewsMatrix <- as.matrix(ReviwesDTM)

# SVM classifier (accuracy = 83.3%)
svmClassifier <- svm(ReviewsMatrix[1:240,], as.factor(sentimentAll[1:240]))
svmPredicted <- predict(svmClassifier, ReviewsMatrix[241:288,])
table(svmPredicted, sentimentTest)
confusionMatrix(as.factor(svmPredicted), as.factor(sentimentTest))

