# Classifying companies using LLM embeddings, an experiment

![header](https://raw.githubusercontent.com/darfire/stock-embedding-classification/master/blog/assets/stock-classification-embeddings-header.webp)

Large language models(LLM) have ushured in new tools an opportunities in text processing and undestanding. One of the most interesting applications is the ability to embed documents into a vector space and use these embeddings to look-up similar documents, or answers that are relevant to a particular question. This is at the base of  Retrieval-Augmented Generation (RAG) models, which have been shown to be very effective in question answering tasks.

In short, embeddings are a way to represent a document in a vector space, with a list of numbers. The idea is that documents that are similar will be close to each other in this vector space. This is a very powerful idea, as it allows us to use the same tools that have been developed for vector spaces, such as nearest neighbors, to find similar documents. They are learned by training a neural network to predict the next word in a sequence of words, given the previous words. The embeddings are the weights of the neural network, and are learned by backpropagation. Usually they are pretrained on a very large corpus of generic text, such as Common Crawl, and then optionally fine-tuned on a specific task.

In this article we've set up to study whether we can figure out how much semantic information (if any) about public companies has been learned by these LLM models during training. There is an accompanuing github repository and notebook with all the code and data used in this article.

## The data

We will be using a simple dataset of NYSE and NASDAQ companies, which contains, among other things, the company name, symbol, sector, industry and market capitalization. The data is available in the companion github repository.

| Symbol   | Name                                            | Sector                 | Industry                                       |
|:---------|:------------------------------------------------|:-----------------------|:-----------------------------------------------|
| AAPL     | Apple Inc. Common Stock                         | Technology             | Computer Manufacturing                         |
| MSFT     | Microsoft Corporation Common Stock              | Technology             | Computer Software: Prepackaged Software        |
| GOOG     | Alphabet Inc. Class C Capital Stock             | Technology             | Computer Software: Programming Data Processing |
| AMZN     | Amazon.com Inc. Common Stock                    | Consumer Discretionary | Catalog/Specialty Distribution                 |
| NVDA     | NVIDIA Corporation Common Stock                 | Technology             | Semiconductors                                 |
| META     | Meta Platforms Inc. Class A Common Stock        | Technology             | Computer Software: Programming Data Processing |
| BRK/A    | Berkshire Hathaway Inc.                         | Unknown                | Unknown                                        |
| BRK/B    | Berkshire Hathaway Inc.                         | Unknown                | Unknown                                        |
| HSBC     | HSBC Holdings plc. Common Stock                 | Finance                | Savings Institutions                           |
| TSLA     | Tesla Inc. Common Stock                         | Consumer Discretionary | Auto Manufacturing                             |
| LLY      | Eli Lilly and Company Common Stock              | Health Care            | Biotechnology: Pharmaceutical Preparations     |
| TSM      | Taiwan Semiconductor Manufacturing Company Ltd. | Technology             | Semiconductors                                 |
| JPM      | JP Morgan Chase & Co. Common Stock              | Finance                | Major Banks                                    |


## The task

Given the data above, we will try to classify the companies into their respective sectors and industries. We will use the company name and symbol as input, and the sector and industry as the target. We will use the embeddings of the company names and symbols as input to a simple classifier, and train it on a subset of the data, and test it on the rest.


## The approach

We will compute the embeddings of the company names and symbols using the [sentence-transformers](https://www.sbert.net/) library, which provides a very simple interface to a number of pretrained LLM models. We will then use the embeddings to classify the companies' sectors and industries.

The model we will be using will be a simple logistic regression model, which will be trained on a subset of the data and tested on the rest. The classification classes will be weighted by the inverse of their frequency in the training set, to account for the class imbalance.

We will use both the accuracy and the top-5 accuracy as metrics to evaluate the model's performance. After classifying the companies using either the company name or symbol embeddings, we get the following results:

[ table here]

As can be seen, classifying sectors yields better results than classifying industries. This is to be expected, as there are fewer sectors than industries, and the classes are more balanced. The top-5 accuracy is also higher, which indicates that the model is able to correctly classify the company in the top 5 most likely classes more often than not.

Also, as expected, using the company name embeddings yields better results than using the symbol embeddings. This is because the company name contains more information than the symbol, and the embeddings are able to capture this information. "Advanced Micro Devices, Inc." is more informative than "AMD", for example. To test this, let's see a sample of companies and the index of the correct class:

[ sample table here ]

However, we can see that in both cases the model has a much higher accuracy than random guessing, suggesting that the embeddings do contain some information about the companies, even if we only use the stock symbol.

## Experiment #1: Does prompting matter?

In the end, the SequenceTransformer takes in a sequence of tokens and returns an embedding of the same size. The question is: can we embelish the string that we input so we guide the model towards embeddings that improve the classification performace? And conversely, cand we make the model perform worse by providing it with strings that, while still containing the company name or symbol, mislead it and makes it harder to classify the company?

This is related to the recent practice introduced by large language models, that of prompt engineering. The idea is that we can guide the model towards a particular task by providing it with a prompt, which is a string that is prepended to the input. For example, if we want to use GPT-3 to translate from English to Spanish, we can prepend the string "Translate from English to Spanish:" to the input. This is a very powerful idea, as it allows us to use the same model for a variety of tasks, by simply changing the prompt.

We will be using 4 prompts:
* the default prompt, which is just he original input, be it the company name or symbol
* a prompt that indicates the task we are classifying for, such as "The sector that the company {value} operates in is:"
* a bad prompt (bad prompt 1), which misleads the model by indicating the wrong task, such as "I'd like a {value} with a side of fries"
* finally, another bad prompt, which just surrounds the input value with random strings

After running the experiment the results are as follows:

| X      | Y        | variation      |   accuracy |   accuracy_top5 |
|:-------|:---------|:---------------|-----------:|----------------:|
| Name   | Industry | Default prompt | 0.400835   |        0.642589 |
| Name   | Industry | Better prompt  | 0.405428   |        0.639248 |
| Name   | Industry | Bad prompt 1   | 0.344468   |        0.589979 |
| Name   | Industry | Bad prompt 2   | 0.245511   |        0.475157 |
| Name   | Sector   | Default prompt | 0.572443   |        0.912735 |
| Name   | Sector   | Better prompt  | 0.579123   |        0.915658 |
| Name   | Sector   | Bad prompt 1   | 0.54405    |        0.903132 |
| Name   | Sector   | Bad prompt 2   | 0.491441   |        0.871816 |
| Symbol | Industry | Default prompt | 0.217954   |        0.374948 |
| Symbol | Industry | Better prompt  | 0.132359   |        0.28643  |
| Symbol | Industry | Bad prompt 1   | 0.120251   |        0.260543 |
| Symbol | Industry | Bad prompt 2   | 0.00626305 |        0.034238 |
| Symbol | Sector   | Default prompt | 0.270981   |        0.759081 |
| Symbol | Sector   | Better prompt  | 0.219207   |        0.711065 |
| Symbol | Sector   | Bad prompt 1   | 0.210438   |        0.717745 |
| Symbol | Sector   | Bad prompt 2   | 0.0960334  |        0.553653 |


## Experiment #2: Using a more complex model

Up to now we used a simple logistic regression model to classify the companies. We will now use a more complex model, a neural network with 2 hidden layers. The results are as follows:

| X      | Y        | variation          |   accuracy |   accuracy_top5 |
|:-------|:---------|:-------------------|-----------:|----------------:|
| Name   | Industry | LogisticRegression |   0.4      |        0.643006 |
| Name   | Sector   | LogisticRegression |   0.57286  |        0.912735 |
| Symbol | Industry | LogisticRegression |   0.217954 |        0.374948 |
| Symbol | Sector   | LogisticRegression |   0.271399 |        0.759499 |
| Name   | Industry | MLPClassifier      |   0.475992 |        0.653027 |
| Name   | Sector   | MLPClassifier      |   0.696033 |        0.949896 |
| Symbol | Industry | MLPClassifier      |   0.387474 |        0.505637 |
| Symbol | Sector   | MLPClassifier      |   0.513987 |        0.881837 |


## Experiment #3: Using various embedding models

Finally, let's compare how various embeddings models perform on name->sector classification. We will use various models supported by the sentence-transformers library, and compare their performance. The results are as follows:

![experiment 3](https://raw.githubusercontent.com/darfire/stock-embedding-classification/master/blog/assets/exp3.png)