# Sentiment Classification with BERT

In this project, I train a sentiment classifier using a dataset of restaurant reviews from [Yelp](https://raw.githubusercontent.com/mayank100sharma/Sentiment-Analysis-on-Yelp-Reviews/master/yelp.csv). I classify all 5-star reviews as positive and classify all other reviews as negative.

I use a compact, pre-trained BERT model presented by Google Research in [this paper](https://arxiv.org/pdf/1908.08962.pdf) as a base. [This model](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) is much smaller than the original BERT model, allowing it to easily fit in the 1 GB RAM limit of [Streamlit Sharing](https://streamlit.io/sharing), which I use to host the model in a web app.

I tokenize the reviews, send them through the BERT model, and retrieve the output vectors in the [CLS] position.

I train a classifer to take these output vectors and classify them as positive or negative.

[train_classifier.ipynb](https://github.com/DanOKeefe/BERT_Sentiment_Classification/blob/main/train_classifier.ipynb) - Notebook used to train the classifier.

[app.py](https://github.com/DanOKeefe/BERT_Sentiment_Classification/blob/main/app.py) - Web app used to serve the model.

Train the model yourself by running the [training notebook in Google Colab](https://colab.research.google.com/github/DanOKeefe/BERT_Sentiment_Classification/blob/main/train_classifier.ipynb).
