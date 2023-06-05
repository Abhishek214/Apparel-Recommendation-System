**Title:** Apparel Recommendation: Enhancing Shopping Experience with AI

**Introduction:**
With the ever-increasing popularity of online shopping, personalized product recommendations have become essential for enhancing the customer experience. In this project, we have developed an apparel recommendation engine that leverages the Amazon API, Natural Language Toolkit (NLTK), and Keras to extract apparel details and recommend similar products to users. By utilizing various techniques such as bag of words, TF-IDF, word2vec, and convolutional neural networks, we provide accurate and personalized recommendations to customers.

**Problem Statement:**
Our goal is to recommend similar apparel products based on a given product's details. We extract data from a JSON file containing over 180,000 apparel images from Amazon.com. The recommendations are generated based on fields such as ASIN, brand, color, product type, image URL, title, and price. We employ seven different approaches to ensure diverse and effective recommendations.

**Approaches Used:**
1. Bag of Words Model: Utilizes a bag of words representation to find similar products based on textual information.

2. TF-IDF Model: Implements the TF-IDF algorithm to measure the importance of words in the product descriptions and suggests similar items.

3. IDF Model: Focuses on the inverse document frequency (IDF) of words to identify related products.

4. Word2Vec Model: Employs word embeddings to capture semantic relationships between words and generate recommendations based on similarity.

5. IDF Weighted Word2Vec Model: Combines the IDF and Word2Vec models, weighting the word vectors based on their IDF scores for improved recommendation accuracy.

6. Weighted Similarity using Brand and Color: Uses the brand and color information to calculate weighted similarity scores and recommend relevant apparel products.

7. Visual Features Based on Convolutional Neural Networks: Extracts visual features from apparel images using a pre-trained CNN model and recommends visually similar items.

**Datasets and Inputs:**
- tops_fashion.json: JSON file containing the details of over 180,000 apparel images from Amazon.com.
- 16k_apparel_preprocessed pickle file: Preprocessed data of 16,000 apparel items for training and evaluation.
- Trained Word2Vec Model: Pre-trained Word2Vec model used for generating word embeddings.
- Trained CNN Model: Pre-trained Convolutional Neural Network (CNN) model for extracting visual features from apparel images.
- 16k_data_features_asins: ASINs (Amazon Standard Identification Numbers) of 16,000 apparel items.
- 16k_data_cnn_features.npy: Extracted CNN features for the 16,000 apparel items.

**Software Requirements:**
- Anaconda with additional packages: TensorFlow, Plotly, PIL.
- GPU for training the CNN and Word2Vec models.

**Execution and Running Code:**
1. Clone the repository by executing the following command in the terminal: `git clone https://github.com/Abhinav1004/Apparel-Recommendation.git`.
2. Launch Jupyter Notebook by executing the command: `jupyter notebook Apparel_Recommendation.ipynb`.
3. Run the notebook cells by pressing Shift+Enter to execute the code.
4. Observe the results, including the recommended apparel items and the evaluation metrics.

**Observations:**
During the project, we trained and evaluated seven different models for recommending similar apparel products. For each model, we recommended the top 20 apparel items with the least Euclidean distance. We calculated the average Euclidean distance for each model and compared their performance using line plots and bar graphs.

**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF

7.CNN
