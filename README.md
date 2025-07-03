# NLP: Sentiment Analysis -  <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/twitter.svg" alt="Twitter" width="32" height="32" style="vertical-align:middle;"/>  Twitter & <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/amazon.svg" alt="Amazon" width="32" height="32" style="vertical-align:middle;"/>  Amazon Reviews
## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Business Case](#business-case)
- [Datasets](#datasets)
- [Project Structure & Key Tasks](#project-structure--key-tasks)
- [Methodology](#methodology)
- [Results & Insights](#results--insights)
- [How to Run the Project](#how-to-run-the-project)
- [Dependencies](#dependencies)
- [Coursera Project Network](#coursera-project-network)
- [Author](#author)
- [License](#license)

---

## 1. Project Overview
This project focuses on Sentiment Analysis using Natural Language Processing (NLP) techniques. The goal is to classify text data (tweets and Amazon product reviews) into sentiment categories (e.g., positive, negative, neutral). This involves a complete machine learning pipeline from data loading and exploration to cleaning, feature extraction, model training, and evaluation. The project was completed as part of the "NLP: Twitter Sentiment Analysis" course on Coursera Project Network.

## 2. Problem Statement
The core problem addressed is the automatic identification of emotional tone within text. Given a piece of text (like a tweet or a product review), can we accurately determine if the expressed sentiment is positive, negative, or neutral? This is crucial for understanding public opinion, customer feedback, and brand perception at scale.

## 3. Business Case
Sentiment analysis has numerous real-world applications across various industries:

- **Customer Service:** Quickly identify dissatisfied customers from reviews or social media to provide timely support.
- **Brand Monitoring:** Track public perception of a brand, product, or campaign in real-time.
- **Market Research:** Understand consumer preferences and trends from large volumes of text data.
- **Product Development:** Gain insights from product reviews to inform future feature enhancements or address common complaints.
- **Social Media Analysis:** Monitor and react to public discourse around specific topics or events.

## 4. Datasets
This project utilizes two distinct datasets to demonstrate the versatility of sentiment analysis techniques:

### Twitter Sentiment Dataset
- **Source:** [Specify where you got the Twitter dataset, e.g., Kaggle, a specific Coursera link, etc. If it's a course provided dataset, mention that.]
- **Description:** Contains tweets labeled with sentiment (e.g., 0 for negative/neutral, 1 for positive). This dataset helps in understanding short, informal text often containing slang and abbreviations.
- **Filename:** `twitter.csv`

### Amazon Reviews Dataset
- **Source:** [Specify where you got the Amazon dataset, e.g., Kaggle, UCI ML Repository, etc.]
- **Description:** Comprises product reviews from Amazon, typically longer and more structured than tweets. It's used to analyze customer feedback on products. The sentiment might be implied from rating or explicit 'feedback' column.
- **Filename:** `amazon_reviews.csv` (or your actual file name, e.g., `reviews.csv`)

## 5. Project Structure & Key Tasks
The project follows a standard machine learning workflow, organized into the following tasks:

1. **Understand the Problem Statement and Business Case:** Defined the core problem and real-world applications.
2. **Import Libraries and Datasets:** Loaded necessary Python libraries (Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn, WordCloud) and both `twitter.csv` and `amazon_reviews.csv` into Jupyter Notebooks.
3. **Perform Exploratory Data Analysis (EDA):**
    - Inspect dataset structure (`.head()`, `.info()`, `.describe()`).
    - Check for missing values.
    - Calculate and visualize the length of reviews/tweets (e.g., character count) for initial insights.
    - Analyze the distribution of sentiment labels (countplot).
4. **Plot the Word Cloud:** Generated word clouds for different sentiment categories (e.g., positive vs. negative) to visually identify frequently occurring words.
5. **Perform Data Cleaning - Removing Punctuation:** Implemented a function to strip punctuation from text data.
6. **Perform Data Cleaning - Remove Stop Words:** Utilized NLTK's stopwords list to remove common words that do not carry significant sentiment (e.g., "the", "is", "and").
7. **Perform Count Vectorization (Tokenization):** Converted cleaned text into numerical feature vectors using `CountVectorizer`, representing word frequencies.
8. **Create a Pipeline for Preprocessing & Tokenization:** Combined the cleaning steps and Count Vectorization into a reusable pipeline for efficient data transformation.
9. **Understand the Theory and Intuition behind Naive Bayes Classifiers:** Explored the probabilistic foundations of Naive Bayes, particularly its application in text classification.
10. **Train a Naive Bayes Classifier:**
    - Split the processed data into training and testing sets.
    - Trained a Multinomial Naive Bayes model on the vectorized text data.
11. **Assess Trained Model Performance:**
    - Evaluated the model's performance using metrics such as confusion matrix, classification report (precision, recall, f1-score), and accuracy.

## 6. Methodology
The project adopts a supervised machine learning approach for sentiment classification.

- **Data Loading:** Datasets are loaded into Pandas DataFrames.
- **Exploratory Data Analysis (EDA):** Initial understanding of data distribution, text lengths, and sentiment balance.
- **Text Preprocessing:**
    - Handling Missing Values: Replaced NaN values in text columns with empty strings.
    - Punctuation Removal: Stripped standard punctuation.
    - Stop Word Removal: Eliminated common, less informative words.
    - Lowercasing: Converted all text to lowercase.
- **Feature Extraction:** `CountVectorizer` transforms text into a sparse matrix of token counts ("bag-of-words" representation).
- **Model Training:** A Multinomial Naive Bayes classifier is chosen for its suitability in text classification.
- **Model Evaluation:** Standard classification metrics are used to assess the model's ability to generalize to unseen data.

## 7. Results & Insights
[Summarize your key findings here. Be specific! Replace examples below with your actual results.]

- The EDA revealed a significant class imbalance in the Twitter dataset (approx. 90% neutral/negative vs. 10% positive), which impacted initial model performance.
- Word clouds clearly showed common positive terms like 'love', 'great', 'happy' and negative terms like 'bad', 'problem', 'fail' for their respective categories.
- The Naive Bayes classifier achieved an accuracy of approximately **[X]%** on the Twitter dataset and **[Y]%** on the Amazon reviews dataset.
- Precision and recall varied across classes, indicating [e.g., better performance on positive class than negative due to data imbalance / misclassification of certain types of reviews].
- The length of reviews significantly varied, with Amazon reviews being much longer on average than tweets, requiring robust text cleaning.

## 8. How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### Place Datasets
Ensure your `twitter.csv` and `amazon_reviews.csv` (or your actual Amazon reviews CSV name) are placed in the root directory of the cloned repository, or update the file paths in the notebooks accordingly.

### Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
(You'll need to create a `requirements.txt` file. See the next section.)

### Download NLTK Data
Open a Python interpreter or a new Jupyter cell and run:
```python
import nltk
nltk.download('stopwords')
```

### Launch Jupyter Notebook
```bash
jupyter notebook
```

### Open and Run Notebooks
- For Twitter Sentiment Analysis: Open `Twitter_Sentiment_Analysis - Skeleton (1).ipynb`
- For Amazon Reviews Sentiment Analysis: Open `Amazon Reviews Sentiment Analysis.ipynb`

Run all cells sequentially in each notebook.

## 9. Dependencies
The project relies on the following Python libraries. You can install them using `pip install -r requirements.txt` after creating the file.

Create a `requirements.txt` file in your project root with these contents:
```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
wordcloud
jupyter
jupyterthemes  # If you used it for styling
```

## 10. Coursera Project Network
This project was developed as part of the "NLP: Twitter Sentiment Analysis" guided project on Coursera Project Network. It provided the foundational tasks and guidance to implement the sentiment analysis pipeline.

## 11. Author
[Fouzia Ashfaq] 

## 12. License
This project is licensed under the MIT License - see the LICENSE.md file for details (Optional, but good practice. If you don't have one, you can remove this section or choose a license).
