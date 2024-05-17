# **Sentiment Analysis for Product Reviews**  

## Project Overview  

This project aims to develop a sentiment analysis model to classify customer reviews from Flipkart into neutral, positive, or negative sentiments. Leveraging machine learning techniques, the model will analyze textual reviews and assign appropriate sentiment labels to each review.Flipkart is a leading e-commerce platform in India, offering a wide range of products across categories like mobiles, fashion, electronics, home appliances, groceries, and more. It provides a convenient and secure online shopping experience with features like cash on delivery, easy returns, and multiple payment options.

## Business Understanding

Understanding customer sentiment is crucial for businesses to gauge product reception, identify areas of improvement, and tailor marketing strategies. By analyzing customer reviews, businesses can gain valuable insights into customer satisfaction levels, product strengths, and weaknesses. This sentiment analysis project will provide actionable intelligence to enhance customer experience and product offerings.

### Objectives

- Develop a robust sentiment analysis model capable of accurately categorizing customer reviews into neutral, positive, or negative sentiments.
- Sentiment Analysis and text data analysis to derive insights from it
- Provide actionable insights to stakeholders for informed decision-making, such as product improvements, marketing strategies, and customer engagement tactics.
- Deploy the sentiment analysis model.

## Data Understanding

The data is obtained from Kaggle: [Flipkart Product reviews with sentiment Dataset](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset). The dataset consists of customer reviews sourced from Flipkart, a prominent Indian e-commerce platform.

### Data Columns

- **Product_name:** Name of the product.
- **Product_price:** Price of the product.
- **Rate:** Customer's rating on product(Between 1 to 5).
- **Review:** Customer's review on each product.
- **Summary:** This column include descriptive information of customer's thought on each product.

## Data preprocessing

- Text Analysis with NLP Preprocessing
- Sentiment Analysis - Sentiment Analysis

## Modeling  

- Logistic Regression
- SVM Model
- Random Forest model
- Tuned Logistic Regression

## Deployment

App: [Link](https://flipkart.streamlit.app/)

## Recommendation

- Leveraging Positive Keywords for Product Insights 
- Enhancing Customer Engagement through Positive Sentiment Recognition
- Utilizing Negative Keywords for Product Improvement
- Addressing Negative Feedback for Improved Perception
- Harnessing Sentiment Analysis for Predictive Insights
- using the deployed Sentiment Analysis App for Review Classification

## Project Requirements

To run this project, you need to install the following dependencies:  

```python
# requirements.txt 
streamlit
nltk
pandas
matplotlib
seaborn
numpY
scikit-learn
imbalanced-learn
wordcloud
```