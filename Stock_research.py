#1 install and import baseline deopendecies
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
from bs4 import BeautifulSoup
import requests
import re
import torch
#import csv 
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name="gpu")

device = "cuda" if torch.cuda.is_available() else "cpu"

#2 setup the summarization model
model_name = "human-centered-summarization/financial-summarization-pegasus" #the model we use for the project
tokenizer = PegasusTokenizer.from_pretrained(model_name) #encode and decode the downloaded texts like a translator
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device) #loading the model
print("Setting up the model.")
""""
#3 summarize a single article
url = "https://finance.yahoo.com/news/tesla-reverses-model-y-price-204727601.html" #will be automated
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
paragraphs = soup.find_all('p')
#print(paragraphs)
text = [paragraphs.text for paragraphs in paragraphs]
words = ' '.join(text).split(' ')[:400]
ARTICLE = ' '.join(words)

input_ids = tokenizer.encode(ARTICLE, return_tensors = 'pt')
output = model.generate(input_ids, max_length = 55, num_beams = 5, early_stopping = True)
summary = tokenizer.decode(output[0], skip_special_tokens = True) #translating back from tokens to words
#
# print(summary)
"""

#4 building a news and sentiment pipeline
monitored_tickers = ['TSLA', 'BTC', 'AAPL', 'AMZN']

#4.1 search for stock news using google and yahoo finance
def search_for_stock_news_urls(ticker):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
    #search_url = "https://www.ecosia.org/news?q=yahoo%20finance%20{}".format(ticker)
    r = requests.get(search_url, cookies = {"CONSENT": "YES+cb.20210720-07-p0.en+FX+410"})
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]
    return hrefs 

raw_urls = {ticker:search_for_stock_news_urls(ticker) for ticker in monitored_tickers}

exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
print("Cleaning up URLs.")
#4.2 strip out unwanted URLs
def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls: 
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}

print("Removing unwanted URLs.")
#4.3 search and scrape unwanted  URLs
def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs: 
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:200]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES

articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}
#print(len(articles['BTC'][0]))
#print(articles['BTC'][0])
#4.4 summarize articles 
print("Summarizing articles.")
def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors = 'pt').to(device)
        output = model.generate(input_ids, max_length = 50, num_beams = 5, early_stopping = True)
        summary = tokenizer.decode(output[0], skip_special_tokens = True)
        summaries.append(summary)
    return summaries

summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}
#print(summaries)

#5 adding sentiment analysis
print("Commencing sentiment analysis.")
sentiment = pipeline('sentiment-analysis')

scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers}

#print(summaries['TSLA'][0], scores['TSLA'][0]['label'], scores['TSLA'][0]['score'])

#6 exporting results to csv

def create_output_array(summaries, scores, urls):
    output = []
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])): 
            output_this = [
                ticker,
                summaries[ticker][counter],
                scores[ticker][counter]['label'],
                scores[ticker][counter]['score'],
                urls[ticker][counter]
            ]    
            output.append(output_this)
    return output 
    
print("Creating final output.")

final_output = create_output_array(summaries, scores, cleaned_urls)
final_output.insert(0, ['Ticker', 'Summary', 'Label', 'Confidence', 'URL'])

#with open('assetsummaries.csv', mode='w', newline='') as f:
 #   csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  #  csv_writer.writerows(final_output)