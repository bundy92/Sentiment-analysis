#1 install and import baseline deopendecies

from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests

#2 setup summarization model

model_name = "human-centered_summarization/financial-summarization-pegasus" #the model we use for the project
tokenizer = PegasusTokenizer.from_pretrained(model_name) #encode and decode the downloaded texts like a translator
model = PegasusForConditionalGeneration.from_pretrained(model_name) #loading the model



#3 summarize a single article

#4 building a news and sentiment pipeline

#4.1 strip out unwanted URLs

