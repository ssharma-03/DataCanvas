from transformers import pipeline
import os 
summarizer = pipeline("summarization")

def generate_insights(text):
    summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
    return summary[0]['summary_text']
