
#!pip install -U transformers==3.0.0
#!pip install PIL
#!pip install pytesseract
#!pip install pdf2image
#!sudo apt-get install tesseract-ocr

#!python -m nltk.downloader punkt
#!pip install allennlp
#python -m spacy download en_core_web_sm

from keras.models import load_model
import tensorflow
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
import pickle
import random
import json
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from flask_cors import CORS
from googletrans import Translator


import itertools
import logging
from typing import Optional, Dict, Union

from nltk import sent_tokenize

import torch
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,)

from transformers import AutoModelWithLMHead, AutoTokenizer
logger = logging.getLogger(__name__)
# Importing package and summarizer
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import spacy
import re

import spacy
from spacy.matcher import Matcher 
from spacy import displacy 
#import visualise_spacy_tree
from IPython.display import Image, display

# load english language model
import en_core_web_sm
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))                  

# Import libraries for pdf reading
from PIL import Image 
import pytesseract 
import sys 
from pdf2image import convert_from_path 
import os 
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
from tkinter import Tk, Label, Frame, Entry, Button,Text,PhotoImage
import tkinter as tk 
# To get the dialog box to open when required  
from tkinter import filedialog 

#loading other files 
from QGAPipline import GenAnsPipline
from ReadingpdfFile import PdfReading

#uploading pdf file 
def upload_image():
    try:
        file_path=filedialog.askopenfilename(title ='pdf file uploading')
        return file_path
    
    except:
        pass

pdffile= upload_image()
readingpdf=PdfReading(pdffile)  #this function will convert pdf file named uploadedpdf to a text file named 

#reading text file after converting from pdf to text out_text
f = open("out_text.txt", "r")
uploadedtext = f.read()

# valhalla folder must be at the same directory with the code
modelPath_e2e_qg_small='valhalla/t5-small-e2e-qg/'  
modelPath_qg_small='valhalla/t5-small-qg-hl/'
modelPath_qa_qg_small='valhalla/t5-small-qa-qg-hl/'

modelPath_e2e_qg_base='valhalla/t5-base-e2e-qg/'
modelPath_qg_base='valhalla/t5-base-qg-hl/'
modelPath_qa_qg_base='valhalla/t5-base-qa-qg-hl/'

#downloading and saving model files one time only
#model.save_pretrained(modelPath_e2e_qg_small)  #downloading model and save it to folder valhalla then folder t5-small-e2e-qg
#model.save_pretrained(modelPath_qg_small)  #downloading model and save it to folder valhalla then folder t5-small-qg-hl
#model.save_pretrained(modelPath_qa_qg_small)  #downloading model and save it to folder valhalla then folder t5-small-qa-qg-hl
#model.save_pretrained(modelPath_e2e_qg_base)  #downloading model and save it to folder valhalla then folder t5-base-e2e-qg
#model.save_pretrained(modelPath_qg_base)  #downloading model and save it to folder valhalla then folder t5-base-qg-hl
#model.save_pretrained(modelPath_qa_qg_base)  #downloading model and save it to folder valhalla then folder t5-base-qa-qg-hl

#tokenizer.save_pretrained(modelPath_e2e_qg)   #downloading model and save it to folder valhalla then folder t5-small-e2e-qg
#tokenizer = AutoTokenizer.from_pretrained(modelPath_qa_qg_small)
#model = AutoModelForSeq2SeqLM.from_pretrained(modelPath_qa_qg_small)




"""# qenerating and answering task loading"""

#generating Questions using model="valhalla/t5-small-qg-hl"
QGSmall = GenAnsPipline("question-generation")

#generating Questions using model="valhalla/t5-base-qg-hl"
QGBase = GenAnsPipline("question-generation", model=modelPath_qg_base)

#generating Questions and answering  using model="valhalla/t5-small-qa-qg-hl"
MultiTaskQAQGsmall = GenAnsPipline("multitask-qa-qg")

#generating Questions and answering  using model="valhalla/t5-base-qa-qg-hl"
MultiTaskQAQGbase = GenAnsPipline("multitask-qa-qg", model=modelPath_qa_qg_base)

#generating Questions using model="valhalla/t5-small-e2e-qg"
E2EQGsmall = GenAnsPipline("e2e-qg")

#generating Questions using model="valhalla/t5-base-e2e-qg"
E2EQGbase = GenAnsPipline("e2e-qg", model=modelPath_e2e_qg_base)

# function to clean text
def clean(text):
    # removing paragraph numbers
    #text = re.sub('[0-9]+.\t','',str(text))
    # removing new line characters
    text = re.sub('\n ','',str(text))
    text = re.sub('\n',' ',str(text))
    # removing apostrophes
    text = re.sub("'s",'',str(text))
    # removing hyphens
    text = re.sub("-",' ',str(text))
    text = re.sub("— ",'',str(text))
    # removing quotation marks
    text = re.sub('\"','',str(text))
    # removing salutations
    text = re.sub("Mr\.",'Mr',str(text))
    text = re.sub("Mrs\.",'Mrs',str(text))
    # removing any reference to outside text
    text = re.sub("[\\[]*[\\]]", "", str(text))
    
    return text

# split inputtext to several sentences 
def sentences(text):
    text = re.split('[.?]', text)
    clean_sent = []
    for sent in text:
        clean_sent.append(sent)
    return clean_sent

def GenerateQuestions(inputtext):
    cleanedtext=clean(inputtext)
    listedtext=sentences(cleanedtext)
    questions=E2EQGbase(listedtext)
    return questions
  
def GetQuestionsAnswers(inputtext):
    cleanedtext=clean(inputtext)
    listedtext=sentences(inputtext)
    questions=E2EQGbase(listedtext)
    print('questions list is : \n' , questions)
    print('====================================================================')
    for question in questions:
        answer = MultiTaskQAQGbase({"question": question, "context": cleanedtext})
        print( ' question  is : ' ,question  )
        print( ' answer  is : ' ,answer  )
        print('------------------------------------------------------------------')
    return questions , answer

def getkeywords(inputtext):
    cleantext = clean(inputtext)
    textKeywords=keywords(cleantext, words=30, lemmatize=True)
    print(textKeywords)
    #summerizedtext=summarize(inputtext, ratio=0.6)
    #print(summerizedtext)
    return textKeywords 

def gettextsummerization(inputtext):
    summerizedtext=summarize(inputtext, ratio=0.6)
    print(summerizedtext)
    return summerizedtext


def getEntity(inputtext):
    doc=nlp(inputtext)
    entities= []
    for ent in doc.ents:
        entword = ent.text 
        enttype=ent.label_
        entities.append((entword , enttype))
        #print(ent.text, ' .. and word type is ..',  ent.label_)
    #enydisplay = displacy.render(nlp(str(inputtext)), jupyter=True, style='ent') 
    #print(enydisplay)    
    return entities     

def chatbot_response(input_text):
    QesGenresp=GenerateQuestions(input_text)
    QesAnsres = GetQuestionsAnswers(input_text)
    keywordres = getkeywords(input_text)
    entityres= getEntity(input_text)
    return QesGenresp , QesAnsres , entityres , keywordres

#getting results
input_text = str(uploadedtext)
chatbot_reply = chatbot_response(input_text)
response = chatbot_reply
#response = jsonify(reply_back = chatbot_reply)
print (response)


