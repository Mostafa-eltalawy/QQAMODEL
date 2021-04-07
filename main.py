
#!pip install -U transformers==3.0.0
#!pip install PIL
#!pip install pytesseract
#!pip install pdf2image
#!sudo apt-get install tesseract-ocr

#!python -m nltk.downloader punkt
#!pip install allennlp
#python -m spacy download en_core_web_sm

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,)

from transformers import AutoModelWithLMHead, AutoTokenizer
# Importing package and summarizer
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import spacy
import re
# load english language model
import en_core_web_sm
from nltk.corpus import stopwords                  
# Import libraries for pdf reading
from PIL import Image 
import pytesseract 
import sys 
from pdf2image import convert_from_path 
import os 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
from tkinter import Tk, Label, Frame, Entry, Button,Text,PhotoImage
# To get the dialog box to open when required  
from tkinter import filedialog 
#==============================================
#loading main files
from QGAPipline import GenAnsPipline
from ReadingpdfFile import PdfReading
#==============================================
nlp = spacy.load('en_core_web_sm')
stops = set(stopwords.words("english"))
nltk.download('punkt')
nltk.download('wordnet')
#==============================================
#uploading pdf file 
def upload_image():
    try:
        file_path=filedialog.askopenfilename(title ='pdf file uploading')
        print('uploading done..')
        return file_path
    except:
        pass
#getting pdf and converting it to text
pdffile= upload_image()   #this will request upload_image function to upload pdf file 
readingpdf=PdfReading(pdffile)  #this function will convert pdf file named uploadedpdf to a text file named out_text

#reading text file after converting from pdf to text out_text
textfile = open("out_text.txt", "r")
uploadedtext = textfile.read()
#===============================================
""" valhalla folder must be at the same directory with the code """
#defining trained model path
modelPath_e2e_qg_small='valhalla/t5-small-e2e-qg/'  
modelPath_qg_small='valhalla/t5-small-qg-hl/'
modelPath_qa_qg_small='valhalla/t5-small-qa-qg-hl/'

modelPath_e2e_qg_base='valhalla/t5-base-e2e-qg/'
modelPath_qg_base='valhalla/t5-base-qg-hl/'
modelPath_qa_qg_base='valhalla/t5-base-qa-qg-hl/'

#==========================================================================
""" downloading pretrained model in valhalla folder  """
model4 = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-base-e2e-qg')
model4.save_pretrained(modelPath_e2e_qg_base)  #downloading model and save it to folder valhalla then folder t5-base-e2e-qg

tokenizer4 = AutoTokenizer.from_pretrained('valhalla/t5-base-e2e-qg')
tokenizer4.save_pretrained(modelPath_e2e_qg_base)
#==========================================================================
model6 = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-base-qa-qg-hl')
model6.save_pretrained(modelPath_qa_qg_base)  #downloading model and save it to folder valhalla then folder t5-base-qa-qg-hl

tokenizer6 = AutoTokenizer.from_pretrained('valhalla/t5-base-qa-qg-hl')
tokenizer6.save_pretrained(modelPath_qa_qg_base)
#==========================================================================
"""qenerating and answering task loading models"""
#generating Questions using model="valhalla/t5-small-qg-hl"
#QGSmall = GenAnsPipline("question-generation")
#generating Questions using model="valhalla/t5-base-qg-hl"
#QGBase = GenAnsPipline("question-generation", model=modelPath_qg_base)
#generating Questions and answering  using model="valhalla/t5-small-qa-qg-hl"
#MultiTaskQAQGsmall = GenAnsPipline("multitask-qa-qg")    
#generating Questions and answering  using model="valhalla/t5-base-qa-qg-hl"
MultiTaskQAQGbase = GenAnsPipline("multitask-qa-qg", model=modelPath_qa_qg_base)    
#generating Questions using model="valhalla/t5-small-e2e-qg"
#E2EQGsmall = GenAnsPipline("e2e-qg")
#generating Questions using model="valhalla/t5-base-e2e-qg"
E2EQGbase = GenAnsPipline("e2e-qg", model=modelPath_e2e_qg_base)

#===============================================
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
#===============================================
# split inputtext to several sentences 
def sentences(text):
    text = re.split('[.?]', text)
    clean_sent = []
    for sent in text:
        clean_sent.append(sent)
    return clean_sent
#===============================================
# API definition
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
logger = logging.getLogger(__name__)

@app.route('/main/QuestionsGeneration', methods=['POST'])
def QuestionsGeneration(uploadedtext):
    inputtext = str(uploadedtext)        
    cleanedtext=clean(inputtext)
    listedtext=sentences(cleanedtext)
    questions=E2EQGbase(listedtext)
    return jsonify(questions)
    #return questions
    
    
@app.route('/main/QuestionsAnswers', methods=['POST'])      
def QuestionsAnswers(uploadedtext):
    inputtext = str(uploadedtext)            
    cleanedtext=clean(inputtext)
    listedtext=sentences(inputtext)
    questions=E2EQGbase(listedtext)
    QesAndAnsList=[]
    for question in questions:
        answer = MultiTaskQAQGbase({"question": question, "context": cleanedtext})
        QesAndAnsList.append(( ' Question  is : ' ,question  ))
        QesAndAnsList.append(( ' Answer  is : ' ,answer  ))
        QesAndAnsList.append('------------------------------------------------------------------')
    return jsonify(QesAndAnsList)
    #return QesAndAnsList

@app.route('/main/KEYWORDS', methods=['POST'])
def KEYWORDS(uploadedtext):
    inputtext = str(uploadedtext)        
    cleantext = clean(inputtext)
    textKeywords=keywords(cleantext, words=30, lemmatize=True)
    #print(textKeywords)
    return jsonify(textKeywords) 
    #return textKeywords

@app.route('/main/TextSummerization', methods=['POST'])
def TextSummerization(uploadedtext):
    inputtext = str(uploadedtext)    
    summerizedtext=summarize(inputtext, ratio=0.6)
    #print(summerizedtext)
    return jsonify(summerizedtext)
    #return summerizedtext
    
@app.route('/main/Entities', methods=['POST'])
def Entities(uploadedtext):
    inputtext = str(uploadedtext)    
    doc=nlp(inputtext)
    entities= []
    for ent in doc.ents:
        entword = ent.text 
        enttype=ent.label_
        entities.append((entword , 'word type is : ' , enttype))   
    return jsonify(entities) 
    #return entities  

    

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
        



