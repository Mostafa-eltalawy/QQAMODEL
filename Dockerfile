FROM python:3.6.4
WORKDIR /app
ADD main.py .
COPY requirements.txt .

RUN pip install numpy
RUN pip install gensim
RUN pip install logging

RUN pip install spacy 
RUN pip install pytesseract
RUN pip install transformers


RUN pip install pdf2image
RUN pip install typing
RUN pip install PIL
RUN pip install torch==1.7.1
RUN pip install flask
RUN pip install flask-cors
RUN pip install  nltk
RUN pip install re
RUN pip install itertools
RUN pip install tkinter

#RUN /bin/sh -c pip install -r requirements.txt

COPY . /app
EXPOSE 5000
CMD ["python","./main.py"]
