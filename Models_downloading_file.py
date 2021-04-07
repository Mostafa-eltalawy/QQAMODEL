'''
important note:
    we will use this code only if we have no any downloaded models in valhalla folder
    and valhalla folder must be at the same folder at downloading to use it again by loading it
    
'''

from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,)

from transformers import AutoModelWithLMHead, AutoTokenizer
#==========================================================================
#defining downloading path
# valhalla folder must be at the same directory with the code
modelPath_e2e_qg_small='valhalla/t5-small-e2e-qg/'  
modelPath_qg_small='valhalla/t5-small-qg-hl/'
modelPath_qa_qg_small='valhalla/t5-small-qa-qg-hl/'

modelPath_e2e_qg_base='valhalla/t5-base-e2e-qg/'
modelPath_qg_base='valhalla/t5-base-qg-hl/'
modelPath_qa_qg_base='valhalla/t5-base-qa-qg-hl/'

#==========================================================================
#rin below code step by step 
#downloading and saving model files one time only
model1 = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-small-e2e-qg')
model1.save_pretrained(modelPath_e2e_qg_small)  #downloading model and save it to folder valhalla then folder t5-small-e2e-qg

tokenizer1 = AutoTokenizer.from_pretrained('valhalla/t5-small-e2e-qg')
tokenizer1.save_pretrained(modelPath_e2e_qg_small)

#==========================================================================
model2 = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-small-qg-hl')
model2.save_pretrained(modelPath_qg_small)  #downloading model and save it to folder valhalla then folder t5-small-qg-hl

tokenizer2 = AutoTokenizer.from_pretrained('valhalla/t5-small-qg-hl')
tokenizer2.save_pretrained(modelPath_qg_small)
#==========================================================================
model3 = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-small-qa-qg-hl')
model3.save_pretrained(modelPath_qa_qg_small)  #downloading model and save it to folder valhalla then folder t5-small-qa-qg-hl

tokenizer3 = AutoTokenizer.from_pretrained('valhalla/t5-small-qa-qg-hl')
tokenizer3.save_pretrained(modelPath_qa_qg_small)
#==========================================================================
model4 = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-base-e2e-qg')
model4.save_pretrained(modelPath_e2e_qg_base)  #downloading model and save it to folder valhalla then folder t5-base-e2e-qg

tokenizer4 = AutoTokenizer.from_pretrained('valhalla/t5-base-e2e-qg')
tokenizer4.save_pretrained(modelPath_e2e_qg_base)
#==========================================================================
model5 = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-base-qg-hl')
model5.save_pretrained(modelPath_qg_base)  #downloading model and save it to folder valhalla then folder t5-base-qg-hl

tokenizer5 = AutoTokenizer.from_pretrained('valhalla/t5-base-qg-hl')
tokenizer5.save_pretrained(modelPath_qg_base)
#==========================================================================
model6 = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-base-qa-qg-hl')
model6.save_pretrained(modelPath_qa_qg_base)  #downloading model and save it to folder valhalla then folder t5-base-qa-qg-hl

tokenizer6 = AutoTokenizer.from_pretrained('valhalla/t5-base-qa-qg-hl')
tokenizer6.save_pretrained(modelPath_qa_qg_base)
#==========================================================================