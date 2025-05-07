import time
import random
import csv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import call_llm
import json
import argparse
import pickle
import os
#ibm_bam/meta-llama/llama-2-70b-chat
#openai/gpt-3.5-turbo

class QaTester:
    def __init__(self, save_folder, llm_model, has_examples = False, roleplay = False, sbs = False, ca = False):

        self.save_folder = save_folder
        self.llm_model = llm_model


        self.start_text = """You are an analog circuit designer"""
        if roleplay: 
            self.start_text = "You are an analog circuit design assistant. " + self.start_text
        if sbs: 
            self.start_text = self.start_text + "\nLet’s talk about this in a step-by-step way."
        if ca: 
            self.start_text = self.start_text + "\nPlease be sure you have the correct answer."
        self.examples = []
        if has_examples: 
            self.examples = []



    def get_ans(self, input_text, convid=None):
        if convid:
            return call_llm.call_llm(self.llm_model, input_text, convid)
        else:
            return call_llm.call_llm(self.llm_model, input_text)
    
    def mc_tester(self, qa_filename, write_filename):
        write_filename = self.save_folder + write_filename

        f = open(qa_filename)

        w = open(write_filename, "w")


        qa_data = json.load(f)

        c = 0
        
        messages=[]

        question1 = qa_data[0]['netlist']
        question2 = qa_data[0]['prompt']
        question_string = question1 + "\n" + question2
        messages.append({"role": "system", "content":question1})
        messages.append({"role": "user", "content": question2})

        out, convid = self.get_ans(messages) 



        w.writelines(out)
        return convid

    def mc_tester1(self, qa_filename, write_filename, convid):
        write_filename = self.save_folder + write_filename

        f = open(qa_filename)

        w = open(write_filename, "w")

        qa_data = json.load(f)
        c = 0
        
        messages=[]
        messages1=[]
        question = qa_data[0]['prompt']
        question_string = question
        messages.append({"role": "user", "content": question_string})
       # w.writelines(str(messages))

        out, convid2 = self.get_ans(messages, convid) 
        messages.append({"role": "assistant", "content": out})
        with open('/path/to/LLM/my_variablel3.pkl', 'rb') as file:
            messages1 = pickle.load(file)
        messages1.append({"role": "user", "content": question_string})
        messages1.append({"role": "assistant", "content": out})
        with open('/path/to/LLM/my_variablel3.pkl', 'wb') as file:
            pickle.dump(messages1, file)

        w.writelines(out)

if __name__ == "__main__":
    f = open("/path/to/LLM/convid.txt", "r")
    convid = f.read()
    save_folder = ""
    modelname = "ibm_bam/meta-llama/llama-3-70b-instruct"
    has_examples = False
    tester = QaTester(save_folder, modelname, True)
    tester.mc_tester1("/path/to/LLM/sample-qa-data_l.json", "/path/to/LLM/qa_output_amp_l.txt", convid)



