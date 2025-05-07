import re
import requests
import time
import random

import openai
import json


EMBEDDING_MODEL = "text-embedding-ada-002"






#TODO: model call also generic (ex: llama [INST])

def call_llm(model, messages, convid=None):
    if "ibm_bam/" in model: 
        model = re.sub(r'ibm_bam/', "", model)
        ibm_input = ""
        if "llamax" in model:
            for m in messages: 
                if m["role"] == "system": 
                    ibm_input += "\n<<SYS>>\n"+ m["content"] + "\n<</SYS>>\n"
                elif m["role"] == "user": 
                    ibm_input += "\n[INST]\n" + m["content"] + "\n[/INST]\n"
                else: 
                    ibm_input += "\nAnswer: \n" + m["content"] + "\n"

        elif "llamax" in model: 
            for m in messages: 
                if m["role"] == "system": 
                    ibm_input += "\n<<SYS>>\n"+ m["content"]
                elif m["role"] == "user": 
                    ibm_input += "\n\nHuman: " + m["content"]
                else: 
                    ibm_input += "\n\n" + m["content"] + "\n"
        # print("ibm input", ibm_input)
        if convid:
            return get_ibm_ans(model, messages, convid)
        else:
            return get_ibm_ans(model, messages)
    elif "openai/" in model: 
        model = re.sub(r'openai/', "", model)
        # print(model, messages)
        return get_openai_ans(model, messages)
    else: 
        print("model name not valid")
        return None



def get_ibm_ans(model, input_text, convid=None):
    url = 'https://bam-api.res.ibm.com/v2/text/chat?version=2024-03-19' #'https://bam-api.res.ibm.com/v2/text/generation?version=2024-02-02'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_API_KEY'  # Replace YOUR_API_KEY with your actual API key
    }
    # print(input_text)
    if convid:
        data = {
            "model_id": model,
            "parameters": {
                "temperature": 0.8,
                "decoding_method": "greedy",
                "min_new_tokens": 1,
                "max_new_tokens": 1000
            },
            "conversation_id": convid,
            "messages": input_text #input
        }
    else:
        data = {
            "model_id": model,
            "parameters": {
                "temperature": 0.8,
                "decoding_method": "greedy",
                "min_new_tokens": 1,
                "max_new_tokens": 1000
            },
            "messages": input_text #input
        }


# data = {
#         "model_id": model,
#         "parameters": {
#             # "decoding_method": "greedy",
#             # "min_new_tokens": 1,
#             # "max_new_tokens": 1000,
#             "temperature": 0.8
#         },
#         # "moderations": {
#         #     "hap": {
#         #         "threshold": 0.75,
#         #         "input": True,
#         #         "output": True
#         #     },
#         #     "stigma": {
#         #         "threshold": 0.75,
#         #         "input": True,
#         #         "output": True
#         #     }
#         # },
#         "messages": input_text #input
#     }

    out = "failed"
    convid1 = ""
    try:
        response = requests.post(url, headers=headers, json=data)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print("Request was successful!")
            # print("Response:")
            # print(response.json())
            out = response.json()["results"][0]["generated_text"]
            convid1 = response.json()["conversation_id"]

        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response content:")
            print(response.text)
     

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


    return out, convid1

