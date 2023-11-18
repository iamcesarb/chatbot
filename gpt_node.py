#!/urs/bin/env python3
import rospy
import openai
import os
import requests
def n_context_response(pre)->str:
    openai.api_key = "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"
    pregunta = pre + ' respondeme de manera corta como si fuera una conversacion'
    res = openai.Completion.create(engine = "text-davinci-002", prompt = pregunta, max_tokens = 35)
    return res["choices"][0]["text"]

pregunta = input()
print(n_context_response(pregunta))
