#%%
import datetime
import json
import os

import openai

from retrieval import RetrievalSystem

openai.api_key = os.environ["OPENAI_API_KEY"]

# db can be found here: https://github.com/adactio/TheSession-data/blob/main/json/tunes.json
DB_PATH = "data/tunes.json"

TEMPERATURE=1
MAX_TOKENS=256
TOP_P=1
FREQUENCY_PENALTY=0
PRESENCE_PENALTY=0

SEP = "@"

system_prompt=f'''
You are a highly advanced folk music generation and editing system that outputs folk music in abc notation.
Prior to generating a piece you will describe the piece you will generate, first in words, and then commenting on how this will be reflected in the abc notation.
Always output the commentary first followed by {SEP} and then the abc notation.
Don't write anything else except the commentary and abc notation.
Keep the commentary very short and to the point (less than 140 characters).
Here is a reference example of folk music:
M:4/4
K:Dmajor
DE|F2ED EDEF|A4 B2dA|B2AF E3D|B,4 A,2DE|
F2ED EDEF|A4 B2dA|B2AF E3D|D6||
A2|d2cA B3d|c3A F2FA|B2AF ABde|d6 DE|
F2ED EDEF|A4 B2dA|B2AF E3D|D6||
Occasionally, the assistant will provide more reference tunes to help you, you may use these if you wish.
'''
#%%
messages = [
            {
            "role": "system",
            "content": system_prompt
            },  
        ]

def call_gpt(messages):
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    top_p=TOP_P,
    frequency_penalty=FREQUENCY_PENALTY,
    presence_penalty=PRESENCE_PENALTY
    )
    return response

def dummy_gpt(messages):
    return {
        "choices": [
            {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I don't feel like making music right now."
            },
            "finish_reason": "stop"
            }
        ]
    }
USE_DUMMY = False
retrieval_system = RetrievalSystem(openai_api_key=openai.api_key, db_path=DB_PATH)

print("Starting new SCRAP session")
while True:
    text = input("User: ")
    # remove the user prompt from the system out
    print("\033[A                             \033[A")

    # if the input is empty or "quit" then exit the loop
    if text == "quit" or text == "":
        print("SCRAP: Goodbye")
        break
    print("User:", text)
    messages.append({"role": "user", "content": text})
    references = retrieval_system.call(text)
    if  references is not None:
        messages.append({"role": "assistant", 
                        "content": "I found these reference tunes:\n" + references})
    if USE_DUMMY:
        response = dummy_gpt(messages)
    else:
        response = call_gpt(messages)
    return_message = response["choices"][0]["message"]
    messages.append(return_message)
    print("SCRAP:\n", return_message["content"])

# make artefacts directory if it doesn't exist
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists("artefacts"):
    os.makedirs("artefacts")
with open(f"artefacts/conversation_{timestamp}.json", "w", newline="") as f:
    json.dump(messages, f, indent=4)
# %%

