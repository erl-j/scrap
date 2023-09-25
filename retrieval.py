#%%
import pandas as pd
import numpy as np
import re
import openai
import os
import sklearn.metrics
from sklearn import preprocessing
import json

def handle_the(title):
    # if trailing the, remove it and put it in front
    thes = [", The"]
    for the in thes:
        if title.endswith(the):
            title = the.strip(", ") +" " +title[:-len(the)]
    return title

def obj2abc_form(obj):
    # convert object to abc form
    abc_form = f'''T:{obj["name"]}
M:{obj["meter"]}
K:{obj["key"]}{obj["mode"]}
{obj["abc"]}
'''
    return abc_form

class RetrievalSystem:
    def __init__(self, openai_api_key, db_path) -> None:
        openai.api_key = openai_api_key
        # load database
        df = pd.read_json(db_path)
        # drop with duplicate abc notation and name
        df.drop_duplicates(subset=["abc"]).drop_duplicates(subset=["name"]).drop_duplicates(subset=["tune_id"])
        df = df[["name", "type", "mode", "meter", "abc"]]
        # split mode into key (first char) and mode (rest of string)
        df["key"] = df["mode"].str[0]
        df["mode"] = df["mode"].str[1:]
        df["name"] = df["name"].apply(handle_the)

        # prepare database description
        n_tunes = len(df)
        tune_types= df["type"].unique()
        tune_keys = df["key"].unique()
        tune_modes = df["mode"].unique()
        tune_meters = df["meter"].unique()
        db_description = f'''
        The database contains {n_tunes} tunes with attributes type, mode, and meter.
        The tune types are {tune_types}.
        The modes are {tune_modes}.
        The meters are {tune_meters}.
        ''' 
        # future work: Include abc notation and prompt "You may use regex to rank according to the presene of specific patterns in the abc notation if needed."

        # prepare system prompt
        self.system_prompt = f'''
        You are a highly advanced folk music retireval system.
        You are tasked with with assisting a folk music musician in finding tunes in a database for inspiration in composing new tunes or editing existing tunes.
        You are given a natural language text and you must return prototypical object(s) which we will use to rank the tunes by similarity in the database.{db_description}
        '''

        self.TEMPERATURE=1
        self.MAX_TOKENS=256
        self.TOP_P=1
        self.FREQUENCY_PENALTY=0
        self.PRESENCE_PENALTY=0

        # put all labels in a long list
        self.all_tags = list(df["type"].unique()) +  list(df["mode"].unique()) + list(df["meter"].unique())
        self.encoder = preprocessing.MultiLabelBinarizer()
        self.encoder.fit([self.all_tags])
        self.db_encoded = self.encoder.transform(df[["type", "mode", "meter"]].values.tolist())
        self.df = df

        print(f"Retrieval system initialized with {n_tunes} tunes in database.")

        self.K = 3 # number of tunes to return per query

    def jaccard_similarity(self,query,db):
        # expand query in numpy array
        query = query.repeat(db.shape[0],axis=0)
        # compute intersection
        intersection = np.sum(query * db, axis=1)
        # compute union
        union = np.sum(query+db>0, axis=1)
        # compute jaccard similarity
        jaccard_score = intersection/union
        return jaccard_score

    def select_tunes(self,query_obj):
        # get tags for query
        query_tags = list(query_obj.values())
        # check if query tags are in database and print items not in database
        for tag in query_tags:
            if tag not in self.all_tags:
                print(f"{tag} not in database")
        # encode query tags
        query_encoded = self.encoder.transform([query_tags])
        # compute jaccard similarity
        jaccard_score = self.jaccard_similarity(query_encoded,self.db_encoded)
        # get index by jaccard distance
        sort = np.argsort(-jaccard_score)
        # sort tunes by jaccard score
        sorted_df = self.df.iloc[sort]
        # get K top tunes
        top_tunes = sorted_df.iloc[:self.K]
        tunes = []
        for i, tune in top_tunes.iterrows():
            # turn back into dict
            comment = f"This tune is a {tune['type']} in {tune['key']}{tune['mode']} and {tune['meter']}."
            tune_dict = tune.to_dict()
            tunes.append({"comment": comment, "content": tune_dict, "abc_form":obj2abc_form(tune_dict)})
        return tunes

    def translate_query(self,natural_language_query):
        messages = [
            {
            "role": "system",
            "content": self.system_prompt
            },  
            {
            "role": "user",
            "content": "I want to rewrite this as a jig or a reel in 4/4 and Dminor."
            },
            {"role": "assistant",
            "content": "[\{\"type\": \"jig\", \"meter\": \"4/4\", \"mode\": \"minor\"\}]"
            },
             {
            "role": "user",
            "content": "Give me a slow hornpipe."
            },
            {"role": "assistant",
            "content": "[\{\"type\": \"hornpipe\"}]"
            },
            {
            "role": "user",
            "content": "Make the song faster."
            },
            {"role": "assistant",
            "content": "Not applicable."
            },
            {"role": "assistant",
            "role": "user",
            "content": natural_language_query
            }
        ]

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=self.TEMPERATURE,
        max_tokens=self.MAX_TOKENS,
        top_p=self.TOP_P,
        frequency_penalty=self.FREQUENCY_PENALTY,
        presence_penalty=self.PRESENCE_PENALTY
        )
        message = response["choices"][0]["message"]
        # sanitize message
        message["content"] = re.sub(r'\\','',message["content"])
        # parse json
        query_obj = json.loads(message["content"])
        return query_obj

    def call(self,natural_language_query):
        try:
            query_obj = self.translate_query(natural_language_query)
            if query_obj == "Not applicable.":
                return None
            # rank tunes
            tunes = []
            for obj in query_obj:
                tunes += self.select_tunes(obj)
            # output format
            out = ""
            for tune in tunes:
                out += tune["comment"] + "\n"
                out += tune["abc_form"] + "\n\n"
            return out
        except:
            return None

# %%
