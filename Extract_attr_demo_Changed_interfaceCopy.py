
import glob
import ntpath
import sys, os
from os import listdir
from os.path import isfile, join
import re
import ntpath
from tkinter import filedialog, ttk
from tkinter import *
from tkinter import messagebox
import scispacy
import spacy
import pandas as pd
import en_ner_bc5cdr_md
#import en_core_web_lg
from itertools import chain
import glob
import ntpath
import os
from os import listdir
from os.path import isfile, join
from spacy import displacy
from pathlib import Path
import os
from PIL import Image,  ImageTk
import pandas as pd
import numpy as np
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.util import constants
import sent2vec

from nltk import word_tokenize
#from nltk.corpus import stopwords
from string import punctuation
from gensim.utils import simple_preprocess
from scipy.spatial import distance
from nltk.stem import WordNetLemmatizer
#from stanfordnlp.server import CoreNLPClient

root = Tk()
root.geometry('800x400')
root.title('Text2EL')
folder_path = StringVar()
WIDTH, HEIGHT = 800, 500
img = ImageTk.PhotoImage(Image.open("logo.png").resize((WIDTH, HEIGHT), Image.ANTIALIAS))
lbl = Label(root, image=img)
lbl.img = img  # Keep a reference in case this code put is in a function.
lbl.place(relx=0.5, rely=0.5, anchor='center')

top_frame = Frame(root, width=607, height=40)

left_frame=Frame(root, highlightbackground="white", highlightthickness=1, bg='grey', width=75, height=150 )
center_frame = Frame(root, highlightbackground="white", highlightthickness=1,bg='grey', width=75, height=150 )
right_frame = Frame(root, highlightbackground="white", highlightthickness=1,bg='grey', width=75, height=150 )
corner_frame = Frame(root, highlightbackground="white", highlightthickness=1,bg='grey', width=75, height=150 )


left_frame.grid_rowconfigure(0, weight=1)
left_frame.grid_columnconfigure(0, weight=1)

top_frame.grid(row=0, columnspan=5,  padx=2, pady=2)
left_frame.grid(row=2, column=0, sticky="ns",  padx=3, pady=10)
center_frame.grid(row=2, column=1, sticky="ns" ,  padx=3, pady=10)
right_frame.grid(row=2, column=2, sticky="ns",  padx=3, pady=10 )
corner_frame.grid(row=2, column=3, sticky="ns",  padx=3, pady=10)

p_label1 = Label(left_frame, text='Extraction' , font =("Candara", 11,"bold") ,bg='grey', fg= 'black')
p_label2 = Label(center_frame, text='Verification',  font =("Candara", 11,"bold") , bg='grey', fg= 'black')
p_label3 = Label(right_frame, text='Expert Guidance', font =("Candara", 11,"bold"), fg= 'black' , bg='grey')
p_label4 = Label(corner_frame, text='Enrichment', font =("Candara", 11,"bold") , fg= 'black' , bg='grey')


# layout the widgets in the top frame
p_label1.grid(row=1, column=0,  sticky="ns")
p_label2.grid(row=1, column=0,  sticky="ns")
p_label3.grid(row=1, column=0, sticky="ns")
p_label4.grid(row=1, column=0, sticky="ns")



def preprocess(doc):

    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]



def tag_NER_lookup(filepath, lookup_p1, lookup_p2, out_csv):
    
    nlp = spacy.load("en_core_sci_lg")
    nlp_ = spacy.load("en_ner_bc5cdr_md")
    #NER = spacy.load("en_core_web_sm")

    stop = open("smart_stopword.txt", "r")
    k = stop.readlines()
    k = ([i.strip('\n') for i in k])
    stopwords = [p.lower() for p in k]

    lookup_path_1 = open(lookup_p1, 'r')
    word = lookup_path_1.read().lower().split("\n")

    lookup_path_2 = open(lookup_p2, 'r')
    word1 = lookup_path_1.read().lower().split("\n")
    
    
    df = pd.DataFrame()

    dicfilepath = filepath
    case_id =[]
    attribute=[]
    value=[]
    source=[]
    folders =[f for f in listdir(filepath) ]
    outputfile=open(out_csv, 'w')
    for dir in folders:

        
        for f in glob.glob(dicfilepath+'/'+dir + "/*.txt"):
            
            with open(f, "r+") as inputfile:
              
                file1 = inputfile.read().lower()
                text = file1
                        
                doc = nlp(text)
                doc1 = nlp_(text)
                #doc2 = NER(text)
                ent_bc = {}
                ner_bc = {}

 
                for x in doc1.ents:
                    ent_bc[x.text] = x.label_

                for key in ent_bc:
                    case_id.append(dir)
                    attribute.append(ent_bc[key])
                    value.append(key)
                    source.append(os.path.basename(f))

                for i in word:
                    if (i in file1):
                        case_id.append(dir)
                        attribute.append(i)
                        value.append("hospital activity")
                        source.append(os.path.basename(f))
                                        
                for j in word1:
                    if (j in file1):
                        case_id.append(dir)
                        attribute.append(j)
                        value.append("medical activity")
                        source.append(os.path.basename(f))
   
        myvars = {}
        
        #Get key-value pairs
        for f in glob.glob(dicfilepath + '/' + dir + "/*.txt"):
            
            with open(f, "r+") as myfile:
                
                for line in myfile:
                    v = "\n".join([x.strip() for x in line.splitlines() if ": " in x])
                    result = [x for x in re.split("\s{4,}", v) if x]

                    for r in result:
                        name, var = r.partition(": ")[::2]
                        myvars[name.strip()] = var
                        case_id.append(dir)
                        attribute.append(name)
                        value.append(var)
                        source.append(os.path.basename(f))
   
        myvars2 = {}

        for f in glob.glob(dicfilepath + '/' + dir + "/*Discharge summary.txt"):
            with open(f, "r+") as myfile:

                print(myfile)
                text = myfile.read()
                #text.pop()
                    
                per_results = text.partition("Pertinent Results:")[2].partition("Brief Hospital Course:")[0]
                lines = per_results.split("\n")
                l= list(filter(None, lines))
         
                keep_these =[]
                w=[item for item in l if '-' in ''.join(item) and ' ' in ''.join(item)]
                for line in w:
                    v = "\n".join([x.strip() for x in line.splitlines() if " " in x])

                    result = [x for x in re.split(" ", v) if x]
      
                    myvars = {}
                    for r in result:
                        if ('-' in r):
                            ex=r
                            name, var = r.partition("-")[::2]
                            myvars2[name.strip()] = var
                            case_id.append(dir)
                            attribute.append(name)
                            value.append(var)
                            source.append(os.path.basename(f))

    df['HADM_ID']=case_id
    df['Entity']=attribute
    df['Value']=value
    df['Note']=source
    df = df[df['Entity'].astype(str).str.len()>1]
    df['Entity']=df['Entity'].astype(str).str.replace('^[^a-zA-Z]*', '')
    
    df.dropna()
    
    df.Entity = df.Entity.str.replace(' +', ' ')
 
    df['Entity']=df['Entity'].str.lower()

    df['Entity'] = df['Entity'].str.strip()
    df_uni = df[df['Value'].notnull()]
    df_uni = df_uni[df_uni['Entity'].notnull()]
    df_uni=df_uni.dropna()
    

    df_uni=df_uni.drop_duplicates( keep='first')
    df_uni.to_csv(outputfile)
    outputfile.close()


    '''word = open('/mnt/c/Research/Text2EL/MIMIC_Exp/hospital_activity.txt', 'r')
    word = word.read().lower().split("\n")

    word1 = open('/mnt/c/Research/Text2EL/MIMIC_Exp/medical_activity.txt', 'r')
    word1 = word1.read().lower().split("\n")
    
    # print(word)
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()
    df6=pd.DataFrame()
    dicfilepath = filepath
    
    folders =[f for f in listdir(filepath) ]
    outputfile=open('/mnt/c/Research/Text2EL/MIMIC_Exp/Test/attributes_d1.csv', 'w')
    for dir in folders:
        #print(dir)
        #print(dicfilepath+'/'+dir)
        
        for f in glob.glob(dicfilepath+'/'+dir + "/*.txt"):
            
            with open(f, "r+") as inputfile:
              
                file1 = inputfile.read().lower()
                text = file1
                        
                doc = nlp(text)
                doc1 = nlp_(text)
                #doc2 = NER(text)
                ent_bc = {}
                ner_bc = {}

                svg = displacy.render(doc1, style="dep")
                file_name = '-'.join([w.text for w in doc if not w.is_punct]) + ".svg"
                #output_path = Path("/mnt/c/Research/MIMIC_Exp/plot.svg")
                #output_path.open("w", encoding="utf-8").write(svg)

                for x in doc1.ents:
                    ent_bc[x.text] = x.label_
                table = {"T":[],"HADM_ID":[], "Entity": [], "Value": [], "Note": []}
                for key in ent_bc:
                    table["T"].append('T1')
                    table["HADM_ID"].append(dir)
                    table["Value"].append(key)
                    table["Entity"].append(ent_bc[key])
                    table["Note"].append(os.path.basename(f))
                table1 = {"T":[],"HADM_ID":[], "Entity": [], "Value": [], "Note": []}
                for i in word:
                    if (i in file1):
                        table1["T"].append('T2')
                        table1["HADM_ID"].append(dir)
                        table1["Value"].append(i)
                        table1["Entity"].append("hospital activity")
                        table1["Note"].append(os.path.basename(f))

                table2 = {"T":[],"HADM_ID":[],"Entity": [], "Value": [], "Note": []}
                #Comment from here
                ''''''
                for w in doc2.ents:
                    ner_bc[w.text] = w.label_

                for key in ner_bc:
                    table2["T"].append('T3')
                    table2["HADM_ID"].append(dir)
                    table2["Entity"].append(key)
                    table2["Value"].append(ner_bc[key])
                    table2["Note"].append(os.path.basename(f))
                ''''''
                #to here
                table3 = {"T":[],"HADM_ID":[],"Entity": [], "Value": [], "Note": []}
                
                for j in word1:
                    if (j in file1):
                        table3["T"].append('T4')
                        table3["HADM_ID"].append(dir)
                        table3["Value"].append(j)
                        table3["Entity"].append("medical activity")
                        table3["Note"].append(os.path.basename(f))
                
                
                df1 = df1.append(pd.DataFrame(table), ignore_index=True)
                df2 = df2.append(pd.DataFrame(table1), ignore_index=True)
                df3 = df3.append(pd.DataFrame(table2), ignore_index=True)
                df4 = df4.append(pd.DataFrame(table3), ignore_index=True)
        
        table4 = {"T":[], "HADM_ID":[], "Entity": [], "Value": [], "Note":[]}
        myvars = {}
        
        #print(dicfilepath)
        #Get key-value pairs
        for f in glob.glob(dicfilepath + '/' + dir + "/*Discharge summary.txt"):
            
            with open(f, "r+") as myfile:
                
                for line in myfile:
                    v = "\n".join([x.strip() for x in line.splitlines() if ": " in x])
                    result = [x for x in re.split("\s{4,}", v) if x]

                    for r in result:
                        name, var = r.partition(": ")[::2]
                        myvars[name.strip()] = var
                        table4["T"].append('T5')
                        table4["HADM_ID"].append(dir)
                        table4["Entity"].append(name)
                        table4["Value"].append(var)
                        table4["Note"].append(os.path.basename(f))
            df5 = df5.append(pd.DataFrame(table4), ignore_index=True)

        table7 = {"HADM_ID":[], "Entity": [], "Value": [], "Note":[]}
        myvars2 = {}


    # print(dir)
    # print(dicfilepath+'/'+dir)
        for f in glob.glob(dicfilepath + '/' + dir + "/*Discharge summary.txt"):
            with open(f, "r+") as myfile:

                print(myfile)
                text = myfile.read()
                #text.pop()
                    
                per_results = text.partition("Pertinent Results:")[2].partition("Brief Hospital Course:")[0]
                lines = per_results.split("\n")
                l= list(filter(None, lines))
                table7 = {"T":[], "HADM_ID":[], "Entity": [], "Value": [], "Note":[]}
                keep_these =[]
                w=[item for item in l if '-' in ''.join(item) and ' ' in ''.join(item)]
                for line in w:
                    v = "\n".join([x.strip() for x in line.splitlines() if " " in x])

                    result = [x for x in re.split(" ", v) if x]
                    #print(result)
                    myvars = {}
                    for r in result:
                        if ('-' in r):
                            ex=r
                            name, var = r.partition("-")[::2]
                            myvars2[name.strip()] = var
                            table7["T"].append('T7')
                            table7["HADM_ID"].append(dir)
                            table7["Entity"].append(name)
                            table7["Value"].append(var)
                            table7["Note"].append(os.path.basename(f))

            df6 = df6.append(pd.DataFrame(table7), ignore_index=True)
            
    
    
          
    # print(list(doc.sents))
    # print(doc.ents)
    #df = pd.concat([df1, df2, df3, df4, df5, df6])
    df = pd.concat([ df1, df2, df3, df4, df5, df6])
    df = df[df['Entity'].astype(str).str.len()>1]
    df['Entity']=df['Entity'].astype(str).str.replace('^[^a-zA-Z]*', '')
    print(df)
    df.dropna()
    df_uni=df.drop_duplicates( keep='first')
    df.Entity = df.Entity.str.replace(' +', ' ')
    df_uni = df_uni[df_uni['Value'].notnull()]
    df_uni=df_uni.dropna()
    
    df_uni.to_csv(outputfile)
    outputfile.close()
    messagebox.showinfo("Case attributes extracted")'''
'''
def dep_parse(filepath):
    os.environ["CORENLP_HOME"]='C:\\Users\\kapugama\\Downloads\\stanford-corenlp-latest\\stanford-corenlp-4.4.0'

    
    sub_process_table = {"HADM_ID":[], "Entity": [],  "Note":[]}

    df = pd.DataFrame()

    folders = [f for f in listdir(filepath)]

    for dir in folders:

        print(dir)
        
        file_ext = ["/*Radiology.txt" , "/*Echo.txt"]
        for i in file_ext:
            for f in glob.glob(filepath + '/' + dir + str(i)):

                with open(f, "r+") as inputfile:

                    text = inputfile.read().split(".")
                    text.pop()

                    
                    for k in text:
                        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'depparse'], timeout=60000, memory='16G') as client:
                    # set up the client

                        # submit the request to the server
                            ann = client.annotate(str(k))

                        # get the first sentence
                            sentence = ann.sentence[0]


                        # get the dependency parse of the first sentence
                            dependency_parse = sentence.basicDependencies

                        #print(dir(sentence.token[0])) #to find all the attributes and methods of a Token object
                        #print(dir(dependency_parse)) #to find all the attributes and methods of a DependencyGraph object
                        #print(dir(dependency_parse.edge))

                        #get a dictionary associating each token/node with its label
                            token_dict = {}
                            for i in range(0, len(sentence.token)) :
                                token_dict[sentence.token[i].tokenEndIndex] = sentence.token[i].word

                            #get a list of the dependencies with the words they connect
                            list_dep=[]
                            for i in range(0, len(dependency_parse.edge)):

                                source_node = dependency_parse.edge[i].source
                                source_name = token_dict[source_node]

                                target_node = dependency_parse.edge[i].target
                                target_name = token_dict[target_node]

                                dep = dependency_parse.edge[i].dep

                                list_dep.append((dep,
                                    str(source_node)+'-'+source_name,
                                    str(target_node)+'-'+target_name))

                            #print(list_dep)
                            for k in list_dep:
                                
                                #if (k[0] =='appos') or  (k[0] == 'nmod') or (k[0]=='dep'):
                                
                                if(k[0]=='obj') or (k[0]=='obl')or (k[0]=='nsubj:pass') :
                                    a1=k[1].split("-",1)
                                    b1=k[2].split("-",1)
                                    a1=a1[1]
                                    b1=b1[1]
                                    c1=a1 + " "+ b1
                                    sub_process_table["HADM_ID"].append(dir)
                                    sub_process_table["Entity"].append(c1)
                                    sub_process_table["Note"].append(os.path.basename(f))
                            df = df.append(pd.DataFrame(table1), ignore_index=True)

                            #print(df)
                            df=df.drop_duplicates()
                            df.to_csv("/mnt/c/Research/MIMIC_Exp/output/dep_par_d.csv")
'''

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def apply_regex_rules(collection_filepath,output_file1):
    time = []
    activity = []
    caseID = []
    folders = [f for f in listdir(collection_filepath)]
    df = pd.DataFrame()

    for dir in folders:
        #print(dir)
        for f in glob.glob(collection_filepath + '/' + dir + "/*Discharge summary.txt"):
            print(f)
            with open(f, "r+") as inputfile:
                #print(inputfile)
                con = inputfile.read()
                x = con.split("\n")

                #MIMIC-III
                pattern = re.compile(r"(.*[[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]*[*]{2}\])")
                frontkeyonlypattern = re.compile(
                    r"(.*)[[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\]([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?")
                backkeyonlypattern = re.compile(
                    r"[[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\]([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?(.*)")
                datetimepattern = re.compile(
                    r"([[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\]([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?)")
                pattern1 = re.compile(r"([[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\].*)")

                #Demo
                '''pattern = re.compile(r"(.*[0-9]{4}[-][0-9]*[-][0-9]*)")
                frontkeyonlypattern = re.compile(
                    r"(.*)[0-9]{4}[-][0-9]*[-][0-9]{1,2}([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?")
                backkeyonlypattern = re.compile(
                    r"[0-9]{4}[-][0-9]*[-][0-9]{1,2}([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?(.*)")
                datetimepattern = re.compile(
                    r"([0-9]{4}[-][0-9]*[-][0-9]{1,2}([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?)")
                pattern1 = re.compile(r"([0-9]{4}[-][0-9]*[-][0-9]{1,2}.*)")'''

                for k in x:
                    m = pattern.findall(k)
                    m1 = pattern1.findall(k)
                    for i in m:
                        caseID.append(dir)
                        if not datetimepattern.match(i):
                            rgx = frontkeyonlypattern.search(i)
                            if rgx.group(1) is not None:
                                activity.append(rgx.group(1))
                            rgx1 = datetimepattern.search(i)
                            if rgx1.group(1) is not None:
                                time.append(rgx1.group(1))

                        else:
                            for i1 in m1:
                                rgx = backkeyonlypattern.search(i1)
                                if rgx.group(3) is not None:
                                    activity.append(rgx.group(3))
                                rgx1 = datetimepattern.search(i1)
                                if rgx1.group(1) is not None:
                                    time.append(rgx1.group(1))


    df["HADM_ID"] = caseID
    df['Activity'] = activity
    df['Timestamp'] = time
    df['Note']="Discharge summary"


    df.to_csv(output_file1)

#Extract events from event specific notes in MIMIC-III 
def extract_specific_notes(collection_filepath, output_file2):
    folders = [f for f in listdir(collection_filepath)]
    
    df = pd.DataFrame()

    table1 = {"HADM_ID": [], "Activity": [], "Timestamp": [], "Note": []}
    for dir in folders:
        
        for f in glob.glob(collection_filepath + '/' + dir + "/*Echo.txt"):
            with open(f, "r+") as inputfile:
                #print(inputfile)
                con = inputfile.read()

                pattern1 = re.compile(r"Date/Time: (.*)")
                pattern2 = re.compile(r"Test:(.*)")

                rem_characters= ["['","']", "at"]
                m1 = pattern1.findall(con)
                m2 = pattern2.findall(con)

                for i in rem_characters:
                    m1 = str(m1).replace(i, "")
                    m2 = str(m2).replace(i, "")

                table1["HADM_ID"].append(dir)
                table1["Activity"].append(m2)
                table1["Timestamp"].append(m1)
                table1["Note"].append(os.path.basename(f))

        for f in glob.glob(collection_filepath + '/' + dir + "/*Radiology.txt"):
            with open(f, "r+") as inputfile:

                con2 = inputfile.readlines()
                pattern = re.compile(r"([[0-9]{4}[-][0-9]*[-][0-9]{1,2}.*)")

                n1 = con2[:1]
                for k in n1:
                    m = pattern.findall(k)

                n2 = con2[1]

                pattern3 = re.compile(r"(.*)Clip \# \[\*\*Clip Number")
                pattern4 = re.compile(r"(.*);")
                mm = pattern3.findall(n2)
                mm1 = pattern4.findall(n2)

                rem_characters = ["['", "']", ]
                for i in rem_characters:
                    m = str(m).replace(i, "")
                    mm = str(mm).replace(i, "")
                    mm1 = str(mm1).replace(i, "")

                mm = str(mm).replace(") ", ")")
                mm = re.sub('\s+', " ", mm)
                mm = re.sub("[\[].*?[\]]", "", mm)
                mm1 = str(mm1).replace(") ", ")")
                mm1 = re.sub('\s+', " ", mm1)
                mm1 = re.sub("[\[].*?[\]]", "", mm1)


                table1["HADM_ID"].append(dir)
                table1["Activity"].append(mm + mm1)
                table1["Timestamp"].append(m)
                table1["Note"].append(os.path.basename(f))

        df = df.append(pd.DataFrame(table1), ignore_index=True)

    df = df[["HADM_ID", "Activity", "Timestamp", "Note"]]
    df = df.dropna()
    #df['Note'] = df['Note'].replace('^[^a-zA-Z]*', '')
    #df['Note'] = df['Note'].replace('.txt', '')

    df = df.drop_duplicates(keep='first')
    df = df.dropna(how = 'all')
    df = df[df.Activity != '']
    df.to_csv(output_file2)
    
    
#Refine the extracted events by excluding the activities which can’t be events     
def refine_events(all_events, exclude_list, stopword_list, output_csv):
    d = pd.read_csv(all_events)
    f1 = open(exclude_list , "r")
    w_list = f1.read().split('\n')

    suffix_list = ['in', 'on', 'of', 'On']

    d['Activity'] = d['Activity'].str.strip()

    d['Activity'] = d['Activity'].str.rstrip(',')
    d['Activity'] = d['Activity'].str.rstrip(',')
    d['Activity'] = d['Activity'].str.lstrip(',')

    '''for j in suffix_list:
        d['Activity'] = d['Activity'].removesuffix(j)
        remove_suffix(d['Activity'], j)

    '''
    d['Activity'] = d['Activity'].replace('^[^a-zA-Z]*', '')
    d['Activity'] = d['Activity'].replace('^and*', '')

    d = d[d['Activity'].str.len() > 0]
    d = d.dropna()

    out_df = d[~d['Activity'].str.lower().isin([x.lower() for x in w_list])]
    out_df['Activity']=out_df['Activity'].str.lower()
    out_df.loc[out_df['Activity'].str.contains('blood'), 'Activity'] = 'blood'
    #if out_df['Activity'].str.contains('BLOOD').any() :
     #   out_df['Activity']=out_df['Activity'].str.split().str[0].drop_duplicates()
    
    '''
    f2 = open(stopword_list, "r")
    among_list = f2.readlines()
    rem=str.maketrans('', '', '\n')
    b=[s.translate(rem) for s in among_list]
    #print(b)
    n_l = []
    for i in b:
        out_df['Activity'] = out_df['Activity'].str.replace(re.escape(i),'')
    '''

    stop_words = [':', 'she', 'the', 'was', '.']
    rem=str.maketrans('', '', '\n')
    b=[s.translate(rem) for s in stop_words]
    #print(b)
    n_l = []
    for i in b:
        out_df['Activity'] = out_df['Activity'].str.replace(re.escape(i),'')
    out_df.dropna(how='all')
    
        
        
    out_df['Activity'] = out_df['Activity'].str.replace('^[^a-zA-Z]*', '')
    out_df = out_df[out_df['Activity'].str.len() > 0]
    out_df=out_df[~out_df['Activity'].str.lower().isin([x.lower() for x in suffix_list])]
    out_df=out_df[['HADM_ID', 'Activity','Timestamp',	'Note']]
    out_df.to_csv(output_csv)

#merge all events, events extracted from the event-specific notes and case notes
def get_all_events(general_events_file, specific_event_file, all_events):
    d1=pd.read_csv(general_events_file)
    d2=pd.read_csv(specific_event_file)
    d1=d1[["HADM_ID",	"Activity",	"Timestamp",	"Note"]]
    d2 = d2[["HADM_ID", "Activity", "Timestamp", "Note"]]

    d3 = pd.concat([d1, d2])

    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("[**"), '')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("**]"), '')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("-"), '/')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("at"), ' ')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("**"), '')
    d3=d3.drop_duplicates()
    d3=d3.sort_values(['HADM_ID', 'Timestamp'], ascending=[True, True])
    d3=d3[["HADM_ID", "Activity", "Timestamp", "Note"]]
    
    d3.to_csv(all_events)

def extract_case_attributes(dependenc_csv, extracted_attr_csv,all_events_csv, all_case_att_csv):
    d3=pd.DataFrame()
    ''' extract dependency_csv
    d1=pd.read_csv(dependenc_csv)
    d1=d1[d1['Note'].str.contains("Discharge summary")]
    
    Entitiy_ls =[]
    value_ls=[]
    case_ls=[]
    for index, row in d1.iterrows():
        ea_list = str(row.Dependency).split('***')
        for i in ea_list:
            k = tuple(re.findall(r'[\w]+', i))
            if len(k) == 5:
                if (k[0] == 'nummod') :
                    Entitiy_ls.append(k[2])
                    value_ls.append(k[4])
                    case_ls.append(str(row.HADM_ID))
        d3['HADM_ID']= case_ls
        d3['Entity']=Entitiy_ls
        d3['Value']=value_ls
        '''
    d2=pd.read_csv(extracted_attr_csv)
    case_df=d2[d2['Note'].str.contains("Discharge summary.txt")]
    df_e=pd.read_csv(all_events_csv)
    events_l = df_e['Activity'].drop_duplicates().to_list()
    exclude_l=['birth:  [', 'birth:  [' ,'admit' , 'discharge', 'Discharge Date', 'Date of Birth']
    exclude_v = ['+2  Left:+2']
    
    #events_l.extend(exclude_v)
      

    new_df = pd.concat([case_df , d3])
    
    new_df = new_df[~new_df['Value'].isin(exclude_l)]
 
    new_df=new_df[~new_df['Entity'].isin(exclude_v)]
  
    new_df=new_df.drop_duplicates(keep='first')
    new_df=new_df[new_df['Value'].notna()]
    new_df=new_df[new_df['Entity'].notna()]
    new_df=new_df.dropna(how='all')
    new_df.dropna(how='all')
    new_df=new_df[['HADM_ID','Entity', 'Value', 'Note']]
    new_df.to_csv(all_case_att_csv)


def extract_event_attributes( dependency_csv, extracted_attributes_csv, specific_events_csv, all_events_csv, all_event_att_csv ):

    d3=pd.DataFrame()

    ''' extract from dependency_csv
    e_df1=pd.read_csv(dependency_csv)
    e_df1['Source_edi']=e_df1['Note'].str.replace('^[^a-zA-Z]*', '')
    e_df1['Source_edi']=e_df1['Source_edi'].str.replace('.txt', '')
    #print(e_df1)
    
    
    Entitiy_ls =[]
    value_ls=[]
    case_ls=[]
    for index, row in e_df1.iterrows():
        ea_list = str(row.Dependency).split('***')
        for i in ea_list:
            k = tuple(re.findall(r'[\w]+', i))
            if len(k) == 5:
                if (k[0] == 'nummod') :
                    Entitiy_ls.append(k[2])
                    value_ls.append(k[4])
                    case_ls.append(str(row.HADM_ID))
        d3['HADM_ID']= case_ls
        d3['Entity']=Entitiy_ls
        d3['Value']=value_ls
    e_df1 = e_df1[e_df1['Source_edi'].isin(event_notes)]
    '''
    event_notes=['Radiology' , 'Echo']
    e_df2=pd.read_csv(extracted_attributes_csv)
    e_df2['Source_edi']=e_df2['Note'].str.replace('^[^a-zA-Z]*', '')
    e_df2['Source_edi']=e_df2['Source_edi'].str.replace('.txt', '')
    
    #print(e_df2)
    e_df2 = e_df2[e_df2['Source_edi'].isin(event_notes)]
    #print(e_df2)
    df_e=pd.read_csv(all_events_csv)
    events_l = df_e['Activity'].drop_duplicates().to_list()
    e_df2 = e_df2[~e_df2['Value'].isin(events_l)]
    new_df = pd.concat([e_df2 , d3])
    #print(new_df)
    new_df.dropna()
    e_df3=pd.read_csv(specific_events_csv)
    new_df=new_df.merge(e_df3, on=["Note", 'HADM_ID'])
    new_df = new_df[['HADM_ID','Activity', 'Timestamp', 'Entity', 'Value', "Note"]]

    new_df=new_df.drop_duplicates(keep='first')

    new_df.to_csv(all_event_att_csv)
 
def extract_echo_event_attr(filepath, echo_events):

    folders = [f for f in listdir(filepath)]
    echo_event = pd.DataFrame()
    c1 = 0
    hadm_ID, activity, time, note, e_type, reason, diagnosis, indication, status, doppler, contrast, tecnical_quality, reason, admitting_diagnosis, reason_exam = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for dir in folders:
        
        for f in glob.glob(filepath + '/' + dir + "/*Echo.txt"):
            print("Echo note ", f)
            with open(f, "r+") as inputfile:
                
                c1 = c1 + 1
                con = inputfile.read()
                x = con.split("\n")

                pattern1 = re.compile(r"Date/Time:(.*)")
                pattern2 = re.compile(r"Test:(.*)")
                p3 = re.compile(r"Indication:(.*)")
                p4 = re.compile(r"Doppler:(.*)")
                p5 = re.compile(r"Contrast:(.*)")
                p6 = re.compile(r"Technical Quality:(.*)")
                p7 = re.compile(r"Status:(.*)")
                m1 = pattern1.findall(con)
                m2 = pattern2.findall(con)
                m3 = p3.findall(con)
                m4 = p4.findall(con)
                m5 = p5.findall(con)
                m6 = p6.findall(con)
                m7 = p7.findall(con)

                hadm_ID.append(dir)
                activity.append(m2)
                time.append(m1)
                note.append(os.path.basename(f))
                indication.append(m3)
                doppler.append(m4)
                contrast.append(m5)
                tecnical_quality.append(m6)
                status.append(m7)

    echo_event["HADM_ID"] = hadm_ID
    echo_event['Activity'] = activity
    echo_event['Timestamp'] = time
    echo_event['Type'] = "Echo"
    echo_event['Note'] = note
    echo_event["Indication"] = indication
    echo_event["Doppler"] = doppler
    echo_event["Contrast"] = contrast
    echo_event["Technical Quality"] = tecnical_quality
    echo_event["Status"] = status
    echo_event.to_csv(echo_events)

def extract_radiology_event_attr(filepath, radio_events):
    folders = [f for f in listdir(filepath)]
    radio_event=pd.DataFrame()
    c2=0
    hadm_ID, activity, time, note, e_type, reason, diagnosis, indication, status, doppler, contrast, tecnical_quality, reason, admitting_diagnosis, reason_exam = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for dir in folders:
        for f in glob.glob(filepath + '/' + dir + "/*Radiology.txt"):
            with open(f, "r+") as inputfile:
                con3 = inputfile.read()
                x = con3.split("\n")
                con2 = inputfile.readlines()

                c2 = c2 + 1
                n1 = x[:1]
                n1 = str(n1).replace("['", "")
                n1 = str(n1).replace("]", "")

                n2 = x[1]

                pattern3a = re.compile(r'(.*)Clip \# \[\*\*Clip Number')
                pattern3b = re.compile(r'(.*)\[\*\*Name')
                mm = pattern3a.findall(n2)
                mm2 = pattern3b.findall(n2)
                mm = str(mm).replace("['", "")
                mm = str(mm).replace("']", "")
                mm = str(mm).replace(") ", ")")
                mm = re.sub('\s+', " ", mm)
                mm = re.sub("[\[].*?[\]]", "", mm)
                v = str(mm) + str(mm2)
                # print(v)

                p3 = re.compile(r"Reason:(.*)")
                p4 = re.compile(r"Admitting Diagnosis:(.*)")
                m3 = p3.findall(con3)
                m4 = p4.findall(con3)
                hadm_ID.append(dir)
                activity.append(v)
                time.append(n1)
                note.append(os.path.basename(f))
                reason.append(m3)
                diagnosis.append(m4)

    radio_event["HADM_ID"] = hadm_ID
    radio_event['Activity'] = activity
    radio_event['Timestamp'] = time
    radio_event['Type'] = "Radiology"
    radio_event["Reason"]=reason
    radio_event["Admitting Diagnosis"]= diagnosis
    radio_event["Note"]=note
    radio_event.to_csv(radio_events)

def extract_event_attr_event_notes(echo_events, radiology_notes, all_event_attr_csv, final_event_attributes):
    echo_df = pd.read_csv(echo_events)
    radio_df = pd.read_csv(radiology_notes)
    f_event_df = pd.concat([echo_df, radio_df])
    f_event_df.dropna()
    f_event_df = f_event_df.drop('Unnamed: 0', axis=1)
    f_event_df = f_event_df.melt(id_vars=["HADM_ID", "Activity", "Timestamp" , "Note", "Type"])
    f_event_df = f_event_df.rename(columns={'variable': 'Entity', 'value': 'Value'})
    #f_event_df=f_event_df.set_index(['HADM_ID',"Activity", "Timestamp" , "Note", "Type" ]).stack()
    #f_event_df.to_csv("C:\Research\Text2EL\MIMIC_Exp\Test\\merged_events.csv")
    extracted_event_attr=pd.read_csv(all_event_attr_csv)
    #print(extracted_event_attr.info())
    extracted_event_attr = extracted_event_attr[['HADM_ID', 'Activity', 'Timestamp','Entity', 'Value']]
    event_merge = extracted_event_attr.groupby(['HADM_ID', 'Activity', 'Timestamp','Entity'], as_index=False).agg(list)
    #event_merge.to_csv("C:\Research\Text2EL\MIMIC_Exp\Test\\ex_events_1.csv")

    out_df_f = event_merge.groupby(['HADM_ID','Activity','Timestamp']).apply(lambda x: dict(zip(x['Entity'], x['Value']))).reset_index().rename(
        columns={"HADM_ID": "HADM_ID", 0: "Medical Info"})
    out_df_ff = out_df_f.melt(id_vars=["HADM_ID",'Activity','Timestamp'])
    out_df_ff = out_df_ff.rename(columns={'variable': 'Entity', 'value': 'Value'}) 
    ff_event_df=pd.concat([f_event_df, out_df_ff])
    ff_event_df=pd.concat([f_event_df])
    ff_event_df=ff_event_df[ff_event_df['Value'].notna()]
    ff_event_df['Value'] = ff_event_df['Value'].str.replace('[|]', '')
    # to group the event attributes and get all into a list.
    #ff_event_df=ff_event_df.groupby(['HADM_ID','Activity','Timestamp'])['Entity',  'Value'].agg(list)
    ff_event_df.dropna(inplace = True)
    ff_event_df.to_csv(final_event_attributes)

def merge_case_attributes(all_case_att_csv,  list_case_attributes):
    d1 = pd.read_csv(all_case_att_csv)
  
    final_filtered_attr = ["service", "sex", "date of birth"]
    d1["Value"]=d1["Value"].str.strip()
    d1['Entity'] = d1['Entity'].str.strip()
    d1["Value"]=d1["Value"].str.lstrip()
    d1['Entity'] = d1['Entity'].str.lstrip()
    d1 = d1[['HADM_ID', 'Entity', 'Value']]
    d1=d1.drop_duplicates(keep='first')
    event_merge = d1.groupby(['HADM_ID', 'Entity'], as_index=False).agg(list)

    out_df_f = event_merge[~event_merge['Entity'].str.lower().isin([x.lower() for x in final_filtered_attr])]
    include_v=['chemical', 'disease']
    out_df_f=out_df_f[out_df_f['Entity'].isin(include_v)]

    out_merg_f = event_merge[event_merge['Entity'].str.lower().isin([x.lower() for x in final_filtered_attr])]
    out_merg_f = out_merg_f[['HADM_ID', 'Entity', 'Value']]
    out_df_f = out_df_f.groupby('HADM_ID').apply(lambda x: dict(zip(x['Entity'], x['Value']))).reset_index().rename(
        columns={"HADM_ID": "HADM_ID", 0: "Medical Info"})

    out_df_ff = out_df_f.melt(id_vars="HADM_ID")
    out_df_ff = out_df_ff.rename(columns={'variable': 'Entity', 'value': 'Value'})
    
    out_df_f = pd.concat([out_df_ff, out_merg_f])
    out_df_f.to_csv(list_case_attributes)

def merge_event_attributes(final_event_attributes, list_event_attributes):
    d1 = pd.read_csv(final_event_attributes)

    final_filtered_attr = ["Indication", "Admitting Diagnosis", "Contrast", "Doppler", "Indication", "Reason", "Status",
                           "Technical Quality"]
    medical_info =['disease', 'chemical']
    d1["Value"] = d1["Value"].str.strip()
    d1['Entity'] = d1['Entity'].str.strip()
    d1["Value"] = d1["Value"].str.lstrip()
    d1['Entity'] = d1['Entity'].str.lstrip()
    d1 = d1[['HADM_ID', 'Activity', 'Timestamp','Entity', 'Value']]
    d1 = d1.drop_duplicates(keep='first')
    event_merge = d1.groupby(['HADM_ID', 'Activity', 'Timestamp','Entity'], as_index=False).agg(list)

    out_df_f = event_merge[~event_merge['Entity'].str.lower().isin([x.lower() for x in final_filtered_attr])]
    #out_df_f.to_csv("C:\Research\Text2EL\RunUbuntu\\out1.csv")
    out_merg_f = event_merge[event_merge['Entity'].str.lower().isin([x.lower() for x in final_filtered_attr])]
    out_merg_f = out_merg_f[['HADM_ID','Activity', 'Timestamp','Entity', 'Value']]
    out_df_f = out_df_f.groupby(['HADM_ID','Activity', 'Timestamp']).apply(lambda x: dict(zip(x['Entity'], x['Value']))).reset_index().rename(
        columns={"HADM_ID": "HADM_ID", 0: "Medical Info"})
    out_df_ff = out_df_f.melt(id_vars=["HADM_ID", 'Activity', 'Timestamp'])
    out_df_ff = out_df_ff.rename(columns={'variable': 'Entity', 'value': 'Value'})

    out_df_f = pd.concat([out_df_ff, out_merg_f])
    out_df_f.to_csv(list_event_attributes)


def delete_garbage_files():
    file_list=["./output1.csv", "./specific_events.csv", "./all_events.csv", "./all_attributes.csv" , "./echo_events.csv","./radio_events.csv" ,"./case_attr_final.csv", "./event_attr_final.csv" ]
    for i in file_list:
        if(os.path.exists(i) and os.path.isfile(i)):
            os.remove(i)
            print("file deleted")
            
def get_expert_inputs():
    
    #Expert input extraction and annotation is conducted in Google Colab
    #Annotate confirmed events and attributes
    '''
    The code is given below
    import spacy
    !python -m spacy download en_core_web_lg
    nlp = spacy.load("en_core_web_lg")
    pip install spacy[transformers]
    from google.colab import drive
    drive.mount('/content/drive/')
    import json
 
    # This is the output of the expert which extracted interms of annotations
    with open('/content/drive/My Drive/Colab Notebooks/j5_annotations.json', 'r') as f:
        data = json.load(f)
    
    #training_data = {'classes' : ['MEDICINE', "MEDICALCONDITION", "PATHOGEN"], 'annotations' : []}
    training_data = {'tags' : ['activityLabel', 'eventAttribute', 'caseAttribute', 'timestamp', 'caseID', 'caseAttributeValue', 'eventAttributeValue' ], 'annotations' : []}
    for example in data['examples']:
      temp_dict = {}
      temp_dict['text'] = example['content']
      temp_dict['entities'] = []
      for annotation in example['annotations']:
        start = annotation['start']
        end = annotation['end']
        label = annotation['tag'].upper()
        temp_dict['entities'].append((start, end, label))
      training_data['annotations'].append(temp_dict)

    
    from spacy.tokens import DocBin
    from tqdm import tqdm

    nlp = spacy.blank("en") # load a new spacy model
    doc_bin = DocBin() # create a DocBin object

    from spacy.util import filter_spans

    for training_example  in tqdm(training_data['annotations']): 
        text = training_example['text']
        labels = training_example['entities']
        doc = nlp.make_doc(text) 
        ents = []
        for start, end, label in labels:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents 
        doc_bin.add(doc)

    doc_bin.to_disk("training_data.spacy")

    !python -m spacy train config.cfg --output ./ --paths.train ./training_data.spacy --paths.dev ./training_data.spacy --gpu-id 0

    from os import listdir
    import glob
    filepath = '/content/drive/My Drive/Colab Notebooks/Text2EL_test - Sample Data/'
    folders = [f for f in listdir(filepath)]

    All_docs=[]
    for dir in folders:
  
      for f in glob.glob(filepath + '/' + dir + "/*.txt"):
    
        with open(f, "r+") as inputfile:
          con3 = inputfile.read()
      
          All_docs.append(con3)

    nlp_ner = spacy.load("model-best")

    for i in All_docs:
        doc = nlp_ner(i)
        colors = {"activityLabel": "#7DF6D9", "caseAttribute":"#FFFFFF"}
        options = {"colors": colors} 

        spacy.displacy.render(doc, style="ent", options= options, jupyter=True)'''
    
    #remove rejected events and attributes
    d1=pd.read_csv("./new_events.csv")
    print(d1.info())
    d2=pd.read_csv("./case_attr_final_list.csv")
    d3=pd.read_csv("./final_event_attr_final_list.csv")
    print(d2.info())
    print(d3.info())
    
    reject_event_list = open("./reject_event.txt" , "r")
    re_list = reject_event_list.read().split('\n')

    reject_case_attr_list = open("./reject_case_attr.txt" , "r")
    re_ca_list = reject_case_attr_list.read().split('\n')

    reject_event_attr_list = open("./reject_event_attr.txt" , "r")
    re_ea_list = reject_event_attr_list.read().split('\n')
    
    event_df = d1[~d1['Extracted_Activity'].str.lower().isin([x.lower() for x in re_list])]
    case_df = d2[~d2['Entity'].str.lower().isin([x.lower() for x in re_ca_list])]
    eventattr_df = d3[~d3['Entity'].str.lower().isin([x.lower() for x in re_ea_list])]
    event_df.to_csv("./new_events.csv")
    case_df.to_csv("./case_attr_final_list.csv")
    eventattr_df.to_csv("./final_event_attr_final_list.csv")
    

def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    apply_regex_rules(filename,"./output1.csv")
    extract_specific_notes(filename, "./specific_events.csv")
    get_all_events("./output1.csv", "./specific_events.csv",  "./all_events.csv")
    refine_events("./all_events.csv", "./exclude_list.txt", "./smart_stopword_among.txt", "./final_event_list.csv")
    tag_NER_lookup(filename , "./hospital_activity.txt" , "./medical_activity.txt", './all_attributes.csv')
    extract_case_attributes("./all_dependency.csv", "./all_attributes.csv", './all_events.csv', "./case_attr_final.csv")
    extract_echo_event_attr(filename, "./echo_events.csv")
    extract_radiology_event_attr(filename, "./radio_events.csv")
    extract_event_attributes("./all_dependency.csv", "./all_attributes.csv", './specific_events.csv', './all_events.csv', "./event_attr_final.csv")
    extract_event_attr_event_notes("./echo_events.csv", "./radio_events.csv","./event_attr_final.csv", "./final_event_attr_list.csv" )
    merge_case_attributes("./case_attr_final.csv", "./case_attr_final_list.csv")
    merge_event_attributes("./event_attr_final.csv", "./final_event_attr_final_list.csv")
    delete_garbage_files()
    print("Phase 1 completed")
    messagebox.showinfo(message="Extraction is completed")
    

def extract_case():
    # Allow user to select a directory and store it in global var
    # called folder_path
    frame2=LabelFrame(root, text="Load Case Attributes")
    frame2.place(height=380, width=600)
    tv1=ttk.Treeview(frame2)
    tv1.place(relheight=1, relwidth=1)
    #jsonfile ='/mnt/c/Research/MIMIC_Exp/output/case_attributes_d.json'
    csvfile= './case_attr_final_list.csv'
    
    def back():
        frame2.destroy()
        fileframe1.destroy()
    def Load_csv(csvfile):
        
        d= pd.read_csv('./case_attr_final_list.csv')
        #d = d[d['Note'].str.contains("Discharge summary.txt")]
        #ed= pd.read_csv('./case_attr_final_list.csv')
        #events_l = ed['concept:name'].drop_duplicates().to_list()
        
        exclude_v = ['+2  Left:+2']            
        #events_l.extend(exclude_l)
        #d1 = d[~d['Entity'].isin(events_l)]
        
        d['Entity']=d['Entity'].str.replace('^[^a-zA-Z]*', '')
        
        d1 = d[d['Entity'].str.len()>1]
        d1 = d1[['HADM_ID', 'Entity', 'Value']]
        df1=d1[['HADM_ID', 'Entity', 'Value']]
        df=d1.groupby(['HADM_ID', 'Entity', 'Value']).size()
        de = (d1.groupby(['HADM_ID']).apply(lambda x: x[['Entity','Value']].to_dict('records')).reset_index().rename(columns={0:'Case Attributes'}).to_json(orient='records'))
        
        df.to_csv(csvfile)
        jsonfile='./case_attributes_d.json'
        f2=open('./case_attributes_d.json', "w")
        f2.write(de)
        f2.close()
        
        df1= df1.sort_values(['HADM_ID'], ascending=[True])
        df1['Value']=df1['Value'].str.replace('[', '')
        df1['Value']=df1['Value'].str.replace(']', '')
        df1=df1.rename(columns={'HADM_ID':'Case_ID'})
        tv1["column"] = list(df1.columns)
        tv1.column('Case_ID' ,width=5, stretch=False)
        tv1.column('Entity' ,width=5, stretch=False)
        tv1.column('Value' ,width=45, stretch=True)
        tv1["show"]="headings"
        for column in tv1["column"]:
            tv1.heading(column, text=column)
        df_rows = df1.to_numpy().tolist()
        for row in df_rows:
            tv1.insert("", "end", values=row)
        return None
    '''def clear_frame():
        for widgets in frame2.winfo_children():
            widgets.destroy()
       
    def load_json(jsonfile):
        
        f= open(jsonfile, 'r')
        s=f.read()
        #print("file Content",s)
        my_text= Text(frame2, width=100, height=80, font=("Calibri" , 12))
        my_text.pack(pady=20)
        my_text.insert(1.0, s)
        f.close()
    '''  
    fileframe1=LabelFrame(root, text="*******")
    fileframe1.place(height=150, width=600,rely=0.75, relx=0)
    button5=Button(fileframe1, text="Load Case attributes", command=lambda:Load_csv(csvfile))
    button5.place(rely=0.5, relx=0.2)
    #button6=Button(fileframe1, text="Load Json", command=lambda:load_json(jsonfile))
    #button6.place(rely=0.65, relx=0.45)
    #button7=Button(fileframe1, text="Clear", command=lambda:clear_frame())
    #button7.place(rely=0.65, relx=0.65)
    button8=Button(fileframe1, text="Back", command=lambda:back())
    button8.place(rely=0.5, relx=0.6)
    treescolly=Scrollbar(frame2, orient="vertical", command=tv1.yview)
    treescollx = Scrollbar(frame2, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescollx.set ,yscrollcommand=treescolly.set )
    treescollx.pack(side="bottom", fill="x")
    treescolly.pack(side="right", fill="y")   

    
def process_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    #df=pd.read_csv("./final_event_list.csv")
    infile="./final_event_list.csv"
    frame1=LabelFrame(root, text="Load Events")
    frame1.place(height=380, width=600)
    tv1=ttk.Treeview(frame1)
    tv1.place(relheight=1, relwidth=1)
    def Load_csv(infile):
        df = pd.read_csv(infile)
        df=df[['HADM_ID','Activity', 'Timestamp']]
        df = df.sort_values(['HADM_ID', 'Timestamp'], ascending=[True, True])
        df=df.rename(columns={'HADM_ID':'Case_ID'})
        
        tv1["column"] = list(df.columns)
        tv1["show"]="headings"
        for column in tv1["column"]:
            tv1.heading(column, text=column)
        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            tv1.insert("", "end", values=row)
        return None
    '''
    def clear_frame():
       for widgets in frame1.winfo_children():
          widgets.destroy()
 
    def extract_events(infile):
    # Allow user to select a directory and store it in global var
    # called folder_path
        #clear_frame()
        log_csv=pd.read_csv(infile)
        log_csv=log_csv[["HADM_ID", "Activity", "Timestamp" , "Note"]]
        log_csv.rename(columns = {'HADM_ID' : 'case:concept:name', 'Activity':"concept:name" , 'Timestamp':"time:timestamp"}, inplace=True)
        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case'}
        log = log_converter.apply(log_csv, parameters={constants.PARAMETER_CONSTANT_CASEID_KEY: "case:concept:name",
                                                       constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "concept:name",
                                                       constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"})
        out=open('./log_d.xes', 'w')
        out.write(str(log))
        out.close()
        
        f= open('./log_d.xes', 'r')
        s=f.read()
        #print("file Content",s)
        my_text= Text(frame1, width=100, height=80, font=("Calibri" , 12))
        my_text.pack(pady=20)
        my_text.insert(1.0, s)
        f.close()
        #t is a Text widget
    '''  
    def back():
        frame1.destroy()
        fileframe.destroy()
    
    fileframe=LabelFrame(root, text="Show Events")
    fileframe.place(height=150, width=600,rely=0.75, relx=0)
    button5 = Button(fileframe, text="Load events", command=lambda:Load_csv(infile))
    button5.place(rely=0.4, relx=0.2)
    #button6 = Button(fileframe, text="Clear All", command=lambda:clear_frame())
    #button6.place(rely=0.4, relx=0.3)
    #button7 = Button(fileframe, text="Load xes", command=lambda:extract_events(infile))
    #button7.place(rely=0.4, relx=0.5)
    button8 = Button(fileframe, text="Back", command=lambda:back())
    button8.place(rely=0.4, relx=0.6)
    treescolly=Scrollbar(frame1, orient="vertical", command=tv1.yview)
    treescollx = Scrollbar(frame1, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescollx.set ,yscrollcommand=treescolly.set )
    treescollx.pack(side="bottom", fill="x")
    treescolly.pack(side="right", fill="y")   

    
def Generate_lookup():
    
    filename = filedialog.askopenfile()
    
    #showinfo( title='Selected File', message=filename)
    
def extract_event_attr():
    frame3=LabelFrame(root, text="Load Event Attributes")
    frame3.place(height=380, width=600)
    tv1=ttk.Treeview(frame3)
    tv1.place(relheight=1, relwidth=1)
    
    def back():
        frame3.destroy()
        fileframe2.destroy()
    def Load_csv():
        d= pd.read_csv('./final_event_attr_list.csv')
        d['Timestamp']=d['Timestamp'].str.replace('[', '')
        d['Timestamp']=d['Timestamp'].str.replace(']', '')
        d['Activity']=d['Activity'].str.replace('[', '')
        d['Activity']=d['Activity'].str.replace(']', '')
        d['Activity']=d['Activity'].str.replace("'", '')
        d['Timestamp']=d['Timestamp'].str.replace("'", '')
        d=d.dropna()
        d=d.drop_duplicates()
        
        #d['HADM_ID_x'] =d['HADM_ID_x'].astype(str)
        d=d.rename(columns={'HADM_ID_x':'HADM_ID'})
        
        d=d[['HADM_ID', 'Activity','Timestamp',  'Entity', 'Value']]
        de = (d.groupby(['HADM_ID']).apply(lambda x: x[['Activity', 'Timestamp','Entity','Value']].to_dict('records')).reset_index().rename(columns={0:'Event Attributes'}).to_json(orient='records'))
        d=d.rename(columns={'HADM_ID':'Case_ID'})
        f2=open("./event_attributes_d.json", "w")
        f2.write(de)
        f2.close()
        tv1["column"] = list(d.columns)
        tv1["show"]="headings"
        for column in tv1["column"]:
            tv1.heading(column, text=column)
        df_rows = d.to_numpy().tolist()
        for row in df_rows:
            tv1.insert("", "end", values=row)
        return None
    '''def clear_frame():
        for widgets in frame3.winfo_children():
            widgets.destroy()
       
    def load_json():
        

        f= open('./event_attributes_d.json', 'r')
        s=f.read()
        print("file Content",s)
        my_text= Text(frame3, width=100, height=80, font=("Calibri" , 12))
        my_text.pack(pady=20)
        my_text.insert(1.0, s)
        f.close()'''
        
    fileframe2=LabelFrame(root, text="*******")
    fileframe2.place(height=150, width=600,rely=0.75, relx=0)
    button5=Button(fileframe2, text="Load event attributes", command=lambda:Load_csv())
    button5.place(rely=0.5, relx=0.2)
    #button6=Button(fileframe2, text="Load Json", command=lambda:load_json())
    #button6.place(rely=0.65, relx=0.45)
    #button7=Button(fileframe2, text="Clear", command=lambda:clear_frame())
    #button7.place(rely=0.65, relx=0.65)
    button8=Button(fileframe2, text="Back", command=lambda:back())
    button8.place(rely=0.5, relx=0.6)
    treescolly=Scrollbar(frame3, orient="vertical", command=tv1.yview)
    treescollx = Scrollbar(frame3, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescollx.set ,yscrollcommand=treescolly.set )
    treescollx.pack(side="bottom", fill="x")
    treescolly.pack(side="right", fill="y")   


def time_match(event_log, extracted_events, time_matched_events):
    d1= pd.read_csv(extracted_events)

    d2=pd.read_csv(event_log )

    d1['Timestamp'] = d1['Timestamp'].str.replace(re.escape("[' [**"),'')
    d1['Timestamp'] = d1['Timestamp'].str.replace(re.escape("**]"),'')
    d1['Timestamp'] = d1['Timestamp'].str.replace(re.escape("]"),'')
    d1['Timestamp'] = d1['Timestamp'].str.replace(re.escape("-"),'/')
    d1['Timestamp'] = d1['Timestamp'].str.replace(re.escape("at"),' ')
    d1['Timestamp'] = d1['Timestamp'].str.replace(re.escape("**"),'')
    d1['Timestamp'] = d1['Timestamp'].str.replace(re.escape("'"),'')
    d2['Timestamp'] = d2['Timestamp'].str.replace(re.escape("-"),'/')
    d2['Timestamp'] = d2['Timestamp'].str.replace(re.escape("/0"),'/')

    d1[['Date','Time']] = d1["Timestamp"].str.split(" ", expand=True, n=1)
    d2[['Date','Time']] = d2["Timestamp"].str.split(" ", expand=True, n=1)

    d3o2=d1.merge(d2, on =['HADM_ID'])

    d3o2['Date_x'] = pd.to_datetime(d3o2['Date_x'])

    d3o2['Date_x'] = pd.to_datetime(d3o2['Date_x'])
    d3o2['Date_y'] = pd.to_datetime(d3o2['Date_y'], dayfirst=True)
    d3o2['Date Difference'] = (d3o2['Date_x'] - d3o2['Date_y']).abs()

    pat = re.compile(r"([AP])")

    li = []
    for index, row in d3o2.iterrows():
        v = pat.sub(" \\1", str(row.Time_x))
        li.append(v)

    d3o2['Time_x'] = li
    d3o2['Time_x'] = d3o2['Time_x'].replace("nan", "")
    d3o2['Time_x'] = d3o2['Time_x'].replace("AM", "")

    d3o21=d3o2.loc[d3o2['Date Difference']=='0 days']

    d3o21.to_csv(time_matched_events)

def semantic_match(time_matched_events, semantic_similairty):
    lemmatizer = WordNetLemmatizer()

    model_path = "/mnt/c/Research/PAH/PAH_experiments/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
    #model_path = "/mnt/c/Research/PAH_experiments/RunUbuntu/wiki_bigrams.bin"
    #model_path = "/mnt/c/Research/PAH_experiments/RunUbuntu/twitter_bigrams.bin"
    model = sent2vec.Sent2vecModel()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(e)
    print('model successfully loaded')
    
    stop = open("smart_stopword.txt", "r")
    k = stop.readlines()
    k = ([i.strip('\n') for i in k])
    stop_words=[p.lower() for p in k]

    def preprocess_sentence(text):
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()
        tokens= [token for token in simple_preprocess(text, min_len=0, max_len=float("inf")) if token not in stop_words]
        #tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

        return ' '.join(tokens)

    d0=pd.read_csv(time_matched_events)

    event_sim_li, source_sim_li =[], []
    d0 = d0.rename(columns={'Activity_x': 'Extracted_Activity', 'Activity_y': 'Eventlog_Activity'})
    for index, row in d0.iterrows():
        w_l1=preprocess_sentence(str(row.Extracted_Activity))
        w_l2 = preprocess_sentence(str(row.Eventlog_Activity))
        w_21=preprocess_sentence(str(row.Note))
        w_22 = preprocess_sentence(str(row.Source))
        w_l1.lower()
        w_l2.lower()
        w_21.lower()
        w_22.lower()
        vec_1 = model.embed_sentence(w_l1)
        vec_2=model.embed_sentence(w_l2)
        vec_3 = model.embed_sentence(w_21)
        vec_4=model.embed_sentence(w_22)
        cosine_sim1 = 1 - distance.cosine(vec_1,vec_2)
        event_sim_li.append(cosine_sim1)
        cosine_sim2 = 1 - distance.cosine(vec_3,vec_4)
        source_sim_li.append(cosine_sim2)

    d0['Sim_Bio']=event_sim_li
    d0['Source_similarity']=source_sim_li
    #d0=d0[(d0["Similarity"]<1)]
    #d0_uni=d0.drop_duplicates( keep='first')
    d0.to_csv(semantic_similairty)


def calculate_threshold(similarity_csv, recorded_csv, unrecorded_csv):
    d1 = pd.read_csv(similarity_csv)
    avg = d1['Sim_Bio'].mean()
    std_dev= d1['Sim_Bio'].std()

    threshod = avg + std_dev
    d2 = d1[(d1['Sim_Bio'] >= threshod)]
   
    #Further filter lower time difference
    '''
    d2['Timestamp_x'] = pd.to_datetime(d2['Timestamp_x'], dayfirst=True)
    d2['Timestamp_y'] = pd.to_datetime(d2['Timestamp_y'], dayfirst=True)
    d2['Time_Diff'] = (d2['Timestamp_x'] - d2['Timestamp_y']).abs()'''

    d2=d2.loc[d2.groupby(['Extracted_Activity' ,'HADM_ID','Timestamp_x'])['Sim_Bio'].idxmax()]

    d2.to_csv(recorded_csv)
    #time_matched_recorded = d2[
    sem_sim=pd.read_csv(similarity_csv)
    sem_sim=sem_sim[['HADM_ID',	'Extracted_Activity',	'Timestamp_x',	'Note']]
    rec_=d2[['HADM_ID',	'Extracted_Activity',	'Timestamp_x',	'Note']]
    
    sem_sim.Timestamp_x.astype('datetime64[ns]')
    rec_.Timestamp_x.astype('datetime64[ns]')
   
    unrecorded = sem_sim.merge(rec_, indicator=True, how='outer')
    unrecorded = unrecorded[unrecorded['_merge'] == 'left_only']
    unrecorded = unrecorded[['HADM_ID', 'Extracted_Activity', 'Timestamp_x', 'Note']]
    unrecorded = unrecorded.drop_duplicates(keep='first')
    unrecorded.to_csv(unrecorded_csv)
    
    
    
def compare():
    #time_match('/mnt/c/Research/Text2EL/MIMIC_Exp/Event Log/video_event_log.csv','./final_event_list.csv', './time_matched_events.csv')
    time_match('/mnt/c/Research/Text2EL/MIMIC_Exp/Event Log/event_log_no_duplicates_wo wierd tables.csv','./final_event_list.csv', './time_matched_events.csv')
    semantic_match('./time_matched_events.csv', './semantic_similairty.csv')
    prepare_attributefor_comparisson()
    messagebox.showinfo(message="Validation is completed")
    frame5=LabelFrame(root, text="Validate Event Logs")
    frame5.place(height=370, width=600)
    
    tv1=ttk.Treeview(frame5)
    tv1.place(relheight=1, relwidth=1)

    def validation():
        calculate_threshold('./semantic_similairty.csv', './recorded_events.csv', './unrecorded_events.csv')
        tv1=ttk.Treeview(frame5)
        tv1.place(relheight=1, relwidth=1)
            
        d0=pd.read_csv('./recorded_events.csv')
        #d0=d0.dropna()
        d0=d0.drop_duplicates()
        d0= d0[['HADM_ID'   ,'Extracted_Activity',	'Eventlog_Activity', 'Timestamp_y']]
        d0=d0.rename(columns={'HADM_ID':'Case_ID', 'Timestamp_y' : 'Timestamp' })


        tv1["column"] = list(d0.columns)
        tv1["show"]="headings"
        for column in tv1["column"]:
            tv1.heading(column, text=column)
        df_rows = d0.to_numpy().tolist()
        for row in df_rows:
            tv1.insert("", "end", values=row)
        
    def back_1():
        frame5.destroy()
        button18.destroy()
        fileframe12.destroy()
        
    def clear_frame():
        for widgets in frame5.winfo_children():
            widgets.destroy()

    fileframe12=LabelFrame(root, text="*******Validate eventlog**********")
    fileframe12.place(height=150, width=600,rely=0.75, relx=0)
        
    button18=Button(fileframe12, text="Back", command=lambda:back_1())
    button18.place(rely=0.3, relx=0.5)
    button19=Button(fileframe12, text="Show Matched events", command=lambda:validation())
    button19.place(rely=0.3, relx=0.1)
    treescolly=Scrollbar(frame5, orient="vertical", command=tv1.yview)
    treescollx = Scrollbar(frame5, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescollx.set ,yscrollcommand=treescolly.set )
       
    treescollx.pack(side="bottom", fill="x")
    treescolly.pack(side="right", fill="y")
    
         
def show_new():
    frame5=LabelFrame(root, text="New Events")
    frame5.place(height=300, width=600)
    
    tv1=ttk.Treeview(frame5)
    tv1.place(relheight=1, relwidth=1)

    # take simiilarit -recorded unique
    d0=pd.read_csv('./unrecorded_events.csv')
    d1=pd.read_csv('./recorded_events.csv')
    d0=d0[['HADM_ID','Extracted_Activity',	'Timestamp_x',	'Note']]
    d0['Timestamp_x'] = pd.to_datetime(d0['Timestamp_x'], dayfirst=True)
    d0=d0.drop_duplicates()
    #print(d0)
    d1=d1[['HADM_ID','Extracted_Activity','Timestamp_x','Note']]
    d1=d1.drop_duplicates()
    d1['Timestamp_x'] = pd.to_datetime(d1['Timestamp_x'], dayfirst=True)
    #print(d1)
    set_diff_df = pd.concat([d0, d1, d1]).drop_duplicates(keep=False)
    set_diff_df=set_diff_df.rename(columns={'Timestamp_x':'Timestamp', 'HADM_ID':'Case_ID'})
    set_diff_df.to_csv("./new_events.csv")
            
    #d0=d0.dropna()
    set_diff_df=set_diff_df.drop_duplicates()

    
    tv1["column"] = list(set_diff_df.columns)
    tv1["show"]="headings"
    for column in tv1["column"]:
        tv1.heading(column, text=column)
    df_rows = set_diff_df.to_numpy().tolist()
    for row in df_rows:
        tv1.insert("", "end", values=row)
    #return None
        
    def back_1():
        frame5.destroy()
        button18.destroy()
        fileframe12.destroy()
        
             
    fileframe12=LabelFrame(root, text="******New Events*********")
    fileframe12.place(height=140, width=600,rely=0.75, relx=0)
        
    button18=Button(fileframe12, text="Back", command=lambda:back_1())
    button18.place(rely=0.4, relx=0.9)
    treescolly=Scrollbar(frame5, orient="vertical", command=tv1.yview)
    treescollx = Scrollbar(frame5, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescollx.set ,yscrollcommand=treescolly.set )
    treescollx.pack(side="bottom", fill="x")
    treescolly.pack(side="right", fill="y")

def update_inconsitent_time():
    da0=pd.read_csv('./recorded_events.csv')

    #original event log
    d1= pd.read_csv('/mnt/c/Research/Text2EL/MIMIC_Exp/Event Log/video_event_log.csv')
    d2=pd.read_csv('./new_events.csv')
    d2=d2[['Case_ID','Extracted_Activity', 'Timestamp',	'Note']]
    d2 = d2.rename(columns={'Note': 'Source', 'Extracted_Activity' : 'Activity'})
    df0=da0[['HADM_ID',	'Eventlog_Activity',	'Timestamp_y',	'Source']]
    df0 = df0.rename(columns={'Eventlog_Activity': 'Activity', 'Timestamp_y': 'Timestamp'})
    d0 = da0[['HADM_ID', 'Extracted_Activity', 'Timestamp_x', 'Note', 'Eventlog_Activity', 'Timestamp_y', 'Source']]
    d0['Timestamp_y'] = np.where(d0['Note']!='Discharge summary', d0['Timestamp_x'], d0['Timestamp_y'])
    updated_time=d0[['HADM_ID',	'Eventlog_Activity'	,'Timestamp_y',	'Source']]
    updated_time=updated_time.rename(columns={'HADM_ID': 'Case_ID', 'Eventlog_Activity':'Activity', 'Timestamp_y':'Timestamp' })
    updated_time=updated_time.drop_duplicates(keep='first')
    df0=df0.drop_duplicates(keep='first')

    d1['Timestamp'] = pd.to_datetime(d1['Timestamp'], dayfirst=True)
    df0['Timestamp'] = pd.to_datetime(df0['Timestamp'], dayfirst=True)
    original_min_updated_time=pd.concat([df0,d1]).drop_duplicates(keep=False)

    original_min_updated_time = original_min_updated_time[['HADM_ID', 'Activity', 'Timestamp', 'Source']]
    original_min_updated_time = original_min_updated_time.rename(columns={'HADM_ID': 'Case_ID'})
    updated_time = updated_time.rename(columns={'HADM_ID': 'Case_ID', 'Eventlog_Activity': 'Activity', 'Timestamp_y': 'Timestamp'})


    eventlog_with_updated = pd.concat([updated_time,original_min_updated_time, d2])
    eventlog_with_updated =eventlog_with_updated [['Case_ID', 'Activity', 'Timestamp', 'Source']]
    eventlog_with_updated=eventlog_with_updated.drop_duplicates(keep='first')
    eventlog_with_updated= eventlog_with_updated.sort_values(['Case_ID', 'Timestamp'],
              ascending = [True, True])
    eventlog_with_updated.to_csv('./enriched_log2.csv')
    return eventlog_with_updated
     
def enrich_log():
    frame5=LabelFrame(root, text="Enriched Log")
    frame5.place(height=300, width=600)
    
    tv1=ttk.Treeview(frame5)
    tv1.place(relheight=1, relwidth=1)

    
    d0=pd.read_csv('/mnt/c/Research/Text2EL/MIMIC_Exp/Event Log/video_event_log.csv')
  
    d0=d0.rename(columns={ 'HADM_ID':'Case_ID'})
    d0=d0[['Case_ID', 'Activity', 'Timestamp', 'Source']]
    d0=d0.rename(columns={'Source':'Note'})
    
    d1=pd.read_csv('./new_events.csv')
    d1=d1[['Case_ID','Extracted_Activity',	'Timestamp', 'Note']]
    d1=d1.rename(columns={'Extracted_Activity':'Activity'})
    
    d2=pd.concat([d0,d1])
    d2.to_csv('./enriched_log.csv')
    d2=pd.read_csv('./enriched_log.csv')
    d2=d2[['Case_ID','Activity',	'Timestamp']]
    #d2=d2.sort_values(['Case_ID', 'Timestamp'])
    d2=d2.drop_duplicates()

    '''d0=d0[['HADM_ID','Extracted_Activity',	'Timestamp_x',	'Note']]
    d0['Timestamp_x'] = pd.to_datetime(d0['Timestamp_x'], dayfirst=True)
    d0=d0.drop_duplicates()
    #print(d0)
    d1=d1[['HADM_ID','Extracted_Activity',	'Timestamp_x',	'Note']]
    d1=d1.drop_duplicates()
    d1['Timestamp_x'] = pd.to_datetime(d1['Timestamp_x'], dayfirst=True)
    #print(d1)
    set_diff_df = pd.concat([d0, d1, d1]).drop_duplicates(keep=False)
    set_diff_df=set_diff_df.rename(columns={'Timestamp_x':'Timestamp', 'HADM_ID':'Case_ID'})
    set_diff_df.to_csv("./new_events.csv")
            
    #d0=d0.dropna()
    set_diff_df=set_diff_df.drop_duplicates()'''

    
    tv1["column"] = list(d2.columns)
    tv1["show"]="headings"
    for column in tv1["column"]:
        tv1.heading(column, text=column)
    df_rows = d2.to_numpy().tolist()
    for row in df_rows:
        tv1.insert("", "end", values=row)
    #return None

    def update_timestamp():
        eventlog_with_updated=update_inconsitent_time()
        eventlog_with_updated=eventlog_with_updated[['Case_ID', 'Activity', 'Timestamp']]
        tv1["column"] = list(eventlog_with_updated.columns)
        tv1["show"]="headings"
        for column in tv1["column"]:
            tv1.heading(column, text=column)
        df_rows = eventlog_with_updated.to_numpy().tolist()
        for row in df_rows:
            tv1.insert("", "end", values=row)
        
    def back_1():
        frame5.destroy()
        button18.destroy()
        fileframe12.destroy()
        
             
    fileframe12=LabelFrame(root, text="******Enriched event log*********")
    fileframe12.place(height=140, width=600,rely=0.75, relx=0)
    button17=Button(fileframe12, text="Update inconsistent timestamps", command=lambda:update_timestamp())
    button17.place(rely=0.3, relx=0.3)    
    button18=Button(fileframe12, text="Back", command=lambda:back_1())
    button18.place(rely=0.3, relx=0.7)
    treescolly=Scrollbar(frame5, orient="vertical", command=tv1.yview)
    treescollx = Scrollbar(frame5, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescollx.set ,yscrollcommand=treescolly.set )
    treescollx.pack(side="bottom", fill="x")
    treescolly.pack(side="right", fill="y")      

def prepare_attributefor_comparisson():

    #compare with original MIMIC
    '''
    d0=pd.read_csv('/mnt/c/Research/TEXT2EL/MIMIC_Exp/Case_list_fil.csv')

    case_list= d0['HADM_ID'].drop_duplicates().to_list()
    p1 = pd.read_csv("/mnt/c/Research/mimic-iii-clinical-database-1.4/admissions.csv")
    p1.dropna(subset=['HADM_ID'], inplace=True)
    d1=p1[p1['HADM_ID'].isin (case_list)]

    p2 = pd.read_csv("/mnt/c//Research/mimic-iii-clinical-database-1.4/patients.csv")

    p1 =d1[["SUBJECT_ID","HADM_ID"]]
    p2=p2[["SUBJECT_ID","GENDER",	"DOB"]]
    p3=d1.merge(p2, on =["SUBJECT_ID"])
    p3=p3[["HADM_ID", "GENDER", "DOB"]]
    '''
    # For Demo
    p3=pd.read_csv('./unstacked_filtered.csv')
    p4 = pd.read_csv("./case_attr_final_list.csv")
    p4=p4[['HADM_ID',	'Entity',	'Value']]
    p4['Value'] = p4['Value'].astype(str).str.replace("[']", "", regex=True)
    p4['Value'] = p4['Value'].astype(str).str.replace("[", "", regex=True)
    p4['Value'] = p4['Value'].astype(str).str.replace("]", "", regex=True)
    case_attribute_list=['date of birth', 'sex']
    p4 = p4[p4['Entity'].str.lower().isin([x.lower() for x in case_attribute_list])]
    p4=p4.pivot_table(values=['Value'], index=p4.HADM_ID, columns=['Entity'], aggfunc='first')

    merged=p3.merge(p4, on='HADM_ID')
    merged.columns = [str(s) for s in merged.columns]
    merged=merged.rename(columns={"('Value', 'date of birth')":'DoB' })
    merged=merged.iloc[:, :-1].join(merged.iloc[:, -1].rename('Sex'))
    merged.to_csv("./compare_attributes.csv")

def attribute_comp(df):
    merged=pd.read_csv(df)
    merged['Gender Similarity'] = (merged['Gender'] == merged['Sex'])
    merged['Date of birth'] = pd.to_datetime(merged['Date of birth'], dayfirst=True)
    merged['DoB'] = pd.to_datetime(merged['DoB'], dayfirst=True)
    merged['DoB Similarity'] = (merged['DoB'] == merged['Date of birth'])
    
    merged.to_csv("./compare_attributes.csv")
    return merged

    
def compare_attribute():
    frame6=LabelFrame(root, text="Compare Case Attributes")
    frame6.place(height=370, width=600)
    
    tv1=ttk.Treeview(frame6)
    tv1.place(relheight=1, relwidth=1)


    def show_matced():
        d0=attribute_comp("./compare_attributes.csv")
        d0=d0.dropna()
        d0=d0.drop_duplicates()
        d0=d0[['HADM_ID', 'Gender', 'Sex','Gender Similarity', 'DoB' , 'Date of birth', 'DoB Similarity' ]]
        tv1["column"] = list(d0.columns)
        tv1["show"]="headings"
        for column in tv1["column"]:
            tv1.heading(column, text=column)
        df_rows = d0.to_numpy().tolist()
        for row in df_rows:
            tv1.insert("", "end", values=row)
    def update_incorrect():
        for row in tv1.get_children():
            tv1.delete(row)
        merged=pd.read_csv("./compare_attributes.csv")
        merged['Gender']=merged['Sex']
        merged['Date of birth'] = merged['DoB']
        merged.to_csv("./compare_attributes.csv")
        d0=attribute_comp("./compare_attributes.csv")
        d0=d0[['HADM_ID', 'Gender', 'Sex','Gender Similarity', 'DoB' , 'Date of birth', 'DoB Similarity' ]]
        tv1["column"] = list(d0.columns)
        tv1["show"]="headings"
        for column in tv1["column"]:
            tv1.heading(column, text=column)
        df_rows = d0.to_numpy().tolist()
        for row in df_rows:
            tv1.insert("", "end", values=row)
        
    def back_1():
        frame6.destroy()
        button18.destroy()
        fileframe12.destroy()

     

    fileframe12=LabelFrame(root, text="Validate Case attributes")
    fileframe12.place(height=140, width=600,rely=0.75, relx=0)
    button16=Button(fileframe12, text="Show matched case attributes", command=lambda:show_matced())
    button16.place(rely=0.3, relx=0.1)
    button17=Button(fileframe12, text="Update inconsistent", command=lambda:update_incorrect())
    button17.place(rely=0.3, relx=0.5)
    button18=Button(fileframe12, text="Back", command=lambda:back_1())
    button18.place(rely=0.3, relx=0.8)
   
        
    #Label_1=Label(root, textvariable="hhhhhh").pack()
    #Label_1.place(rely = 0.2, relx = 0.2) 
    treescolly=Scrollbar(frame6, orient="vertical", command=tv1.yview)
    treescollx = Scrollbar(frame6, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescollx.set ,yscrollcommand=treescolly.set )
    treescollx.pack(side="bottom", fill="x")
    treescolly.pack(side="right", fill="y")
    
    def back():
        frame4.destroy()
        button12.destroy()
        button19.destroy()
        button20.destroy()
        button21.destroy()

  
lbl2=Label(master=top_frame,text="Text2EL: Exploiting Unstructured Text for Event Log Enrichment" , font =("Candara", 12,"bold") ,  fg='black'  )
lbl2.grid(row=0, columnspan =4, pady = (0,5))

button2 = Button(master=left_frame, text="Select Text Collection",  width=17, font =("Candara", 9,"bold") , height=2, command=browse_button)
button2.grid(row=6, column=0,pady = (5,5))

button10 = Button(master=left_frame, text="Extract Events",width=17 , height=2, font =("Candara", 9,"bold") , command=lambda:process_button())
button10.grid(row=12, column=0,pady = (5,5))
button4 = Button(master=left_frame, text="Extract Case Attributes", width=17 , height=2,font =("Candara", 9,"bold")  , command=lambda:extract_case())
button4.grid(row=18, column=0,pady = (5,5))
button3 = Button(master=left_frame, text="Extract Event Attributes", width=17 , height=2,font =("Candara", 9,"bold")  ,command=lambda:extract_event_attr())
button3.grid(row=15, column=0,pady = (5,5))

button11 = Button(master=center_frame,text="Validate with eventlog", width=17 , height=2,font =("Candara", 9,"bold")  ,command=lambda:validate())
button11.grid(row=6, column=0,pady = (5,5))
button12 = Button(master=center_frame, text="Compare Events",  width=17 , height=2, font =("Candara", 9,"bold") ,command=lambda:compare())
button12.grid(row=6, column=0 , pady = (5,5))
    
button19 = Button(master=center_frame, text="Compare Case Attributes",  width=17 , height=2,font =("Candara", 9,"bold") , command=lambda:compare_attribute())
button19.grid(row=12, column=0, pady = (5,5))
button20 = Button(master=center_frame, text="Show New Events",  width=17 , height=2, font =("Candara", 9,"bold") ,command=lambda:show_new())
button20.grid(row=15, column=0, pady = (5,5))
button21 = Button(master=right_frame,text="Incorporate User Input",  width=17 , height=2, font =("Candara", 9,"bold"),command=lambda:get_expert_inputs() )
button21.grid(  row=6, column=0, pady = (5,5))
button22 = Button(master=corner_frame,text="Show Event Log",  width=17 , height=2, font =("Candara", 9,"bold") ,command=lambda:enrich_log())
button22.grid(  row=6, column=0, pady = (5,5))
mainloop()


