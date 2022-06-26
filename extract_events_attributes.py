import re
import pandas as pd
import glob
import ntpath
import sys, os
from os import listdir
from os.path import isfile, join

import scispacy
import spacy

import en_ner_bc5cdr_md
from itertools import chain

from spacy import displacy
from pathlib import Path


from stanfordnlp.server import CoreNLPClient

os.environ["CORENLP_HOME"] = 'C:\\Users\\kapugama\\Downloads\\stanford-corenlp-latest\\stanford-corenlp-4.4.0'


#Extract events based on temporal expressions from MIMIC-Discharge summaries
def apply_regex_rules(collection_filepath, output_file1):
    time = []
    activity = []
    caseID = []
    folders = [f for f in listdir(collection_filepath)]
    df = pd.DataFrame()

    for dir in folders:
        for f in glob.glob(collection_filepath + '\\' + dir + "\\*Discharge summary.txt"):
            with open(f, "r+") as inputfile:
                con = inputfile.read()
                x = con.split("\n")

                pattern = re.compile(r"(.*[[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]*[*]{2}\])")
                frontkeyonlypattern = re.compile(
                    r"(.*)[[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\]([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?")
                backkeyonlypattern = re.compile(
                    r"[[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\]([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?(.*)")
                datetimepattern = re.compile(
                    r"([[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\]([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?)")
                pattern1 = re.compile(r"([[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\].*)")

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
    df['Source']="Discharge summary"


    df.to_csv(output_file1)

#Extract events from event specific notes in MIMIC-III 
def extract_specific_notes(collection_filepath, output_file2):
    folders = [f for f in listdir(collection_filepath)]
    df = pd.DataFrame()

    table1 = {"HADM_ID": [], "Activity": [], "Timestamp": [], "Source": []}
    for dir in folders:

        for f in glob.glob(collection_filepath + '\\' + dir + "\\*Echo.txt"):
            with open(f, "r+") as inputfile:

                con = inputfile.read()

                pattern1 = re.compile(r"Date/Time:(.*)")
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
                table1["Source"].append(os.path.basename(f))

        for f in glob.glob(collection_filepath + '\\' + dir + "\\*Radiology.txt"):
            with open(f, "r+") as inputfile:

                con2 = inputfile.readlines()
                pattern = re.compile(r"([[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\].*)")

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
                table1["Source"].append(os.path.basename(f))

        df = df.append(pd.DataFrame(table1), ignore_index=True)

    df = df[["HADM_ID", "Activity", "Timestamp", "Source"]]
    df = df.dropna()
    df['Source'] = df['Source'].str.replace('^[^a-zA-Z]*', '')
    df['Source'] = df['Source'].str.replace('.txt', '')

    df = df.drop_duplicates(keep='first')
    df = df.dropna()
    df = df[df.Activity != '']
    df.to_csv(output_file2)
    
    
#Refine the extracted events by excluding the activities which can’t be events     
def refine_events(input_csv, exclude_list, stopword_list, output_csv):
    d = pd.read_csv(input_csv)
    f1 = open(exclude_list , "r")
    w_list = f1.read().split('\n')

    suffix_list = ['in', 'on', 'of', 'On']

    d['Activity'] = d['Activity'].str.strip()

    d['Activity'] = d['Activity'].str.rstrip(',')
    d['Activity'] = d['Activity'].str.rstrip(',')
    d['Activity'] = d['Activity'].str.lstrip(',')

    for j in suffix_list:
        d['Activity'] = d['Activity'].str.removesuffix(j)


    d['Activity'] = d['Activity'].str.replace('^[^a-zA-Z]*', '')
    d['Activity'] = d['Activity'].str.replace('^and*', '')

    d = d[d['Activity'].str.len() > 0]
    d = d.dropna()

    out_df = d[~d['Activity'].str.lower().isin([x.lower() for x in w_list])]
    f2 = open(stopword_list, "r")
    among_list = f2.readlines()
    rem=str.maketrans('', '', '\n')
    b=[s.translate(rem) for s in among_list]
    print(b)
    n_l = []
    for i in b:
        out_df['Activity'] = out_df['Activity'].str.replace(re.escape(i),'')

    out_df.dropna()
    out_df['Activity'] = out_df['Activity'].str.replace('^[^a-zA-Z]*', '')
    out_df = out_df[out_df['Activity'].str.len() > 0]

    out_df.to_csv(output_csv)

#merge all events, events extracted from the event-specific notes and case notes
def get_all_events(general_events_file, specific_event_file, all_events):
    d1=pd.read_csv(general_events_file)
    d2=pd.read_csv(specific_event_file)
    d1=d1[["HADM_ID",	"Activity",	"Timestamp",	"Source"]]
    d2 = d2[["HADM_ID", "Activity", "Timestamp", "Source"]]

    d3 = pd.concat([d1, d2])

    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("[**"), '')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("**]"), '')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("-"), '/')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("at"), ' ')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("**"), '')
    d3=d3.drop_duplicates()
    d3=d3.sort_values(['HADM_ID', 'Timestamp'], ascending=[True, True])
    d3.to_csv(all_events)

    
def preprocess(doc):

    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]


#Apply Named Entity Recognition and lookup list tagging 
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
    df['Source']=source
    df = df[df['Entity'].astype(str).str.len()>1]
    df['Entity']=df['Entity'].astype(str).str.replace('^[^a-zA-Z]*', '')
    
    df.dropna()
    
    df.Entity = df.Entity.str.replace(' +', ' ')
 
    df['Entity']=df['Entity'].str.lower()

    df['Entity'] = df['Entity'].str.strip()
    df_uni = df[df['Value'].notnull()]
    df_uni = df_uni[df_uni['Entity'].notnull()]
    df_uni=df_uni.dropna()
    

    df_uni=d1.drop_duplicates( keep='first')
    df_uni.to_csv(outputfile)
    outputfile.close()

#apply dependency parsing and extract attributes  
def dep_parse_attribute(filepath, out_dep_csv):
    df = pd.DataFrame()
    case_id = []
    dep_p= []
    source = []

    folders = [f for f in listdir(filepath)]

    for dir in folders:

        event_notes = ["/*Radiology.txt", "/*Echo.txt"]
        case_notes = ["/*Discharge summary.txt"]

        for i in case_notes:
            for f in glob.glob(filepath + '/' + dir + str(i)):

                with open(f, "r+") as inputfile:

                    text = inputfile.read().split(".")
                    text.pop()
                    list_dep=[]
                    add_dp = ""
                    for k in text:

                        if len(k) > 3:

                            with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'depparse'], timeout=60000, memory='16G') as client:
                        # set up the client
                                add_dp = ""

                            # submit the request to the server
                                ann = client.annotate(str(k))

                            # get the first sentence
                                sentence = ann.sentence[0]


                            # get the dependency parse of the first sentence
                                dependency_parse = sentence.basicDependencies

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

                                #print(list_dep)'''
                                    dp_r = ""
                                    for k in list_dep:
                                        if (k[0] == 'nsubj') or (k[0] == 'obl') or (k[0] == 'obj') or (k[0] == 'advmod') or (k[0] == 'compound') or (k[0] == 'acomp') or (k[0] == 'appos') or ( k[0] == 'neg') or (k[0] == 'nsubjpass') or (k[0] == 'nummod') or ( k[0] == 'obl:tmod'):
                                            dp_r = dp_r + str(k) + "***"

                        add_dp = add_dp + dp_r

                        case_id.append(dir)
                        dep_p.append(add_dp)
                        source.append(os.path.basename(f))

            df['HADM_ID']=case_id
            df['Dependency']=dep_p
            df['Source']=source

            df = df.drop_duplicates()

            n_df=df.groupby(['HADM_ID', 'Source'], as_index=False).agg({'Dependency': ' '.join})
            n_df.to_csv(out_dep_csv)
            

def extract_case_attributes(dependenc_csv, extracted_attr_csv,all_events_csv, all_case_att_csv):
    d1=pd.read_csv(dependenc_csv)
    d1=d1[d1['Source'].str.contains("Discharge summary")]
    d2=pd.read_csv(extracted_case_csv)
    
    d3=pd.DataFrame()
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
    case_df=d2[d2['Source'].str.contains("Discharge summary.txt")]
    df_e=pd.read_csv(all_events_csv)
    events_l = df_e['Activity'].drop_duplicates().to_list()
    exclude_l=['birth:  [', 'birth:  [' ,'admit' , 'discharge', 'Discharge Date', 'Date of Birth']
    exclude_v = ['+2  Left:+2']            
    #events_l.extend(exclude_v)
      

    new_df = pd.concat([case_df , d3])
    new_df.dropna()
    new_df = new_df[~new_df['Value'].isin(exclude_l)]
 
    new_df=new_df[~new_df['Entity'].isin(exclude_v)]
    new_df=new_df.drop_duplicates(keep='first')
    new_df=new_df[['HADM_ID','Entity', 'Value', 'Source']]
    new_df.to_csv(all_case_att_csv)


def extract_event_attributes( dependency_csv, extracted_attributes_csv, specific_events_csv, all_events_csv, all_event_att_csv ):

    e_df1=pd.read_csv(dependency_csv)
    
    e_df1['Source_edi']=e_df1['Source'].str.replace('^[^a-zA-Z]*', '')
    e_df1['Source_edi']=e_df1['Source_edi'].str.replace('.txt', '')
    #print(e_df1)
    event_notes=['Radiology' , 'Echo']
    e_df2=pd.read_csv(extracted_attributes_csv)
    e_df2['Source_edi']=e_df2['Source'].str.replace('^[^a-zA-Z]*', '')
    e_df2['Source_edi']=e_df2['Source_edi'].str.replace('.txt', '')
    e_df1 = e_df1[e_df1['Source_edi'].isin(event_notes)]
    #print(e_df2)
    e_df2 = e_df2[e_df2['Source_edi'].isin(event_notes)]
    #print(e_df2)
    d3=pd.DataFrame()
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
    
    df_e=pd.read_csv(all_events_csv)
    events_l = df_e['Activity'].drop_duplicates().to_list()
    e_df2 = e_df2[~e_df2['Value'].isin(events_l)]
    new_df = pd.concat([e_df2 , d3])
    #print(new_df)
    new_df.dropna()
    e_df3=pd.read_csv(specific_events_csv)
    new_df=new_df.merge(e_df3, on=["Source", 'HADM_ID'])
    new_df = new_df[['HADM_ID','Activity', 'Timestamp', 'Entity', 'Value', "Source"]]

    new_df=new_df.drop_duplicates(keep='first')

    new_df.to_csv(all_event_att_csv)
    
    
if __name__ == '__main__':
    apply_regex_rules("extracted_notes_MIMIC-III_Evaluation", "extracted_events.csv")
    extract_specific_notes("extracted_notes_MIMIC-III_Evaluation", "specific_events.csv")
    refine_events("extracted_events.csv", "text2el\models\\exclude_list.txt", "text2el\models\\stopwords.txt", "refined_events.csv")
    get_all_events("refined_events.csv", "Cspecific_events.csv", "all_events.csv")
    tag_NER_lookup("extracted_notes_MIMIC-III_Evaluation", 'lookup_hospital_activity.txt' , 'lookup_medical_activity.txt', 'all_attributes.csv')
    dep_parse_attribute("extracted_notes_MIMIC-III_Evaluation", "all_dependency.csv")
    extract_case_attributes("all_dependency.csv", "all_attributes.csv", 'all_events.csv', "ase_attr_final.csv")
    extract_event_attributes("all_dependency.csv", "all_attributes.csv", 'specific_events.csv', 'all_events.csv', "event_attr_final.csv")
