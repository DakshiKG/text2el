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
    df['Note']="Discharge summary"


    df.to_csv(output_file1)

#Extract events from event specific notes in MIMIC-III 
def extract_specific_notes(collection_filepath, output_file2):
    folders = [f for f in listdir(collection_filepath)]
    df = pd.DataFrame()

    table1 = {"HADM_ID": [], "Activity": [], "Timestamp": [], "Note": []}
    for dir in folders:

        for f in glob.glob(collection_filepath + '\\' + dir + "\\*Echo.txt"):
            with open(f, "r+") as inputfile:

                con = inputfile.read()

                pattern1 = re.compile(r"Date/Time:(.*)")
                pattern2 = re.compile(r"Test:(.*)")

                rem_characters = ["['", "']", "at"]
                m1 = pattern1.findall(con)
                m2 = pattern2.findall(con)

                for i in rem_characters:
                    m1 = str(m1).replace(i, "")
                    m2 = str(m2).replace(i, "")

                table1["HADM_ID"].append(dir)
                table1["Activity"].append(m2)
                table1["Timestamp"].append(m1)
                table1["Note"].append(os.path.basename(f))

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
                table1["Note"].append(os.path.basename(f))

        df = df.append(pd.DataFrame(table1), ignore_index=True)

    df = df[["HADM_ID", "Activity", "Timestamp", "Note"]]
    df = df.dropna(how='all')


    df = df.drop_duplicates(keep='first')

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

    #for j in suffix_list:
      #  d['Activity'] = d['Activity'].str.removesuffix(j)


    d['Activity'] = d['Activity'].str.replace('^[^a-zA-Z]*', '')
    d['Activity'] = d['Activity'].str.replace('^and*', '')

    d = d[d['Activity'].str.len() > 0]
  

    out_df = d[~d['Activity'].str.lower().isin([x.lower() for x in w_list])]
    '''f2 = open(stopword_list, "r")
    among_list = f2.readlines()
    rem=str.maketrans('', '', '\n')
    b=[s.translate(rem) for s in among_list]
    print(b)
    n_l = []
    for i in b:
        out_df['Activity'] = out_df['Activity'].str.replace(re.escape(i),'')
    '''
    out_df= out_df.dropna(how='all')
    out_df['Activity'] = out_df['Activity'].str.replace('^[^a-zA-Z]*', '')
    out_df = out_df[out_df['Activity'].str.len() > 0]
    out_df=out_df[~out_df['Activity'].str.lower().isin([x.lower() for x in suffix_list])]
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
            df['Note']=source

            df = df.drop_duplicates()

            n_df=df.groupby(['HADM_ID', 'Note'], as_index=False).agg({'Dependency': ' '.join})
            n_df.to_csv(out_dep_csv)
            

def extract_case_attributes(dependenc_csv, extracted_attr_csv,all_events_csv, all_case_att_csv):
    d1=pd.read_csv(dependenc_csv)
    d1=d1[d1['Note'].str.contains("Discharge summary")]
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

        for f in glob.glob(filepath + '\\' + dir + "\\*Echo.txt"):
            with open(f, "r+") as inputfile:
                # print(f)
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
        for f in glob.glob(filepath + '\\' + dir + "\\*Radiology.txt"):
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
    ff_event_df=ff_event_df[ff_event_df['Value'].notna()]
    # to group the event attributes and get all into a list.
    ff_event_df=ff_event_df.groupby(['HADM_ID','Activity','Timestamp'])['Entity',  'Value'].agg(list)
    ff_event_df.dropna(inplace = True) 
    ff_event_df.to_csv(final_event_attributes)
    
def merge_attributes(all_case_att_csv,  list_case_attributes):
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

    out_merg_f = event_merge[event_merge['Entity'].str.lower().isin([x.lower() for x in final_filtered_attr])]
    out_merg_f = out_merg_f[['HADM_ID', 'Entity', 'Value']]
    out_df_f = out_df_f.groupby('HADM_ID').apply(lambda x: dict(zip(x['Entity'], x['Value']))).reset_index().rename(
        columns={"HADM_ID": "HADM_ID", 0: "Medical Info"})

    out_df_ff = out_df_f.melt(id_vars="HADM_ID")
    out_df_ff = out_df_ff.rename(columns={'variable': 'Entity', 'value': 'Value'})
    
    out_df_f = pd.concat([out_df_ff, out_merg_f])
    out_df_f.to_csv(list_case_attributes)

def delete_garbage_files():
    file_list=["./output1.csv", "./specific_events.csv", "./all_events.csv", "./all_attributes.csv" , "./echo_events.csv","./radio_events.csv","./case_attr_final.csv", "./event_attr_final.csv" ]
    for i in file_list:
        if(os.path.exists(i) and os.path.isfile(i)):
            os.remove(i)
            print("file deleted")


if __name__ == '__main__':
    apply_regex_rules("extracted_notes_MIMIC-III_Evaluation\\", ".extracted_events.csv")
    extract_specific_notes("extracted_notes_MIMIC-III_Evaluation\\", "specific_events.csv")
    refine_events("extracted_events.csv", "text2el\models\\exclude_list.txt", "text2el\models\\stopwords.txt", "refined_events.csv")
    get_all_events("refined_events.csv", "specific_events.csv", "all_events.csv")
    tag_NER_lookup("extracted_notes_MIMIC-III_Evaluation\\", 'lookup_hospital_activity.txt' , 'lookup_medical_activity.txt', 'all_attributes.csv')
    dep_parse_attribute("extracted_notes_MIMIC-III_Evaluation\\", "all_dependency.csv")
    extract_case_attributes("all_dependency.csv", "all_attributes.csv", 'all_events.csv', "case_attr_final.csv")
    extract_event_attributes("all_dependency.csv", "all_attributes.csv", 'specific_events.csv', 'all_events.csv', "event_attr_final.csv")
    extract_echo_event_attr("extracted_notes_MIMIC-III_Evaluation\\", "echo_events.csv")
    extract_radiology_event_attr("extracted_notes_MIMIC-III_Evaluation\\", "radio_events.csv")
    extract_event_attr_event_notes("echo_events.csv", "radio_events.csv","event_attr_final_f.csv", "final_event_attr_list.csv" )
    merge_attributes("case_attr_final.csv", "case_attr_final_list.csv")
    delete_garbage_files()


