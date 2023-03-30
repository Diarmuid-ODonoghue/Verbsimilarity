import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile, common_texts
from gensim import matutils
import gensim.downloader as api
import numpy as np
import customtkinter
import tkinter
from tkinter import *
from tkinter.font import Font
from datetime import datetime

class UI: # class to control project ui
    def __init__(self, parent, customtkinter):
        self.parent = parent
        self.parent.minsize(1700,700)
        self.parent.maxsize(1920,1080)
        self.parent.title("Verb based similarity's")
        self.parent.geometry(f"{1300}x{500}")
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")
        customtkinter.deactivate_automatic_dpi_awareness()
        self.parent.grid_columnconfigure(1, weight=1)
        self.parent.grid_columnconfigure((2, 3), weight=1)
        self.parent.grid_rowconfigure((0, 1, 2), weight=1)
        self.start_interface(customtkinter)

    def start_interface(self,customtkinter):    #only run once at the launch
        print()
        self.parent.frame_sidebar = customtkinter.CTkFrame(self.parent, width=140, corner_radius=20)
        self.parent.frame_sidebar.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.parent.frame_sidebar.grid_rowconfigure(4, weight=1)
        self.parent.frame_title = customtkinter.CTkFrame(self.parent.frame_sidebar, width=140, corner_radius=20)
        self.parent.frame_title.grid(row=0, column=0, sticky="nsew")
        self.parent.label_title = customtkinter.CTkLabel(self.parent.frame_title, text="Verb based similarity's", font=("Helvetica", 20, "bold"))
        self.parent.label_title.grid(row=0, column=0, padx=(20, 20), pady=(20, 20))

        self.parent.switch_highlightall = customtkinter.CTkSwitch(master=self.parent.frame_sidebar, text="Highlight All Verbs", command=self.switch_highlightall)
        self.parent.switch_highlightall.grid(row=1, column=0, padx=(40,20), pady=(30, 20), sticky="nsew")

        self.parent.switch_showonlyverbs = customtkinter.CTkSwitch(master=self.parent.frame_sidebar, text="Show only Verbs",command=self.switch_showonlyverbs)
        self.parent.switch_showonlyverbs.grid(row=2, column=0, padx=(40,20), pady=(0, 20), sticky="nsew")

        self.parent.switch_highlightcommon = customtkinter.CTkSwitch(master=self.parent.frame_sidebar, text="Highlight Common Verbs",command=self.switch_highlightcommon)
        self.parent.switch_highlightcommon.grid(row=3, column=0, padx=(40,20), pady=(0, 20), sticky="nsew")

        self.parent.checkbox_stop = customtkinter.CTkCheckBox(master=self.parent.frame_sidebar, text='Filter stop-words')
        self.parent.checkbox_stop.grid(row=4, column=0, pady=(20, 0), padx=(20, 20), sticky="n")

        self.parent.checkbox_auto = customtkinter.CTkCheckBox(master=self.parent.frame_sidebar, text='Auto-compare', command=self.auto_compare)
        self.parent.checkbox_auto.grid(row=5, column=0, pady=(20, 0), padx=(20, 20), sticky="n")

        self.parent.sidebar_button_2 = customtkinter.CTkButton(self.parent.frame_sidebar, text="Compare Similarity", command=self.similarity_button)
        self.parent.sidebar_button_2.grid(row=6, column=0, padx=(20,20), pady=(20,30))

        self.parent.sidebar_button_2 = customtkinter.CTkButton(self.parent.frame_sidebar, text="Compare all", command=self.parent.compare_all)
        self.parent.sidebar_button_2.grid(row=7, column=0, padx=(20,20), pady=(00,30))

        self.parent.frame_textbox1 = customtkinter.CTkFrame(self.parent, width=140, corner_radius=20)
        self.parent.frame_textbox1.grid(row=0, column=1, sticky="nsew")
        self.parent.frame_textbox1.grid_rowconfigure(4, weight=1)

        self.parent.frame_textbox2 = customtkinter.CTkFrame(self.parent, width=140, corner_radius=20)
        self.parent.frame_textbox2.grid(row=0, column=2, sticky="nsew")
        self.parent.frame_textbox2.grid_rowconfigure(4, weight=1)

        self.parent.frame_identresults = customtkinter.CTkFrame(self.parent, width=140, corner_radius=20)
        self.parent.frame_identresults.grid(row=1, column=1, sticky="nsew")
        self.parent.frame_identresults.grid_rowconfigure(4, weight=1)

        self.parent.frame_simresults = customtkinter.CTkFrame(self.parent, width=140, corner_radius=20)
        self.parent.frame_simresults.grid(row=1, column=2, sticky="nsew")
        self.parent.frame_simresults.grid_rowconfigure(4, weight=1)

        self.parent.frame_entrys = customtkinter.CTkFrame(self.parent, width=140, corner_radius=20)
        self.parent.frame_entrys.grid(row=0, column=3, rowspan=5, sticky="nsew")
        self.parent.frame_entrys.grid_rowconfigure(0, weight=1)

        self.parent.entry_filepath = customtkinter.CTkEntry(self.parent, placeholder_text="Path to file")
        self.parent.entry_filepath.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(00, 00), sticky="nsew")

        self.parent.button_filepath = customtkinter.CTkButton(master=self.parent.frame_entrys, command=self.submit_directory, fg_color="transparent", border_width=2, text="Path",text_color=("gray10", "#DCE4EE"))
        self.parent.button_filepath.grid(row=3, column=3, padx=(20, 20), pady=(0, 00), sticky="nsew")

        self.parent.entry_dirpath = customtkinter.CTkEntry(self.parent, placeholder_text="Path to Directory with files to compare")
        self.parent.entry_dirpath.grid(row=4, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.parent.button_dirpath = customtkinter.CTkButton(master=self.parent.frame_entrys, command=self.submit_directory, fg_color="transparent", border_width=2, text="Submit Path",text_color=("gray10", "#DCE4EE"))
        self.parent.button_dirpath.grid(row=4, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.parent.scrollframe_filelist = customtkinter.CTkScrollableFrame(self.parent, label_text="File List:")
        self.parent.scrollframe_filelist.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.parent.scrollframe_filelist.grid_columnconfigure(0, weight=1)
        self.parent.radio_val = tkinter.IntVar(value=0)

    def update_ui(self, num):   # nearly every function calls this to update the ui with any changes
        self.parent.doc_a()
        self.parent.textbox1 = customtkinter.CTkTextbox(master=self.parent, width=500, corner_radius=15, height=500)
        self.parent.textbox1.grid(row=0, column=1, padx=(20, 20), pady=(80, 0), sticky="nsew")
        textbox = self.parent.textbox1
        tokenized_text = self.parent.word_tokenized_doca
        verbs = self.parent.verbs_doca
        if self.parent.checkbox_auto.get() == 1:
            self.parent.compare = 1
        textbox.delete('0.0', END)
        if num == 0:
            if self.parent.switch_showonlyverbs.get() == 0:
                textbox = self.label_verbs(textbox, tokenized_text, verbs)
            elif self.parent.switch_showonlyverbs.get() == 1:
                textbox = self.label_verbs(textbox, verbs, verbs)
        elif num == 1:
            print()
        else:
            if self.parent.switch_showonlyverbs.get() == 0:
                textbox = self.unlabel_verbs(textbox, tokenized_text, verbs)
            elif self.parent.switch_showonlyverbs.get() == 1:
                textbox = self.unlabel_verbs(textbox, verbs, verbs)

        self.parent.scrollframe_filelist = customtkinter.CTkScrollableFrame(self.parent, label_text="File List:")
        self.parent.scrollframe_filelist.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.parent.scrollframe_filelist.grid_columnconfigure(0, weight=1)

        for ind, file in enumerate(self.parent.file_list):
            if ind <= 50:
                self.parent.radiobutton_filelist = customtkinter.CTkRadioButton(master=self.parent.scrollframe_filelist, variable=self.parent.radio_val, value=self.parent.file_list.index(file), command=self.selected, text=os.path.basename(file))
                self.parent.radiobutton_filelist.grid(row=self.parent.file_list.index(file)+1, column=0, pady=10, padx=20, sticky="w")
            if self.parent.radio_val.get()==self.parent.file_list.index(file):
                self.parent.word_tokenized_docb = NLTK.nltk_word_tokenize(self.parent,NLTK.parse_file(self.parent,file))
                if self.parent.checkbox_stop.get() == 1:
                    #self.parent.stemmed_docb = self.parent.nltk_stemming(self.parent.filtered_docb)
                    self.parent.pos_tagged_docb = NLTK.nltk_tagging(self.parent, self.parent.word_tokenized_docb)
                    self.parent.pos_tagged_docb = NLTK.nltk_stopwords(self.parent,self.parent.pos_tagged_docb)
                else:
                    self.parent.pos_tagged_docb = NLTK.nltk_tagging(self.parent,self.parent.word_tokenized_docb)
                verb_data = NLTK.extract_verbs(self.parent,[self.parent.pos_tagged_docb])
                self.parent.verbs_tagged_docb = [verb_data[0]]
                self.parent.verbs_docb = verb_data[1]

                if self.parent.compare == 1:
                    self.remove_similarity()
                    self.parent.get_similarity()
                else:
                    self.remove_similarity()

                self.parent.textbox2 = customtkinter.CTkTextbox(self.parent, width=500, height= 500, corner_radius=15)
                self.parent.textbox2.grid(row=0, column=2, padx=(20, 20), pady=(80, 0), sticky="nsew")
                if num == 0:
                    if self.parent.switch_showonlyverbs.get() == 0:
                        self.parent.textbox2 = self.label_verbs(self.parent.textbox2, self.parent.word_tokenized_docb, self.parent.verbs_docb)
                    elif self.parent.switch_showonlyverbs.get() == 1:
                        self.parent.textbox2 = self.label_verbs(self.parent.textbox2, self.parent.verbs_docb, self.parent.verbs_docb)
                elif num == 1:
                    if self.parent.switch_showonlyverbs.get() == 0:
                        self.parent.textbox2 = self.label_common_verbs(self.parent.textbox2, self.parent.word_tokenized_docb, self.parent.verbs_docb,'b')
                        textbox = self.label_common_verbs(textbox, tokenized_text, verbs,'a')
                    elif self.parent.switch_showonlyverbs.get() == 1:
                        self.parent.textbox2 = self.label_common_verbs(self.parent.textbox2, self.parent.verbs_docb, self.parent.verbs_docb,'b')
                        textbox = self.label_common_verbs(textbox, verbs, verbs,'a')
                else:
                    if self.parent.switch_showonlyverbs.get() == 0:
                        self.parent.textbox2 = self.unlabel_verbs(self.parent.textbox2, self.parent.word_tokenized_docb, self.parent.verbs_docb)
                    elif self.parent.switch_showonlyverbs.get() == 1:
                        self.parent.textbox2 = self.unlabel_verbs(self.parent.textbox2, self.parent.verbs_docb, self.parent.verbs_docb)

    def remove_similarity(self):    # this removes the scores underneath the textbox's
        self.parent.logo = tkinter.StringVar(value=f'                                        ')
        self.parent.logo1 = tkinter.StringVar(value=f'                                                     ')
        self.parent.logo_label2 = customtkinter.CTkLabel(master = self.parent,textvariable=self.parent.logo1 ,wraplength=400, text='',font=customtkinter.CTkFont(size=20, weight="bold"))
        self.parent.logo_label2.grid(row=1, column=1, padx=(0,0), pady=(00,10))
        self.parent.logo_lab = customtkinter.CTkLabel(master = self.parent,textvariable=self.parent.logo1 ,wraplength=400, text='', font=customtkinter.CTkFont(size=20, weight="bold"))
        self.parent.logo_lab.grid(row=1, column=1, padx=(0,0), pady=(60,10))
        self.parent.logo_label3 = customtkinter.CTkLabel(master = self.parent,textvariable=self.parent.logo1 ,wraplength=400, text='',font=customtkinter.CTkFont(size=20, weight="bold"))
        self.parent.logo_label3.grid(row=1, column=2, padx=(0,0), pady=(60,10))
        self.parent.logo_label4 = customtkinter.CTkLabel(master = self.parent,textvariable=self.parent.logo1 ,wraplength=400, text='',font=customtkinter.CTkFont(size=20, weight="bold"))
        self.parent.logo_label4.grid(row=1, column=2, padx=(0,0), pady=(00,10))

    def auto_compare(self): # check if auto compare is clicked
        if self.parent.checkbox_auto.get() == 1:
            self.parent.compare = 1
        if self.parent.checkbox_auto.get() == 0:
            self.parent.compare = 0

    def switch_highlightall(self):  #check if highlight all is clicked
        if self.parent.switch_highlightall.get() == 0:
            self.update_ui(2)
        else:
            self.parent.switch_highlightcommon.deselect()
            self.update_ui(0)

    def switch_highlightcommon(self):   # check if highlight common is switch is clicked
        if self.parent.switch_highlightcommon.get() == 0:
            self.update_ui(2)
        else:
            self.parent.switch_highlightall.deselect()
            self.update_ui(1)

    def switch_showonlyverbs(self): # check if show only verbs switch is clicked
        if self.parent.switch_highlightall.get() == 1:
            if self.parent.switch_showonlyverbs.get() == 1:
                self.parent.only_verbs = 1
            self.update_ui(0)
            if self.parent.checkbox_auto.get() == 0:
                self.parent.compare = 0
        elif self.parent.switch_highlightcommon.get() == 1:
            if self.parent.switch_showonlyverbs.get() == 1:
                self.parent.only_verbs = 1
            self.update_ui(1)
            if self.parent.checkbox_auto.get() == 0:
                self.parent.compare = 0
        else:
            if self.parent.switch_showonlyverbs.get() == 1:
                self.parent.only_verbs = 1
            self.update_ui(2)
            if self.parent.checkbox_auto.get() == 0:
                self.parent.compare = 0

    def submit_directory(self):     # called if user inputs a directory
        print()
        run = False
        print(self.parent.entry_dirpath.get())
        if os.path.isdir(self.parent.entry_dirpath.get()):
            print('real path1')
            self.parent.dir_path = self.parent.entry_dirpath.get()
            self.parent.file_list = NLTK.check_dir(self.parent)
            run = True
        elif os.path.isfile(self.parent.entry_dirpath.get()):
            print('real file1')
        else:
            print('Not a real directory1')
        if os.path.isdir(self.parent.entry_filepath.get()):
            print('real path2')
        elif os.path.isfile(self.parent.entry_filepath.get()):
            print('real file2')
            print(self.parent.entry_filepath.get())
            self.parent.path2 = self.parent.entry_filepath.get()
            run = True
        else:
            print('Not a real directory2')
        if run == True:
            self.update_ui(0)

    def similarity_button(self):    # checks if similarity button is clicked
        if self.parent.switch_highlightall.get() == 1:
            self.parent.compare = 1
            self.update_ui(0)
            if self.parent.checkbox_auto.get() == 0:
                self.parent.compare = 0
        elif self.parent.switch_highlightcommon.get() == 1:
            self.parent.compare = 1
            self.update_ui(1)
            if self.parent.checkbox_auto.get() == 0:
                self.parent.compare = 0
        else:
            self.parent.compare = 1
            self.update_ui(2)
            if self.parent.checkbox_auto.get() == 0:
                self.parent.compare = 0

    def selected(self): #checks what file in scrollable frame is selected
        for file in self.parent.file_list:
            if self.parent.radio_val.get()==self.parent.file_list.index(file):
                if self.parent.switch_highlightall.get() == 1:
                    self.update_ui(0)
                elif self.parent.switch_highlightcommon.get() == 1:
                    self.update_ui(1)
                else:
                    self.update_ui(2)

    def unlabel_verbs(self, textbox, tokenized_doc, verbs): #removes and highlights
        for word in tokenized_doc:
            textbox.insert(END, word+" ")
            self.parent.bold_font = ("Helvetica", 14, "bold")
            textbox.configure(font=self.parent.bold_font)

    def label_verbs(self, textbox, tokenized_doc, verbs):   #highlights verbs
        string = ''
        for word in tokenized_doc:
            textbox.insert(END, word+" ")
            self.parent.bold_font = ("Helvetica", 14, "bold")
            textbox.configure(font=self.parent.bold_font)
            string= string + word+" "
            if word in verbs:
                index = len(string)-len(word)-1
                length = len(word)
                textbox.tag_add("start", f"1.{index}", f"1.{index+length}")
                textbox.tag_config("start", background="yellow", foreground="black")

    def label_common_verbs(self, textbox, tokenized_doc, verbs, doc):   #highlights common verbs to both docs
        print()
        string = ''
        if doc == 'a':
            verb_doc = self.parent.verbs_docb
        else:
            verb_doc = self.parent.verbs_doca
        for word in tokenized_doc:
            textbox.insert(END, word+" ")
            self.parent.bold_font = ("Helvetica", 14, "bold")
            textbox.configure(font=self.parent.bold_font)
            string= string + word+" "
            if word in verbs:
                for verb in verb_doc:
                    if verb == word:
                        index = len(string)-len(word)-1
                        length = len(word)
                        textbox.tag_add("start", f"1.{index}", f"1.{index+length}")
                        textbox.tag_config("start", background="orange", foreground="black")


class NLTK: # class called for any NLTK purposes
    def __init__(self, parent):
        print('nltk')

    def parse_file(self, path):   # open and read file
        with open(path, 'r') as f:
            return(f.read())

    def nltk_word_tokenize(self, text):
        return(word_tokenize(text))

    def nltk_stopwords(self, text):   # remove stop words
        filtered = []
        stop_words = set(stopwords.words("english"))
        for word in text:
            if word[0] not in stop_words:
                filtered.append(word)
        return(filtered)

    def nltk_stemming(self, word_tokenized_text):   # stemming words in tokenized text
        ps = PorterStemmer()
        stemmed = []
        for word in word_tokenized_text:
            stemmed.append(ps.stem(word))
        return(stemmed)

    def nltk_tagging(self, text):   # pos tagging words
        return(nltk.pos_tag(text))

    def extract_verbs(self, tagged):    # extracting verbs from tagged strings
        verbs_tagged = []
        verbs = []
        for sentence_tags in tagged:
            for word_tag in sentence_tags:
                if word_tag[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
                    verbs_tagged.append(word_tag)
        for pair in verbs_tagged:
            if pair[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
                new_string = pair[0]
                verbs.append(new_string)
                new_string = ''
        return([verbs_tagged, verbs])

    def check_dir(self):
        file_list = []
        self.directory = os.listdir(self.dir_path)
        for file in self.directory:
            if file.endswith(".txt"):
                file_list.append(self.dir_path+'\\'+file)
        return(file_list)

class Verb_Similarity(customtkinter.CTk):   # Main Class of project, where all similaritys are calculated
    def __init__(self):   # initialising variables
        super().__init__()
        self.ui = UI(self, customtkinter)
        self.nltk = NLTK(self)
        #nltk.download()
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.directory = os.listdir(self.dir_path)
        self.path = f'{self.dir_path}\\testfile.txt'
        self.path2 = f'{self.dir_path}\\exercise2.txt'
        self.file_contents = ''
        self.file_contents = NLTK.parse_file(self,self.path)
        self.logo = tkinter.StringVar(value='')
        self.filtered = []
        self.stemmed = []
        self.tagged = []
        self.verbs_tagged = []
        self.verbs = []
        self.on = 1
        self.only_verbs = 0
        self.compare = 0
        self.logo1 = ''
        self.logo = ''
        self.file_list = NLTK.check_dir(self)

    def doc_a(self):    # calls everything needed for doc_a in NLTK
        self.file_contents2 = NLTK.parse_file(self,self.path2)
        self.word_tokenized_doca = NLTK.nltk_word_tokenize(self,self.file_contents2)
        if self.checkbox_stop.get() == 1:
            #self.stemmed_doca = self.nltk_stemming(self.filtered_doca)
            self.pos_tagged_doca = NLTK.nltk_tagging(self,self.word_tokenized_doca)
            self.pos_tagged_doca = NLTK.nltk_stopwords(self,self.pos_tagged_doca)
        else:
            self.pos_tagged_doca = NLTK.nltk_tagging(self,self.word_tokenized_doca)
        verb_data_doca = NLTK.extract_verbs(self,[self.pos_tagged_doca])
        self.verbs_tagged_doca = [verb_data_doca[0]]
        self.verbs_doca = verb_data_doca[1]

    def compare_all(self):  # only called if button to compare all is clicked
        print('compare all')
        results = {}
        self.doc_a()
        tokenized_text = self.word_tokenized_doca
        verbs = self.verbs_doca
        for file in self.file_list:
            #try:
            self.word_tokenized_docb = NLTK.nltk_word_tokenize(self,NLTK.parse_file(self,file))
            if self.checkbox_stop.get() == 1:
                #self.stemmed_docb = self.nltk_stemming(self.filtered_docb)
                self.pos_tagged_docb = NLTK.nltk_tagging(self,self.word_tokenized_docb)
                self.pos_tagged_docb = NLTK.nltk_stopwords(self,self.pos_tagged_docb)
            else:
                self.pos_tagged_docb = NLTK.nltk_tagging(self,self.word_tokenized_docb)
            verb_data = NLTK.extract_verbs(self,[self.pos_tagged_docb])
            self.verbs_tagged_docb = [verb_data[0]]
            self.verbs_docb = verb_data[1]
            if len(self.verbs_docb) > 10:
                size = int(min([len(self.verbs_doca),len(self.verbs_docb)])/10)
                if size < 1:
                    size = 2

                self.doca_adjacent = self.return_adjacent_words(self.verbs_doca,size)
                self.docb_adjacent = self.return_adjacent_words(self.verbs_docb,size)

                sequences_a,sequences_b,identicalities = self.adjacent_identicality(self.doca_adjacent, self.docb_adjacent)
                #self.most_identical = self.n_highest_in_list(sequences_a,sequences_b,identicalities)
                average = self.average_identicalitys(sequences_a,sequences_b,identicalities)
                #print(f'Average Identicality : {str(round(average, 4))}')

                self.model = self.word2vec([self.verbs_docb])

                #how many verbs they have in common / how many verbs in total between them
                set_a = set(self.verbs_doca)
                set_b = set(self.verbs_docb)
                jaccard_identicality = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
                #print(f'Jaccard Identicality: #intersecting verbs / #total verbs: {jaccard_identicality}')

                # Get Word2Vec vectors for the words in each document
                doc_a_vecs = []
                for word in self.verbs_doca:
                    if word in list(self.model.key_to_index.keys()):
                        doc_a_vecs.append(self.model[word])

                doc_b_vecs = []
                for word in self.verbs_docb:
                    if word in list(self.model.key_to_index.keys()):
                        doc_b_vecs.append(self.model[word])

                distance = self.model.wmdistance(self.verbs_doca, self.verbs_docb)

                #identicalitydistance = self.levenshtein_distance(self.verbs_doca, self.verbs_docb)
                #print(f'Levenshtein distance: {identicalitydistance}')
                print('..')

                doc_a_sum = np.zeros(self.model.vector_size)
                for i in self.verbs_doca:
                    if i in list(self.model.key_to_index.keys()):
                        doc_a_sum += self.model[i]

                doc_b_sum = np.zeros(self.model.vector_size)
                for i in self.verbs_docb:
                    if i in list(self.model.key_to_index.keys()):
                        doc_b_sum += self.model[i]
                results[file] = str(round(np.dot(doc_a_sum, doc_b_sum)/(np.linalg.norm(doc_a_sum)* np.linalg.norm(doc_b_sum)), 3))
                with open(f'{os.path.basename(self.path2)}-{os.path.basename(self.dir_path)}.txt', 'a') as resultsfile:
                    resultsfile.write(f'\n{file}\n{str(round(jaccard_identicality, 3))}\n{str(round(average, 4))}\n{str(round(((self.model.n_similarity(self.verbs_doca, self.verbs_docb)- 0.5) * 2), 3))}\n{str(round(distance, 3))}\n{str(round(np.dot(doc_a_sum, doc_b_sum)/(np.linalg.norm(doc_a_sum)* np.linalg.norm(doc_b_sum)), 3))}\n')

            #except:
            #    print("An exception occured")
            #    print(Exception)
        sorted_results = sorted(results, key=results.get, reverse=True)
        top_10 = sorted_results[:10]
        print(top_10)
        print()
        self.file_list = top_10
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)

    def get_similarity(self):   # called if buttton to compare the two docs on display
        print()
        size = int(min([len(self.verbs_doca),len(self.verbs_docb)])/10)
        if size < 1:
            size = 2
        self.doca_adjacent = self.return_adjacent_words(self.verbs_doca,size)
        self.docb_adjacent = self.return_adjacent_words(self.verbs_docb,size)

        sequences_a,sequences_b,identicalities = self.adjacent_identicality(self.doca_adjacent, self.docb_adjacent)
        self.most_identical = self.n_highest_in_list(sequences_a,sequences_b,identicalities)
        average = self.average_identicalitys(sequences_a,sequences_b,identicalities)

        self.model = self.word2vec([self.verbs_docb])
        doc_a_sum = np.zeros(self.model.vector_size)
        for i in self.verbs_doca:
            if i in list(self.model.key_to_index.keys()):
                doc_a_sum += self.model[i]

        doc_b_sum = np.zeros(self.model.vector_size)
        for i in self.verbs_docb:
            if i in list(self.model.key_to_index.keys()):
                doc_b_sum += self.model[i]

        print()
        print(f'Order-based Identicality : {str(round(average, 3))}')
        #how many verbs they have in common / how many verbs in total between them
        identicality = len(set(self.verbs_doca).intersection(set(self.verbs_docb))) / len(set(self.verbs_doca).union(set(self.verbs_docb)))
        print(f'Jaccard Identicality: #intersecting verbs / #total verbs: {identicality}')

        distance = self.model.wmdistance(self.verbs_doca, self.verbs_docb)
        print("WMD between list1 and list2:", distance)

        print(f'Cosine similarity : {np.dot(doc_a_sum, doc_b_sum)/(np.linalg.norm(doc_a_sum)* np.linalg.norm(doc_b_sum))}')

        #identicalitydistance = self.levenshtein_distance(self.verbs_doca, self.verbs_docb)
        #print(f'Levenshtein distance: {identicalitydistance}')

        #qwdistance = self.compute_distance(self.verbs_doca, self.verbs_docb)
        #print(f'QW distance: {qwdistance}')

        self.logo1 = tkinter.StringVar(value=f'Jaccard Identicality : {str(round(identicality, 3))}')
        self.logo_lab = customtkinter.CTkLabel(master = self,textvariable=self.logo1 , text='', font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_lab.grid(row=1, column=1, padx=(20,20), pady=(00, 10))
        self.logo1 = tkinter.StringVar(value=f'Order-based Identicality: {str(round(average, 3))}')
        self.logo_lab = customtkinter.CTkLabel(master = self,textvariable=self.logo1 , text='', font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_lab.grid(row=1, column=1, padx=(20,20), pady=(60, 10))

        self.logo2 = tkinter.StringVar(value=f"Word Mover's Distance : {str(round(distance, 3))}")
        self.logo_labe3 = customtkinter.CTkLabel(master = self,textvariable=self.logo2 , text='', font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_labe3.grid(row=1, column=2, padx=(20,20), pady=(60, 10))

        self.logo = tkinter.StringVar(value=f'Cosine Similarity : {str(round(np.dot(doc_a_sum, doc_b_sum)/(np.linalg.norm(doc_a_sum)* np.linalg.norm(doc_b_sum)), 3))}')
        self.logo_label2 = customtkinter.CTkLabel(master = self,textvariable=self.logo ,wraplength=300, text='',font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label2.grid(row=1, column=2, padx=(20,20), pady=(00, 10))

    def n_highest_in_list(self, sequence_a, sequence_b, sim):   # returns top 10 most similar vectors
        final_list = []
        list = sim.copy()
        for i in range(10):
            max1 = 0
            for j in range(len(list)):
                if list[j] > max1:
                    max1 = list[j];
            if max1 in list:
                list.remove(max1);
            final_list.append(max1)
        return(final_list)

    def average_identicalitys(self, sequence_a, sequence_b, sim): # returns the average of the vectors
        final_list = []
        list = sim.copy()
        sum = 0
        for i in range(len(list)):
            sum = sum + list[i]
        return(sum / len(list))

    def adjacent_identicality(self, sequences_a, sequences_b): # computes an identicality score
        adjacent_words_a = []
        adjacent_words_b = []
        similaritys = []
        for sequence_a in sequences_a:
            for sequence_b in sequences_b:
                vocab = {}
                i = 0
                for word in sequence_a:
                    if word not in vocab:
                        vocab[word] = i
                        i += 1
                for word in sequence_b:
                    if word not in vocab:
                        vocab[word] = i
                        i += 1

                a = np.zeros(len(vocab))
                b = np.zeros(len(vocab))
                for word in sequence_a:
                    index = vocab[word]
                    a[index] += 1
                for word in sequence_b:
                    index = vocab[word]
                    b[index] += 1

                cosine_sim = np.dot(a, b)/(np.linalg.norm(a)* np.linalg.norm(b))

                adjacent_words_a.append(sequence_a)
                adjacent_words_b.append(sequence_b)
                similaritys.append(cosine_sim)
        return(adjacent_words_a,adjacent_words_b,similaritys)

    def return_adjacent_words(self, word_tokenized_text, num):  # returns a list of lists of verb sequences
        list = []
        # Input format: ['fostering', 'achieving', 'promotes', 'communicate', 'work', 'achieve', ...]
        for i, word in enumerate(word_tokenized_text):
            if not i > len(word_tokenized_text)-num:
                sequence = []
                for j in range(num):
                    sequence.append(word_tokenized_text[i+j])
                list.append(sequence)
        # Output format: [['fostering', 'achieving', 'promotes'], ['achieving', 'promotes', 'communicate'], ['promotes', 'communicate'], ...]
        return(list)

    def identical_sequences(self, sequences_a, sequences_b): # returns identical verb sequences
        exact_matches = []
        for sequence_a in sequences_a:
            for sequence_b in sequences_b:
                if sequence_a == sequence_b:
                    exact_matches.append(sequence_a)
        return(exact_matches)

    def word2vec(self, word_tokenized_text):    # mapping words to vectors with word2vec, downloading model
        if os.path.exists('gensim_model.d2v'):
                model = KeyedVectors.load("gensim_model.d2v")
        else:
            print('Downloading Model...')
            model = gensim.downloader.load('glove-wiki-gigaword-300')
            model.save('gensim_model.d2v')
        #self.word2vec_testing(word_tokenized_text, model)
        return(model)

    def word2vec_testing(self, word_tokenized_text, model):    # testing word2vec
        vector_list = []
        word_a = 'king'
        word_b = 'queen'
        word_c = 'table'

        print(f"\nCosine similarity - '{word_a}' and '{word_b}' - : {model.similarity(word_a, word_b)}\n")
        print(f"\nCosine similarity - '{word_a}' and '{word_b}' - : {model.similarity(word_a, word_b)}\n")
        print(f"\nCosine similarity - '{word_a}' and '{word_c}' - : {model.similarity(word_a, word_c)}\n")
        print(f"\nCosine similarity - '{word_b}' and '{word_c}' - : {model.similarity(word_b, word_c)}\n")

        a = model['queen']
        b = model['king']
        c = model['woman']
        d = model['man']
        e = np.subtract(b,d)
        f = np.add(e,c)     # f = king - man + woman
        woman = 'woman'
        man = 'man'
        king = 'king'
        university = 'university'
        print(f'Most similar words to = "university" - \n{model.most_similar(positive=[university], topn = 3)}\n')

        print(f'Most Similar - (Woman + King - Man) - \n{model.most_similar(positive=[woman, king], negative=[man])}\n')
        print(self.cosine_similarity1(a,f))
        print(self.cosine_similarity1(b,c))
        print(self.cosine_similarity1(a,b))
        print(self.cosine_similarity1(a,model['table']))
        return(model)

    def cosine_similarity1(self, vector1, vector2):  # returns cosine similarity of two vectors
        return(np.dot(vector1, vector2)/(np.linalg.norm(vector1)* np.linalg.norm(vector2)))

if __name__ == "__main__":
    app = Verb_Similarity()
    app.mainloop()
