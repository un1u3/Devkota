# data cleaning 

import re 
import argparse
from pathlib import Path
from pathlib import Path 
import sentencepiece as spm 



class CorpusBuilder:
    # cleans and prepares text dat for toekniier trainng 
    # what it does 
    #  remvoes junk char 
    # keeps only good line 


    def __init__(self, min_chars=5):
        self.min_chars = min_chars
        self.control_chars = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
        self.multiple_spaces = re.compile(r"[ \t\u00A0]+")

    def clean_line(self, text):
        # zero width space 
        text = text.replace('\u200b',"")
        # remove bom byte oreder mark k
        text = text.replace("\uffeff","")
        # remove contril characters 
        text = self.control_chars.sub("",text)
        # remove extraa spaces at start/en 
        text = text.strip()
        return text 
    

    # def build_corpus(self, input_files, output_file):
    #     # c