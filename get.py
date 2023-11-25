from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field
from util.tokenizer import Tokenizer
from conf import *
ROOT = '~/Python/DATASETS/Multi30k/'
Multi30k.download(ROOT)
tokenizer = Tokenizer()
source = Field(tokenize=tokenizer.tokenize_de, init_token='<sos>', eos_token='<eos>',
                                lower=True, batch_first=True)
target = Field(tokenize=tokenizer.tokenize_en, init_token='<sos>', eos_token='<eos>',
                                lower=True, batch_first=True)
(trnset, valset, testset) = TranslationDataset.splits(   
                                      path       = ROOT,  
                                      exts       = ['.en', '.de'],   
                                      fields     = [('src', source), ('trg',target)],
                                      test       = 'test2016'
                                      )