import json
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.tag.stanford import StanfordPOSTagger
st = StanfordPOSTagger('/home/ubuntu/workspace/postagger/stanford/models/english-caseless-left3words-distsim.tagger', '/home/ubuntu/workspace/postagger/stanford/stanford-postagger.jar')


def get_new_chunks(tags):
    chunks = ne_chunk(tags)
    simple = []
    paired = []
    fulltextaswords = ""
    fulltextastags = ""
    for elt in chunks:
        if isinstance(elt, Tree):
            new = ""
            for tag in elt:
                new = new + " " + tag[0]
            fulltextaswords = fulltextaswords + " " + new.strip()
            fulltextastags = fulltextastags + " {{" + elt.label() + "}}"
        else:
            if any(ext in elt[0] for ext in ['.', ',', '!', '\'', '\"', '?']):
                fulltextaswords = fulltextaswords + elt[0]
                fulltextastags = fulltextastags + elt[0]
            else:
                fulltextaswords = fulltextaswords + " " + elt[0]
                fulltextastags = fulltextastags + " {{" + elt[1] + "}}"
    paired.append([fulltextaswords.strip(), fulltextastags.strip()])
    return paired
    
# return     
def get_chunk_tuples(tags):
    chunks = ne_chunk(tags)
    simple = []
    paired = []
    fulltextaswords = ""
    fulltextastags = ""
    for elt in chunks:
        this_tuple = []
        if isinstance(elt, Tree):           # found named entities
            new = ""
            for tag in elt:
                new = new + " " + tag[0]
            #fulltextaswords = fulltextaswords + " " + new.strip()
            #fulltextastags = fulltextastags + " {{" + elt.label() + "}}"
            this_tuple.append(new.strip());
            this_tuple.append(elt.label());
            paired.append(this_tuple);
        else:
            if any(ext in elt[0] for ext in ['.', ',', '!', '\'', '\"', '?', ';', ':']):      # punctuation
                fulltextaswords = fulltextaswords + elt[0]
                fulltextastags = fulltextastags + "{{PUNCT}}"
                this_tuple.append(elt[0]);
                this_tuple.append("PUNCT");
                paired.append(this_tuple);
            else:       # regular words and POS tags
                fulltextaswords = fulltextaswords + " " + elt[0]
                fulltextastags = fulltextastags + " {{" + elt[1] + "}}"
                this_tuple.append(elt[0]);
                this_tuple.append(elt[1]);
                paired.append(this_tuple);
    
    #paired.append([fulltextaswords.strip(), fulltextastags.strip()])
    return paired    
    
def use_new_method (sentence):
    for sent in nltk.sent_tokenize(sentence):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                print(chunk.label(), ' '.join(c[0] for c in chunk))    

my_sent = "Khashoggi Killing Overshadows Saudis' Grand Economic Ambitions. Google Retreats from Berlin Plan Opposed by Local Groups. Why We Are Publishing Haunting Photos of Emaciated Yemeni Children."
my_sent = 'They know President Trump has vowed to block them yet they press on. "We prefer to die on the American border than die in Honduras from hunger," one said.  One New Zealand lawmaker has accused another of trying to disguise a campaign contribution from a businessman with ties to the Chinese Communist Party.'

#my_sent = "Premier Li Keqiang of China, right, and Prime Minister Shinzo Abe of Japan attended a signing ceremony at the Great Hall of the People in Beijing on Friday."
#NEarry = get_continuous_chunks(my_sent)

tokens = word_tokenize(my_sent)

# Use nltk POStagger
tags = pos_tag(tokens)
#NEarry = get_new_chunks(tags)
NEarry = get_chunk_tuples(tags)
print(json.JSONEncoder().encode(NEarry))
print("########################################")
print("########################################")
# Use Stanford POStagger <--- WINNING COMBO!
tags = st.tag(tokens)
NEarry = get_chunk_tuples(tags)
#NEarry = get_new_chunks(tags)
print(json.JSONEncoder().encode(NEarry))
print("########################################")
print("########################################")
print(tags)


