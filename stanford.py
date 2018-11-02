import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.tag.stanford import StanfordPOSTagger
#english_nertagger = NERTagger('classifiers/english.all.3class.distsim.crf.ser.gz', 'stanford-ner.jar')
sn = StanfordNERTagger('/home/ubuntu/workspace/nerftagger/stanford/classifiers/english.nowiki.3class.distsim.crf.ser.gz', '/home/ubuntu/workspace/nerftagger/stanford/stanford-ner.jar')
st = StanfordPOSTagger('/home/ubuntu/workspace/postagger/stanford/models/english-caseless-left3words-distsim.tagger', '/home/ubuntu/workspace/postagger/stanford/stanford-postagger.jar')


source = 'They know President Trump has vowed to block them yet they press on. "We prefer to die on the American border than die in Honduras from hunger," one said.  One New Zealand lawmaker has accused another of trying to disguise a campaign contribution from a businessman with ties to the Chinese Communist Party.'
source = "Cornell Cuts Ties With Chinese School After Crackdown on Students"

tags = nltk.word_tokenize(source)


#source = source.split()
#mine = ['Trump', 'American', 'Honduras', 'New Zealand', 'Chinese Communist Party']




print(sn.tag(tags))
print("########################################")
print("########################################")
print(st.tag(tags))
print("########################################")
print("########################################")
print(nltk.pos_tag(tags))

