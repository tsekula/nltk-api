# - *- coding: utf- 8 - *-

import sys
import os
import logging
import json
from flask import Flask, jsonify, request
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.tag.stanford import StanfordPOSTagger
st = StanfordPOSTagger('/home/ubuntu/workspace/postagger/stanford/models/english-caseless-left3words-distsim.tagger', '/home/ubuntu/workspace/postagger/stanford/stanford-postagger.jar')

app = Flask(__name__)

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
            this_tuple.append(new.strip());
            this_tuple.append(elt.label());
            paired.append(this_tuple);
        else:
            if any(ext in elt[0] for ext in ['.', ',', '!', '\'', '\"', '?', ';', ':', u'’']):      # punctuation
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
    return paired    
    


@app.route('/nlp/parse-sentences',methods=['POST','GET'])
def nlp_tokenise():

    try:
        req_json = request.get_json()
        if req_json is None:

                return jsonify( error = 'this service require A JSON request' )

        else:
                if not ('text' in req_json):
                    raise Exception('Missing mandatory paramater "text"')

        text = req_json['text']

        tokens = word_tokenize(text)

        tags = st.tag(tokens)     # Stanford POS Tagger  GOOD
        #tags = pos_tag(tokens)     # Standard NLTK Tagger Meh
        processed = get_chunk_tuples(tags)

        app.log.info(json.dumps(processed))
	    
        return (json.dumps(processed))

    except Exception as ex:
       app.log.error(type(ex))
       app.log.error(ex.args)
       app.log.error(ex)
       return jsonify(error = str(ex))

if __name__ == '__main__':

    LOG_FORMAT = "'%(asctime)s - %(name)s - %(levelname)s - %(message)s'"
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT)
    app.log = logging.getLogger(__name__)

    #app.run(host="0.0.0.0",port="8080",debug=False)
    
    app.run(host=os.environ["IP"],port=os.environ["PORT"],debug=False)