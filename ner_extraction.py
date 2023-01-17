import re
from flair.data import Sentence
from flair.models import SequenceTagger
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
import json
import numpy as np


class NER_extractor:
    def __init__(self):

        print('Loading NER models...')
        self.ner_large = SequenceTagger.load('models/ner_large')
        self.ner_base = SequenceTagger.load('models/ner_base')

        print('Loading Entity name similarity model...')
        self.ent_name_sim_model = SentenceTransformer('models/ent_name_sim/')

        print('Loading Title embeddings')
        with open("Data/titles.json", "r") as f:
            self.ent_codes = json.load(f)

        self.title_embeddings = np.load('Data/title_embeddings.npy')

        with open("Data/ent2name.json", "r") as f:
            self.ent2name = json.load(f)

        with open("Data/name2ent.json", "r") as f:
            self.name2ent = json.load(f)


    def _get_model_res(self, model, text):
        '''
        run a model to identify NER and Other words
        Only person and movies are accepted
        '''

        sentence = Sentence(text)
        model.predict(sentence)

        Owords = []
        ent_words = []
        idx = []
        for entity in sentence.get_spans('ner'):
            if entity.get_labels('ner')[0].value in ['PER', 'MISC']:
                ent_words.append(text[entity.start_position:entity.end_position])
                idx.append((entity.start_position,entity.end_position))

        if len(idx) == 1:
            Owords.append(text[0:idx[0][0]])
            Owords.append(text[idx[0][1]:])
        else:
            for i, v in enumerate(idx):
                if i == 0:
                    Owords.append(text[0:v[0]])
                    continue
                Owords.append(text[idx[i-1][1]:v[0]])
                if i == len(idx)-1:
                    Owords.append(text[v[1]:])

        return  ent_words, Owords

    def get_entities(self, text):
        '''
        Method to run NER models on input Text
        Uses 2 ner models large and base sized, for backup
        '''
        if text[-1] == '?' or text[-1] == '.':
            text = text[:-1]


        ent_words1, Owords1 = self._get_model_res(self.ner_large, text)
        ent_words2, Owords2 = self._get_model_res(self.ner_base, text)

        if len(ent_words1) > len(ent_words2) and ent_words2:
            word_group = ent_words2
            Owords = Owords2
        else:
            word_group = ent_words1
            Owords = Owords1


        print()
        print('NER')
        print(Owords)
        print(word_group)
        print()


        return  word_group, Owords if word_group else [text]



    def _EntityURI_to_ID(self, URI_LIST, WD):
        '''
        Converts a list of URIs to list of URI IDs
        '''
        res = []
        for uri in URI_LIST:

            if WD in uri:
                res.append(re.match("{}(.*)".format(WD), uri)[1])

        return res


    def _getEntity_URI_ID(self, graph, ent, WDT, WD, cat2id):
        '''
        Query search for entity names and returns URI IDs
        Also searches human or film type to entities
        '''

        # query = f'''
        #     prefix wdt: <http://www.wikidata.org/prop/direct/>
        #     prefix wd: <http://www.wikidata.org/entity/>
        #
        #     SELECT ?res
        #     WHERE{{
        #         ?res rdfs:label "{ent}"@en.
        #         }}'''
        # URI_LIST = [str(x[0]) for x in list(graph.query(query))]

        # entities_ids = self._EntityURI_to_ID( URI_LIST, WD)

        # embed for input entity
        inp_emb = self.ent_name_sim_model.encode(ent)

        # calculate nearest answer
        dist = pairwise_distances(inp_emb.reshape(1, -1),
                                  self.title_embeddings).reshape(-1)
        most_likely = dist.argsort()
        most_likely_ent = self.ent_codes[most_likely[0]]

        #check if other entities exist with the same name
        name = self.ent2name[most_likely_ent]
        entities_ids = self.name2ent[name]

        # filter non movie occupation for PERsons
        # filter non movie entities
        res = {}
        for e_id in entities_ids:
            for k, v in cat2id.items():
                g = list(graph.objects(WD[e_id], WDT[v['cat']]))

                instancesOf = self._EntityURI_to_ID(g, WD)
                if instancesOf and set(v['ids']).intersection(instancesOf):
                    if res.get(k):
                        res[k].append({'entity':name, 'id':e_id})
                    else:
                        res[k] = [{'entity':name, 'id':e_id}]



        return res

    def getEntities_URIIDs(self, graph, entities, WDT, WD, cat2id):
        '''
        Converts entity names to URI ids
        Also maps each entity to human or film type
        '''

        qres = []
        for e in entities:
            uri_res = self._getEntity_URI_ID(graph, e, WDT, WD, cat2id)
            qres.append(uri_res)

        entities_uriID = {}
        for q in qres:
            for k, v in q.items():
                if entities_uriID.get(k):
                    entities_uriID[k].extend(v)
                else:
                    entities_uriID[k] = v

        return entities_uriID
