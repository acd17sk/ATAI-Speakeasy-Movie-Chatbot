from flair.data import Sentence
from flair.models import SequenceTagger
import re
import rdflib
from word_forms.word_forms import get_word_forms
import inflect



class POS_extractor:
    def __init__(self):

        print('Loading POS model...')
        self.pos_model = SequenceTagger.load('models/pos1')
        self.inflect_engine = inflect.engine()


        # noun mapper to film properties
        self.noun_mapper = {
        'cast member' : set(['actor', 'actress', 'cast']),
        'genre': set(['type', 'kind']),
        'publication date': set(['release', 'date', 'airdate', 'publication', 'launch', 'broadcast']),
        'executive producer': set(['showrunner']),
        'screenwriter': set(['scriptwriter', 'screenplay', 'teleplay', 'writer', 'script', 'scenarist', 'story']),
        'director of photography': set(['cinematographer', 'DOP', 'dop']),
        'film editor': set(['editor']),
        'production designer': set(['designer']),
        'box office': set(['box', 'office', 'funding']),
        'cost': set(['budget', 'cost']),
        'nominated for': set(['nomination', 'award', 'finalist', 'shortlist', 'selection']),
        'costume designer': set(['costume']),
        'official website' : set(['website', 'site']),
        'filming location' : set(['flocation']),
        'narrative website' : set(['nlocation']),
        'production company' : set(['company']),
        'country of origin': set(['origin', 'country'])

        }

        self.noun_film_properties = set()
        for  v in self.noun_mapper.values():
            self.noun_film_properties.update(v)





    def get_pos(self, text):
        '''
        Method to run POOS tag on input text
        returns the words and their pos tags
        '''
        sentence = Sentence(text)

        self.pos_model.predict(sentence)

        pos_words = []
        pos_tags = []
        for entity in sentence:
            pos_words.append(entity.text)
            pos_tags.append(entity.get_labels('pos')[0].value)

        print('POS')
        print(pos_words)
        print(pos_tags)
        print()

        pop = list(zip(pos_words, pos_tags))

        return pop

    def _pos_to_word_index(self, pos_tags, Osent):
        res = {}
        for i in pos_tags:
            w = i[0]
            p = i[1]
            try:
                if re.search(w.lower(), Osent.lower()):
                    if res.get(p):
                        res[p].append(w)
                    else:
                        res[p] = [w]
            except:
                continue

        return res



    def _getRelation_URI_ID(self, graph, noun, WDT):
        '''
        recieves nouns (relations) and converts them to their URI ID
        '''

        query = f'''
            prefix wdt: <http://www.wikidata.org/prop/direct/>
            prefix wd: <http://www.wikidata.org/entity/>

            SELECT ?res
            WHERE{{
                ?res rdfs:label "{noun}"@en.
                }}'''
        URIs = list(filter(lambda x: WDT in x, [x[0] for x in list(graph.query(query))]))

        res = []
        for uri in URIs:

            if WDT in uri:
                relId = re.match("{}(.*)".format(WDT), uri)[1]
                res.append(relId)
        return res


    def get_relations(self, pos_tags, Owords, graph, WDT, film_properties):
        '''
        Method to retrieve the relations of the Input
        Pattern matching is performed for relations that consist of multiple words
        Verbs are converted to nouns and nouns are used to find a relation
        Plural form of nouns are singularized
        Only nouns that are associated with relational film properties are considered
        '''

        properties = set()
        properties.update(self.noun_film_properties)
        properties.update(film_properties)

        Osent = ' '.join(Owords)

        # pos tags mapping on 'Other' words
        pos_text_dict = self._pos_to_word_index(pos_tags, Osent)
        print(pos_text_dict)

        res = []

        # for filming location property
        fil = [
            re.search('where', Osent.lower()) and re.search('film', Osent.lower()),
            re.search('location', Osent.lower()) and re.search('film', Osent.lower()),
            re.search('place', Osent.lower()) and  re.search('film', Osent.lower()),
            re.search('shooting', Osent.lower()) and re.search('location', Osent.lower()),
            re.search('shot in', Osent.lower()),
            re.search('filmed in', Osent.lower())
        ]

        if any(fil):
            res.append({'relation':'filming location', 'ids': ['P915']})

        #for narrative location property
        narr = [
            re.search('where', Osent.lower()) and re.search('narrat', Osent.lower()),
            re.search('where', Osent.lower()) and re.search('set', Osent.lower()),
            re.search('where', Osent.lower()) and re.search('takes place', Osent.lower()),
            re.search('place', Osent.lower()) and re.search('set', Osent.lower()),
            re.search('location', Osent.lower()) and re.search('set', Osent.lower()),
            re.search('location', Osent.lower()) and re.search('narrat', Osent.lower()),
            re.search('set', Osent.lower()) and re.search('work', Osent.lower()),

        ]
        if any(narr):
            res.append({'relation':'narrative location', 'ids': ['P840']})

        # for MPA rating property
        if re.search('MPA', Osent):
            res.append({'relation':'MPAA rating', 'ids': ['P1657']})


        picture = [re.search('look like', Osent.lower()),
                   re.search('looks like', Osent.lower()),
                   re.search('picture', Osent.lower()),
                   re.search('poster', Osent.lower())]

        # for image property
        if any(picture):
            res.append({'relation': 'IMDb ID', 'ids': ['P345']})

        # for movie recommendation
        recom = [re.search('recommend', Osent.lower()),
                 re.search('recommendation', Osent.lower()),
                 re.search('suggest', Osent.lower()),
                 re.search('suggestion', Osent.lower())]

        if any(recom):
            res.append({'relation': 'recommendation', 'ids': []})

        nouns = []

        # convert verb to noun
        # convert plural noun to singular noun
        # each word, check if noun is in film properties
        for pos in pos_text_dict.keys():

            if pos[:2] == 'VB':
                for w in pos_text_dict[pos]:
                    noun_conversions = get_word_forms(w.lower())['n']
                    matching_conversions = set(properties).intersection(noun_conversions)
                    for noun in matching_conversions:
                        if noun in film_properties:
                            nouns.append(noun)
                        elif noun in self.noun_film_properties:
                            for k, v in self.noun_mapper.items():
                                if noun in v:
                                    nouns.append(k)
                                    break
            elif pos[:2] == 'NN':

                for w in pos_text_dict[pos]:
                    #check if plural
                    if pos[-1] == 'S':
                        noun = self.inflect_engine.singular_noun(w.lower())
                    else:
                        noun = w.lower()

                    if noun in film_properties:
                        nouns.append(noun)
                    elif noun in self.noun_film_properties:
                        for k, v in self.noun_mapper.items():
                            if noun in v:
                                nouns.append(k)
                                break



        for noun in set(nouns):
            res.append({'relation': noun, 'ids': self._getRelation_URI_ID(graph, noun, WDT)})

        return res
