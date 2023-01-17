import time
import atexit
import getpass
import requests  # install the package via "pip install requests"
from collections import defaultdict
import rdflib
import pandas as pd
import json
import csv
import re

from pos_extraction import *
from ner_extraction import *
from intent_decider import *

# url of the speakeasy server
url = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 3


class StefosBot:
    def __init__(self, username, password):
        self.agent_details = self.login(username, password)
        self.session_token = self.agent_details['sessionToken']
        self.chat_state = defaultdict(lambda: {'messages': defaultdict(dict), 'initiated': False, 'my_alias': None})

        self.ner_extractor = NER_extractor()
        self.pos_extractor = POS_extractor()
        self.intent_decider = IntentionDecider()


        # URI IDS for PERson and Movies (MISC) e.g. film, animated film etc...
        self.category2URIID = {
            'PER': {'ids':['Q33999', 'Q10800557', 'Q2526255', 'Q2405480', 'Q28389', 'Q1053574', 'Q47541952', 'Q222344', 'Q7042855', 'Q2962070',  'Q1323191'], 'cat': 'P106'},
            'MISC': {'ids':['Q11424', 'Q20650540', 'Q29168811', 'Q24862', 'Q24865', 'Q24869'], 'cat': 'P31'},

        }

        #genres and their synonyms that will be used for matching
        self.genre_dict = {
            'drama film': {'words': ['drama'], 'id': 'Q130232'},
            'documentary film': {'words': ['documentary', 'factual'], 'id': 'Q93204'},
            'comedy film': {'words': ['funny', 'comedy', 'comedic'], 'id': 'Q157443'},
            'crime film': {'words': ['crime'], 'id': 'Q959790'},
            'action film': {'words': ['action'], 'id': 'Q188473'},
            'romance film': {'words': ['romantic', 'romance'], 'id': 'Q1054574'},
            'horror film': {'words': ['horror', 'scary'], 'id': 'Q200092'},
            'adventure film': {'words': ['adventure'], 'id': 'Q319221'},
            'neo-noir': {'words': ['neo-noir', 'new-black', 'neo noir', 'new black'], 'id': 'Q2421031'},
            'science fiction': {'words': ['science fiction', 'SF', 'scifi', 'sci Fi', 'fantasy'
                                          'sci-Fi', 'science-fiction', 'sci fi', 'sciencefiction'], 'id': 'Q24925'},
            'thriller film': {'words': ['thriller', 'suspense'], 'id': 'Q2484376'},
            'animated film': {'words': ['animated', 'animation', 'cartoon'], 'id': 'Q202866'},

        }

        # film properties were retrieved from wikidata itself
        # no code exists for creating this
        print('Loading film properties...')
        self.film_properties = set(pd.read_csv('Data/Film Properties.csv')['res'])

        RDFS = rdflib.namespace.RDFS
        self.WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
        self.WD = rdflib.Namespace('http://www.wikidata.org/entity/')

        print('Loading Graph...')
        self.graph = rdflib.Graph().parse('Data/14_graph.nt', format='turtle')

        print('Loading Embeddings...')
        self.entity_emb = np.load('Data/ddis-graph-embeddings/entity_embeds.npy')
        self.relation_emb = np.load('Data/ddis-graph-embeddings/relation_embeds.npy')

        # load the dictionaries
        with open('Data/ddis-graph-embeddings/entity_ids.del', 'r') as ifile:
            self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            self.id2ent = {v: k for k, v in self.ent2id.items()}
        with open('Data/ddis-graph-embeddings/relation_ids.del', 'r') as ifile:
            self.rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
            self.id2rel = {v: k for k, v in self.rel2id.items()}

        self.ent2lbl = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(RDFS.label)}
        self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}

        print('Loading images...')
        with open("Data/images.json", "r") as f:
            self.images = json.load(f)

        print('All Set up and ready to roll!!')

        atexit.register(self.logout)



    def create_response(self, message):
        '''
        method that recieves the message and responds with an answer
        takes the message and takes it through the whole pipeline
        '''

        # various remappings for certain relations that can interfere with other
        # relations that consist of the same words
        message = re.sub('executive producer', 'showrunner', message)
        message = re.sub('production designer', 'designer', message)
        message = re.sub('costume designer', 'costume', message)
        message = re.sub('box office', 'box', message)
        message = re.sub('narrative location', 'nlocation', message)
        message = re.sub('filming location', 'flocation', message)
        message = re.sub('production company', 'company', message)

        # extract NER
        entities,  Owords = self.ner_extractor.get_entities(message)

        # extract Relations through POS
        pos = self.pos_extractor.get_pos(message)

        # Get entities URI IDs
        ent = self.ner_extractor.getEntities_URIIDs(self.graph, entities, self.WDT, self.WD, self.category2URIID)

        # Get Relations URI IDs
        rel = self.pos_extractor.get_relations(pos, Owords, self.graph, self.WDT, self.film_properties)

        print()
        print(ent)
        print(rel)
        print()

        # Pass entities and relations to decide answer
        # considering relations as intentions
        final_answer = self.intent_decider.decider(self.graph, self.WD, self.WDT,
                                                   ent, rel, Owords, self.entity_emb,
                                                   self.ent2id, self.ent2lbl,
                                                   self.id2ent, self.relation_emb,
                                                   self.rel2id, self.images,
                                                   self.genre_dict, self.category2URIID)

        # if no answer found
        if final_answer == '':
            return ("Sorry mate, couldn't get you or an answer. " +
                    "In case it is my fault, " +
                    "I just respond to stuff about movies, SO NOT REALLY ME FAULT. " +
                    "By the way, make sure to check for any spelling mistakes " +
                    "because I forgot to learn magic " +
                    "sorceries for spell correction. Better luck next time!")
        else:
            return final_answer




    def listen(self):
        while True:
            # check for all chatrooms
            current_rooms = self.check_rooms(session_token=self.session_token)['rooms']
            for room in current_rooms:
                # ignore finished conversations
                if room['remainingTime'] > 0:
                    room_id = room['uid']
                    if not self.chat_state[room_id]['initiated']:
                        # send a welcome message and get the alias of the agent in the chatroom
                        greeting_message = ("Hi, I am Stefos bot, a rebellious teenager that responds with minimum effort. "
                                            "I don't respond to thank yous or greetings. Just tell me what you want about "
                                            "MOVIES and MOVIES only. Another thing, I'm Case sensitive and i don't respond "
                                            "to spelling mistakes.")
                        self.post_message(room_id=room_id, session_token=self.session_token, message=greeting_message)
                        self.chat_state[room_id]['initiated'] = True
                        self.chat_state[room_id]['my_alias'] = room['alias']

                    # check for all messages
                    all_messages = self.check_room_state(room_id=room_id, since=0, session_token=self.session_token)['messages']

                    # you can also use ["reactions"] to get the reactions of the messages: STAR, THUMBS_UP, THUMBS_DOWN

                    for message in all_messages:
                        if message['authorAlias'] != self.chat_state[room_id]['my_alias']:

                            # check if the message is new
                            if message['ordinal'] not in self.chat_state[room_id]['messages']:
                                self.chat_state[room_id]['messages'][message['ordinal']] = message
                                print('\t- Chatroom {} - new message #{}: \'{}\' - {}'.format(room_id, message['ordinal'], message['message'], self.get_time()))

                                self.post_message(room_id=room_id, session_token=self.session_token, message="...".encode('utf-8'))

                                response = self.create_response(message['message'])

                                self.post_message(room_id=room_id, session_token=self.session_token, message=response.encode('utf-8'))
            time.sleep(listen_freq)

    def login(self, username: str, password: str):
        agent_details = requests.post(url=url + "/api/login", json={"username": username, "password": password}).json()
        print('- User {} successfully logged in with session \'{}\'!'.format(agent_details['userDetails']['username'], agent_details['sessionToken']))
        return agent_details

    def check_rooms(self, session_token: str):
        return requests.get(url=url + "/api/rooms", params={"session": session_token}).json()

    def check_room_state(self, room_id: str, since: int, session_token: str):
        return requests.get(url=url + "/api/room/{}/{}".format(room_id, since), params={"roomId": room_id, "since": since, "session": session_token}).json()

    def post_message(self, room_id: str, session_token: str, message: str):
        tmp_des = requests.post(url=url + "/api/room/{}".format(room_id),
                                params={"roomId": room_id, "session": session_token}, data=message).json()
        if tmp_des['description'] != 'Message received':
            print('\t\t Error: failed to post message: {}'.format(message))

    def get_time(self):
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

    def logout(self):
        if requests.get(url=url + "/api/logout", params={"session": self.session_token}).json()['description'] == 'Logged out':
            print('- Session \'{}\' successfully logged out!'.format(self.session_token))


if __name__ == '__main__':
    username = 'stefanos.konstantinou_bot'
    print('stefanos.konstantinou_bot')
    print('05HH_vef7-3j9Q')
    password = getpass.getpass('Password of the Stefos bot:')
    demobot = StefosBot(username, password)
    demobot.listen()
