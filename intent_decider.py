import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import inflect
import re
import json


class IntentionDecider():

    def __init__(self):

        self.inflect_engine = inflect.engine()

        print('Loading Clean crowd data and rates...')
        self.clean_crowd_pd = pd.read_csv('Data/crowd_data/clean_crowd_data.csv')

        with open("Data/crowd_data/rates.json", "r") as f:
            self.rates = json.load(f)


    def crowdsource_search(self, ent, rel):
        '''
        crowdsource searching
        if answer is deemed correct, the answer, the status (Correct/INCORRECT), approval rate is returned
        if status is deemed incorrect, and there exists a fixing value, the fixing value is returned as an answer and status as CORRECT
        if no answer return None
        '''

        res = self.clean_crowd_pd[(self.clean_crowd_pd['Input1ID'] == f"wd:{ent}") & (self.clean_crowd_pd['Input2ID'] == f"wdt:{rel}")]

        # if there exists an answer
        if bool(list(res.HITId)):

            #aggregate answers from crowd
            agg_dict = {}
            for k,v in list(zip(res.Input3ID, res.AnswerLabel)):
                if agg_dict.get(k):
                    agg_dict[k].append(v)
                else:
                    agg_dict[k] = [v]

            agg_res = {}
            for k, v in agg_dict.items():
                agg_res[k] = max(set(v), key=v.count)

            hitid = list(set(res.HITId))[0]
            rate_ans = self.rates[str(hitid)]
            ans =  str(list(agg_res.keys())[0]).strip('wd:')
            state = list(agg_res.values())[0]

            # filter nan types
            filter_fix_val = set(filter(lambda x: isinstance(x, str), list(res.FixValue)))

            if list(agg_res.values())[0] == 'INCORRECT' and filter_fix_val:
                ans = str(list(filter_fix_val)[0]).strip('wd:')
                state = 'CORRECT'

            return ans, rate_ans, state
        else:
            return None, None, None


    def embeddings(self, WD, WDT, entity_emb, ent2id, ent2lbl, id2ent, relation_emb, rel2id, ent, rel, num_ret):
        '''
        embedding query
        gets entity and relation and searches through the embedddings
        '''

        try:

            #  entity
            head = entity_emb[ent2id[WD[ent]]]
            # relation
            pred = relation_emb[rel2id[WDT[rel]]]
            # add vectors according to TransE scoring function.
            lhs = head + pred
            # compute distance to *any* entity
            dist = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)
            # find most plausible entities
            most_likely = dist.argsort()


            return [{'label':ent2lbl[id2ent[idx]], 'Score': dist[idx]} for idx in most_likely[:num_ret]]
        except:
            return []


    def movie_recom_movie(self, graph, ent, WD, WDT, entity_emb, ent2id, ent2lbl, id2ent, relation_emb, rel2id, cat2id):
        '''
        Recommend movies based on input movies
        returns 3 most suitable recommendations
        '''

        m_ids = [p['id'] for p in ent['MISC']]

        # averaging input movies embeddings
        mean_emb = np.mean([entity_emb[ent2id[WD[idd]]] for idd in m_ids], 0)

        dist = pairwise_distances(mean_emb.reshape(1, -1), entity_emb).reshape(-1)
        # find most plausible entities to the average of the input movies embeddings
        most_likely = dist.argsort()

        res = []
        for idx in most_likely[:15+len(m_ids)]:
            #check if the recommendation is a movie
            # avoid return other types of entities besides movies
            g = list(filter(lambda x: str(x).split('/')[-1] in cat2id['MISC']['ids'],
                            list(graph.objects(id2ent[idx], WDT.P31))))

            if g and id2ent[idx][len(WD):] not in m_ids and ent2lbl.get(id2ent[idx]):
                res.append({'ent':id2ent[idx][len(WD):],
                            'label':ent2lbl[id2ent[idx]],
                            'Score': dist[idx]})

        return res[:3]

    def movie_recom_genre(self, graph, genre):
        '''
        recommend a random movie based on genre
        '''

        query = f'''
        prefix wdt: <http://www.wikidata.org/prop/direct/>
        prefix wd: <http://www.wikidata.org/entity/>

        SELECT ?obj ?lbl
        WHERE {{
          ?obj wdt:P136 wd:{genre} .
          ?obj rdfs:label ?lbl .
          FILTER(LANG(?lbl) = "en").
          }}
        ORDER BY RAND() LIMIT 1

        '''
        qres = list(graph.query(query))[0]

        res = {'ent':str(qres[0]).split('/')[-1], 'label': str(qres[1])}

        return res if res else None


    def movie_recom_actor_genre(self, graph, actor, genre):
        '''
        recommend a randoom movie based on an actor input and a genre
        '''

        query = f'''
        prefix wdt: <http://www.wikidata.org/prop/direct/>
        prefix wd: <http://www.wikidata.org/entity/>

        SELECT ?obj ?lbl
        WHERE {{
          ?obj wdt:P136 wd:{genre} .
          ?obj rdfs:label ?lbl .
          FILTER(LANG(?lbl) = "en").
          }}
        ORDER BY RAND() LIMIT 1

        '''
        qres = list(graph.query(query))[0]

        res = {'ent':str(qres[0]).split('/')[-1], 'label': str(qres[1])}

        return res if res else None

    def movie_recom_actor(self, graph, actor):
        '''
        reccommend a random movie based on an input actor
        '''

        query = f'''
        prefix wdt: <http://www.wikidata.org/prop/direct/>
        prefix wd: <http://www.wikidata.org/entity/>

        SELECT ?obj ?lbl
        WHERE {{
          ?obj wdt:P161 wd:{actor} .
          ?obj rdfs:label ?lbl .
          FILTER(LANG(?lbl) = "en").
        }}
        ORDER BY RAND() LIMIT 1

        '''
        qres = list(graph.query(query))[0]

        res = {'ent':str(qres[0]).split('/')[-1], 'label': str(qres[1])}

        return res if res else None

    def get_movie_year(self, graph, ent):
        '''
        return the year of release of a movie
        '''

        query = f'''
            prefix wdt: <http://www.wikidata.org/prop/direct/>
            prefix wd: <http://www.wikidata.org/entity/>

            SELECT ?obj
            WHERE {{
              wd:{ent} wdt:P577  ?obj.
            }} ORDER BY ASC(?obj)
             LIMIT 1

        '''
        res = list(graph.query(query))
        return f" ({str(max([int(i) for i in res[0][0].split('-')]))})" if res else ''

    def _EntityURI_to_ID(self, URI_LIST, WD='http://www.wikidata.org/entity/'):
        '''
        converts an entity name to a URI ID
        '''

        res = []
        for uri in URI_LIST:

            if WD in uri:
                res.append(re.match("{}(.*)".format(WD), uri)[1])

        return res

    def embeddings_search(self, graph, WD, WDT, entity_emb, ent2id, ent2lbl, id2ent, relation_emb, rel2id, ent, rel, rid, n_to_retr):
        '''
        Call the embedddings search method (embeddings()) to retrieve info on embeddings
        finds 1 movie more than the knowledge graph search result
        returns a response in string for output
        '''


        ent_print = f"{ent['entity']}{self.get_movie_year(graph, ent['id'])}"

        emb_res = self.embeddings(WD, WDT, entity_emb, ent2id, ent2lbl, id2ent, relation_emb, rel2id, ent['id'], rid, n_to_retr)

        if emb_res:
            if len(emb_res) > 1:
                labels = ', '.join([e['label'] for e in emb_res])
                scores = ', '.join([str(e['Score']) for e in emb_res])
                return f" The Embeddings suggest that the {self.inflect_engine.plural_noun(rel)} of {ent_print} could be {labels} (Scores: {scores})."
            else:
                return f" The Embeddings suggest that the {rel} of {ent_print} could be {emb_res[0]['label']} (Score: {emb_res[0]['Score']})."
        else:
            return ''

    def get_uri2label(self, graph, URI_LIST):
        '''
        method that recieves a URI list and returns the name list of the uris
        '''
        res = []
        for r in URI_LIST:

            query = f'''
                    prefix wdt: <http://www.wikidata.org/prop/direct/>
                    prefix wd: <http://www.wikidata.org/entity/>

                    SELECT ?res
                    WHERE
                    {{
                    wd:{r} rdfs:label ?res .
                    FILTER(LANG(?res) = "en").
                    }}'''
            res.extend([str(i[0]) for i in set(graph.query(query))])
        return res

    def knowledge_graph_search(self, graph, g, ent, rel, rid):
        '''
        Method to retrieve results based on knowledge graph
        Crowd sourcing is performed if result conists of oone output
        checks if KG result and CS result match and checks the approval or
        correctness from CS and responds accordingly
        also returns the amount of KG results (will be used for embeddings search)
        '''


        ent_print = f"{ent['entity']}{self.get_movie_year(graph, ent['id'])}"

        kg_result = self._EntityURI_to_ID(g)

        # crowd source for 1 result or no result on KG
        cs_ans, cs_rate, cs_state = None, None, None
        if len(kg_result) < 2:
            cs_ans, cs_rate, cs_state = self.crowdsource_search(ent['id'], rid)
            if cs_ans:
                cs_ans = self.get_uri2label(graph, [cs_ans])[0]


        kg_res = self.get_uri2label(graph, kg_result)


        #if both methods retrieved a result
        if kg_res and cs_ans:
            #if both methods agree on the result
            if kg_res[0] == cs_ans:
                #check the validity
                if cs_state == 'CORRECT' and cs_rate > 0.5:
                    return f" The {rel} of {ent_print} is {cs_ans} (Crowd Approval rate: {cs_rate}).", len(kg_res)
                else:
                    return f" I think the {rel} of {ent_print} is {kg_res[0]} but im not really sure (Crowd Approval rate: {cs_rate}).", len(kg_res)
            else:
                if cs_state == 'CORRECT' and cs_rate > 0.5:
                    return f" The {rel} of {ent_print} is {cs_ans} (Crowd Approval rate: {cs_rate}).", len(kg_res)
                else:
                    return f" I think the {rel} of {ent_print} is {kg_res[0]} but im not really sure, but I know it's not {cs_ans} (Crowd Approval rate: {cs_rate}).", len(kg_res)
        # if only CS result exist
        elif cs_ans:
            if cs_state == 'CORRECT' and cs_rate > 0.5:
                return f" The {rel} of {ent_print} is {cs_ans} (Crowd Approval rate: {cs_rate}).", 0
            else:
                return f" I don't really know the {rel} of {ent_print}. But {cs_ans} is not (Crowd Approval rate: {cs_rate}).", 0
        # if only KG result
        elif kg_res:
            if len(kg_res) > 1:
                return f" The {self.inflect_engine.plural_noun(rel)} of {ent_print} are {', '.join(kg_res)}.", len(kg_res)
            else:
                return f" The {rel} of {ent_print} is {kg_res[0]}.", len(kg_res)
        # if no result at all
        else:
            return '', 0

    def particular_relation_search(self, g, ent, rel, rid):
        '''
        KG and CS search just like knowledge_graph_search()
        But on particular relations since these relations retrieve values instead of entities
        '''

        kg_res = str(g[0]) if g else None

        cs_ans, cs_rate, cs_state = self.crowdsource_search(ent['id'], rid)

        if kg_res and cs_ans:
            if kg_res == cs_ans:
                if cs_state == 'CORRECT' and cs_rate > 0.5:
                    return f" The {rel} of {ent['entity']} is {kg_res} (Crowd Approval rate: {cs_rate})."
                else:
                    return f" I think the {rel} of {ent['entity']} is {kg_res} but im not really sure (Crowd Approval rate: {cs_rate})."
            else:
                if cs_state == 'CORRECT' and cs_rate > 0.5:
                    return f" The {rel} of {ent['entity']} is {cs_ans} (Crowd Approval rate: {cs_rate})."
                else:
                    return f" I think the {rel} of {ent['entity']} is {kg_res} but im not really sure (Crowd Approval rate: {cs_rate})."
        elif cs_ans:
            if cs_state == 'CORRECT' and cs_rate > 0.5:
                return f" The {rel} of {ent['entity']} is {cs_ans} (Crowd Approval rate: {cs_rate})."
            else:
                return f" I don't really know the {rel} of {ent['entity']}. But {cs_ans} is not (Crowd Approval rate: {cs_rate})."
        elif kg_res:
            return f" The {rel} of {ent['entity']} is {kg_res}."
        else:
            return ''


    def decider(self, graph, WD, WDT, ent, rel, Owords, entity_emb, ent2id, ent2lbl, id2ent, relation_emb, rel2id, images, genre_dict, cat2id):
        '''
        main Method for this Class
        given all information trackeed form previous classes it decides the final answer
        if no answer is retrieved returns empty string

        Considers relations to Intents

        First it checks if there is recommendation intention and creates recommendation
        if no else intention  then exits as a final answer
        Else, it uses all other relations (intents) and retrieves the answers
        and returns final answer

        '''


        final_ans = ''

        # recommendation
        if 'recommendation' in [r['relation'] for r in rel]:
            # based on movies
            if ent.get('MISC'):
                res = self.movie_recom_movie(graph, ent, WD, WDT, entity_emb, ent2id, ent2lbl, id2ent, relation_emb, rel2id, cat2id)
                ent_print_list = []
                for r in res:
                    ent_print_list.append(f"{r['label']}{self.get_movie_year(graph, r['ent'])}")
                final_ans += f" Hmm... I could recommend you {', '.join(ent_print_list)}."
            else:

                #pattern match for genres
                Osent = ' '.join(Owords)
                genre = ''
                for v in genre_dict.values():
                    words = v['words']
                    found = False
                    for w in words:
                        if re.search(w, Osent.lower()):
                            genre = v['id']
                            found = True
                            break
                    if found:
                        break

                #if PERson entity  and genre exist
                if ent.get('PER') and genre:
                    for actor in ent['PER']:
                        res = self.movie_recom_actor_genre(graph, actor['id'], genre)
                        if res:
                            ent_print = f"{res['label']}{self.get_movie_year(graph, res['ent'])}"
                            final_ans += f" Hmm... I could recommend you {ent_print}."
                # if only PER
                elif ent.get('PER'):
                    for actor in ent['PER']:
                        res = self.movie_recom_actor(graph, actor['id'])
                        if res:
                            ent_print = f"{res['label']}{self.get_movie_year(graph, res['ent'])}"
                            final_ans += f" Hmm... I could recommend you {ent_print}."
                #if only genre
                elif genre:
                    res = self.movie_recom_genre(graph, genre)
                    if res:
                        ent_print = f"{res['label']}{self.get_movie_year(graph, res['ent'])}"
                        final_ans += f" Hmm... I could recommend you {ent_print}."


        #if no other relation exists, exit with final answer
        rel = list(filter(lambda x:  x['relation'] != 'recommendation', rel))
        if not rel:
            return final_ans[1:] if final_ans != '' else final_ans

        #for every entity type (PER, MISC)
        for k, v in ent.items():
            # for every entity
            for ee in v:
                #for every relation
                for relation in rel:
                    # for every URI ID in relations
                    for rid in relation['ids']:

                        g = list(graph.objects(WD[ee['id']], WDT[rid]))

                        # Intent for image search
                        if relation['relation'] == 'IMDb ID':
                            imbds = []
                            for i in g:
                                imbds.append(str(i))
                            imdb_id = imbds[0]

                            im_id = ''
                            for image in images:

                                if k == 'MISC':
                                    if image['movie'] == [imdb_id] and image['type'] == 'poster':
                                        im_id = image['img']
                                        im_id = 'image:'+im_id.strip('.jpg')
                                        break
                                else:
                                    if image['cast'] == [imdb_id] and image['type'] != 'poster':
                                        im_id = image['img']
                                        im_id = 'image:'+im_id.strip('.jpg')
                                        break


                            if im_id:
                                final_ans += f" There you go... {im_id}"

                        # knowledge graph on these particular relations
                        elif relation['relation'] in ['publication date', 'cost', 'box office']:
                            final_ans += self.particular_relation_search(g, ee, relation['relation'], rid)


                        else:
                            #knowledge graph search
                            kg_res = self.knowledge_graph_search(graph, g, ee, relation['relation'], rid)

                            #number of KG results
                            final_ans += kg_res[0]

                            #embedding search
                            n_to_retr = kg_res[1]

                            final_ans += self.embeddings_search(graph, WD, WDT, entity_emb, ent2id, ent2lbl, id2ent, relation_emb,
                                                                rel2id, ee, relation['relation'], rid, n_to_retr+1)



        return final_ans[1:] if final_ans != '' else final_ans
