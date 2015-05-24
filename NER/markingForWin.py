# -*- coding: utf-8 -*-

import sys    # sys.setdefaultencoding is cancelled by site.py
reload(sys)    # to re-enable sys.setdefaultencoding()
sys.setdefaultencoding('utf-8')
def read_article(path_to_article):
    with open(path_to_article) as my_file:
        article = my_file.read()
    return article.decode('utf-8')

article = read_article(sys.argv[1])
#article = read_article('.\\AML\\1\\article')

from pprint import pprint
import json
def read_json(path_to_json):
    with open(path_to_json) as json_file:
        json_of_article = json.load(json_file)
    return json_of_article

json_of_article = read_json(sys.argv[2])
#json_of_article = read_json('2.out')

from collections import OrderedDict
def get_entities(json_of_article):
    entities = {}
    for j in json_of_article:
        if j != {}:
            s = (j['Type'])
            for pair in j['Boundaries']:
                entities[ pair[0]] = list([pair[1], entity_color[s] ])
    entities = OrderedDict(sorted(entities.items(), key=lambda t:t[0]))
    return entities

color1 ='#' + sys.argv[4][2:];
color2 ='#' + sys.argv[5][2:];
color3 ='#' + sys.argv[6][2:];




entity_color = {'Person':color1, 'PopulatedPlace':color2, 'Organisation':color3}

entities = get_entities(json_of_article)
#print(entities)

from yattag import Doc, indent

def make_html(article, entities, path_to_html):
    doc, tag, text = Doc().tagtext()

    with tag('html'):
        with tag('head'):
            with tag('meta'):
                doc.attr(charset='utf-8')
            with tag('title'):
                text('Marking of article')

        with tag('body'):
            with tag('table'):
                doc.attr(border='2')
                with tag('tr'):
                    with tag('th'):
                        doc.attr(style='padding: 10px;font-size:160%')
                        with tag('p'):
                            doc.attr(style='word-spacing:20px;font-style:italic;')
                            with tag('font'):
                                if(color2 != '#000000'):
                                    doc.attr(color=color2)
                                    text(' Organisation ')
                            with tag('font'):
                                if(color3 != '#000000'):
                                    doc.attr(color=color3)
                                    text(' PopulatedPlace ')
                            with tag('font'):
                                if(color1 != '#000000'):
                                    doc.attr(color=color1)
                                    text(' Person ')
            with tag('pre'):
                doc.attr(style='white-space: pre-wrap')
                with tag('p'):
                    doc.attr(style='width:3000px;font-size:160%')
                    k = 0
                    for d in entities.items():
                        #print article[d[0]: d[1][0]]
                        if (article[k: d[0]] == ' '):
                            text(' ')
                        else:
                            text(article[k: d[0]])
                        with tag('font'):
                            if(d[1][1]!='#000000'):
                                doc.attr(color=d[1][1])
                                with tag('b'):
                                    text(article[d[0]: d[1][0]])
                                    k = d[1][0]
                            else:
                                text(article[d[0]: d[1][0]])
                                k = d[1][0]
                    text(article[k:])

    result = ((doc.getvalue()))
    #print(result)
    Html_file = open(path_to_html, 'w+')
    Html_file.write(result)
    Html_file.close()


make_html(article, entities, sys.argv[3])
#make_html(article, entities, 'h.html')
#print (len(article))
