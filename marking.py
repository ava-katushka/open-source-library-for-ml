def read_article(path_to_article):
    with open(path_to_article) as my_file:
        article = my_file.read()
    return article

article = read_article('pushkin.txt')

from pprint import pprint
import json
def read_json(path_to_json):
    with open(path_to_json) as json_file:
        json_of_article = json.load(json_file)
    return json_of_article

json_of_article = read_json('1.json')

from collections import OrderedDict
def get_entities(json_of_article):
    entities = {}
    for j in json_of_article:
        s = (j['Type'])
        for pair in j['Boundaries']:
            entities[ pair[0]] = list([pair[1], entity_color[s] ])
    entities = OrderedDict(sorted(entities.items(), key=lambda t:t[0]))
    return entities

entity_color = {'Person':'green', 'PopulatedPlace':'blue', 'Organisation':'red'}

entities = get_entities(json_of_article)
print(entities)

from yattag import Doc, indent

def make_html(article, entities):
    doc, tag, text = Doc().tagtext()

    with tag('html'):
        with tag('head'):
            with tag('title'):
                text('Marking of article')

        with tag('body'):
            with tag('H4'):
                doc.attr(style='margin-bottom: -0.05em')
                text('Legend')
            with tag('H5'):
                doc.attr(style='font-style:italic')
                with tag('font'):
                    doc.attr(color='red')
                    text('Organisation : red')
            with tag('H5'):
                doc.attr(style='font-style:italic')
                with tag('font'):
                    doc.attr(color='blue')
                    text('PopulatedPlace : blue')
            with tag('H5'):
                doc.attr(style='font-style:italic')
                with tag('font'):
                    doc.attr(color='green')
                    text('Person : green')
            with tag('p'):
                k = 0
                for d in entities.items():
                    text(article[k: d[0]])
                    with tag('font'):
                        doc.attr(color=d[1][1])
                        text(article[d[0]: d[1][0]])
                        k = d[1][0]
                text(article[k:])

    result = indent(doc.getvalue())
    Html_file = open('my_html.html', 'w+')
    Html_file.write(result)
    Html_file.close()


make_html(article, entities)







'''from yattag import Doc, indent
doc, tag, text = Doc().tagtext()

str = 'The RGB color model is an additive color model in which red, green, and blue light are added together in various ways to reproduce a broad array of colors.\
 The name of the model comes from the initials of the three additive primary colors, red, green, and blue.'
str = str.split(' ')
with tag('html'):
    with tag('head'):
        with tag('title'):
            text('Example 1')
        with tag('style'):
            text('H1 {\n')
            text('margin-bottom: -0.5em\n')
            text('}\n')
        with tag('style'):
            text('H4 {\n')
            text('margin-bottom: -0.05em\n')
            text('}\n')
    with tag ('body'):
        with tag('H1'):
            text('Title')
        with tag('table'):
            doc.attr(border='2')
            doc.attr(sellpadding = '0')
            with tag('caption'):
                with tag('H4'):
                    text('Legend')
            with tag('tr'):
                with tag('th'):
                    text('type of essence')
                with tag('th'):
                    text('color')
            with tag('tr'):
                with tag('td'):
                    text('Person')
                with tag('td'):
                    doc.attr(BGCOLOR = 'green')
                    text('Green')
        with tag('p'):
            for s in str:
                if (s[:3] == 'red'):
                    with tag ('font'):
                        doc.attr(color = 'red')
                        text(s + ' ')
                elif (s[:4] == 'blue'):
                    with tag ('font'):
                        doc.attr(color = 'blue')
                        text(s + ' ')
                elif (s[:5] == 'green'):
                    with tag ('font'):
                        doc.attr(color = 'green')
                        text(s + ' ')
                elif (s[:5] == 'model'):
                    with tag ('font'):
                        doc.attr(color = 'pink')
                        text(s + ' ')
                elif (s[:7] == 'primary'):
                    with tag ('font'):
                        doc.attr(color = 'yellow')
                        text(s + ' ')
                else:
                    text(s + ' ')


print (indent(doc.getvalue()))
result = indent(doc.getvalue())
Html_file = open('my_html.html', 'w+')
Html_file.write(result)
Html_file.close()'''



