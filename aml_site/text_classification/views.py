import os
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseForbidden

import pickle
from AML_TextClassification.textclassifier import TextClassifier
import site_user_interaction.email_sending as email_sending

SCRIPT_ROOT = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_ROOT))
CLASSIFIER_ROOT = os.path.join(ROOT, "AML_TextClassification")


def index(request):
    context = {}
    return render(request, 'index.html', context)


def submit_text(request):
    if request.method == 'POST':
        try:
            text = request.POST['input_text']
            filename_names = os.path.join(CLASSIFIER_ROOT, "names.pkl")
            with open(filename_names, "rb") as f:
                names_loaded = pickle.load(f)

            text_classifier = TextClassifier()
            text_classifier.load(os.path.join(CLASSIFIER_ROOT, "class.pkl"))

            predicted = text_classifier.predict([text])
            offset = 0
            mask = text_classifier.get_support()

            tags = []
            for i in range(len(predicted[0])):
                while not mask[offset]:
                    offset += 1
                if predicted[0][i] != 0:
                    tags.append(names_loaded[offset])
                offset += 1
            return HttpResponse(', '.join(tags))
        except Exception as e:
            return HttpResponse(e.message)


def send_feedback_email(request):
    if request.method != 'POST':
        return HttpResponseForbidden()
    try:
        text = request.POST['input_text']
        email_sending.sendMail(text)
    except Exception as e:
        return HttpResponse(e.message)
    return HttpResponse("OK")


