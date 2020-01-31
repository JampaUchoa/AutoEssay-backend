from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

#from ML.algorithms import Classifier
from ML.algorithms import CoherenceClf
from traditional.Classifiers import Classifiers as Traditional
import json

@csrf_exempt
def index(request):

    if request.method == 'POST':
        essay = json.loads(request.body)["essay"]
    else:
        essay = "Historicamente causadores de inúmeras vítimas, os acidentes de trânsito vêm ocorrendo com frequência cada vez menor"

    print(essay)
    redacao = Traditional(essay)
    score_svm = redacao.SVM_Modeler()
    score_dt = redacao.Tree_Modeler()
    coh_clf = CoherenceClf(essay)
    scores = [
        {"algorithm": "SVM", "score": score_svm},
        {"algorithm": "Decision Tree", "score": score_dt},
        {"algorithm": "coh_mean_score", "score": coh_clf.mean_score()}
    ]

    return JsonResponse({"success": True, "scores": scores})
