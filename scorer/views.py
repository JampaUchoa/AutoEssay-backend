from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

#from ML.algorithms import Classifier
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

    scores = [
        {"algorithm": "SVM", "score": score_svm},
        {"algorithm": "Decision Tree", "score": score_dt},
    ]

    return JsonResponse({"success": True, "scores": scores})
