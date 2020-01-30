from django.shortcuts import render
# Create your views here.
from django.http import HttpResponse, JsonResponse
#from ML.algorithms import Classifier
from traditional.Classifiers import Classifiers as Traditional

def index(request):

    notas = []

    redacao = Traditional("Historicamente causadores de inúmeras vítimas, os acidentes de trânsito vêm ocorrendo com frequência cada vez menor")
    nota_svm = redacao.SVM_Modeler()
    nota_dt = redacao.Tree_Modeler()

    scores = [
        {"algorithm": "SVM", "total_score": nota_svm},
        {"algorithm": "Decision Tree", "total_score": nota_dt},
    ]

    return JsonResponse({"success": True, "scores": scores})
