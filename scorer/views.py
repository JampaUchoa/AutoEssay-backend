from django.shortcuts import render
# Create your views here.
from django.http import HttpResponse, JsonResponse
from ML.algorithms import Classifier

def index(request):

    redacao = Classifier(["Hello, world. You're at the polls index."])
    print(redacao)
    print(redacao.linear_score())

    return JsonResponse({"success": True})
