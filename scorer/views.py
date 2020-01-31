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
        essay = "A liberdade de expressão, assim como o direito à intimidade são conquistas históricas. Quando um entra em contato com o outro, é preciso pesar com bom senso os dois lados." \
                "Em épocas de regimes autoritários como as ditaduras militares na América Latina e fascistas na Europa, a imprensa não tinha liberdade para veicular notícias diversas, esbarrando comumente na censura. Nessas épocas, as pessoas também sofriam graves violações em suas vidas privadas por meio de tortura e constrangimentos, por exemplo, mas com a ascenção da democracia no ocidente , esses direitos foram amplamente protegidos por lei." \
                "Porém, há casos em que essas garantias são postas em conflito, como em que figuras públicas, principalmente políticas, são pauta de notícias que esbarram em sua intimidade. Assim, o alvo dessas matérias, pode requerer sigilo por se tratar da sua vida privada enquanto a imprensa vê cerceada sua liberdade de expressão." \
                "Portanto, é preciso avaliar até onde o direito de expressão pode ser abusivo, lesar a honra do cidadão, e até mesmo causar transtornos psicológicos ao alvo das notícias. Dessa forma, o bem estar do ser humano deve estar à frente da liberdade de expressão quando esta não for de suma necessidade ao conhecimento público."

    print(essay)
    redacao = Traditional(essay)
    score_svm = redacao.SVM_Modeler()
    score_dt = redacao.Tree_Modeler()
    coh_clf = CoherenceClf(essay)
    scores = [
        {"algorithm": "SVM", "score": score_svm},
        {"algorithm": "Decision Tree", "score": score_dt},
        {"algorithm": "coh_mean_score", "score": coh_clf.random_forest_score()}
    ]

    return JsonResponse({"success": True, "scores": scores})
