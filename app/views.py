from django.shortcuts import render
from django.http import HttpResponse
from app.forms import FaceDetectorForm
from app.Pipeline_FaceDetector import facedetector
from django.conf import settings
from app.models import FaceDetector
import os

# Create your views here.
def index(request):
    form = FaceDetectorForm()
    if request.method == 'POST':
        form = FaceDetectorForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)
            primary_key = save.pk
            imgobj = FaceDetector.objects.get(pk = primary_key)
            fileroot = str(imgobj.image)
            filepath = os.path.join(settings.MEDIA_ROOT,fileroot)
            results = facedetector(filepath)
            print(results)
            return render(request,"index.html",{'form':form,"upload" : True ,"result" : results})
    return render(request,"index.html",{'form':form,"upload" : False})