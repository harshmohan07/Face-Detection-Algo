from django import forms
from app.models import FaceDetector

class FaceDetectorForm(forms.ModelForm):
    class Meta:
        model = FaceDetector
        fields = ['image']
