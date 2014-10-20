from django.db import models

class File(models.Model):
    path = models.FilePathField("/home/local/data", match="*\.csv", recursive=True, allow_files=True, allow_folders=False)

class Run(models.Model):
    date = models.DateTimeField()
