from django.db import models

class PredictionStats(models.Model):
    correct_count = models.IntegerField(default=0)
    incorrect_count = models.IntegerField(default=0)

    def __str__(self):
        return f'PredictionStats - Correct: {self.correct_count}, Incorrect: {self.incorrect_count}'