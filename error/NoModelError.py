# -*- coding: utf-8 -*-


class NoModelError(Exception):
    """
    No model is specified during training.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
