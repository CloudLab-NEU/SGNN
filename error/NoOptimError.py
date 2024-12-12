# -*- coding: utf-8 -*-


class NoOptimError(Exception):
    """
    No optim is specified during training.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
