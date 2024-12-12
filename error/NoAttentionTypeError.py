# -*- coding: utf-8 -*-


class NoAttentionTypeError(Exception):
    """
    No tfe_type is specified during training.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
