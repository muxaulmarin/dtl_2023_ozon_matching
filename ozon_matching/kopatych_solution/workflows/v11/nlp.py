import string
from difflib import SequenceMatcher


def longest_common_prefix(strs):
    if len(strs) == 0:
        return ""
    current = strs[0]
    for i in range(1, len(strs)):
        temp = ""
        if len(current) == 0:
            break
        for j in range(len(strs[i])):
            if j < len(current) and current[j] == strs[i][j]:
                temp += current[j]
            else:
                break
        current = temp
    return current


def longest_common_subsequence(s1, s2) -> int:
    return len(
        "".join(
            [
                s1[block.a : (block.a + block.size)]
                for block in SequenceMatcher(None, s1, s2).get_matching_blocks()
            ]
        )
    )


class FilterToken:
    def __init__(self, brands):
        self.brands = brands
        self.possible = set("qazwsxedcrfvtgbyhnujmikolp1234567890")
        self.punctuation = string.punctuation

    def is_token(self, token):
        if len(token) == 1:
            return False
        for letter in token.lower():
            if letter not in self.possible:
                return False
        return True

    def replace_punctuation(self, input_string):
        for punctuation in self.punctuation:
            input_string = input_string.replace(punctuation, " ")
        return " ".join(input_string.split())

    def get_compatible_devices(self, text):
        tokens = set(self.replace_punctuation(text).lower().split(" ")) - self.brands
        tokens = [t for t in tokens if self.is_token(t)]
        return set(tokens)
