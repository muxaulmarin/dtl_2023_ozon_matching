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
