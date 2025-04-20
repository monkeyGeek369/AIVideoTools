
def count_common_chars(text1, text2):
    '''Count the number of common characters in two strings'''
    set1 = set(text1)
    set2 = set(text2)
    common_chars = set1 & set2
    return len(common_chars)

def is_str_contain_list_strs(text:str, strs:list[str]) -> bool:
    if not text or not strs:
        return False
    for str in strs:
        if str in text:
            return True
    return False
