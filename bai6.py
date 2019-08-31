def count _characters(senterce):
    dictionary ={}
    for character is senterce :
        Keys = dictionary.Keys()
        if character in Keys:
            dictionary[character]+=1
        else :
            dictionary[character]=1

return dictionary

senterce = input("input your senterce")
print(count_character(senterce))