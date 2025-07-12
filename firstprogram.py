#first converting string into lower case and check ehether vowels are present or not otherwise it is a consonant
def vowels_consonants(s):
    v=0
    c=0
    for i in s:
        if i.lower() in 'aeiou':#converting it into lower case and checking whether it is a vowel or not
            v=v+1
        else:
            c=c+1
    return v,c
def main():
    word=input("enter a word")
    vowels, consonants=vowels_consonants(word)
    print("vowels:", vowels)
    print("consonants:", consonants)

if __name__=="__main__":
    main()