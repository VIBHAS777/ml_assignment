#first create a new list and then append the numbers in the list if the two list contain same element
def common(l1,l2):
    sameelements = []
    for i in l1:
        for j in l2:
            if i == j and i not in sameelements:#by using nested loop iterating and checking whether the elements are same or not
                sameelements.append(i)
    return sameelements
def main():
    l1 = list(map(int,input("enter elements of first list").split()))
    l2 = list(map(int,input("enter elements of second list").split()))
    result=common(l1,l2)
    print(result)

if __name__=="__main__":
    main()