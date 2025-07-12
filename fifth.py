import random
def memdmo(s):
    total = 0
    for num in s:#for mean we are adding all the numbers and dividing it by size
        total += num
    mean = total / len(s)

    n = len(s)
    sorted_nos = sorted(s) #for mode if it is even sized then finding two mid elements adding and dividing it by 2
    if n % 2 == 0:
        median = (sorted_nos[n // 2 - 1] + sorted_nos[n // 2]) / 2
    else:
        median = sorted_nos[n // 2]#if odd directly mid element

    new_nos = []
    mode_nos = []
    for i in s:
        if i not in new_nos:#for mode by checking the number occurance appending to the other list
            new_nos.append(i)
        else:
            mode_nos.append(i)
   
    return mean, median, mode_nos

def main():
    randomlist = [random.randint(100, 150) for _ in range(100)]
    mean,median,mode_nos = memdmo(randomlist)
    print("randomlist\n", randomlist)
    print("mean\n", mean)
    print("median\n", median)
    print("Mode\n", mode_nos)

if __name__ == "__main__":
    main()