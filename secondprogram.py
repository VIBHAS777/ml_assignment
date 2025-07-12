#first check whether two matrix is multiple or not and next using numpy library and do the product of the matrices
import numpy as np
def mul(mat1,mat2):
    return np.dot(mat1,mat2)#by using numpy directly 2 matrices are multiplied
def main():
    mr1=int(input("enter no of rows for 1st matrix"))
    mc1=int(input("enter no of cols for 1st matrix"))
    mr2=int(input("enter no of rows for 2nd matrix"))
    mc2=int(input("enter no of cols for 2nd matrix"))
    print("enter the elements for matrix 1")
    mat1=[]
    for _ in range(mr1):
        mat1.append([int(x) for x in input().split()])
    mat1=np.array(mat1)
    print("enter the elements for matrix 2")
    mat2=[]
    for _ in range(mr2):
        mat2.append([int(x) for x in input().split()])
    mat2=np.array(mat2)
    if mat1.shape[1] !=mat2.shape[0]:
        print("matrix cannot be multiplied")#if coloumns of a and rows of b are different then we cant multiply
        return
    else:
        result=mul(mat1,mat2)
    print(result)


if __name__=="__main__":
    main()