#first create a new matrix and interchange the cell number of the matrix and append the nubers into new matrix
def mul(mat,mr,mc):
    transposematrix = []
    for i in range(mc):
        updatedrow= []
        for j in range(mr):#interchanging the cell numbers and appending it to new matrix
            updatedrow.append(mat[j][i])
        transposematrix.append(updatedrow)
    return transposematrix
def main():
    mr=int(input("enter no of rows for matrix"))
    mc=int(input("enter no of cols for matrix"))
    
    print("enter the elements for matrix")
    mat=[]
    for _ in range(mr):
        mat.append([int(x) for x in input().split()])
    result=mul(mat,mr,mc)
    print(result)


if __name__=="__main__":
    main()
