import pickle as pkl


if __name__ == "__main__":

    filename = 'solutionComboDRL'
    fileObject = open(filename,'rb')
    route = pkl.load(fileObject)
    fileObject.close()

    print('Route',route)