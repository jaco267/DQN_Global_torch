import matplotlib.pyplot as plt


def plot2data(data1,label1,data2,label2,y_label,savepath):
    plt.figure()
    plt.plot(data1,'r',label=label1)
    plt.plot(data2,'b',label=label2)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(savepath)
    plt.cla();    plt.clf();    plt.close();  #* added to prevent warning
    return 
def imshow(img,title,savepath):
    plt.figure()
    plt.imshow(img,cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig(savepath)
    plt.cla();    plt.clf();    plt.close();  #* added to prevent warning
    return
def barplot(x,title,savepath,colorbar = False):
    plt.figure()
    # plt.plot(utilization_frequency[:,0],utilization_frequency[:,1])
    plt.bar(x[:,0],x[:,1])
    plt.plot(x[:,0],x[:,1],'r-')
    plt.title(title)
    if colorbar:
        plt.colorbar()
        plt.gca().invert_yaxis()
    plt.savefig(savepath)
    plt.cla();    plt.clf();    plt.close();  #* added to prevent warning
    return 