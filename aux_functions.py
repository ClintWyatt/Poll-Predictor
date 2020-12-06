# auxillary functions

def strip_columns(dataset, list_columns): # strip a list of columns from a numpy dataframe
    for column in list_columns:
        dataset = dataset.drop(columns=[column])
    return dataset


# hopefully used to make the correct results lines of code look smaller
#def selectwhere(dataset,columns,query): # select columns and specific samples where query is true 
    #ADD CODE


def decision_point(clf): #
    coef = clf.coef_
    intercept = clf.intercept_
    return (-intercept[0])/coef[0,0]

def to_int(x): #convert variable -> int
    
    x = int(x)
    return x

def to_float(x): #convert variable -> float

    x = float(x)
    return x