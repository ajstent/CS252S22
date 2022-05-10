import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import scipy.linalg as sp_la
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

def getSummaryStatistics(data, columns):
    "Get the max, min, mean, std for each variable in the data."
    return pd.DataFrame(np.array([data.max(axis=0), data.min(axis=0), data.mean(axis=0), np.sqrt(data.var(axis=0))]), columns=columns)

# Random shuffle
def randomSplit(data, depVar, indVars):
    np.random.shuffle(data)
    train, dev, test = np.split(data, [int(.8 * len(data)), int(.9 * len(data))])
    train, trainY = train[np.ix_(np.arange(train.shape[0]), indVars)], train[:, depVar]
    dev, devY = dev[np.ix_(np.arange(dev.shape[0]), indVars)], dev[:, depVar]
    test, testY = test[np.ix_(np.arange(test.shape[0]), indVars)], test[:, depVar]
    return train, dev, test, trainY, devY, testY

# Stratified sampling
def stratifiedSplit(data, depVar, indVars, numberValsToKeep=-1):
    if numberValsToKeep > -1:
        byCategory = [data[data[:,-1]==k] for k in np.unique(data[:,-1])][0:numberValsToKeep]
    else:
        byCategory = [data[data[:,-1]==k] for k in np.unique(data[:,-1])]
    print(depVar, indVars, len(byCategory))

    trainFeats = []
    devFeats = []
    testFeats = []
    trainYs = []
    devYs = []
    testYs = []
    for category in byCategory:
        train, dev, test = np.split(category, [int(.8 * len(category)), int(.9 * len(category))])
        train, trainY = train[np.ix_(np.arange(train.shape[0]), indVars)], train[:, depVar]
        dev, devY = dev[np.ix_(np.arange(dev.shape[0]), indVars)], dev[:, depVar]
        test, testY = test[np.ix_(np.arange(test.shape[0]), indVars)], test[:, depVar]
        trainFeats.append(train)
        devFeats.append(dev)
        testFeats.append(test)
        trainYs.append(trainY)
        devYs.append(devY)
        testYs.append(testY)
    return np.vstack(trainFeats), np.vstack(devFeats), np.vstack(testFeats), np.concatenate(trainYs), np.concatenate(devYs), np.concatenate(testYs)

def homogenizeData(data):
    return np.append(data, np.array([np.ones(data.shape[0], dtype=float)]).T, axis=1)
   
# yaw
def rotateTransformX(x):
    return np.array([1, 0, 0, 0, np.cos(np.radians(x)), -np.sin(np.radians(x)), 0, np.sin(np.radians(x)), np.cos(np.radians(x))]).reshape(3, 3)

# pitch
def rotateTransformY(y):
    return np.array([np.cos(np.radians(y)), 0, np.sin(np.radians(y)), 0, 1, 0, -np.sin(np.radians(y)), 0, np.cos(np.radians(y))]).reshape(3, 3)

# roll
def rotateTransformZ(z):
    return np.array([np.cos(np.radians(z)), -np.sin(np.radians(z)), 0, np.sin(np.radians(z)), np.cos(np.radians(z)), 0, 0, 0, 1]).reshape(3, 3)

def minmaxGlobal(data):
    "Global max-min normalization."
    scaleTransform = np.eye(data.shape[1], data.shape[1])
    for i in range(data.shape[1]):
        scaleTransform[i, i] = 1 / (data.max() - data.min())
    return (scaleTransform@data.T).T

def minmaxLocal(data):
    "Local max-min normalization."
    scaleTransform = np.eye(data.shape[1], data.shape[1])
    for i in range(data.shape[1]):
        if data[:, i].max() - data[:, i].min() != 0:
            scaleTransform[i, i] = 1 / (data[:, i].max() - data[:, i].min())
    return (scaleTransform@data.T).T

def zScore(data, translateTransform=None, scaleTransform=None):
    "z score."
    homogenizedData = np.append(data, np.array([np.ones(data.shape[0], dtype=float)]).T, axis=1)
    if translateTransform is None:
        translateTransform = np.eye(homogenizedData.shape[1])
        for i in range(homogenizedData.shape[1]):
            translateTransform[i, homogenizedData.shape[1]-1] = -homogenizedData[:, i].mean()
    if scaleTransform is None:
        diagonal = [1 / homogenizedData[:, i].std() if homogenizedData[:, i].std() != 0 else 1 for i in range(homogenizedData.shape[1])]
        scaleTransform = np.eye(homogenizedData.shape[1], dtype=float) * diagonal
    data = (scaleTransform@translateTransform@homogenizedData.T).T
    return translateTransform, scaleTransform, data[:, :data.shape[1]-1]

class PCA:
    def __init__(self, centered=False, plot=False):
        self.eigenvalues = None
        self.eigenvectors = None
        self.principalComponents = 0
        self.centered = centered
        self.plot = plot
    
    def fit(self, data, columns):
        # center
        if not self.centered:
            data = data - np.mean(data, axis=0)

        # covariance
        covarianceMatrix = (data.T @ data) / (data.shape[0] - 1)
        if self.plot:
            sns.heatmap(pd.DataFrame(covarianceMatrix, columns=columns), xticklabels=True, yticklabels=True, annot=False, cmap='PuOr')
            plt.title("Covariance Matrix")
            plt.show()

        # svd
        (evals, evectors) = np.linalg.eig(covarianceMatrix)

        # sort
        evalsOrder = np.argsort(evals)[::-1]
        self.eigenvalues = evals[evalsOrder]
        self.eigenvectors = evectors[:, evalsOrder]

        # proportional variance
        evalsSum = np.sum(self.eigenvalues)
        proportionalVars = [e / evalsSum for e in self.eigenvalues]

        # cumulative sum of proportional variance
        cumulativeSum = np.cumsum(proportionalVars)

        if self.plot:
            plt.figure(figsize=(6, 4))
            plt.bar(range(len(proportionalVars)), proportionalVars, alpha=0.5, align='center',
                    label='Proportional variance')
            plt.xticks(list(range(len(proportionalVars))))
            plt.ylabel('Proportional variance ratio')
            plt.xlabel('Ranked Principal Components')
            plt.title("Scree Graph")
            plt.show()

            plt.figure(figsize=(6,4))
            plt.plot(range(len(cumulativeSum)), cumulativeSum)
            plt.ylim((0,1.1))
            plt.xticks(list(range(len(proportionalVars))))
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Cumulative explained variance')
            plt.title('Elbow Plot')
            plt.show()

    def project(self, data, numberOfComponents):
        self.principalComponents = numberOfComponents
        # center
        if not self.centered:
            data = data - np.mean(data, axis=0)
        v = self.eigenvectors[:, :numberOfComponents]
        projected = data@v
        return projected
    
    
def convertLabel(label):
    labels = {'SEKER': 0, 'BARBUNYA': 1, 'BOMBAY': 2, 'CALI': 3, 'HOROZ': 4, 'SIRA': 5, 'DERMASON': 6}
    return float(labels[str(label)])

def convertLevel(feat):
    options = ['low', 'mid', 'high']
    return options.index(feat)

def convertStatus(feat):
    options = ['poor', 'fair', 'good', 'excellent']
    return options.index(feat)

def convertStability(feat):
    options = ['unstable', 'mod-stable', 'stable']
    return options.index(feat)

def convertDecision(feat):
    options = ['S', 'A', 'I']
    return options.index(feat)

def convertRaisin(feat):
    options = ['Kecimen', 'Besni']
    return float(options.index(feat))

def prepData(type="regression", dataName="beans", fractionToKeep=1.0):
    # Load the data
    print("\nLoad the data")
    if dataName == "beans":
        if type=="regression":
            depVar = 3
        elif type=="classification":
            depVar = 16
        else:
            depVar = -1
        columns = ['area', 'perimeter', 'major_axis_length', 'minor_axis_length', 'aspect_ratio', 'eccentricity', 'convex_area', 'equivalent_diameter', 'extent', 'solidity', 'roundness', 'compactness', 'shapefactor1', 'shapefactor2', 'shapefactor3', 'shapefactor4', 'class']
        # This dataset comes from https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset and the data sheet is there
        data = np.array(np.genfromtxt('data/Dry_Bean_Dataset.arff', delimiter=',', converters={16: convertLabel}, skip_header=25, dtype=float, encoding='utf-8')) 
    elif dataName == "heart_attacks":
        columnsToPlot = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
        if type=="regression":
            depVar = 11
            columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
            # This dataset comes from https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records and the data sheet is there
            data = np.array(np.genfromtxt('data/heart_failure_clinical_records_dataset.csv', delimiter=',', usecols=[0,1,2,3,4,5,6,7,8,9,10,11], skip_header=1, dtype=float, encoding='utf-8')) 
        elif type=="classification":
            depVar = 12
            columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT']
            # This dataset comes from https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records and the data sheet is there
            data = np.array(np.genfromtxt('data/heart_failure_clinical_records_dataset.csv', delimiter=',', skip_header=1, dtype=float, encoding='utf-8')) 
        else:
            depVar = -1
            columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT']
            # This dataset comes from https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records and the data sheet is there
            data = np.array(np.genfromtxt('data/heart_failure_clinical_records_dataset.csv', delimiter=',', skip_header=1, dtype=float, encoding='utf-8')) 
    elif dataName == "mammography":
        if type=="regression":
            depVar = 0
        elif type=="classification":
            depVar = 5
        else:
            depVar = -1
        columns = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity']
        # This dataset comes from https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass and the data sheet is there
        data = np.array(np.genfromtxt('data/mammographic_masses.data', delimiter=',', skip_header=0, missing_values='?', dtype=float, encoding='utf-8')) 
    elif dataName == "post-op":
        if type=="regression":
            depVar = 7
        elif type=="classification":
            depVar = 8
        else:
            depVar = -1
        columns = ["l-core", "l-surf", "l-o2", "l-bp", "surf-stbl", "core-stbl", "bp-stbl", "comfort", "adm-decs"]
        # This dataset comes from https://archive.ics.uci.edu/ml/datasets/Challenger+USA+Space+Shuttle+O-Ring and the data sheet is there
        data = np.array(np.genfromtxt('data/post-operative.data', converters={0: convertLevel, 1: convertLevel, 2: convertStatus, 3: convertLevel, 4: convertStability, 5: convertStability, 6: convertStability, 8: convertDecision}, delimiter=',', skip_header=0, dtype=int, encoding='utf-8')) 
        print("\nClean the data by filling in column means, before length is ", len(data))
        data = np.where(np.isnan(data), ma.array(data, mask=np.isnan(data)).mean(axis=0), data)  
        print("After length is ", len(data))
    elif dataName == "o-ring":
        if type=="regression":
            depVar = 1
        elif type=="classification":
            depVar = 1
        else:
            depVar = -1
        columns = ['number-o-rings', 'number-distressed', 'launch-temp', 'leak-check-pressure']
        # This dataset comes from https://archive.ics.uci.edu/ml/datasets/Post-Operative+Patient and the data sheet is there
        data = np.array(np.genfromtxt('data/o-ring-erosion-only.data', usecols=[0,1,2,3], delimiter=' ', skip_header=0, dtype=float, encoding='utf-8')) 
    elif dataName == "happiness":
        if type=="regression":
            depVar = 2
        elif type=="classification":
            depVar = 0
        else:
            depVar = -1
        columns = ['happiness', 'services', 'housing', 'schools', 'police', 'maintenance', 'events']
        # This dataset comes from https://archive.ics.uci.edu/ml/datasets/Somerville+Happiness+Survey and the data sheet is there
        data = np.array(np.genfromtxt('data/SomervilleHappinessSurvey2015.csv', delimiter=',', skip_header=1, dtype=int, encoding='utf-8'))         
    elif dataName == "seeds":
        if type=="regression":
            depVar = 6
        elif type=="classification":
            depVar = 7
        else:
            depVar = -1
        columns = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry-coefficient', 'length-groove', 'variety']
        # This dataset comes from https://archive.ics.uci.edu/ml/datasets/seeds and the data sheet is there
        data = np.array(np.genfromtxt('data/seeds_dataset.txt', delimiter='\t', skip_header=0, dtype=float, encoding='utf-8'))         
    elif dataName == "raisin":
        if type=="regression":
            depVar = 6
        elif type=="classification":
            depVar = 7
        else:
            depVar = -1
        columns = ['area', 'majoraxislength', 'minoraxislength', 'eccentricity', 'convexarea', 'extent', 'perimeter', 'class']
        # This dataset comes from https://archive.ics.uci.edu/ml/datasets/Raisin+Dataset and the data sheet is there
        data = np.array(np.genfromtxt('data/Raisin_Dataset.csv', converters={7: convertRaisin}, delimiter=',', skip_header=1, dtype=float, encoding='utf-8'))         
    elif dataName == "cc":
        columnsToPlot = ["debt", "yearsemployed", "creditscore", "zip", "income"]
        if type=="regression":
            depVar = 6
            #columns = ['gender', 'age', 'debt', 'married', 'bankcustomer', 'industry', 'ethnicity', 'yearsemployed', 'priordefault', 'employed', 'creditscore', 'driverslicense', 'citizen', 'zip', 'income', 'approved']
            columns = ['debt', 'married', 'bankcustomer', 'yearsemployed', 'priordefault', 'employed', 'creditscore', 'driverslicense', 'zip', 'income']
            # This dataset comes from https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data and the data sheet is there
            data = np.array(np.genfromtxt('data/clean_dataset.csv', usecols=[2,3,4,7,8,9,10,11,13,14], delimiter=',', skip_header=1, dtype=float, encoding='utf-8'))         
        elif type=="classification":
            depVar = 10
            #columns = ['gender', 'age', 'debt', 'married', 'bankcustomer', 'industry', 'ethnicity', 'yearsemployed', 'priordefault', 'employed', 'creditscore', 'driverslicense', 'citizen', 'zip', 'income', 'approved']
            columns = ['debt', 'married', 'bankcustomer', 'yearsemployed', 'priordefault', 'employed', 'creditscore', 'driverslicense', 'zip', 'income', 'approved']
            # This dataset comes from https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data and the data sheet is there
            data = np.array(np.genfromtxt('data/clean_dataset.csv', usecols=[2,3,4,7,8,9,10,11,13,14,15], delimiter=',', skip_header=1, dtype=float, encoding='utf-8'))         
        else:
            depVar = -1
            #columns = ['gender', 'age', 'debt', 'married', 'bankcustomer', 'industry', 'ethnicity', 'yearsemployed', 'priordefault', 'employed', 'creditscore', 'driverslicense', 'citizen', 'zip', 'income', 'approved']
            columns = ['debt', 'married', 'bankcustomer', 'yearsemployed', 'priordefault', 'employed', 'creditscore', 'driverslicense', 'zip', 'income', 'approved']
            # This dataset comes from https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data and the data sheet is there
            data = np.array(np.genfromtxt('data/clean_dataset.csv', usecols=[2,3,4,7,8,9,10,11,13,14,15], delimiter=',', skip_header=1, dtype=float, encoding='utf-8'))         
         
    else:
        print("Data set ", dataName, " not found!")
        return None

    print("data columns\n", columns)
    # Clean the data
#    print("\nClean the data by dropping values, before length is ", len(data))
    ## No missing values
#    data = data[~np.any(np.isnan(data), axis=1)]
#    print("After length is ", len(data))

    # Look at the data
    print("\nInspect the data")
    print("data shape\n", data.shape, "\ndata type\n", data.dtype)
    print("missing data: none")
    print("data max, min, mean, std\n", getSummaryStatistics(data))
    if not columnsToPlot:
        columnsToPlot = columns
    if depVar != -1:
        sns.pairplot(pd.DataFrame(data, columns=columns), y_vars = [columns[depVar]], vars=columnsToPlot, kind = "scatter")
        plt.show()
    else:
        sns.pairplot(pd.DataFrame(data, columns=columns), vars=columnsToPlot, kind = "scatter")
        plt.show()
        
    if depVar > -1:
        # Set the dependent and independent variables
        indVars = list(range(data.shape[1]))
        indVars.pop(depVar)
        depName = columns.pop(depVar)
        # Split the data, reducing it if necessary
        print("\nSplit the data, dependent variable ", depVar, ", to keep ", fractionToKeep)
        train, dev, test, trainY, devY, testY = stratifiedSplit(data, depVar, indVars, numberValsToKeep=fractionToKeep)
        print("training data shape", "\n", train.shape, "\ntraining data max, min, mean, std\n", getSummaryStatistics(train))
        print("\ndev data shape", "\n", dev.shape, "\ndev data max, min, mean, std\n", getSummaryStatistics(dev))
        print("\ntest data shape", "\n", test.shape, "\ntest data max, min, mean, std\n", getSummaryStatistics(test))
        # Transform the data
        print("\nTransform the data")
        translateTransform, scaleTransform, trainTransformed = zScore(train)
        print("training data shape", "\n", trainTransformed.shape, "\ntraining data max, min, mean, std\n", getSummaryStatistics(trainTransformed))
        _, _, devTransformed = zScore(dev, translateTransform=translateTransform, scaleTransform=scaleTransform)
        print("\ndev data shape", "\n", devTransformed.shape, "\ndev data max, min, mean, std\n", getSummaryStatistics(devTransformed))
        _, _, testTransformed = zScore(test, translateTransform=translateTransform, scaleTransform=scaleTransform)
        print("\ntest data shape", "\n", testTransformed.shape, "\ntest data max, min, mean, std\n", getSummaryStatistics(testTransformed))
        return trainTransformed, devTransformed, testTransformed, trainY, devY, testY, columns
    else:
        # Reduce the data if necessary
        if fractionToKeep < 1.0:
            print("\nReducing the data by", (1.0-fractionToKeep))
            data, _ = np.split(data, [int(fractionToKeep*len(data))])
        # Transform the data
        print("\nTransform the data")
        print("data shape", "\n", data.shape, "\ndata max, min, mean, std\n", getSummaryStatistics(data))
        translateTransform, scaleTransform, transformed = zScore(data)
        print("data shape", "\n", transformed.shape, "\ndata max, min, mean, std\n", getSummaryStatistics(transformed))
        return transformed, columns
    
def projectData(data, principalComponents, columns, *args):
    # Dimensionality reduction
    print("\nDo PCA on the data, keep principal components")
    pca = PCA(centered=True, plot=True)
    pca.fit(data, columns)
    projected = pca.project(data, principalComponents)
    return projected, [pca.project(arg, principalComponents) for arg in args]

def fitExploreKMeans(data, minK, maxK, byK):
    inertiaByK = []
    for k in range(minK, maxK, byK):
        print(k)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        inertiaByK.append([k, kmeans.inertia_])
    fig = plt.figure(figsize=(6,4))
    inertiaByK = np.array(inertiaByK)
    plt.plot(inertiaByK[:, 0], inertiaByK[:, 1])
    plt.xticks([x[0] for x in inertiaByK])
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Plot')
    plt.show()
    
def fitExploreKNN(train, trainY, dev, devY, minK, maxK, byK):
    accuracyByK = []
    for k in range(minK, maxK, byK):
        print(k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train, trainY)
        accuracyByK.append([k, knn.score(dev, devY)])
    fig = plt.figure(figsize=(6,4))
    accuracyByK = np.array(accuracyByK)
    plt.plot(accuracyByK[:, 0], accuracyByK[:, 1])
    plt.xticks([x[0] for x in accuracyByK])
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Elbow Plot')
    plt.show()
    
def aucRoc(y, yhat):
    "Only works for binary classification"
    auc = roc_auc_score(y, yhat, multi_class="ovr")
    FPR, TPR, thresholds = roc_curve(y, yhat)
    plt.figure(figsize=(10, 8), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(FPR, TPR, 'g')
    plt.fill_between(FPR, TPR, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    
class RBFNetwork:
    def __init__(self, type='regression'):
        self.prototypes = None
        self.clusters = None
        self.epsilon = 1e-8
        self.prototypeStandardDeviations = None
        self.regression = None
        self.type = type
        
    def explorePrototypes(self, data, minK, maxK, byK):
        inertiaByK = []
        for k in range(minK, maxK, byK):
            print(k)
            kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
            inertiaByK.append([k, kmeans.inertia_])
        fig = plt.figure(figsize=(6,4))
        inertiaByK = np.array(inertiaByK)
        plt.plot(inertiaByK[:, 0], inertiaByK[:, 1])
        plt.xticks(range(maxK+minK))
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Plot')
        plt.show()
    
    def fitPrototypes(self, data, k, epsilon=1e-8):
        km = KMeans(n_clusters=k, random_state=0).fit(data)
        self.prototypes = km.cluster_centers_
        self.clusters = km.labels_
        self.calculatePrototypeStandardDeviations(data, epsilon=epsilon)

    def calculateActivation(self, datum, prototypeIndex):
        return np.exp(-np.sum(np.square(datum-self.prototypes[prototypeIndex])) / (2*self.prototypeStandardDeviations[prototypeIndex]**2 + self.epsilon))
    
    def calculateActivations(self, datum):
        return np.array([self.calculateActivation(datum, prototypeIndex) for prototypeIndex in range(len(self.prototypes))])
    
    def calculateStandardDeviation(self, cluster, prototypeIndex):
        cluster = cluster[:, :-1]
        return (1/len(cluster))*np.sum([np.linalg.norm(datum-self.prototypes[prototypeIndex])**2 for datum in cluster])
            
    def calculatePrototypeStandardDeviations(self, data, epsilon=1e-8):
        self.epsilon = epsilon
        with_clusters = np.hstack((data, np.array([self.clusters]).T))
        indices = np.argsort(with_clusters[:, -1])
        with_clusters_sorted = with_clusters[indices]
        self.prototypeStandardDeviations = [self.calculateStandardDeviation(x, i) for i, x in enumerate(np.array_split(with_clusters_sorted, np.where(np.diff(with_clusters_sorted[:, -1])!=0)[0]+1))]
                        
    def fitClassification(self, data, labels):
        allYs = []
        for value in np.unique(labels):
            allYs.append([1 if x == value else 0 for x in labels])
        y = np.vstack(allYs).T
        self.regression = LinearRegression().fit(data, y)
                        
    def fitRegression(self, data, labels):
        self.regression = LinearRegression().fit(data, labels)
                        
    def fitOutputNodes(self, data, labels):
        activations = np.array([self.calculateActivations(datum) for datum in data])
        if self.type == 'regression':
            self.fitRegression(activations, labels)
        else:
            self.fitClassification(activations, labels)  
      
    def fit(self, data, labels, k, epsilon=1e-8):
        self.fitPrototypes(data, k, epsilon=epsilon)
        self.fitOutputNodes(data, labels)

    def predict(self, data):
        activations = np.array([self.calculateActivations(datum) for datum in data])
        predictions = self.regression.predict(activations)
        if self.type == 'regression':
            return predictions
        else:
            return [np.argmax(x) for x in predictions]

    def accuracy(self, y, yhat):
        return np.sum([1 if y[i]==yhat[i] else 0 for i in range(len(y))]) / len(y)

    def rsquared(self, y, yhat):
        return 1 - (((y - yhat)**2).sum() / ((y - y.mean())**2).sum())
 
    def score(self, y, yhat):
        if self.type == 'regression':
            return self.rsquared(y, yhat)
        else:
            return self.accuracy(y, yhat)