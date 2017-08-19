#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt

## Uncomment below line when submitting
#sys.path.append("../tools/")
## Cooment out below line when submitting
sys.path.append("..\\Intro-to-Mchine_learning\\ud120-projects-master_2\\tools")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

''' 
## Followings are all the features available in the  dataset
Main features
'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
'director_fees',

'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi'

'poi '


expences = deferral_payments. expenses director_fees, total_payments'
income = salary, bonus, loan_advances, differed_income, 
'''
#############################################################################
#data_dict = {} ## Create an empty dict
features_list = ['poi','salary', 'total_payments', 'bonus',   
		'deferred_income', 'expenses', 'to_messages', 'from_poi_to_this_person', 
		'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#############################################################################
### Task 2: Remove outliers

## This function take a specific dictionary
## and generates boxplots
def create_boxplot(dic):
	plt.figure()
	for key in dic.keys():
		vals = dic[key]
		## Remove zeros, because most zeros were for NaNs
		vals = list(filter(lambda a: a != 0.0, vals))

		plt.boxplot(vals)
		plt.xlabel(key)
		plt.ylabel("Values")
		plt.show()
	
## This function take a specific dictionary
## and generates scatter plots with each feature with the other
def create_scatter(dic):
	plt.figure()
	tempkeylst = []
	for key1 in dic.keys():
		for key2 in dic.keys():
			if key1 != key2 and key2:
		
				val1 = dic[key1]
				val2 = dic[key2]				
				## Remove zeros, because most zeros were for NaNs
				#val1 = list(filter(lambda a: a != 0.0, val1))
				#val2 = list(filter(lambda a: a != 0.0, val2))
				plt.scatter(val1, val2)
				plt.xlabel(key1)
				plt.ylabel(key2)
				plt.show()	
	
## By running following codes, I idendified 'TOTAL' is not a name of a person,
## and it deviates all the other values. Because 'TOTAL' values are
## very high compare to other values.

## Please cnange False to True in the folowing block if needed to see the values of 'TOTAL'			
	
## To identify mismattching items in the data_dic	
if False:
	nan_count = 0	
	nan_count1 = 0	
	for name in data_dict.keys():
		bons = float(data_dict[name]['bonus'])
		salry = float(data_dict[name]['salary'])
		if bons > 20000000:		
			print(name, bons)
			data_dict.pop( name, 0 )
		if bons > 5000000 and salry > 1000000:
			print(name)
		if name != 'TOTAL':
			if data_dict[name]['salary'] == 'NaN':
				nan_count += 1
			if data_dict[name]['bonus'] == 'NaN':
				nan_count1 += 1		

	print('nan count salary', nan_count)	
	print('nan count bonus', nan_count1)


## Therefore, TOTAL need to be removed	
data_dict.pop('TOTAL', None)
	

## Please cnange False to True to see the boxplots, scatter plots and number of NaN values

if False:
	## Getting labels and features as numpy arrays 		
	data_array = featureFormat(data_dict, features_list, sort_keys = False)	
	print('Size of the dataset before,'+ str(len(data_dict))+' and, after, ' \
		+str(len(data_array)) +',running featureFormat function' )
	value_dic = {}
	## convert the same feature values append into the fature name. give a dict 
	## Navigate through each each poi record 
	for j, point in enumerate(data_array):
		val_list = []
		## Navigate through each value in a record and collect similar values 
		## to corresponding feature name. This gives a dictionary
		for i, featu in enumerate(features_list):
			if featu != 'poiii': ## to remove 'poi' record
				if j == 0: ## Add the first value
					value_dic[featu] = [point[i]]
				else: ## Append after the value
					value_dic[featu] = value_dic[featu] + [point[i]]

	print('Number of miising values')
	for j, point in enumerate(value_dic):
		z_count = value_dic[point].count(0)		
		if z_count > 0:
			print(point, z_count)
			
	print('Poi stats')
	print('number of poi:', value_dic['poi'].count(1))
	print('number of nonpoi:', value_dic['poi'].count(0))
	print('total persons:', len(value_dic['poi']))
		
	## Uncomment the following line to see plots for each feature for all the persons
	#create_boxplot(value_dic)
	#create_scatter(value_dic)


#############################################################################
### Task 3: Create new feature(s)
## Creating a new features to standadize the from_this_person_to_poi and 
## from_poi_to_this_person by dividing corresponding total emails
		
if True:
	for name in data_dict.keys():
		to_poi = data_dict[name]['from_this_person_to_poi']
		from_poi = data_dict[name]['from_poi_to_this_person']
		to_msg = data_dict[name]['to_messages']
		from_msg = data_dict[name]['from_messages']		
		
		## New standardized features
		if to_poi != 'NaN' or from_msg != 'NaN':
			data_dict[name]['std_from_this_person_to_poi'] = float(to_poi)/float(from_msg)
		else:
			data_dict[name]['std_from_this_person_to_poi'] = 'NaN'
		if from_poi != 'NaN' or to_msg != 'NaN':		
			data_dict[name]['std_from_poi_to_this_person'] = float(from_poi)/float(to_msg)
		else:
			data_dict[name]['std_from_poi_to_this_person'] = 'NaN'

## New feature list	
features_list = ['poi', 'salary', 'total_payments', 'bonus',   
		'deferred_income', 'expenses',  'shared_receipt_with_poi',
		'std_from_this_person_to_poi', 'std_from_poi_to_this_person']

from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## Split into a training and testing set
features_train, features_test, labels_train, labels_test = \
		train_test_split(features, labels, test_size=0.25, random_state=42)
		
## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

'''

## Aplying PCA to feature extract
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
features_train = pca.fit_transform(features_train)
features_test = pca.transform(features_test)
explained_variance = pca.explained_variance_ratio_
#print(explained_variance)

#print(features_train)
#selector = SelectPercentile(f_classif, percentile=10)
#selector.fit(features, features_list)

#print(selector.pvalues_)



'''






'''
#############################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#############################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#############################################################################	
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

'''