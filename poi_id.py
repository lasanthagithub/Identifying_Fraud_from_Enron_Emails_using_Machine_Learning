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
Main features
'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
'director_fees',

'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi'

'poi'
'''
#data_dict = {} ## Create an empty dict
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus',  'deferred_income', 'total_stock_value', 
'expenses', 'long_term_incentive', 'to_messages',  'from_poi_to_this_person', 
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

## This function take a specific dictionary
## and generates boxplots
def create_boxplot(dic):
	plt.figure()
	for key in dic.keys():
		vals = dic[key]
		plt.boxplot(vals)
		plt.xlabel(key)
		plt.ylabel("Values")
		plt.show()
	
def remove_outliers():
	pass
'''	
for name in data_dict.keys():
	sub_dic = data_dict[name]
	sub = ''
	value_dic = []
	for sub_key in sub_dic.keys():
		if sub_key in features_list and sub_key != 'poi':
			value_dic.append(sub_dic[sub_key])
			sub = sub_key
	#		print(name)
	#		print(sub_key)
	#print(value_list)
'''
	
## Getting labels and features as numpy arrays 		
data_array = featureFormat(data_dict, features_list, sort_keys = False)	
label, features = targetFeatureSplit(data_array)
print(len(features))

### your code below
#print(data_dict[1])
value_dic = {}

## convert the same feature values append into the fature name. give a dict 
## Navigate through each each poi record 
for j, point in enumerate(data_array):
	val_list = []
	## Navigate through each value in a record and collect similar values 
	## to corresponding feature name. This gives a dictionary
	for i, featu in enumerate(features_list):
		if j == 0: ## to remove 'poi' record
			value_dic[featu] = [point[i]]
		else:
			value_dic[featu] = value_dic[featu] + [point[i]]

create_boxplot(value_dic)
	
'''
	plt.boxplot(point)
	#print(point)

	plt.xlabel("salary")
	plt.ylabel("bonus")
plt.show()

'''

'''
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

'''