import os
import shutil
import sys
from split_and_shuffle_data import Split_Data

datasetName = sys.argv[1]
Split_Data.splitdata(datasetName)


from bayes import Bayes_Classifier

trainDir = "training/"
testDir = "testing/"


bc = Bayes_Classifier(trainDir)
print("\n ------------------------Running the Bayes Classifier-------------------")

iFileList = []

for fFileObj in os.walk(testDir + "/"):
    iFileList = fFileObj[2]
    break

results = {"negative": 0, "neutral": 0, "positive": 0}
true_negatives_count = 0
true_positives_count = 0
false_negatives_count = 0
false_positive_count = 0

for filename in iFileList:

    try:
        fileText = bc.loadFile(testDir + filename)
        classifier_result = bc.classify(fileText)
        original_result = filename.split('-')[1]
    except:
        print(filename)

    if original_result == '1' and classifier_result == 'negative':
        true_negatives_count += 1
    elif original_result == '1' and classifier_result != 'negative':
        false_negatives_count += 1
    elif original_result == '5' and classifier_result == 'positive':
        true_positives_count += 1
    elif original_result == '5' and classifier_result != 'positive':
        false_positive_count += 1
    results[classifier_result] += 1

print("\nResults Summary:")
for r in results:
    print("%s: %d" % (r, results[r]))

total_positives_count = true_positives_count + true_negatives_count
total_negatives_count = false_positive_count + false_negatives_count

# Calculate and return accuracy, precision, and recall
Accuracy = 100 * total_positives_count / (total_negatives_count + total_positives_count)
precision = 100 * true_positives_count / (true_positives_count + false_positive_count)
Recall = 100 * (true_positives_count) / (true_positives_count + false_negatives_count)
fMeasure = (2 * precision * Recall) / (precision + Recall)
print("Classification Accuracy:  %.2f%%" % Accuracy)
print("Classification Precision: %.2f%%" % precision)
print("Classification Recall:    %.2f%%" % Recall)
print("Classification F-measure: %.2f%%" % fMeasure)

print("\n -------------------------Bayes Classifier End---------------------------")

#-----------------------------------------------------------------------

import os
from bayes_best import Bayes_Best_Classifier

trainDir = "training/"
testDir = "testing/"

bc = Bayes_Best_Classifier(trainDir)

print("\n ------------------------Running Best Bayes Classifier-------------------")

iFileList = []

for fFileObj in os.walk(testDir + "/"):
    iFileList = fFileObj[2]
    break

results = {"negative": 0, "neutral": 0, "positive": 0}
true_negatives_count = 0
true_positives_count = 0
false_negatives_count = 0
false_positive_count = 0

for filename in iFileList:

    try:
        fileText = bc.loadFile(testDir + filename)
        classifier_result = bc.classify(fileText)
        original_result = filename.split('-')[1]
    except:
        print(filename)

    if original_result == '1' and classifier_result == 'negative':
        true_negatives_count += 1
    elif original_result == '1' and classifier_result != 'negative':
        false_negatives_count += 1
    elif original_result == '5' and classifier_result == 'positive':
        true_positives_count += 1
    elif original_result == '5' and classifier_result != 'positive':
        false_positive_count += 1
    results[classifier_result] += 1

print("\nBest Results Summary:")
for r in results:
    print("%s: %d" % (r, results[r]))

total_positives_count = true_positives_count + true_negatives_count
total_negatives_count = false_positive_count + false_negatives_count

# Calculate and return accuracy, precision, and recall
Accuracy = 100 * total_positives_count / (total_negatives_count + total_positives_count)
precision = 100 * true_positives_count / (true_positives_count + false_positive_count)
Recall = 100 * (true_positives_count) / (true_positives_count + false_negatives_count)
fMeasure = (2 * precision * Recall) / (precision + Recall)
print("Best Classification Accuracy:  %.2f%%" % Accuracy)
print("Best Classification Precision: %.2f%%" % precision)
print("Best Classification Recall:    %.2f%%" % Recall)
print("Best Classification F-measure: %.2f%%" % fMeasure)


print("\n ------------Best Bayes Classifier End--------------------")
