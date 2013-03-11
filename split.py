'''
split a file into two randomly, line by line.
Usage: split.py <input file> <output file 1> <output file 2> <output file 3> [<probability of writing to the first file> <probability of writing to the second file>]'
'''
#based on fastml code for predicting job salaries https://github.com/zygmuntz/kaggle-advertised-salaries.git

import csv
import sys
import random
from data_io import get_paths
from os.path import join as path_join
import os

try:
    P_train = float(sys.argv[5])
    P_validation = float(sys.argv[6])
except IndexError:
    P_train = 0.6
    P_validation = 0.2

print "P train = %s %%" % (P_train * 100)
print "P validation = %s %%" % (P_validation * 100)
print "P test = %s %%" % ((1 - P_validation - P_train) * 100)
paths = get_paths("Settings_submission.json")

input_file = sys.argv[1]
output_file1 = path_join(paths["data_path"], "data/processed", sys.argv[2])
output_file2 = path_join(paths["data_path"], "data/processed", sys.argv[3])
output_file3 = path_join(paths["data_path"], "data/processed", sys.argv[4])

print "Input: %s " % input_file
print "Train file: %s " % output_file1
print "Validation file: %s " % output_file2
print "Test file: %s " % output_file3

run = raw_input("OK (Y/N)?")
print run
if run != "Y":
    os.exit()


i = open(input_file)
o1 = open(output_file1, 'wb')
o2 = open(output_file2, 'wb')
o3 = open(output_file3, 'wb')

reader = csv.reader(i)
writer1 = csv.writer(o1)
writer2 = csv.writer(o2)
writer3 = csv.writer(o3)

#headers = reader.next()
#writer1.writerow(headers)
#writer2.writerow(headers)
#writer3.writerow(headers)

counter = 0

random.seed(42)

for line in reader:
    r = random.random()
    if r < P_train:
        writer1.writerow(line)
    elif r < (P_train + P_validation):
        writer2.writerow(line)
    else:
        writer3.writerow(line)

    counter += 1
    if counter % 100000 == 0:
        print counter
