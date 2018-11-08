# This program is used to parse the contents of the data file into a csv
# file usable by the model.
#
# The input data should be distributed into 5 columns per sample:
# sepal_length, sepal_width, petal_length, petal_width, label
import csv

with open('iris.data', 'r') as in_file:
    clean_line = (line.strip() for line in in_file)
    lines = (line.split(",") for line in clean_line if line)
    with open('data.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('sep_len','sep_wid','pet_len','pet_wid','label'))
        writer.writerows(lines)
