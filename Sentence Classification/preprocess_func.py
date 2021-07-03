import os
import tensorflow as tf

# Create function to read the lines of a document
def get_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()

def preprocess_text_with_line_numbers(filename):
    input_line = get_lines(filename) #get all file from file name
    abstract_line = "" #create an empty abstract
    abstract_samples = [] #create an empty list of abstracts

    # Loop through each line in target file
    for line in input_line:
        if line.startswith("###"):# check to see if line is an ID lines
            abstract_id = line
            abstract_lines = "" # reset abtracts string
        elif line.isspace():
            abstract_line_split = abstract_lines.splitlines() # split abstract into separate lines

            # Iterate through each line in abstract and count them at the same time
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {} # create empty dict to store data from line
                target_text_split = abstract_line.split("\t") # split target label from text
                line_data["target"] = target_text_split[0] # get target label
                line_data["text"] = target_text_split[1].lower() # get target text and lower it
                line_data["line_number"] = abstract_line_number # what number line does the line appear in the abstract?
                line_data["total_lines"] = len(abstract_line_split) - 1 # how many total lines are in the abstract? (start from 0)
                abstract_samples.append(line_data) # add line data to abstract samples list

        else: # if the above conditions aren't fulfilled, the line contains a labelled sentence
            abstract_lines += line
    return abstract_samples


# Make function to split sentences into characters
def split_chars(text):
  return " ".join(list(text))
