import os
import collections
import re
import math
import copy

spam_path = "E:\\UTD\\Semester 2\\Machine Learning\\Assignment-2\\train\\spam"
ham_path = "E:\\UTD\\Semester 2\\Machine Learning\\Assignment-2\\train\\ham"
test_path_ham="E:\\UTD\\Semester 2\\Machine Learning\\Assignment-2\\test\\ham"
test_path_spam="E:\\UTD\\Semester 2\\Machine Learning\\Assignment-2\\test\\spam"
lambda_constant=1.0

training_set = dict()
test_set = dict()

words_training_set = []
weights = {'W0': 0.0}
classes = ["ham", "spam"]
learning_constant = .001
penalty = 0.0

class Document:
    text = ""
    # x0 is assumed to be 1 for all training examples
    word_freqs = {'W0': 1.0}
    true_class = ""
    obtained_class = ""
    def __init__(self, text, counter, true_class):
        self.text = text
        self.word_freq = counter
        self.true_class = true_class
    def getText(self):
        return self.text
    def wordFreq(self):
        return self.word_freq
    def trueClass(self):
        return self.true_class
    def obtainedClass(self):
        return self.obtained_class
    def setLearnedClass(self, result):
        self.obtained_class = result
        
# Function to read all the text files and construct the data set
def make_data_set(storage_dict, directory, true_class):
    for dir_entry in os.listdir(directory):
        dir_entry_path = os.path.join(directory, dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path, 'r') as text_file:
                text = text_file.read()
                storage_dict.update({dir_entry_path: Document(text, word_bag(text), true_class)})

# Function to extract all the words from the data set
def extract_words(dataset):
    word_set = []
    for i in dataset:
        for j in dataset[i].wordFreq():
            if j not in word_set:
                word_set.append(j)
    return word_set

# Remove stop words from data set and stores in dictionary
def removeStopWords(data_set):
    stops = []
    with open("E:\\UTD\\Semester 2\\Machine Learning\\Assignment-2\\stop_words.txt", 'r') as txt:
        stops = (txt.read().splitlines())
    filtered_data_set = copy.deepcopy(data_set)
    for i in stops:
        for j in filtered_data_set:
            if i in filtered_data_set[j].wordFreq():
                del filtered_data_set[j].wordFreq()[i]
    return filtered_data_set

# Function to count the frequency of each word
def word_bag(text):
    word = collections.Counter(re.findall(r'\w+', text))
    return dict(word)

# Function to learn weights by using gradient ascent
def gradient_ascent(training, weights_param, iteration, lam):
    # Adjust weights num_iterations times
    for x in range(0, iteration):
        for w in weights_param:
            sums = 0.0
            for i in training:
                y = 0.0
                if training[i].trueClass() == classes[1]:
                    y = 1.0
                if w in training[i].wordFreq():
                    sums += float(training[i].wordFreq()[w]) * (y - conditional_Probabilty(classes[1], weights_param, training[i]))
            weights_param[w] += ((learning_constant * sums) - (learning_constant * float(lam) * weights_param[w]))

# Function to calculate conditional probability
def conditional_Probabilty(class_prob, weights_param, d):
    # If class = Ham:
    if class_prob == classes[0]:
        sum_wx_0 = weights_param['W0']
        for i in d.wordFreq():
            if i not in weights_param:
                weights_param[i] = 0.0
            sum_wx_0 += weights_param[i] * float(d.wordFreq()[i])
        return 1.0 / (1.0 + math.exp(float(sum_wx_0)))

    # If class is spam:
    elif class_prob == classes[1]:
        sum_wx_1 = weights_param['W0']
        for i in d.wordFreq():
            if i not in weights_param:
                weights_param[i] = 0.0
            sum_wx_1 += weights_param[i] * float(d.wordFreq()[i])
            
        return math.exp(float(sum_wx_1)) / (1.0 + math.exp(float(sum_wx_1)))

# Function to apply Logistic regression
def LRClassifier(data_instance, weights_param):
    ham=spam=0
    ham = conditional_Probabilty(classes[0], weights_param, data_instance)
    spam = conditional_Probabilty(classes[1], weights_param, data_instance)
    if spam > ham:
        return classes[1]
    else:
        return classes[0]

if __name__ == '__main__':

################ Training ###########################
        
    make_data_set(training_set, spam_path, classes[1])
    make_data_set(training_set, ham_path, classes[0])
    make_data_set(test_set, test_path_spam, classes[1])
    make_data_set(test_set, test_path_ham, classes[0])
    inp=''
    num_iterations=0
    num_iterations=input("How many iterations do you want? ")
    inp=raw_input("Do you want the stop words filtered? Type 'yes' or 'no': ")
    
    if inp=="yes":
        training_set = removeStopWords(training_set)
        test_set = removeStopWords(test_set)

    words_training_set = extract_words(training_set)
    for i in words_training_set:
        weights[i] = 0.0
    gradient_ascent(training_set, weights, num_iterations, lambda_constant)

################## Testing #############################

    TP = 0.0
    for i in test_set:
        test_set[i].setLearnedClass(LRClassifier(test_set[i], weights))
        if test_set[i].obtainedClass() == test_set[i].trueClass():
            TP += 1.0
    print "Accuracy:%.4f%%" % (100.0 * float(TP) / float(len(test_set)))