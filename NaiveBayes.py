import io
import os
from nltk.tokenize import wordpunct_tokenize
import string
from nltk.stem import WordNetLemmatizer
import pandas as pd
import math

from os import listdir
from os.path import isfile, join
invalidChars = set(string.punctuation)
inp=''

# path for all the training data sets
spam_path = "E:\\UTD\\Semester 2\\Machine Learning\\Assignment-2\\train\\spam"
ham_path = "E:\\UTD\\Semester 2\\Machine Learning\\Assignment-2\\train\\ham"
test_path_ham="E:\\UTD\\Semester 2\\Machine Learning\\Assignment-2\\test\\ham"
test_path_spam="E:\\UTD\\Semester 2\\Machine Learning\\Assignment-2\\test\\spam"
tot_ham_length=0
tot_spam_length=0
def get_words(message):
    
    word_set=[]
    lemmatizer = WordNetLemmatizer()   
    all_words = wordpunct_tokenize(message.replace('=\\n', '').lower())
    for i in range(len(all_words)):
        if all_words[i].isalpha():
            word_set.append(lemmatizer.lemmatize(all_words[i]))   
    
    return word_set
    
def get_mail_from_file(file_name):
    message = ''
    
    with io.open(file_name, 'r',encoding="mbcs") as mail_file:
        for line in mail_file:
            message += line
    return message

def removeStopWords(data_set):
    stops = []
    with open("E:\\UTD\\Semester 2\\Machine Learning\\Assignment-2\\stop_words.txt", 'r') as txt:
        stops = (txt.read().splitlines())
    for i in stops:
            if i in data_set.keys():
                del data_set[i]       
    return data_set
        
def make_training_set(path):
    
    training_set = {}

    mails_in_dir = [mail_file for mail_file in listdir(path) if isfile(join(path, mail_file))]

    # total number of files in the directory
    total_file_count = len(mails_in_dir)
    
    for mail_name in mails_in_dir:
        
        message = get_mail_from_file(path +'\\'+ mail_name)
        terms = get_words(message)
        for term in terms:
            if term in training_set:
                training_set[term] = training_set[term] + 1
            else:
                training_set[term] = 0
    
    for term in training_set.keys():
        training_set[term] = float(training_set[term])
    
    for term in training_set.keys():
        if term.isdigit() or term in invalidChars:
            del training_set[term]
                            
    return training_set,total_file_count

    
def probability_of_ham(value,hts,sts):
    occurances_in_ham=0
    occurances_in_ham=hts.get(value)  
    total_length=len(set(sts.keys()).union(set(hts.keys()))) 
    length_of_ham=sum(hts.values())
    
    prob_ham=(occurances_in_ham+1)/(length_of_ham+total_length)

    return (prob_ham)
    
def probability_of_spam(value,hts,sts):    
    occurances_in_spam=0
    occurances_in_spam=sts.get(value) 
    total_length=len(set(sts.keys()).union(set(hts.keys())))   
    length_of_spam=sum(sts.values())
    prob_spam=(occurances_in_spam+1)/(length_of_spam+total_length)
    
    return(prob_spam) 
      
def training():
     ham=[]
     spam=[]
     table_ham=pd.DataFrame()
     table_spam=pd.DataFrame()
     
     inp=raw_input("Do you want the stop words filtered? Type 'yes' or 'no': ")
     print 'Loading training sets...'
     
     global tot_spam_length
     spam_training_set,tot_spam_length = make_training_set(spam_path)
     
     global tot_ham_length
     ham_training_set,tot_ham_length = make_training_set(ham_path)

     if inp=="yes":
        spam_training_set = removeStopWords(spam_training_set)
        ham_training_set = removeStopWords(ham_training_set) 
     print 'done.'
    
     i=0
     for term in spam_training_set.keys():
        if spam_training_set.get(term) <10 :
            del spam_training_set[term]
        i=i+1
     i=0
     for term in ham_training_set.keys():
        if ham_training_set.get(term) <10 :
            del ham_training_set[term]
        i=i+1
            
     column_list_ham=ham_training_set.keys()
     for val in column_list_ham:
         #print val
         ham_prob=probability_of_ham(val,ham_training_set,spam_training_set)
         ham.append(ham_prob)
         
     column_list_spam=spam_training_set.keys()
     for val in column_list_spam:
         spam_prob=probability_of_spam(val,ham_training_set,spam_training_set)
         spam.append(spam_prob)
     #print ham   
     table_ham=pd.DataFrame(column_list_ham) 
     table_spam=pd.DataFrame(column_list_spam)
     table_ham['Probability']=pd.DataFrame(ham)
     table_ham['Class']='ham'
     table_spam['Probability']=pd.DataFrame(spam)
     table_spam['Class']='spam'
     table = table_ham.append(table_spam, ignore_index=True)
     
     return table, spam_training_set, ham_training_set

def prob_test(tab,email):
    ham_test=[]
    total=tot_ham_length+tot_spam_length
    for mail in email:
        #print mail
        words=[]
        class_ham=0.0
        i=0 
        with io.open(mail,'r',encoding="mbcs") as m:
            for line in m:
                words=line.split()
                if inp=='yes':
                    words=removeStopWords(words)
                for word in words:
                        if word in tab[0][i] and tab['Class'][i]=='ham':
                            class_ham+=math.log(tab['Probability'][i])
                            i+=1
            prob_class_ham=math.log(tot_ham_length/float(total))+class_ham
            ham_test.append(prob_class_ham)  
                  
    spam_test=[]
    for mail in email:
    
        words=[]
        class_spam=0.0
        i=0 
        with io.open(mail,'r',encoding="mbcs") as m:
            for line in m:
                words=line.split()
                if inp=='yes':
                    words=removeStopWords(words)
                for word in words:
                        if word in tab[0][i] and tab['Class'][i]=='spam':
                            class_spam+=math.log(tab['Probability'][i])
                            i=i+1
            prob_class_spam=math.log(tot_spam_length/float(total))+class_spam
            spam_test.append(prob_class_spam)
    return (ham_test,spam_test) 
       
def testing(tab):
    ham=[]
    spam=[]
    print("Testing now...")
    email_ham = [os.path.join(test_path_ham,f) for f in os.listdir(test_path_ham)]
    ham,spam=prob_test(tab,email_ham)
    count_ham=count_spam=0 
           
    for i in range(len(email_ham)):
        if ham[i]>spam[i]:
            count_ham+=1
        else:
            count_spam+=1
            
    email_spam = [os.path.join(test_path_spam,f) for f in os.listdir(test_path_spam)]
    sham,sspam=prob_test(tab,email_spam)
    scount_ham=scount_spam=0        
    for i in range(len(email_spam)):
        if sham[i]>sspam[i]:
            scount_ham+=1
        else:
            scount_spam+=1
            
    accuracy=(scount_spam+count_ham)/(float(len(email_spam)+len(email_ham)))
    print ("Accuracy:%.4f%%" % (100.0 * accuracy))
            
if __name__ == "__main__":    
    
    matrix,sts,hts=training()
    
    testing(matrix)