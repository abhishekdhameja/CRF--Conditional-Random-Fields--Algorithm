from nltk import word_tokenize, pos_tag
from collections import defaultdict
import ast

filename = 'Abhishek_Dhameja_hw4.json'
with open(filename, 'r') as myfile:
    data=myfile.read().replace('\n', '')

data = ast.literal_eval(data)
count = 0
job_desc_list = defaultdict(list)

for json_value in data:
    #json_value = json.loads(jsonline)
    for key in json_value.keys():
        value = json_value[key]
        count += 1
        if value.strip() != "":
            job_desc_list[count].append(value.strip().encode('ascii', 'ignore').decode('ascii'))


for key in job_desc_list.keys():
    job_desc_list[key] = (" ").join(job_desc_list[key])


for key in job_desc_list.keys():
    text = word_tokenize(job_desc_list[key])
    job_desc_list[key] = pos_tag(text)


for key in job_desc_list.keys():
    job_desc_list[key] = [value + tuple('I') for value in job_desc_list[key]]

train_tags = open('Abhishek_Dhameja_train_tags.txt', 'w')
test_tags = open('Abhishek_Dhameja_test_tags.txt', 'w')

for count in range(1, 52):
    train_tags.write(str(job_desc_list[count]))
    train_tags.write("\n")
for count in range(52, 74):
    test_tags.write(str(job_desc_list[count]))
    test_tags.write("\n")
