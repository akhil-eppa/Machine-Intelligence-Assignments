import re
import sys

def mymatchindex(pat, lists):
    index=-1
    for i in lists:
        index+=1
        m=re.search(pat,i)
        if m:
            return index
def mymatchfind(pat, lists):
    find=0
    for i in lists:
        m=re.search(pat,i)
        if m:
            find+=1
            return find
    return find

def mymatch(pat, texts):
	print("pattern : ", pat)
	for text in texts:
		m = re.search(pat, text)
		if m :
			#print(text, ":", m.group())
			return m.groups()  # tuple of strings
path="sample_shell.txt"
#path="sample_shell_test1.txt" #For testing purposes
f = open(path,"r",encoding="UTF8")
lines=f.readlines()
for line in lines:
    print(line)
pat_if="^if.*$"
pat_fi="^fi$"
pat_shebang="^#!.*$"
pat_comment="^#.*$"
start_if=0 #Counter to keep track of the nested ifs
l=[]
for line in lines: #Removing the if-fi blocks
    line=line.strip() #Removes whitespaces(Needed for if-fi otherwise whitespaces will cause an issue)
    if(re.search(pat_if,line)):
        line=re.sub(pat_if,"",line)
        start_if+=1
    elif(re.search(pat_fi,line)):
        line=re.sub(pat_fi,"",line)
        start_if-=1
    elif(start_if!=0):
        line=""
    if line!="": #Removes empty lines stripped of whitespaces
        l.append(line)
#print("\n".join(l)) # Printing in 1 line without using loop
for i in l:
    print(i)
pat_fun="\(\)$"
pat_funop="^\{$"
pat_funcl="^\}$"
match_index_prev=0
match_index_new=0
#(NOTE: Functions will not be nested unlike the if-fi blocks)
while mymatchfind(pat_fun,l)!=0: #Removing empty/redundant functions
    match_index_new=mymatchindex(pat_fun, l)
    if(match_index_prev!=match_index_new):
        a = match_index_new
        b = mymatchindex(pat_funop, l)
        c = mymatchindex(pat_funcl, l)
        if(b-a==1 and c-b==1 and c-a==2):
            del l[a:c+1]
        match_index_prev=match_index_new
    else:
        break
for i in l:
    print(i)
dict = {}

new_l=[]   
for line in l: #
    line=line.strip() #Removes whitespaces
    if(re.search(pat_comment,line) and not re.search(pat_shebang,line)):
        line=re.sub(pat_comment,"",line)
    if line!="": #Removes empty lines stripped of whitespaces also
        new_l.append(line)
for i in new_l:
    print(i)  

a= mymatchindex(pat_fun, new_l)
b= mymatchindex(pat_funop, new_l) 
c= mymatchindex(pat_funcl, new_l)

if(a and b and c and b-a==1 and c-b!=1 and c-a!=2):
    for j in range(b+1,c):
        new_l[j]="\t"+new_l[j] # Rearranging tab-spaces in non-redundant functions (Considering no nested functions)

for i in new_l:
    print(i)
f.close()