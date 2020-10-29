import re
import sys

path=sys.argv[1]
f = open(path,"r",encoding="UTF8")
lines=f.readlines()
for line in lines:
    print(line)
pat_if="^if.*"
pat_fi="^fi$"
start_if=0
for line in lines:
    line=line.strip()
    if(re.search(pat_if,line)):
        line=re.sub(pat_if,"",line)
        start_if+=1
    elif(re.search(pat_fi,line)):
        line=re.sub(pat_fi,"",line)
        start_if-=1
    elif(start_if!=0):
        line=""
    print(line) 


#pat_if="^if"
#lines=re.sub(pat_if,"",lines)
#lines1=re.sub(pat_if,"",lines)
#print(lines)