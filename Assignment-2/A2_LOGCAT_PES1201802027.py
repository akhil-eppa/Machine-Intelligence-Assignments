import re
import datetime as dt
f = open("sample_logcat.txt", "r")
lines=f.readlines()
pat1="Starting Service .*$"
pat2="Ending Service .*$"
d={}
for line in lines:
    m=re.search(pat1,line)
    if(m):
        l=line.split()
        date,time,ser=l[0],l[1],l[6]
        datetime=date+" "+time
        d[ser]=datetime
    m1=re.search(pat2,line)
    if(m1):
        l1=line.split()
        date,time,ser=l1[0],l1[1],l1[6]
        datetime1=date+" "+time
        if ser in d:
            print(ser+","+d[ser]+","+datetime1+","+"duration(msec)")
        #https://www.geeksforgeeks.org/python-difference-between-two-dates-in-minutes-using-datetime-timedelta-method/#:~:text=To%20find%20the%20difference%20between,difference%20between%20two%20datetime%20objects.
