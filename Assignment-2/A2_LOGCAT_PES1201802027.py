import re
import datetime as dt
import pandas as pd
import sys
path=sys.argv[1]
f = open(path,"r")
lines=f.readlines()
pat1="Starting Service .*$" # Regex for service starting using anchor concept
pat2="Ending Service .*$" # Regex for service ending using anchor concept
d={}
for line in lines:
    m=re.search(pat1,line) # Search for service start
    if m:
        l=line.split()
        date,time,ser=l[0],l[1],l[6]
        timestamp_start=date+" "+time # Joining in the same format as the output is required
        d[ser]=timestamp_start
    m1=re.search(pat2,line) # Search for service end
    if m1:
        l=line.split()
        date,time,ser=l[0],l[1],l[6]
        timestamp_end=date+" "+time
        if ser in d:
            a=pd.to_datetime(d[ser])
            b=pd.to_datetime(timestamp_end)
            print(ser+","+d[ser]+","+timestamp_end+","+str((b-a).total_seconds()*1000))
f.close()
# Input in the directory of the files must be : python A2_LOGCAT_PES1201802027.py sample_logcat.txt