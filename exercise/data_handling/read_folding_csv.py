# import statements
import matplotlib.pyplot as plt
import os   
import csv

# TODO: read "intelliperf/data/mg2/mg2.slope.csv"
userhome = os.path.expanduser('~')

csvfile= userhome + r'/Documents/intelliperf/data/mg2/mg2.slope.csv'

# TODO: generate a matplotlib line plot of a particular hardware counter data
time_stamp = set()
hardware_counter_dict = {}
temp_list = []

with open(csvfile,'r') as f:
    plots=csv.reader(f,delimiter=';')
    for row in plots:
        time_stamp.add(row[3])
        key = row[2]
        if key in hardware_counter_dict:
            temp_list = hardware_counter_dict[key]
            temp_list.append(row[4])
            temp_list = []
        else:
            temp_list.append(row[4])
            hardware_counter_dict[key] = temp_list
            temp_list = []
   
for key in hardware_counter_dict:
        print key

choice = raw_input("Please enter the hardware counter name")

time_stamp_list = list(time_stamp)
time_stamp_list.sort()

if choice in hardware_counter_dict:
    temp_list = hardware_counter_dict[choice]       
else:
    print "Entered wrong name"

#Plot and label Graph
plt.plot(temp_list,time_stamp_list)
plt.ylabel("time")
plt.xlabel("Usage")
plt.title("")
plt.show()

# TODO: save the plot as an image file in a file system.
plt.savefig("plot.png")