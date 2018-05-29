# reading extrae raw data

import sys
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import operator
import collections
import pprint
# event type ranges
FOLDED_SAMPLING_CALLER_LINE = range(631000100, 631000200)

class ExtraeRowData(object):

    def __init__(self, path):
        # NOTE: not used for now
        pass

class ExtraePcfData(object):

    def __init__(self, path):

        self.events = {}

        BLANK, HEAD, VALUE = 0, 1, 2
        state = BLANK

        with open(path, 'r') as fh:

            eventtypes = []
            eventvalues = {}

            for line in fh:

                line = line.strip()
                if len(line) == 0:
                    state = BLANK
                    continue

                if state == BLANK:
                    if line == "EVENT_TYPE":
                        for eventtype in eventtypes:
                            eventtype['values'] = eventvalues
                        eventtypes = []
                        eventvalues = {}
                        state = HEAD
                elif state == HEAD:
                    if line == "VALUES":
                        state = VALUE
                    else:
                        items = line.split()
                        if len(items) > 2:
                            eventtype = {'desc': ' '.join(items[2:])}
                            eventtypes.append(eventtype)
                            self.events[int(items[1])] = eventtype
                elif state == VALUE:
                    items = line.split()
                    if len(items) > 2:
                        eventvalues[int(items[0])] = (items[1], ' '.join(items[2:]))
                    elif len(items) == 2:
                        eventvalues[int(items[0])] = ' '.join(items[1:])

            for eventtype in eventtypes:
                eventtype['values'] = eventvalues

class ExtraePrvData(object):

    def __init__(self, path):

        self.events = OrderedDict()

        with open(path, 'r') as fh:

            for line in fh:

                line = line.strip()

                if len(line) == 0:
                    continue

                if line[0] != "2":
                    continue

                items = line.split(":")

                timestamp = int(items[5])

                prvevent = OrderedDict()
                self.events[timestamp] = prvevent

                for etype, evalue in zip(items[6::2], items[7::2]):
                    prvevent[int(etype)] = int(evalue)

class ExtraeRawData(object):


    def __init__(self, path):

        self.row = None
        self.pcf = None
        self.prv = None

        # check path
        root, ext = os.path.splitext(path)
        if ext in (".prv", ".pcf", ".row"):
            path = root

        # read row
        rowpath = path + ".row"
        if os.path.isfile(rowpath):
            self.row = ExtraeRowData(rowpath)

        # read pcf
        pcfpath = path + ".pcf"
        if os.path.isfile(pcfpath):
            self.pcf = ExtraePcfData(pcfpath)

        # read prv
        prvpath = path + ".prv"
        if os.path.isfile(prvpath):
            self.prv = ExtraePrvData(prvpath)

    def get_prv_events(self, eventrange):
        for timestamp, events in self.prv.events.items():
            for eventtype, eventvalue in events.items():
                if eventtype in eventrange:
                    yield timestamp, eventtype, eventvalue

    def get_prv_events_by_timestamp(self, timestamp):
        return self.prv.events[timestamp]

    def get_pcf_events(self, eventrange):
        for eventype, (desc, values) in self.pcf.events.items():
            if eventype in eventrange:
                yield desc, values

    def get_pcf_event(self, eventtype):
        return self.pcf.events[eventtype]


    def get_folded_sampling_caller_lines(self):

        lineidcounts = {}
        callerlevels = {}

        prev_timestamp = None
        deepest_callstack = None 

        for timestamp, callerlevel, lineid in self.get_prv_events(FOLDED_SAMPLING_CALLER_LINE):

            if callerlevel not in callerlevels:
                callerlevels[callerlevel] = {}
            if lineid not in callerlevels[callerlevel]:
                callerlevels[callerlevel][lineid] = None

            if timestamp != prev_timestamp:
                if prev_timestamp is not None:
                    if deepest_callstack[1] in lineidcounts:
                        lineidcounts[deepest_callstack[1]] += 1
                    else:
                        lineidcounts[deepest_callstack[1]] = 1
                deepest_callstack = (callerlevel, lineid)
            elif callerlevel < deepest_callstack[0]:
                deepest_callstack = (callerlevel, lineid)
            prev_timestamp = timestamp

        if prev_timestamp is not None:
            if deepest_callstack[1] in lineidcounts:
                lineidcounts[deepest_callstack[1]] += 1
            else:
                lineidcounts[deepest_callstack[1]] = 1

        lineidsource = self.get_pcf_event(callerlevel)['values']

        return lineidcounts, callerlevels, lineidsource

def main():

    # per every paths provided in a command line
    for path in sys.argv[1:]:

        # read extrae raw data
        extraeraw = ExtraeRawData(path)
        
        # get mappings of:
        # lineidcounts: lineid -> number of samples
        # callerlevels: eventtype -> { lineid -> None }
        # lineidsource: lineid -> a string that contains line number and source file name
        lineidcounts, callerlevels, lineidsource = extraeraw.get_folded_sampling_caller_lines()
        fig,axs = plt.subplots(2,1)
 
        # TODO: plot bar graph of top10 lineidcounts
        newlineidcounts = dict(sorted(lineidcounts.iteritems(),key = operator.itemgetter(1),reverse = True)[:10])
        
        newlineidcounts = collections.OrderedDict(sorted(newlineidcounts.items()))
        resultIdList = []
        resultValueList = []
        for key,value in newlineidcounts.iteritems():
            resultIdList.append(str(key))
            resultValueList.append(value)
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        axs[0].bar(resultIdList,resultValueList)
        axs[0].text(left, 0.5*(bottom+top), 'left center',
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=axs[0].transAxes)
        plt.xlabel('Lined ID')
        plt.ylabel('Number of Sample')
        plt.title('Top 10 Line Id and Samples',y = 2.17 )
        
       # TODO: plot text for lineid to source file and linenum mapping for the top10 lineids
       
        lineIdLineNum = []
        lineIdFileName = []
        for x in resultIdList:
            val = lineidsource.get(int(x))
            lineIdLineNum.append(int(val[0]))
            if '[' in val[1]:
                fileName = str(val[1]).split('[')[0].replace("(","").replace(")","").strip()
            else :
                fileName = str(val[1]).split(',')[0].replace('(','').strip()
            lineIdFileName.append(fileName)    

        w, h = 3, 10;
        Matrix = [[0 for x in range(w)] for y in range(h)] 
        labelr = ['1','2','3','4','5','6','7','8','9','10']
        labelc = ['Line ID', 'Line Number','Source File']
        
        for i in range(len(Matrix)):
            for j in range(len(Matrix[i])):
                if j == 0 :
                    Matrix[i][j] = resultIdList[i]
                if j == 1 :
                    Matrix[i][j] = lineIdLineNum[i]
                if j == 2 :
                    Matrix[i][j] = lineIdFileName[i]
        
        lightgrn = (0.5, 0.8, 0.5)
        axs[1].table(cellText = Matrix,
                  rowLabels=labelr,
                  colLabels=labelc,
                  rowColours=[lightgrn] * 16,
                  colColours=[lightgrn] * 16,
                  cellLoc='center',
                  loc='center')
        axs[1].axis('off')
        axs[0].axis('tight')
        plt.show()
        
        
        return 0
          
if __name__ == "__main__":
    sys.exit(main())
