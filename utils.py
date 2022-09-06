
import numpy as np
import matplotlib.pyplot as pt
import csv
import pandas as pd
from matplotlib import style
import time
import math

def getIDs(data):
    x = data.MessageID.unique().dropna()
    x.sort()
    msgIDs = [int(i) for i in x]

    return msgIDs

def compareIDs(data1,data2):
    """Compare the sets of CANIDs in two pandas datasets.
    Returns a exclude b, b exclude a, a union b.
    In general, use python sets of unique values to compare sets of things."""
    a=set(data1.MessageID.unique())
    b=set(data2.MessageID.unique())

    return a-b,b-a,a&b

def printFiles(data_files, labels=[]):
    """Input: data files, as pandas dataframes, which you want to compare the MessageID, HZ, MessageLength, and/or read bus location.
    Keep in mind that the recorded bus location is totally dependent on the composition of the wiring harness used.
    For recorded data, they have i/o in a pair: Bus 1 is a pair with itself, Bus 0/2 are a pair. """

    c=set()
    for d in data_files:
        temp = set(d.MessageID.dropna().unique())
        c = c|temp
    c=sorted(c)

    for i in c:
        print('\tMessageID\t HZ\t MessageLength\t Busses')
        count=1
        for d in data_files:
            if labels == []:
                printFileLine(d,i,label='File '+str(count))
            else:
                printFileLine(d,i,label=labels[count-1])
            count+=1
        print()

def printFileLine(data, ID, label='Data File'):
    i = ID
    my_data = data.loc[
        (data.MessageID == i
        )
        ].dropna()

    try:
        my_data_hz = len(my_data)/(int(my_data.Time.tail(1))- int(my_data.Time.head(1)))
        print(label,': ',i,'\t','\t',round(my_data_hz),'\t',data.loc[data.MessageID == i].MessageLength.unique()[0:],'\t\t',data.loc[data.MessageID == i].Bus.unique()[0:])
    except:
        test = 'test print failed'

def makebits(messageNameID,data):
    """Parameters are the CANID and the pandas dataframe which you want to convert into bits. This function will make a new column called 'bits'
    which contains the bit version of the specified message."""
    messageData = data.loc[data.MessageID == messageNameID].dropna()

    bitLength = data.loc[
        data.MessageID == messageNameID
    ].MessageLength.unique()[0]*8

    mybytes = messageData.Message.apply(lambda x: bin(int(x,16))[2:].zfill(int(bitLength)))
    messageData['bits']=mybytes

    return messageData

def findNewSignals(data, msgID, removeCounters = True, verbose = False, bitLength = 0):
    """Input: dataframe of can data and a single CAN message ID. Hexadecimal format.
    Output: List of estimated big endian sequntial multi-bit signal positions, and bit transformed input data."""

    msgs = data.loc[data.MessageID == msgID].dropna()
    if bitLength == 0:
        bitLength = getBitLength(msgID,data)
    mybytes = msgs.Message.apply(lambda x: bin(int(x,16))[2:].zfill(int(bitLength)))
    ##this is finding the bit flips per bit position
    bitFlips= bitFlipper(mybytes,bitLength, verbose=verbose)

    ## finding signals
    signals = []
    start = 0
    end = 0
    while start < len(bitFlips):
        #if there is a signal starting at this index, find the end
        end = start + 1
        if end < len(bitFlips)-1:
            while (bitFlips[end] > bitFlips[end-1]) and (end < len(bitFlips)-1):
                end +=1
            if end > start + 2:
                signals.append([start,end])
        #update the next place to start looking for a signal to the end of the signal found
        start = end

    ## looking to find counters:
    if removeCounters == True:
        for i in signals:
            bits = bitFlips[i[0]:i[1]]
            ratios = []
            for j in range(0,len(bits)-1):
                if bits[j] !=0:
                    ratios.append(bits[j+1]/bits[j])

            if len(ratios)>0:
                if abs(sum(ratios)/len(ratios) - 2)< 0.05: #removing counters
                    signals.remove(i)
    return signals,mybytes

def signalLocations(IdList,data , verbose = 1):
    """INPUT: List of CANIDs which you want to find some continuous signals, and pandas dataframe to look within.
    OUTPUT: dictionary of possible signal locations."""
    signalsDict = {}
    data = data

    for i in IdList:
        if i > 0: #add something to save a dictionary of msgIDs to signal positions so then you can go back and find them, or save them to load in later!
            try:
                print(i)
                potentialSignals,mybytes = findNewSignals(data,i,verbose=True)
                signalsDict.update({i:potentialSignals})
                if verbose ==1:
                    plotSigs(potentialSignals,mybytes,i)
            except:
                print('failed on ' + str(i))
    return signalsDict

def plotSigs(potentialSignals,mybytes, ID, verbose = 0, save = ''):
    """INPUT: Potential Signals, output from signalLocations function. mybytes is the bit transformed
    version of the CAN data in a pandas dataframe. ID is the CAN ID. verbose will toggle creating the figures at OUTPUT.
    save is a directory location locally to save the output figures, if you want.

    OUTPUT: Figures of signals in the locations specified by the potentialSignals list and found in the mybytes data.
    """

    m = math.ceil(len(potentialSignals)/3)
    if m > 0:
        pt.figure()
        fig,axs = pt.subplots(m,3)
        fig.suptitle("Possible Signals in "+str(ID))
    elif m==0:
        print('m=0, no signals in potentialSignals.')
    count=0
    for j in potentialSignals:
        t = [int(i[j[0]:j[1]],2) for i in mybytes]
        try:
            if m == 1:
                axs[count%3].plot(t,ls='',marker='.',markersize=10,label='index ' + str(j[0])+' to ' +str(j[1]))
                axs[count%3].legend()
            if m > 1:
                axs[math.floor(count/3),count%3].plot(t,ls='',marker='.',markersize=10,label='index ' + str(j[0])+' to ' +str(j[1]))
                axs[math.floor(count/3),count%3].legend()
        except:
            print('failed on ' + str(ID))
        count +=1
    if save != '':
        pt.savefig(save+'%d.png'%(ID))
    if verbose==0:
        pt.close()

def printFlips(ID,data):
    """Print out the bit flips at each bit position for visual inspection.
    INPUTS: CAN ID, and CAN data in a pandas dataframe"""

    m= makebits(ID,data)
    m_len = getBitLength(ID,data)
    mb = bitFlipper(m.bits,m_len,verbose=1, ID = ID)
    count = 0

    for i in range(0,len(mb)):
        if i%8 == 0:
            print('BYTE %d'%(count))
            count+=1
        print(i,mb[i])

def bitConversion(x):
    """For big endian signals, converting bit number to dbc bit position."""
    y = (int(x/8)+1)*8 -(x%8)-1
    return y

def plotMsgBits(data, range=[0,10000]):
    """This function plots the bit flip vs. bit position of canids in given data and range.
    INPUT: pandas dataframe of can data. range of can signals to plot.
    OUTPUT: plots of can bit flips, characterizing messages."""

    data = data
    msgids = np.sort(data.MessageID.loc[(data.MessageID>range[0]) & (data.MessageID<range[1])].unique())
    msgids = msgids.dropna()

    for i in msgids:
        print(i)
        plotSignal(i,data)

def makeCRC(ID,data):
    """This function labels the last 64 bits of a dataframe as 'crc' and creates
    a new column in the input dataframe to reflect this. The rest of the data payload
    is considered the data signals. Useful for analysis splitting up crc or checksum."""
    m= makebits(ID,data)
    CRC64 = [i[-64:] for i in m.bits]
    mydata = [i[:-64] for i in m.bits]
    m['crc'] = CRC64
    m['data'] = mydata
    return m

def findRepeatCRC(ID,data):
    """This function, taking in the canID and can dataframe, finds repeated crc indices.
    Useful if you want to know if crc-type value is based on data payload, or
    also includes a hash key (or other more sophisticated technique which makes
    the checksun not determinable from the data alone)."""
    m = makeCRC(ID,data)
    mypairs = pd.DataFrame()
    for i in m.data.unique():
        pairs = m.loc[
            m.data == i
        ]
        pairs = pairs.reset_index(drop=True)
        if len(pairs.crc.unique()) > 1:
            mypairs = mypairs.append(pairs)
    return mypairs

def getBitLength(ID,data):
    """Return the bit length of the first data message at the CAN ID
     in the dataframe provided."""
    if len(data.loc[data.MessageID == ID].MessageLength.unique()) > 1:
        print("Warning: more than one data length. You are using the first in this complete list by default: ",data.loc[data.MessageID == ID].MessageLength.unique())
    return int(self.makebits(ID,data).MessageLength.head(1))*8

def plotSignal(ID,data):
    """Plot bit flip plot for specified canID."""

    m= makebits(ID,data)
    m_len = getBitLength(ID,data)
    mb = bitFlipper(m.bits,m_len,ID=ID,verbose=1)

def uniqueValueSet(data,a,b, verbose = 0):
    """INPUT: CAN data in pandas dataframe, a and b the position of a signal of interest.
    Could be hexadecimal, binary, or other. E.g. data=df.Message, or data=df.bits.
    Must be subscriptable type.
    OUTPUT: A list of the set of unique """

    if verbose == 1:
        print('Number of unique values in this range: ', len(list(set(data.apply(lambda x: x[a:b])))))

    return np.sort(list(set(data.apply(lambda x: x[a:b]))))

def plotBitSignal(msgID, data, index, index_end=0, verbose = 1):
    """Input: CAN message ID, index as a bit position, and data is a pandas dataframe of hexadecimal CAN data (raw from vehicle).
    Optionally, you can add a second index (index_end) to indicate a slice of a bit CAN message.
    OUTPUT: Plot of signal in specified range. List of signal points plotted."""

    mbits = makebits(msgID,data)
    start = index
    end=index_end
    if end > 0:
        t2 = [int(i[start:end],2) for i in mbits.bits]
    else:
        t2 = [int(i[start],2) for i in mbits.bits]
    t0 = int(mbits.Time.head(1))
    if verbose == 1:
        pt.figure()
        pt.plot(mbits.Time-t0,t2,markersize=10,label='bit2')
        pt.title('Unknown Signal: '+str(msgID)+' at index '+str(index))

    return t2

def make_mybytes(msgID,data):
    """This function creates and returns a bit transform of hexadecimal CAN data
    from the pandas dataframe input at the msgID specified."""
    msgs = data.loc[data.MessageID == msgID].dropna()
    bitLength = getBitLength(msgID,data)#data.loc[data.MessageID == msgID].MessageLength.unique()[0]*8
    mybytes = msgs.Message.apply(lambda x: bin(int(x,16))[2:].zfill(int(bitLength)))

    return mybytes

def bitFlipper(mybytes,bitLength,ID = '',verbose=0):
    """This function takes the bit transform of hexadecimal CAN data (see:make_mybytes()),
    bit length, and canID as input.
    Output: plot of bit flips, and list of number of bit flips.
    """
    bf = []
    last = 0
    for i in range(0,bitLength):
        count = 0
        for index,byte in enumerate(mybytes):
            try:
                now = int(byte[i])
            except:
                pass
            if (index ==0) & (now ==1):
                last=1
            elif index == 0:
                last = 0
            if now != last:
                count +=1
            last = now
        bf.append(count)
    if verbose == True:
        pt.figure()
        pt.plot(bf)
        pt.xlabel('Bit Position')
        pt.ylabel('Bit Flips')
        pt.title(ID)
        pt.show()
        pt.close()
    return bf





def findStuffBits(msgId,data, bigEndian = 1):
    """This function will fine stuff bits in a given dataframe at a given CANID.
    Returns the position of the stuff bits from the end of the message."""

    msgs = data.loc[data.MessageID == msgId].dropna()
    bitLength = data.loc[data.MessageID == msgId].MessageLength.unique()[0]*8
    mybytes = msgs.Message.apply(lambda x: bin(int(x,16))[2:].zfill(int(bitLength)))
    sb = [x for x in range(0,len(mybytes.head(int(bitLength))))]

    for row in mybytes[:]:
        if bigEndian == 1:
            for i in sb[:-1]:
                    if row[i] == row[i+1]:
                        sb.remove(i)
        else:
            for i in sb[1:]:
                if row[i] == row[i-1]:
                    sb.remove(i)

    if bigEndian==1:
        sb.remove(bitLength-1)
    else:
        sb.remove(0)

    reverse_sb = [bitLength-i for i in sb]

    return reverse_sb

def bitFlipPlotter(msgId, data, minimum=0, maximum=0, exact=0):
    """This function looks for a binary signal with a specific number of flips
    in the given dataframe. E.g. I pushed this button 5 times, so there should be 10 or 11 bit flips.

    Input: It takes the CANID, the CAN dataframe, and the min,max, exact number of bit flips you're looking for.
    Output: Print of the plotBitSignal function to run which will plot the candidate signal.
    Plot of the can signals which have close to the same number of bit flips as exact. """
    msgs = data.loc[data.MessageID == msgId].dropna()


    bitLength = data.loc[
        data.MessageID == msgId
    ].MessageLength.unique()[0]*8

    mybytes = msgs.Message.apply(lambda x: bin(int(x,16))[2:].zfill(int(bitLength)))

    bitFlips = []
    last = 0
    for i in range(0,len(mybytes.head(int(bitLength)))):
        count = 0
        for byte in mybytes:
            now = int(byte[i])
            if now != last:
                count +=1
            last = now
        bitFlips.append(count)
    if (maximum > 0):
        for i in range(0,len(bitFlips)):
            if abs(bitFlips[i] - exact) < 5:
                pt.plot(bitFlips,label=str(msgId),ls='',marker='.')
                pt.legend()
                pt.ylim([minimum, maximum])
    else:
        pt.plot(bitFlips)

    if exact > 0:
        for i in range(0,len(bitFlips)):
            if abs(bitFlips[i] - exact) < 3:
                print("t=plotBitSignal(%d,data,%d) #%d bit flips"%(msgId,i,bitFlips[i]))

def findMovingBits(msgId,data, bigEndian = 1):
    """This function will find moving bits in a given dataframe at a given CANID.
    Returns the position of the moving bits from the end of the message."""

    msgs = data.loc[data.MessageID == msgId].dropna()
    bitLength = data.loc[data.MessageID == msgId].MessageLength.unique()[0]*8
    mybytes = msgs.Message.apply(lambda x: bin(int(x,16))[2:].zfill(int(bitLength)))
    mb = [x for x in range(0,len(mybytes.head(int(bitLength))))]

    for row in mybytes[:]:
        if bigEndian == 1:
            for i in sb[:-1]:
                    if row[i] == row[i+1]:
                        sb.remove(i)
        else:
            for i in sb[1:]:
                if row[i] == row[i-1]:
                    sb.remove(i)

    if bigEndian==1:
        sb.remove(bitLength-1)
    else:
        sb.remove(0)

    reverse_sb = [bitLength-i for i in sb]

    return reverse_sb


def find_nissan_radar(dist_files):
    """This function will identify files which have nissan radar data (or not).

    dist_file is a list of filenames."""
    radar_files = []
    for f in dist_files:
        fn = f
        pattern = ",425,"

        myfile,output = grep_search(pattern,fn)
        if output!="No Match":
            radar_files.append(myfile)

    return radar_files

def grep_search(pattern,fn):
    """Looking for a pattern in file 'fn'."""
    file = open(fn, "r")
    match = ''
    for word in file:
        if re.search(pattern, word):
            match = word
        if match != '':
            print(fn,match)
            return fn,match
            break

    return fn, "No Match"
def makeNissanRadar(ID,data,db):
    """make a big dataframe for all the candidate radar signals."""

    sig = db.get_message_by_frame_id(ID)
    names = [i.name for i in sig.signals]
    # print(names)
    cols = ['Time','TrackID']
    cols = cols + names

    df = pd.DataFrame(columns = cols)
    temp = pd.DataFrame(columns = cols)

    for i in names:
        temp[i] = s.convertData(ID,i,data,db).Message

    temp.Time = s.convertData(ID,i,data,db).Time
    temp.TrackID = ID

    return temp
def makeAllNissanRadar(data,db):
    """Using makeNissanRadar, assemble the total radar df."""

    IDlist=[int(i) for i in list(set(data.MessageID)) if (i >= 381) & (i<=425)]
    IDlist.pop(IDlist.index(402))

    print(IDlist)
    a = makeNissanRadar(IDlist[0],data,db)
    for i in IDlist[1:]:
        print(i)
        b = makeNissanRadar(i,data,db)

        a = a.append(b)

    return a
