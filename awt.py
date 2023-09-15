#
# Display US CBP Airport Waiting Time records
#
# (C) 2023 Ian Worthington 
# All rights reserved
#

# First steps with streamlit.

#  pigar generate
# to generate requirements file

import lzma
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import os
import calendar
import re
import pickle
import lz4.frame
import timeit
# import blosc
import time
# new requirments:
import plotly.express as px
import plotly.graph_objects as go
# from st_aggrid import AgGrid, GridOptionsBuilder  
import sys
import fnmatch

debug = False
perf = False


def FindCsvDataFiles():
    fileList = []

    # os.walk() returns subdirectories, file from current directory and 
    # And follow next directory from subdirectory list recursively until last directory


    for root, dirs, files in os.walk(r"..\data"):
        for file in files:
            if file.endswith(".csv"):
                fileList.append(os.path.join(root, file))
    if debug: print(fileList)

    return fileList


def SaveData( allData ):
    SavedDataFile = 'awtdata.{}.pkl.lzma' # 'awtdata.pkl.lz4'

    years = allData["Year"].unique()
    years.sort()

    for year in years:
        print( f"Saving {year}..." )

        yearFn = SavedDataFile.format(year)
        yearData = allData[allData["Year"] == year]

        with lzma.open(yearFn, "wb") as f:
            pickle.dump(yearData, f)

    return


def FindPickledDataFiles():
    SavedDataFileMask = 'awtdata.{}.pkl.lzma'.format("*")

    fileList = []

    # os.walk() returns subdirectories, file from current directory and 
    # And follow next directory from subdirectory list recursively until last directory

    for root, dirs, files in os.walk(r"."):
        for file in fnmatch.filter(files, SavedDataFileMask):
            fileList.append(os.path.join(root, file))
    if debug: print(fileList)

    return fileList


def LoadPickledDataFiles(filelist):
    allData = pd.DataFrame()

    for file in filelist:
        with lzma.open(file, 'rb') as f:
            data = pickle.load(f)
            #allData.append(data)
            allData = pd.concat([allData, data])

    return allData


@st.cache_data  #  Don't reload static data
def getDetailData(airport, terminal):
    if debug: print("Getting detail data...")

    with st.spinner('Loading hourly data...'):
        data = LoadData()

        data = data[ (data['Airport'] == airport) & (data['Terminal'] == terminal) ]

    # data = data.drop_duplicates(
    #     subset = ['Airport', 'Terminal', 'Year', 'Month', 'Weekday']
    #     ) 
    
        return data


@st.cache_data  #  Don't reload static data
def getBasicData():
    data = LoadData()

    data = data.drop_duplicates(
        subset = ['Airport', 'Terminal', 'Year', 'Month', 'Weekday']
        )
        # keep = 'last')  #.reset_index(drop = True)
    
    return data


@st.cache_data  #  Don't reload static data
def LoadAirports():
    data = LoadData()
    AirportChoices = (data["Airport"].unique())
    AirportChoices.sort()

    return AirportChoices


@st.cache_data  #  Don't reload static data
def LoadData():
    if debug: print("Loading data...")

    print( "Checking for CSV data files..." )

    fileList = FindCsvDataFiles()

    if len(fileList) == 0:
        print( "No CSV data files found")

    else:
        print( f"CSV data files found: {fileList}" )
        
        allData = None 

        for file in fileList:
            if debug: print( file )
            data = PreProcessDataFile(file)
            if debug: print(data)

            allData = pd.concat([allData, data])

        print( "Postprocessing data..." )

        allData = PostProcessData( allData )

        print( "Postprocessing data complete" )

        allData = allData[["Airport", "Terminal", "Date", "Time Range", "US Max Wait", "Non US Max Wait", "Flights", 
                           "Time", "Datetime", "Year", "Month", "Weekday", "time50", "time95", "time99"]]

        print( "Saving data..." )

        SaveData( allData )

        print( "Saving complete." )

    # any CSVs now converted.
    # load picked data

    fileList = FindPickledDataFiles()
    allData = LoadPickledDataFiles(fileList)

    # SavedDataFile = 'awtdata.pkl.lzma'

    # if os.path.isfile(SavedDataFile):
    #     # with lz4.frame.open(SavedDataFile, 'rb') as f:
    #     #     allData = pickle.load(f)
    #     with lzma.open(SavedDataFile, 'rb') as f:
    #         allData = pickle.load(f)


        # #allData.loc[mask, 'flag'] = "**"
        # x = allData.index.is_unique
        # x2 = allData.index.duplicated()
        # print( allData.loc[x2])

        # # allData is 2641258,12
        # # 2612565 rows printed here with  duplicate index

        # duplicated_indexes = allData.index.duplicated(keep=False)
        # duplicated_rows = allData[duplicated_indexes]
        # print(duplicated_rows)

        # # only 2641159 printed here.  why?

        # allData.reset_index(drop=False, inplace=True)  # 2641258,13  Index seems to reset every year.  Maybe from merging dataframes?

        # duplicated_indexes = allData.index.duplicated(keep=False)
        # duplicated_rows = allData[duplicated_indexes]
        # print(duplicated_rows)  # this now empty

        # reindex for merged duplicatesstrs
        # allData.reset_index(drop=True, inplace=True)

        # # Check for data which has not been split out and replace those zeros with the merged value
        # mask = (allData['US Max Wait'] == 0) & (allData['Non US Max Wait'] == 0) & (allData['All Max Wait'] != 0)

        # allData.loc[mask, 'US Max Wait']     = allData['All Max Wait']
        # allData.loc[mask, 'Non US Max Wait'] = allData['All Max Wait']

        # print( allData )


    # # Deserialization of the file
    # file = open('model.pkl','rb')
    # new_model = pickle.load(file)    
    
    #
    # Test compressing data by indexing strings
    # Conclusion: this saves only 0.9 MB in a 30 MB file, not worth the effort
    #

    # stringsList1a = set(allData["Airport"]) #.unique() # + allData["Terminal"].unique() #+ allData["Date"].unique() + allData["Time Range"].unique() 
    # stringsList1b = set(allData["Terminal"])
    # stringsList1c = set(allData["Date"])
    # stringsList1d = set(allData["Time Range"])
    # stringsList = list( stringsList1a.union(stringsList1b).union(stringsList1c).union(stringsList1d) )
 
    # stringsMap = {u: i for i, u in enumerate(stringsList)}
    # indicesMap = {v: k for k, v in stringsMap.items()}     

    # allData["AirportIndex"] = allData["Airport"].map(stringsMap)
    # allData["TerminalIndex"] = allData["Terminal"].map(stringsMap)
    # allData["DateIndex"] = allData["Date"].map(stringsMap)
    # allData["TimeRangeIndex"] = allData["Time Range"].map(stringsMap)

    # allData["AirportIndexDec"] = allData["AirportIndex"].map(indicesMap)
    # allData["TerminalIndexDec"] = allData["TerminalIndex"].map(indicesMap)
    # allData["DateIndexDec"] = allData["DateIndex"].map(indicesMap)
    # allData["TimeRangeIndexDec"] = allData["TimeRangeIndex"].map(indicesMap)

    # print( allData )

    # pickleData = allData[["AirportIndex", "TerminalIndex", "DateIndex", "TimeRangeIndex", "US Max Wait", "Non US Max Wait", "All Max Wait"]] 

    # with lz4.frame.open('compdata.pkl.lz4', 'wb') as f:
    #     pickle.dump(pickleData, f)

    #
    # Moving this postprocessing to after loading
    # decreases the size of the pickeled df from 52 MB to 32 MB 
    # but makes application startup much slower
    #

    dumplz4 = ''' 
with lz4.frame.open('test.pkl.lz4', 'wb') as f:
    pickle.dump(allData, f)
'''    

    loadlz4 = ''' 
with lz4.frame.open('test.pkl.lz4', 'rb') as f:
    allData = pickle.load(f)
'''   

    dumplzma = ''' 
with lzma.open('test.pkl.lzma', 'wb') as f:
    pickle.dump(allData, f)
'''   

    loadlzma = ''' 
with lzma.open('test.pkl.lzma', 'rb') as f:
    allData = pickle.load(f)
'''   

    dumpblosc = '''
with open('test.pkl.blosc', 'wb') as f:
    f.write( blosc.compress(pickle.dumps(allData)) )
'''

    loadblosc = '''
with open("test.pkl.blosc", "rb") as f:
     allData = pickle.loads(blosc.decompress(f.read()))
'''  


    # print("dump lz4: ", timeit.repeat(stmt=dumplz4,  globals={'allData': allData}, setup="import lz4; import pickle",  number=1, repeat = 10) )  # avg ~ 0.6 s, 31 MB
    # print("load lz4: ", timeit.repeat(stmt=loadlz4,                                setup="import lz4; import pickle",  number=1, repeat = 10) )  # avg ~ 0.7 s

    # print("dump lzma: ", timeit.repeat(stmt=dumplzma, globals={'allData': allData}, setup="import lzma; import pickle", number=1, repeat = 10) )  # avg ~ 43 s, 7MB 
    # print("load lzma: ", timeit.repeat(stmt=loadlzma,                               setup="import lzma; import pickle", number=1, repeat = 10) )  # avg ~ 1.2 s

    # print("dump blosc: ", timeit.repeat(stmt=dumpblosc, globals={'allData': allData}, setup="import blosc; import pickle", number=1, repeat = 10) )  # avg ~ 0.6 s, 22 MB
    # print("load blosc: ", timeit.repeat(stmt=loadblosc,                               setup="import blosc; import pickle", number=1, repeat = 10) )  # avg ~ 0.6 s

    # allData = PostProcessData( allData ) # ~15 secs...

    # print("dump lzma: ", timeit.repeat(stmt=dumplzma, globals={'allData': allData}, setup="import lzma; import pickle", number=1, repeat = 1) )  # avg ~ 55 s, 10 MB 
    # print("load lzma: ", timeit.repeat(stmt=loadlzma,                               setup="import lzma; import pickle", number=1, repeat = 1) )  # avg ~ 2 s

    return allData 



def PreProcessDataFile(file):
    data = pd.read_csv(file, header = None )  # [0,1,2,3])
    if debug: print(data)

    data = data[[0,1,2,3,5,7,9,10,11,12,13,14,15,16,17,18,19]]  #[["Airport", "Terminal", "Date", "Hour", "Unnamed: 5_level_0", "Unnamed: 7_level_0"]]
    data.columns = ["Airport", "Terminal", "Date", "Time Range", "US Max Wait", "Non US Max Wait", "All Max Wait", "0-15", "16-30", "31-45",
                    "46-60", "61-90", "91-120", "120+", "Excluded", "Total", "Flights"] 
    if debug: 
        print( data ) 
        print( data.dtypes )

    return data


def PostProcessData(data):
    print("Postprocessing data...")

    # reindex merged dfs
    data.reset_index(drop=True, inplace=True)

    # Find data that's not been split out and replace those zeros with the merged value
    mask = (data['US Max Wait'] == 0) & (data['Non US Max Wait'] == 0) & (data['All Max Wait'] != 0)
    data.loc[mask, 'US Max Wait']     = data['All Max Wait']
    data.loc[mask, 'Non US Max Wait'] = data['All Max Wait']

    data['Time'] =  [re.search(r'\d+', x).group() for x in data['Time Range']]
    data['Datetime'] =  data['Date'] + " " + data['Time']
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%m/%d/%Y %H%M')
    data['Year'] = data['Datetime'].dt.year
    data['Month'] = data['Datetime'].dt.month_name()
    data['Weekday'] = data['Datetime'].dt.day_name()

    data = calculatePercentiles(data)

    #data.drop( columns=['All Max Wait'], inplace=True)

    return data


def calculatePercentiles(df):
    # df['Processed'] = df['Total'] - df['Excluded']  # Actual numbers in table don't always agree with this!
    df['Processed'] = df['0-15'] + df['16-30'] + df['31-45'] + df['46-60'] + df['61-90'] + df['91-120'] + df['120+']

    df['Prop0-15']   = df['0-15']    / df['Processed']
    df['Prop0-30']   = (df['16-30']  / df['Processed']) + df['Prop0-15']
    df['Prop0-45']   = (df['31-45']  / df['Processed']) + df['Prop0-30']
    df['Prop0-60']   = (df['46-60']  / df['Processed']) + df['Prop0-45']
    df['Prop0-90']   = (df['61-90']  / df['Processed']) + df['Prop0-60']
    df['Prop0-120']  = (df['91-120'] / df['Processed']) + df['Prop0-90']
    df['Prop0-120+'] = (df['120+']   / df['Processed']) + df['Prop0-120']    

    df['percentiles'] = df.apply(
        lambda row: findAllPercentiles( 
            row['Processed'], row['All Max Wait'], row['Prop0-15'], row['Prop0-30'], row['Prop0-45'], row['Prop0-60'], row['Prop0-90'], row['Prop0-120'], row['Prop0-120+']
            ), 
        axis=1
        )
    
    df[['time50', 'time95', 'time99']] = df['percentiles'].apply(pd.Series)
    df.drop(columns='percentiles', inplace=True)

    # dfcheck = df[df['time50'] == None]
    # print(dfcheck)
    # dfcheck = df[df['time95'] == None]
    # print(dfcheck)
    # dfcheck = df[df['time99'] == None]
    # print(dfcheck)

    if debug: print( df ) 

    return df


def findAllPercentiles( processed, maxWait, prop0_15, prop0_30, prop0_45, prop0_60, prop0_90, prop0_120, prop0_120p ):
    parms = locals()

    if processed > 0:
        time50 = findAndInterpolate( percentile=0.5,  **parms)
        time95 = findAndInterpolate( percentile=0.95, **parms)
        time99 = findAndInterpolate( percentile=0.99, **parms)
    else:
        time50 = None
        time95 = None
        time99 = None

    return (time50, time95, time99)


def findAndInterpolate( percentile, **kwargs):
    if debug:
        print( kwargs )

    if ( (kwargs['prop0_15'] == percentile) ):
        adjustedTime = 15

    elif ( (kwargs['prop0_30'] == percentile) ):
        adjustedTime = 30        

    elif ( (kwargs['prop0_45'] == percentile) ):
        adjustedTime = 45

    elif ( (kwargs['prop0_60'] == percentile) ):
        adjustedTime = 60        

    elif ( (kwargs['prop0_90'] == percentile) ):
        adjustedTime = 90  

    elif ( (kwargs['prop0_120'] == percentile) ):
        adjustedTime = 120        

    elif ( (kwargs['prop0_120p'] == percentile) ):
        adjustedTime = kwargs['maxWait']

    elif ( (kwargs['prop0_15'] > percentile) ):
        adjustedTime = interpolateBin( percentile, kwargs['maxWait'], 0, kwargs['prop0_15'], 0, min(15,kwargs['maxWait']) )

    elif ( (kwargs['prop0_15'] < percentile) & (kwargs['prop0_30'] > percentile) ):
        adjustedTime = interpolateBin( percentile, kwargs['maxWait'], kwargs['prop0_15'], kwargs['prop0_30'], 16, min(30,kwargs['maxWait']) - 15 )

    elif ( (kwargs['prop0_30'] < percentile) & (kwargs['prop0_45'] > percentile) ):
        adjustedTime = interpolateBin( percentile, kwargs['maxWait'], kwargs['prop0_30'], kwargs['prop0_45'], 31, min(45,kwargs['maxWait']) - 30 )

    elif ( (kwargs['prop0_45'] < percentile) & (kwargs['prop0_60'] > percentile) ):
        adjustedTime = interpolateBin( percentile, kwargs['maxWait'], kwargs['prop0_45'], kwargs['prop0_60'], 46, min(60,kwargs['maxWait']) - 45 )   

    elif ( (kwargs['prop0_60'] < percentile) & (kwargs['prop0_90'] > percentile) ):
        adjustedTime = interpolateBin( percentile, kwargs['maxWait'], kwargs['prop0_60'], kwargs['prop0_90'], 61, min(90,kwargs['maxWait']) - 60 )

    elif ( (kwargs['prop0_90'] < percentile) & (kwargs['prop0_120'] > percentile) ):
        adjustedTime = interpolateBin( percentile, kwargs['maxWait'], kwargs['prop0_90'], kwargs['prop0_120'], 91, min(120,kwargs['maxWait']) - 90 )

    elif ( (kwargs['prop0_120'] < percentile) & (kwargs['prop0_120p'] > percentile) ):
        adjustedTime = interpolateBin( percentile, kwargs['maxWait'], kwargs['prop0_120'], kwargs['prop0_120p'], 121, kwargs['maxWait'] - 120 )         

    else:
        print(f"Unexpected condition in findAndInterpolate evaluating percentile {percentile} in {kwargs}")
        adjustedTime = None

        # sys.exit()        

    return adjustedTime


def interpolateBin( percentile, maxWait, propl, proph, timeHstart, interval ):
    invRate = interval / (proph - propl)
    additionalMins = (percentile - propl) * invRate
    time = timeHstart + additionalMins
    time = min(time, maxWait)  #  due to the way CBP buckets their time bins, the caculated time may occasionally exceed maxWait by up to 1 min.
    time = round(time)

    if debug: 
        print( percentile, time )

    return time


startMainTime = time.perf_counter()
if perf: print(f"Starting timing")

st.set_page_config(page_title="Airport Wait Times", page_icon=":passport_control:", layout="wide")

# data = LoadData() 
# data.info()

afterLoadTime = time.perf_counter()
if perf: print(f"Loaded data in {afterLoadTime - startMainTime:0.4f} seconds")

# TimeRangeChoices = (data["Time Range"].unique())
# TimeRangeChoices.sort()

# print( TimeRangeChoices )

# MonthChoices = (data["Month"].unique())
# MonthChoices.sort()

# print( MonthChoices )



# ---- HEADER SECTION ----
with st.container():
    st.subheader(":passport_control: US CBP Airport Wait Times")
    st.title("Maximum waiting time to clear immigration")
    st.write(":arrow_left: Select where and when using the sidebar")


st.sidebar.markdown("### Choose:")  

# --- select airport


# AirportChoices = (data["Airport"].unique())
# AirportChoices.sort()


beforeGetBasicLoadTime = time.perf_counter()
basicData = getBasicData()
afterGetBasicTime = time.perf_counter()
if perf: print(f"Loaded basic data in {afterGetBasicTime - beforeGetBasicLoadTime:0.4f} seconds")

# basicData.info()
# print( basicData )

AirportChoices = basicData["Airport"].unique()
AirportChoices.sort()




# beforeAirportLoadTime = time.perf_counter()
# AirportChoices = LoadAirports()
# afterAirportLoadTime = time.perf_counter()
# if perf: print(f"Loaded airport data in {afterAirportLoadTime - beforeAirportLoadTime:0.4f} seconds")

airportSelected = st.sidebar.selectbox( "Airport", 
                                       AirportChoices,
                                       on_change= lambda: {print("> airport changed") if debug else None} 
                                       )
st.write( "Airport:", airportSelected )
airportData = basicData[basicData['Airport'] == airportSelected]

# --- select terminal

TerminalChoices = (airportData["Terminal"].unique())
TerminalChoices.sort()

terminalSelected = st.sidebar.selectbox( "Terminal", 
                                        TerminalChoices,
                                        on_change= lambda: {print("> terminal changed") if debug else None}  
                                        )
st.write( "Terminal:", terminalSelected )
terminalData = airportData[airportData['Terminal'] == terminalSelected]

afterTerminalTime = time.perf_counter()
if perf: print(f"After terminal in {afterTerminalTime - afterLoadTime:0.4f} seconds")

# --- select year

beforeYearsTime = time.perf_counter()

# yearChoices1 = '''
# yearChoices = terminalData["Year"]
# '''  

# rundata = timeit.repeat(stmt=yearChoices1, globals={'terminalData': terminalData}, setup="",  number=1, repeat = 1000) 
# print("Build year choices #1 mean,sd:", np.mean(rundata), np.std(rundata) )  # mean,sd: 1.6948028933256864e-06 2.2229582465486665e-06


yearChoices = terminalData["Year"].unique()
if debug:
    print("terminalData:")
    print(terminalData)
    print( "yearChoices:", yearChoices )

yearChoices = list(yearChoices.astype(str))
yearChoices.sort(reverse=True)
yearChoicesDf = pd.DataFrame(
    {
        "Year": yearChoices,
        "Select": [False]*len(yearChoices)
    }
)
if debug: print( "yearChoicesDf (before):", yearChoicesDf )

yearChoicesDf = st.sidebar.data_editor(
    yearChoicesDf,
    column_config={
        "Select": st.column_config.CheckboxColumn(
            "Select",
            help="Select years",
            default=False,
        )
    },
    disabled=["Year"],
    hide_index=True,
    on_change= lambda: {print("> yearChoices changed") if debug else None}
)

if debug: 
    print( "yearChoicesDf (after):" )
    print( yearChoicesDf )

yearChoicesDf = yearChoicesDf[yearChoicesDf['Select'] == True]
yearsSelected =  list(yearChoicesDf["Year"])
if debug: print( "yearsSelected:", yearsSelected )

# yearsSelectedString = ""
# for y in yearsSelected:
#     yearsSelectedString += y + " "

st.write( "Year(s):", ', '.join(yearsSelected) )

#yearsSelected = yearsSelected.astype(int)
yearsSelected = [int(i) for i in yearsSelected]

testYearsData1 = '''
yearsData = terminalData[terminalData['Year'].isin(yearsSelected)]
'''  

testYearsData2 = '''
mask = terminalData['Year'].values
mask = np.isin(mask, yearsSelected)
yearsData = terminalData.loc[mask]
'''  

    # mask = df['A'].values == 'foo'
    # return df.loc[mask]

# rundata = timeit.repeat(stmt=testYearsData1, globals={'terminalData': terminalData, 'yearsSelected': yearsSelected}, setup="",  number=1, repeat = 1000) 
# print("testYearsData1 mean,sd:", np.mean(rundata), np.std(rundata) )  # mean,sd: 0.0007634021025151014 0.00026238721908491194

# rundata = timeit.repeat(stmt=testYearsData2, globals={'terminalData': terminalData, 'yearsSelected': yearsSelected}, setup="import numpy as np", number=1, repeat = 1000) 
# print("testYearsData2 mean,sd:", np.mean(rundata), np.std(rundata) )  # mean,sd:  0.00047096460149623453 0.00016924217003666165

# yearsData = terminalData[terminalData['Year'].isin(yearsSelected)]   # 

mask = terminalData['Year'].values
# print("mask:", type(mask), mask)
mask = np.isin(mask, yearsSelected)
# print("mask:", type(mask), mask)
yearsData = terminalData.loc[mask]
# print("yearsData:", yearsData)



if debug: 
    print( "yearsData:" )
    print( yearsData )

afterYearsTime = time.perf_counter()
if perf: print(f"Processed years in {afterYearsTime - beforeYearsTime:0.4f} seconds")    

# --- select months

if not yearsData.empty:
    monthChoices = [calendar.month_name[i] for i in range(1, 13)]

    monthChoicesDf = pd.DataFrame(
        {
            "Month": monthChoices,
            "Select": [False]*len(monthChoices)
        }
    )
    if debug: print( monthChoicesDf )

    monthChoicesDf = st.sidebar.data_editor(
        monthChoicesDf,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select months",
                default=False,
            )
        },
        disabled=["Month"],
        hide_index=True,
        on_change= lambda: {print("> monthChoices changed") if debug else None}
    )

    if debug: print( monthChoicesDf )

    monthChoicesDf = monthChoicesDf[monthChoicesDf['Select'] == True]
    monthsSelected =  list(monthChoicesDf["Month"])
    if debug: print( monthsSelected )

    # monthsSelectedString = ""
    # for y in monthsSelected:
    #     monthsSelectedString += y + " "

    st.write( "Month(s):", ', '.join(monthsSelected) )

    monthsData = yearsData[yearsData['Month'].isin(monthsSelected)]    


    if debug: print( monthsData )

# --- select weekday

if 'monthsData' in locals() and not monthsData.empty:
    dayChoices = [calendar.day_name[i] for i in range(0, 7)]

    dayChoicesDf = pd.DataFrame(
        {
            "Weekday": dayChoices,
            "Select": [False]*len(dayChoices)
        }
    )
    if debug: print( dayChoicesDf )

    dayChoicesDf = st.sidebar.data_editor(
        dayChoicesDf,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select weekdays",
                default=False,
            )
        },
        disabled=["Weekday"],
        hide_index=True,
        on_change= lambda: {print("> dayChoices changed") if debug else None}
    )

    if debug: print( dayChoicesDf )

    dayChoicesDf = dayChoicesDf[dayChoicesDf['Select'] == True]
    daysSelected =  list(dayChoicesDf["Weekday"])
    if debug: print( daysSelected )

    # daysSelectedString = ""
    # for y in daysSelected:
    #     daysSelectedString += y + " "

    st.write( "Weekday(s):", ', '.join(daysSelected) )

    weekdaysData = monthsData[monthsData['Weekday'].isin(daysSelected)]    

    if debug: print( weekdaysData )

# --- select time

if 'weekdaysData' in locals() and not weekdaysData.empty:
    beforeGetDetailLoadTime = time.perf_counter()
    detailData = getDetailData( airportSelected, terminalSelected )
    afterGetDetailTime = time.perf_counter()
    if perf: print(f"Loaded detail data in {afterGetDetailTime - beforeGetDetailLoadTime:0.4f} seconds")

    weekdaysData = detailData[(detailData['Year'].isin(yearsSelected)) & (detailData['Month'].isin(monthsSelected)) & (detailData['Weekday'].isin(daysSelected))]   
    if debug: 
        print("weekdaysData:")
        print( weekdaysData )  

    timeChoices = weekdaysData["Time Range"].unique()
    timeChoices.sort()
    if debug: 
        print( "timeChoices:", timeChoices ) 

    timeChoicesDf = pd.DataFrame(
        {
            "Time": timeChoices,
            "Select": [False]*len(timeChoices)
        }
    )
    if debug: 
        print( "timeChoicesDf before:" )
        print( timeChoicesDf )

    timeChoicesDf = st.sidebar.data_editor(
        timeChoicesDf,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select times",
                default=False,
            )
        },
        disabled=["Time"],
        hide_index=True,
        on_change= lambda: {print("> timeChoices changed") if debug else None}
    )

    if debug: 
        print( "timeChoicesDf after:" )
        print( timeChoicesDf )

    timeChoicesDf = timeChoicesDf[timeChoicesDf['Select'] == True]
    timesSelected =  list(timeChoicesDf["Time"])
    if debug: 
        print( "timesSelected:",  timesSelected )


    #timesSelectedString = ', '.join(timesSelected)
    st.write( "Time(s):", ', '.join(timesSelected) )

    if debug: 
        print( "weekdaysData:" )
        print( weekdaysData )
    
    timesData = weekdaysData[weekdaysData['Time Range'].isin(timesSelected)]   # df[df['A'].isin([3, 6])]
    if debug: 
        print( "timesData:" )
        print( timesData )

# Display data

#if not timesData.empty:
#if isinstance(timesData, pd.DataFrame):    
if 'timesData' in locals() and not timesData.empty:
    ChartData = timesData[["Datetime", "Date", "Time Range", "US Max Wait", "Non US Max Wait", "time50", "time95", "time99"]] #, "All Max Wait"]]
    ChartData["Fdt"] = ChartData["Datetime"].dt.strftime("%a %Y-%b-%d %H%M")

    ChartDataUsSorted = ChartData[["US Max Wait"]]
    ChartDataUsSorted = ChartDataUsSorted.copy().sort_values(by=['US Max Wait'], ascending=False)
    ChartDataUsSorted = ChartDataUsSorted.reset_index()

    ChartDataNonUsSorted = ChartData[["Non US Max Wait"]]
    ChartDataNonUsSorted = ChartDataNonUsSorted.copy().sort_values(by=['Non US Max Wait'], ascending=False)
    ChartDataNonUsSorted = ChartDataNonUsSorted.reset_index()    

    # ChartDataAllSorted = ChartData[["All Max Wait"]]
    # ChartDataAllSorted = ChartDataAllSorted.copy().sort_values(by=['All Max Wait'], ascending=False)
    # ChartDataAllSorted = ChartDataAllSorted.reset_index()    

    if debug:
        print("ChartDataUsSorted")
        print(ChartDataUsSorted)

    tab_us, tab_nonus = st.tabs(["US", "Non US"])

    with tab_us:
        # st.bar_chart(
        #     ChartDataUsSorted,  
        #     y = 'US Max Wait',
        #     #color='#FF0000'
        #     )  
        
        fig = px.bar( ChartDataUsSorted, y='US Max Wait' )
        fig.update_layout( xaxis_title="", yaxis_title="Minutes waiting", legend_title_text='')
        fig.update_xaxes(visible=False)
        fig.update_traces(
            hovertemplate="<br>".join([
                "%{y} minutes wait"
            ])
        )

        st.plotly_chart( fig, 
                theme="streamlit",
                use_container_width=True
                )   
    
    with tab_nonus:
        # st.bar_chart(
        #     ChartDataNonUsSorted,  
        #     y = 'Non US Max Wait',
        #     #color='#0000FF' 
        #     )  

        fig = px.bar( ChartDataNonUsSorted, y='Non US Max Wait' )
        fig.update_layout( xaxis_title="", yaxis_title="Minutes waiting", legend_title_text='')
        fig.update_xaxes(visible=False)
        fig.update_traces(
            hovertemplate="<br>".join([
                "%{y} minutes wait"
            ])
        )

        st.plotly_chart( fig, 
                theme="streamlit",
                use_container_width=True
                )  
    
    # st.bar_chart(
    #     ChartDataAllSorted,  
    #     y = 'All Max Wait',
    #     #color='#0000FF' 
    #     )  

    # st.line_chart(
    #     ChartData, 
    #     x = 'Datetime',
    #     y = ['US Max Wait', 'Non US Max Wait'], # , 'All Max Wait'],
    #     #color=['#FF0000','#0000FF'] 
    #     )    
    
    ChartData.reset_index(drop=True, inplace=True)

    tab_inorder, tab_bydt = st.tabs(["In order", "By date/time"])

    with tab_inorder:
        # fig = px.line( ChartData, 
        #               y=['US Max Wait', 'Non US Max Wait'],
        #               custom_data=['Datetime'],
        #               markers=True
        #               )
        # # fig.update_layout(legend_title_text='Variable', xaxis_title="X", yaxis_title="Series")
        # fig.update_layout( xaxis_title="", yaxis_title="Minutes waiting", legend_title_text='')
        # fig.update_xaxes(visible=False)
        # fig.update_traces(
        #     hovertemplate="<br>".join([
        #         #"ColX: %{x}",
        #         "%{y} minutes wait",
        #         "%{customdata[0]}"
        #     ])
        # )
        # st.plotly_chart( fig, 
        #                 theme="streamlit",
        #                 use_container_width=True
        #                 )  

        # customdata = np.stack((df['continent'], df['country']), axis=-1)
        # customdataStack = np.stack((ChartData['Datetime']), axis=-1)

        fig = go.Figure()
        fig.add_trace( go.Scatter( y=ChartData["time50"], name="50% All", line=dict(color="#aecfd1"), fillcolor="#aecfd1", fill='tozeroy', customdata=ChartData['Fdt'] )) # fill down to xaxis
        fig.add_trace( go.Scatter( y=ChartData["time95"], name="95% All", line=dict(color="#bed9da"), fillcolor="#bed9da", fill='tonexty', customdata=ChartData['Fdt'] ))
        fig.add_trace( go.Scatter( y=ChartData["time99"], name="99% All", line=dict(color="#cee2e3"), fillcolor="#cee2e3", fill='tonexty', customdata=ChartData['Fdt'] ))
        fig.add_trace( go.Scatter( y=ChartData["US Max Wait"],     name="US Max Wait",     line=dict(color="#dd3737"), customdata=ChartData['Fdt'] )) 
        fig.add_trace( go.Scatter( y=ChartData["Non US Max Wait"], name="Non US Max Wait", line=dict(color="#6f4c1e"), customdata=ChartData['Fdt'] ))  #customdata=customdataStack )) 
        fig.update_layout( xaxis_title="", yaxis_title="Minutes waiting", legend_title_text='')
        fig.update_xaxes(visible=False)
        fig.update_traces(
            hovertemplate="<br>".join([
                #"ColX: %{x}",
                "%{y} minutes wait",
                "%{customdata}"
                # "%{customdata[0]}"
                ])
            )
 
        # fig = px.area( ChartData,
        #             #   x=index,
        #               y=['time99', 'time95', 'time50']
        #               )
        st.plotly_chart( fig, 
                        theme="streamlit",
                        use_container_width=True
                        )   

    with tab_bydt:
        # fig = px.line( ChartData, 
        #               x='Datetime', 
        #               y=['US Max Wait', 'Non US Max Wait'],
        #               markers=True
        #               )
        # fig.update_layout( xaxis_title="Date/time", yaxis_title="Minutes waiting", legend_title_text='')
        # fig.update_traces(
        #     hovertemplate="<br>".join([
        #         "%{x}",
        #         "%{y} minutes wait"
        #     ])
        # )
        # st.plotly_chart( fig, 
        #                 theme="streamlit",
        #                 use_container_width=True
        #                 )   
        
        fig = go.Figure()
        fig.add_trace( go.Scatter( x=ChartData['Datetime'], y=ChartData["time50"], name="50% All", line=dict(color="#aecfd1"), fillcolor="#aecfd1", fill='tozeroy', customdata=ChartData['Fdt'] )) # fill down to xaxis
        fig.add_trace( go.Scatter( x=ChartData['Datetime'], y=ChartData["time95"], name="95% All", line=dict(color="#bed9da"), fillcolor="#bed9da", fill='tonexty', customdata=ChartData['Fdt'] ))
        fig.add_trace( go.Scatter( x=ChartData['Datetime'], y=ChartData["time99"], name="99% All", line=dict(color="#cee2e3"), fillcolor="#cee2e3", fill='tonexty', customdata=ChartData['Fdt'] ))
        fig.add_trace( go.Scatter( x=ChartData['Datetime'], y=ChartData["US Max Wait"],     name="US Max Wait",     line=dict(color="#dd3737"), customdata=ChartData['Fdt'] )) 
        fig.add_trace( go.Scatter( x=ChartData['Datetime'], y=ChartData["Non US Max Wait"], name="Non US Max Wait", line=dict(color="#6f4c1e"), customdata=ChartData['Fdt'] ))  #customdata=customdataStack )) 
        fig.update_layout( xaxis_title="Date time", yaxis_title="Minutes waiting", legend_title_text='')
        #fig.update_xaxes(visible=False)
        fig.update_traces(
            hovertemplate="<br>".join([
                #"ColX: %{x}",
                "%{y} minutes wait",
                "%{customdata}"
                # "%{customdata[0]}"
                ])
            )
 
        # fig = px.area( ChartData,
        #             #   x=index,
        #               y=['time99', 'time95', 'time50']
        #               )
        st.plotly_chart( fig, 
                        theme="streamlit",
                        use_container_width=True
                        )   

 
    TableDisplayData = ChartData[["Datetime", "Date", "Time Range", "US Max Wait", "Non US Max Wait", "time50", "time95", "time99", "Fdt"]]
    TableDisplayData.columns = ["Datetime", "Date", "Time Range", "US Max Wait Mins", "Non US Max Wait Mins", "50% All Mins", "95% All Mins", "99% All Mins", "Formatted date"]
    st.dataframe(TableDisplayData, hide_index=True)

    # AgGrid(ChartData, height=400)



    # gb = GridOptionsBuilder()   

    # # makes columns resizable, sortable and filterable by default
    
    # gb.configure_default_column(
    #     resizable=True,
    #     filterable=True,
    #     sortable=True,
    #     editable=False,
    # )

    # #configures state column to have a 80px initial width

    # gb.configure_column(field="Datetime", header_name="Datetime") #, width=80)



    # go = gb.build()

    # AgGrid(ChartData, gridOptions=go, height=400)    


    # fig = go.Figure(data=[go.Table(
    #     header=dict(values=list(ChartData.columns),
    #                 fill_color='paleturquoise',
    #                 align='left'),
    #     cells=dict(values=[ChartData.Datetime, ChartData.Date, ChartData["Time Range"], ChartData["US Max Wait"], ChartData["Non US Max Wait"]],
    #                fill_color='lavender',
    #                align='left')
    #                )
    # ])

    # st.plotly_chart(fig, theme="streamlit")

st.write( "(C) 2023 Ian Worthington")    
st.write( "Discussion and brickbats [here](https://www.flyertalk.com/forum/travel-tools/2134497-us-airport-immigration-waiting-times-connection-planning.html)" )
# st.write( "Data from [awt.cbt.gov](https://awt.cbp.gov/). (The exact meaning of their data is undefined, but we believe it means the waiting time for passengers on planes that arrive during the given hour range.)")  
st.write( "Data from [awt.cbt.gov](https://awt.cbp.gov/)")  