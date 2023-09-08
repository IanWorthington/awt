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

    SavedDataFile = 'awtdata.pkl.lzma' # 'awtdata.pkl.lz4'

    if os.path.isfile(SavedDataFile):
        # with lz4.frame.open(SavedDataFile, 'rb') as f:
        #     allData = pickle.load(f)
        with lzma.open(SavedDataFile, 'rb') as f:
            allData = pickle.load(f)


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

    else:
        if debug: 
            print( "Loading csv data, please wait..." )

        fileList = FindCsvDataFiles()

        allData = None 

        for file in fileList:
            if debug: print( file )
            data = PreProcessDataFile(file)
            if debug: print(data)

            allData = pd.concat([allData, data])

        if debug: print( "Postprocessing data..." )

        allData = PostProcessData( allData )

        if debug: print( "Postprocessing data complete" )

        if debug: print( "Saving data..." )

        with lzma.open(SavedDataFile, "wb") as f:
            pickle.dump(allData, f)

        if debug: print( "Saving complete." )


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

    data = data[[0,1,2,3,5,7,9]]  #[["Airport", "Terminal", "Date", "Hour", "Unnamed: 5_level_0", "Unnamed: 7_level_0"]]
    data.columns = ["Airport", "Terminal", "Date", "Time Range", "US Max Wait", "Non US Max Wait", "All Max Wait"] 
    if debug: print( data ) 

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

    data.drop( columns=['All Max Wait'], inplace=True)

    return data

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
    st.title("Maximum waiting times to clear immigration (minutes)")


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

yearsSelectedString = ""
for y in yearsSelected:
    yearsSelectedString += y + " "

st.write( "Year(s):", yearsSelectedString )

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

    monthsSelectedString = ""
    for y in monthsSelected:
        monthsSelectedString += y + " "

    st.write( "Month(s):", monthsSelectedString )

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

    daysSelectedString = ""
    for y in daysSelected:
        daysSelectedString += y + " "

    st.write( "Weekday(s):", daysSelectedString )

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


    timesSelectedString = ', '.join(timesSelected)
    st.write( "Time(s):", timesSelectedString )

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
    ChartData = timesData[["Datetime", "Date", "Time Range", "US Max Wait", "Non US Max Wait"]] #, "All Max Wait"]]

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

    st.bar_chart(
        ChartDataUsSorted,  
        y = 'US Max Wait',
        #color='#FF0000'
        )  
    
    st.bar_chart(
        ChartDataNonUsSorted,  
        y = 'Non US Max Wait',
        #color='#0000FF' 
        )  
    
    # st.bar_chart(
    #     ChartDataAllSorted,  
    #     y = 'All Max Wait',
    #     #color='#0000FF' 
    #     )  

    st.line_chart(
        ChartData, 
        x = 'Datetime',
        y = ['US Max Wait', 'Non US Max Wait'], # , 'All Max Wait'],
        #color=['#FF0000','#0000FF'] 
        )    
    
    st.dataframe(ChartData, hide_index=True)

st.write( "(C) 2023 Ian Worthington")    
st.write( "Data from https://awt.cbp.gov/")  