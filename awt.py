#
# Display US CBP Airport Waiting Time records
#
# (C) 2023 Ian Worthington 
# All rights reserved
#

# First steps with streamlit.

import pandas as pd
#import numpy as np
from datetime import datetime
import streamlit as st
import os
import calendar
import re
import pickle
import lz4.frame

debug = True


def FindDataFiles():
    fileList = []

    # os.walk() returns subdirectories, file from current directory and 
    # And follow next directory from subdirectory list recursively until last directory


    for root, dirs, files in os.walk(r"..\data"):
        for file in files:
            if file.endswith(".csv"):
                fileList.append(os.path.join(root, file))
    print(fileList)

    return fileList




@st.cache_data  #  Don't reload static data
def LoadData():
    SavedDataFile = 'awtdata.pkl.lz4'

    if os.path.isfile(SavedDataFile):
        with lz4.frame.open(SavedDataFile, 'rb') as f:
            allData = pickle.load(f)

    else:
        print( "Loading data, please wait..." )

        fileList = FindDataFiles()

        allData = None 

        for file in fileList:
            print( file )
            data = ProcessDataFile(file)
            print(data)

    #        if isinstance(allData, pd.DataFrame):
    #            allData = data
    #        else:
            allData = pd.concat([allData, data])


        print( "Saving data..." )
        # Open a file to write bytes
        # p_file = open('awtdata.pkl', 'wb')

        # # Pickle the object
        # pickle.dump(allData, p_file)
        # p_file.close()

        with lz4.frame.open(SavedDataFile, 'wb') as f:
            pickle.dump(allData, f)


    # # Deserialization of the file
    # file = open('model.pkl','rb')
    # new_model = pickle.load(file)        

    return allData 

# import pickle
# import lz4.frame

# with lz4.frame.open('AlaskaCoast.lz4', 'wb') as f:
#     pickle.dump(arr, f)

# with lz4.frame.open('AlaskaCoast.lz4', 'rb') as f:
#     arr = pickle.load(f)


def ProcessDataFile(file):
    data = pd.read_csv(file, header = [0,1,2,3])
    print(data)

    data = data[["Airport", "Terminal", "Date", "Hour", "Unnamed: 5_level_0", "Unnamed: 7_level_0"]]
    data.columns = ["Airport", "Terminal", "Date", "Time Range", "US Max Wait", "Non US Max Wait"] 
    print( data ) 

    data['Time'] =  [re.search(r'\d+', x).group() for x in data['Time Range']]
    data['Datetime'] =  data['Date'] + " " + data['Time']
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%m/%d/%Y %H%M')
    data['Year'] = data['Datetime'].dt.year
    data['Month'] = data['Datetime'].dt.month_name()
    data['Weekday'] = data['Datetime'].dt.day_name()

    return data


st.set_page_config(page_title="Airport Wait Times", page_icon=":passport_control:", layout="wide")

data = LoadData()
print( data )

# TimeRangeChoices = (data["Time Range"].unique())
# TimeRangeChoices.sort()

# print( TimeRangeChoices )

# MonthChoices = (data["Month"].unique())
# MonthChoices.sort()

# print( MonthChoices )



# ---- HEADER SECTION ----
with st.container():
    st.subheader("US CBP Airport Wait Times :passport_control:")
    st.title("Waiting times (minutes)")


st.sidebar.markdown("### Choose:")  

# select airport

AirportChoices = (data["Airport"].unique())
AirportChoices.sort()

airportSelected = st.sidebar.selectbox( "Airport", 
                                       AirportChoices,
                                       on_change= lambda: {print("> airport changed")} 
                                       )
st.write( "Airport:", airportSelected )
airportData = data[data['Airport'] == airportSelected]

# select terminal

TerminalChoices = (airportData["Terminal"].unique())
TerminalChoices.sort()

terminalSelected = st.sidebar.selectbox( "Terminal", 
                                        TerminalChoices,
                                        on_change= lambda: {print("> terminal changed")}  
                                        )
st.write( "Terminal:", terminalSelected )
terminalData = airportData[airportData['Terminal'] == terminalSelected]

# select year

yearChoices = terminalData["Year"].unique()
yearChoices.sort()
print( yearChoices )
yearChoices = yearChoices.astype(str)
yearChoicesDf = pd.DataFrame(
    {
        "Year": yearChoices,
        "Select": [False]*len(yearChoices)
    }
)
print( yearChoicesDf )

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
    on_change= lambda: {print("> yearChoices changed")}
)

print( "yearChoicesDf:" )
print( yearChoicesDf )

yearChoicesDf = yearChoicesDf[yearChoicesDf['Select'] == True]
yearsSelected =  list(yearChoicesDf["Year"])
print( "yearsSelected:", yearsSelected )

yearsSelectedString = ""
for y in yearsSelected:
    yearsSelectedString += y + " "

st.write( "Year(s):", yearsSelectedString )

#yearsSelected = yearsSelected.astype(int)
yearsSelected = [int(i) for i in yearsSelected]

yearsData = terminalData[terminalData['Year'].isin(yearsSelected)]   # df[df['A'].isin([3, 6])]

print( "yearsData:" )
print( yearsData )

# --- select months

if not yearsData.empty:
    monthChoices = [calendar.month_name[i] for i in range(1, 13)]

    monthChoicesDf = pd.DataFrame(
        {
            "Month": monthChoices,
            "Select": [False]*len(monthChoices)
        }
    )
    print( monthChoicesDf )

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
        on_change= lambda: {print("> monthChoices changed")}
    )

    print( monthChoicesDf )

    monthChoicesDf = monthChoicesDf[monthChoicesDf['Select'] == True]
    monthsSelected =  list(monthChoicesDf["Month"])
    print( monthsSelected )

    monthsSelectedString = ""
    for y in monthsSelected:
        monthsSelectedString += y + " "

    st.write( "Month(s):", monthsSelectedString )

    monthsData = yearsData[yearsData['Month'].isin(monthsSelected)]    


    print( monthsData )

# select weekday

if 'monthsData' in locals() and not monthsData.empty:
    dayChoices = [calendar.day_name[i] for i in range(0, 7)]

    dayChoicesDf = pd.DataFrame(
        {
            "Weekday": dayChoices,
            "Select": [False]*len(dayChoices)
        }
    )
    print( dayChoicesDf )

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
        on_change= lambda: {print("> dayChoices changed")}
    )

    print( dayChoicesDf )

    dayChoicesDf = dayChoicesDf[dayChoicesDf['Select'] == True]
    daysSelected =  list(dayChoicesDf["Weekday"])
    print( daysSelected )

    daysSelectedString = ""
    for y in daysSelected:
        daysSelectedString += y + " "

    st.write( "Weekday(s):", daysSelectedString )

    weekdaysData = monthsData[monthsData['Weekday'].isin(daysSelected)]    

    print( weekdaysData )

# select time

if 'weekdaysData' in locals() and not weekdaysData.empty:
    timeChoices = weekdaysData["Time Range"].unique()
    timeChoices.sort()
    print( timeChoices ) 

    timeChoicesDf = pd.DataFrame(
        {
            "Time": timeChoices,
            "Select": [False]*len(timeChoices)
        }
    )
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
        on_change= lambda: {print("> timeChoices changed")}
    )

    print( timeChoicesDf )

    timeChoicesDf = timeChoicesDf[timeChoicesDf['Select'] == True]
    timesSelected =  list(timeChoicesDf["Time"])
    print( timesSelected )

    timesSelectedString = ""
    for y in timesSelected:
        timesSelectedString += y + " "

    st.write( "Time(s):", timesSelectedString )

    timesData = weekdaysData[weekdaysData['Time Range'].isin(timesSelected)]   # df[df['A'].isin([3, 6])]

    print( timesData )

# Display data

#if not timesData.empty:
#if isinstance(timesData, pd.DataFrame):    
if 'timesData' in locals() and not timesData.empty:
    ChartData = timesData[["Datetime", "Date", "Time Range", "US Max Wait", "Non US Max Wait"]]

    st.line_chart(
        ChartData, 
        x = 'Datetime',
        y = ['US Max Wait', 'Non US Max Wait']
        )    
    
    st.dataframe(ChartData, hide_index=True)

st.write( "Data from https://awt.cbp.gov/")    
