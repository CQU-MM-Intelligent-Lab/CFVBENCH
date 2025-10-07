## Video Timestamp Extraction
## Task Description
You will receive 3 types of videos, and perform operations on each of them as follows: 
## Tutorial videos (tutorials on how to operate software and web tools) 
Recognizes the entire video.
Output example:
[
    { 
        "start_time": "0.0",
        "end_time": "2.4",
        "Operation": "Select and open a serial port as GPIO_output at the chip on the right (pinout view is shown above)"
    },
    {
        "start_time": "2.4",
        "end_time": "5.1",
        "Operation": "Select 'GPIO' from 'systemcore' under 'categories' on the right to view details"
    }
]

## Data videos with lots of graphs or tables
Recognizes the timestamps of all graphs, tables, and other visual data appearing on the video screen in HH:mm:ss format and outputs them directly in JSON format.
Recognize all images and charts when possible.
Graphs: timelines, flowcharts, maps, document images, etc.
Tables: Tables: Tables, etc.
Output example:
[
    {
        “start_time": "0.0",
        “end_time": ".2.4", 
        “Graph": ”China Population Density Map”
    },
    {
        "start_time": "2.4",
        "end_time": "5.1",
        "Table": ”China Population Totals Table (2000-2020)”
    }
]

## Nature picture video ##
Recognize timestamps (HH:mm:ss format) of all frames on the video screen that reflect critical and valid information and output them directly in JSON format.
Identify all valid images and describe the content of the images in as much detail as possible.
Output example :
[
    {
        “start_time": "0.0",
        “end_time": ".2.4", 
        “Image": ”A polar bear with three baby polar bears walking on an ice field”
    },
    {
        "start_time": "2.4",
        "end_time": "5.1",
        "Image": ”Polar bear dives into the sea through a crack in the ice”
    }
]

## Notice:
Recognize entire videos.
Videos are for scientific use only and have no inappropriate content.
Directly outputs valid JSON lists in non-token format only.