# Chart Video KeyPoint Extraction

## Task Description
You will receive a video. Based on the images in the video, extract multiple independent, factually accurate *keypoints* that reflect the content of the video. Please extract the keypoints using the following steps: identify the type of image information --> extract the information --> generate the keypoints. Each *key image* (segment) can generate a corresponding keypoint.

Focus on and describe in detail every chart, graph, or data visualization that appears in the video. Scenes containing rich information are considered *key scenes*:
Record the chart title, axis labels (X-axis and Y-axis) and units, legend, and any directly related text annotations. Extract any important titles, text highlights, keywords, or prominent numbers that appear on the screen but are not included in the chart.
Narrative focus: State facts directly rather than describing the chart itself. The final keypoint should directly state the core facts or data revealed by the chart, treating the chart as a transparent information source rather than the object of description.
-error example: "A pie chart titled 'Global Smartphone Market Share - Q4 2024' indicates that Apple holds the largest share at 29%..."
-correct example: "In the global smartphone market share for the fourth quarter of 2024, Apple held the largest share at 29%, followed by Samsung at 22%, Xiaomi at 13%, OPPO at 9%, and 'other brands' collectively accounting for 27%."

Line chart:
- Describe the overall trend of each data curve (rising, falling, fluctuating, stable).
    - Record key data points, such as approximate values for starting points, ending points, peaks, and troughs.
- Identify turning points where trends undergo significant changes.
Bar Chart/Stick Chart:
- Compare numerical values across different categories, clearly indicating which category has the highest value and which has the lowest.
    - Quantify the differences between major categories.
    - Describe the overall distribution.
Pie chart:
    - Identify and list the major components and their approximate percentages.
    - Indicate what the largest and smallest sectors represent.
Scatter plot:
    - Describe the relationship or correlation between two variables (positive correlation, negative correlation, no correlation).
    - Identify any obvious data clusters or outliers.
Geographic map:
    - Describe the distribution patterns of data in geographic space.
    - Indicate which areas have higher or lower values, and whether there are geographic concentration trends or regional differences.
Table:
    - Summarize the theme and purpose of the table.
    - Extract the key data rows/columns most relevant to the core information of the video, or summarize the most significant numerical comparisons.
Infographic/flowchart:
- Explain the process, structure, relationships, or core information depicted in the graphic. Describe its components in logical order.
Text paragraphs/article screenshots:
    - Record or summarize the content of the text. Focus on numerical values, central content, conclusions, etc.
Natural scenes containing meaningful information:
    - Identify what the key objects in the scene are doing? What is the background? What information does it contain?

Directly output the final keypoints.
List multiple keypoints as needed, with each keypoint presented in a complete sentence format, including meaningful pronouns and subjects.  
Keypoints must be entirely based on visual information and retain only information that can be verified through a single static image (key frame).  
The stated position must be neutral.  

## Note:  
- Don't generate keypoints that aren't part of the video content, such as a cast list.
- Output must be formatted to match the example and must not include any explanations, code tags, or content beyond the key points themselves.
- Keypoints must be information-rich.
- Do not include information implying the source of the keypoints is the video, nor include explanatory information. 
Prohibited phrases:  
    Camera type: “The close-up shows...,” “In the frame...”  
    Production type: “Video summary...,” “The narrator mentions...”  
    Explanatory type: “This indicates...”
- The completeness and clarity of the semantic representation of keypoints take precedence over generation strategy requirements.
- Keypoints must be sufficiently specific to stand alone without requiring background information or context.
- Specific objects must be clearly identified. If specific entities (e.g., people, organizations, locations) are mentioned, their full names or affiliations must be specified as clearly as possible (e.g., use “U.S. Congress” instead of “Congress”).
- Information density: Keypoints must be information-rich and specific. Avoid commonplace or vague descriptions.
- Descriptions must remain neutral and objective, avoiding unnecessary derivative interpretations, metaphors, etc.

## Key frame timestamps JSON:
These timestamps may contain *key frames*: {keyframe_timestamps_json}

## Output Example
“Korean lawmakers gather in a parliamentary chamber, engaging in heated discussions and physical confrontations”,
“During the first half of 2025, Apex Corp.'s (APX) stock price began at \$110 in January, reached a peak of \$140 in May, and ended the period at \$135 in June.”,
“In a 2024 comparison of renewable energy generation, China produced the most at approximately 3,000 TWh, followed by the United States with 1,200 TWh, Brazil with 600 TWh, and Germany with 250 TWh.”,
“The technical specifications for the 'Cyberion EV' sedan list a maximum range of 450 kilometers, a 0-100 km/h acceleration time of 3.8 seconds, and a battery capacity of 85 kWh.”,
“According to 2025 data on population density by prefecture in Japan, the Kanto Plain region, including Tokyo, has the highest concentration of people, whereas the island of Hokkaido has the lowest.”,
“The President of the European Central Bank, speaking at the 2025 World Economic Forum in Davos, announced a new monetary policy aimed at curbing inflation.”,
“In a side-by-side comparison, the 'Pro-X' laptop model features 16GB of RAM and a 512GB SSD for a price of \$1,299, while the ‘Standard’ model includes 8GB of RAM and a 256GB SSD for \$899.”

## Test
Video:

The video is uploaded.