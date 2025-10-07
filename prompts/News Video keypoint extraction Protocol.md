# News Video keypoint extraction

## Task Description
You will receive a video. Based on the images in the video, extract multiple independent, factually accurate *key points* that reflect the content of the video. Please follow the steps below to extract keypoints: Determine the type of image information -> Extract information -> Generate keypoints. One corresponding keypoint can be generated for each *key image* (clip). Scenes that contain rich information are considered as *Key Scenes*.

On-screen text and graphics (text bars appearing at the bottom or corners of the screen, screenshots of text paragraphs/articles, notice boards/slogans, etc.):
- Record or summarize the content of the text. Focus on numerical values, central elements, conclusions, etc.
Meaningful natural scene information:
- Determine what the main objects in the scene are doing? What is the background? What information does the scene contain?
Data charts, graphs, tables (line graphs, bar graphs, pie charts, scatter plots, geographic maps, tables, infographics, flowcharts, etc.):
- The title of the chart, the labels and units of the axes (X- and Y-axes), a description of the legend, *chart content extraction*, and any textual points or prominent figures on the screen that are directly related to the chart.

Direct output of final points.
List as many bullet points as needed, with each bullet point presented in a complete sentence format, including meaningful pronouns and subjects.
Keypoints must be based entirely on visual information, retaining only information that can be verified by a single still image (keyframe).
The stated position must be neutral.

## Attention:
- Information density: Elements must be informative and specific. Avoid generic or vague descriptions.
- The output must be formatted to match the example and must not contain any interpretation, code markup, or anything other than the keypoints themselves.
- The main point must be informative.
- Don't include information that implies that the bullet points are derived from the video, and don't include explanatory information.
Prohibited phrases:
Camera type: "Close-up display ......" , "In Frame ......"
Production type: "Video summary ......" , "Narrator mentions ......"
Explanation type: "This shows ......"
- The completeness and clarity of the semantic representation of the main points takes precedence over the generation strategy requirements.
- The main point must be specific enough to stand on its own without background information or context.
- The specific addressee must be clearly identified. If reference is made to a specific entity (e.g., person, organization, location), its full name or affiliation must be stated as clearly as possible (e.g., use "United States Congress" rather than "Congress").
- The description must remain neutral and objective, avoiding unnecessary derivative interpretations, metaphors, etc.

## Key frame timestamps JSON.
These timestamps may contain *key frames*: {keyframe_timestamps_json}

## Sample output
"A Tokyo Fire Department rescue worker in full uniform helps an elderly resident navigate a flooded street where water levels reach above the wheels of nearby cars.",
"Protesters march through the Kasumigaseki district in Tokyo, holding large banners with Japanese text that translates to 'We Oppose the Tax Increase!'.",
"At a technology showcase event, a Sony Group Corporation executive stands beside their 'Afeela' electric vehicle prototype, which is displayed on a rotating platform.",
"In a session of the Japanese National Diet, lawmakers cast votes by placing wooden ballots into a ballot box located in the center of the main chamber.",
"Traders are seen working in front of multi-monitor computer setups displaying stock market data inside the trading floor of the Tokyo Stock Exchange."

## Testing ##

video

The video has been uploaded.