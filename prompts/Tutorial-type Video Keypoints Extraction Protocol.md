# Tutorial Video Keypoints Extraction

## Task Description

In this task, you will receive a tutorial-type video (a tutorial on how to operate software, web tools) and its subtitles. Extract the main points by combining the content of the video and the subtitles.
Please extract the key steps in the video in the format 'User Action -> Manipulated Object -> Screen Result'. For each step, please describe it in detail:
1. user-specific actions.
2. Interface elements for operations.
3. The change that occurs on the screen or the information displayed after the action is completed.
Direct output of the final key point.
List as many bullet points as needed, with each bullet point standing alone in a sentence with complete sentence structure and meaningful pronouns and subjects.
Prioritize information that can be verified by the video frame. However, if the subtitle/audio provides a crucial explanation about the outcome or format resulting from a user's action, that information must be included in the 'Screen Result' part of the keypoint.
The key snippet timestamps below contain screen content that you may want to focus on.

## Key Clip Timestamps JSON:
{keyframe_timestamps_json}

## Subtitle
{Subtitle}

## Output Example
"The user clicks on the 'File' menu item located in the top-left corner of the application window",
"From the 'Edit' dropdown menu, the 'Copy' option, which is the third item, is selected by a mouse click",
"In the 'Login' dialog box, the user types 'testuser@example.com' into the text field labeled 'Email Address'",
"The cursor drags the 'Brush Size' slider within the 'Tools' panel to the right, increasing the displayed value from '10px' to '25px'",
"A right-click action is performed on the layer named 'Background Layer' in the 'Layers' panel, causing a context menu to appear",
"The user selects the 'checkbox' next to the 'Enable Feature X' label within the 'Settings' window."

## **Note:**
- Clearly identify the user interface element being interacted with (e.g., "button labeled 'Submit'", "opacity" slider", "floppy disk-like icon"). Describe the action (e.g., "was clicked", "was dragged to 75\% of the value", "entered 'My Document'"). Indicate its size, color, and location (e.g., the 'Start' pink button at the top left of the window). 
- Please extract each step in the video in detail. Include any prerequisites, as well as all optional actions or alternative paths and results that may be directed.
- Retain complete information about the text associated with the target step, including: inputs, outputs, interactive elements (e.g. text on buttons), etc.
- The output must be formatted to match the example without any interpretation or code markup.
- Key points must be well-informed. Information density ≥ 3 valid information units ui.
- Does not contain information that implies that the source of the key point is a video.
  Disable phrases:
    Camera Type: “Close-up showing...” , “In the frame...”
     Production Type: “The video summarizes...” , “The narrator mentions...”
    Explanatory Type: “This suggests...”
- The completeness and clarity of the semantic representation of key points takes precedence over generative strategy requirements.
- Key points must be specific enough to stand on their own without background information or context; identify any particular object, e.g., person, organization, etc. For example, if "Congress" is used in a key point, it should specify which country.
- Don't use common sense or vague information as a keypoint.
- The description must be neutral and objective, avoiding unnecessary derivative interpretations, metaphors, etc.
- The demonstration and instructional operations shown in the video, such as entering “testblog” and “example text,” are only examples and should not be considered mandatory operations for viewers.

## Test

Video:

The video is uploaded.