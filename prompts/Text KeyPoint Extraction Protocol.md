# Text KeyPoint Extraction
## Task Description
- You will receive a piece of text. Based on the text, extract multiple independent and factually accurate *keypoints* to reflect the content.
- List multiple keypoints as needed. Each keypoint should be an independent sentence with a complete sentence structure and meaningful pronouns and subjects.
- Directly output the final keypoints.   

## Note:  
- The output format must match the example and must not include any explanations, code tags, or content beyond the keypoints themselves.
- Each keypoint should contain specific, valuable information (e.g., numbers, names, specific actions, conclusions). Avoid vague or common-sense statements.
- The semantic completeness and clarity of keypoints take precedence over generation strategy requirements.
- Keypoints must be sufficiently specific to stand alone without requiring background information or context.
- Specific objects or entities must be explicitly identified. When referring to specific entities (e.g., people, organizations, locations), their full names or affiliations must be specified as clearly as possible (e.g., use “U.S. Congress” instead of “Congress”).
- Information density: Keypoints must be information-rich and specific. Avoid using common or vague descriptions.
- The expression of positions must remain neutral and objective, avoiding unnecessary derivative interpretations, metaphors, etc. (e.g., adding phrases such as “this indicates...” without reference to the original text). It is strictly prohibited to add any inferences, subjective analyses, or external background knowledge not mentioned in the original text. Keypoints must be objective facts, free of any emotional bias or prejudice.

## Example  
“On July 1, 2025, the Bank of Japan's Policy Board announced its decision to keep the short-term interest rate target at -0.1% and the 10-year bond yield target around 0%.”  ,
“The Bank of Japan attributed its continuation of an ultra-loose monetary policy to uncertainties in the global economy and slow domestic wage growth.” ,
“The BOJ raised its inflation forecast for fiscal year 2025 to 1.9%, an increase from the previous forecast of 1.8%, citing higher import costs.”  ,
“Bank of Japan Governor Kazuo Ueda stated that the bank is prepared to implement additional stimulus measures if there is a decline in economic momentum.”

## Input Text
{Subtitle}