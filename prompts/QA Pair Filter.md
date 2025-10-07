# QA Pair Review Agreement
You will receive some multi-hop QA pairs (from multiple key points) that share a common theme. Remove QA pairs with the following two defects and directly output the QA pairs that pass the check:
1. When selecting multiple key points for a QA pair (multi-hop), only some of the key points are needed to answer the question, resulting in invalid key points (these key points may serve as the problem context, etc.).
Method:
Step 1. Check the answer.
Step 2. Carefully review each keypoint provided in the “Keypoints” field.  
Step 3. Attempt to reconstruct the complete answer using only a subset of the provided keypoints.  
Step 4. Determination: If the answer can be fully and accurately constructed from a subset of keypoints, this indicates the presence of an invalid keypoint (i.e., a keypoint used solely for question context rather than for the answer itself). A QA is only approved through this review if its answer requires information from all listed source key points to be complete. If, after removing the invalid key point, the keypoints still include both video and text items, output the corrected QA; otherwise, delete this QA.  
2. QA that is completely identical to a previous QA.  
3. QA unrelated to the topic: Check all QA to determine their topics; QA that does not align with the topic is defective.
4. The questions are not like those asked by humans: they are either too detailed (e.g., what was written on the notice board of a nearby shop when a certain event occurred?) or too mechanical (e.g., combining two unrelated questions to use multiple key points), which does not conform to human questioning habits and areas of interest.
Directly output a JSON list in the same format as the input, without any additional comments.
Output Format Example:
[
    {{
        "question": "Following the live flight demonstration of the 'Aero-X' drone, what is its official retail price and what key feature justifies this cost?",
        "answer": "The 'Aero-X' drone will have an official retail price of $1,499. This price is justified by its proprietary 'Stable-Flight' AI navigation system, which allows it to maintain stability in winds up to 40 mph.",
        "keypoints": {{
            "video": [
                "The 'Aero-X' drone is shown performing complex maneuvers in a windy outdoor environment during a live demonstration."
            ],
            "text": [
                "The 'Aero-X' will retail for $1,499, and its premium price point is due to the new 'Stable-Flight' AI navigation system, rated for winds up to 40 mph."
            ]
        }}
    }},
]
### Test

Input:

multi-hop QA:
{json.dumps(qa_list, ensure_ascii=False, indent=2)}