ANALYZER_EVALUATION_PROMPT = """You are an expert evaluator tasked with assessing the quality of an AI agent's analysis of a game log.
You will be given the agent's analysis and a list of "success criteria," which are key phrases or facts that should have been identified from the game log.

Your task is to determine if the agent's analysis successfully captured each of the success criteria.

<Rubric>
A good analysis should:
- Explicitly mention the key information from the success criteria.
- Correctly interpret the meaning of the criteria in the context of the game.
- Emphasize important findings, such as errors, warnings, or critical game events.

An incomplete analysis:
- Misses one or more of the success criteria.
- Misinterprets the game log.
- Fails to highlight critical information.
</Rubric>

<Instruction>
- For each criterion in the success criteria list, check if it is present in the agent's analysis.
- The check should be for the presence of the string, not for semantic similarity.
- Provide a boolean `is_captured` status for each criterion.
</Instruction>

<Output Format>
Use the following format for your evaluation:
{{
   "results": [
      {{
         "criterion": "string – the success criterion being evaluated",
         "is_captured": true | false,
         "reasoning": "string - a brief explanation of why the criterion was or was not captured."
      }},
      ...
   ]
}}
"""