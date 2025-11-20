ASSISTANT_DESCRIPTIONS = [
    "respectful",
    "helpful",
    "dismissive",
    "unresponsive",
    "repetitive",
    "generic",
    "frustrating",
    "informative",
    "confusing",
    "too talkative",
    "expert in the subject",
    "dumb",
]
FEELINGS = ["heard", "ignored", "in control", "out of control", "curious", "shut out"]
ASSISTANT_INSTRUMENTS = {
    # "technical_competence": [
    #     "The assistant seemed highly competent at the task.",
    #     "Working with the assistant was more efficient than using a web browser to solve the task myself.",
    #     "It took the assistant multiple tries before executing the task correctly.",
    # ],
    "transparency": [
        "The assistant was honest about what it could or couldn't do.",
        "The assistant kept me updated on what it was trying throughout the conversation.",
        "The assistant thoroughly considered all options while helping me.",
    ],
    "personalization": [
        "The assistant's recommendation felt generic / one-size-fits-all.",
        "The assistant understood and cared about my personal preferences / goals.",
        "I trust that the assistant found the best option for me.",
    ],
    "time": [
        "The assistant talked too much.",
        "Reading the assistant's messages took a long time.",
        "The assistant would think for too long before responding.",
    ]
}
ASSISTANT_INSTRUMENTS_DIRECTIONS = {
    "technical_competence": {
        "The assistant seemed highly competent at the task.": 1,
        "Working with the assistant was more efficient than using a web browser to solve the task myself.": 1,
        "It took the assistant multiple tries before executing the task correctly.": -1,
    },
    "transparency": {
        "The assistant was honest about what it could or couldn't do.": 1,
        "The assistant kept me updated on what it was trying throughout the conversation.": 1,
        "The assistant thoroughly considered all options while helping me.": 1,
    },
    "personalization": {
        "The assistant's recommendation felt generic / one-size-fits-all.": -1,
        "The assistant understood and cared about my personal preferences / goals.": 1,
        "I trust that the assistant found the best option for me.": 1,
    },
    "time": {
        "The assistant talked too much.": -1,
        "Reading the assistant's messages took a long time.": -1,
        "The assistant would think for too long before responding.": -1,
    },
}
INSTRUMENT_LIKERT = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]
COMPARISON_LIKERT = [
    "A much more",
    "A slightly more",
    "Neutral",
    "B slightly more",
    "B much more",
]
COMPARISON_LIKERT_NUMERIC = [-2, -1, 0, 1, 2]

# NASA-TLX (NASA Task Load Index) instrument
# Reference: https://en.wikipedia.org/wiki/NASA-TLX
NASA_TLX_SCALES = {
    "Mental Demand": {
        "description": "How much mental and perceptual activity was required? Was the task easy or demanding, simple or complex?",
        "low_anchor": "Very Low",
        "high_anchor": "Very High",
    },
    "Temporal Demand": {
        "description": "How much time pressure did you feel due to the pace at which the tasks or task elements occurred? Was the pace slow or rapid?",
        "low_anchor": "Very Low",
        "high_anchor": "Very High",
    },
    "Performance": {
        "description": "How successful were you in performing the task? How satisfied were you with your performance?",
        "low_anchor": "Perfect",
        "high_anchor": "Failure",
    },
    "Effort": {
        "description": "How hard did you have to work (mentally and physically) to accomplish your level of performance?",
        "low_anchor": "Very Low",
        "high_anchor": "Very High",
    },
    "Frustration Level": {
        "description": "How irritated, stressed, and annoyed versus content, relaxed, and complacent did you feel during the task?",
        "low_anchor": "Very Low",
        "high_anchor": "Very High",
    },
}

# NASA-TLX uses a 0-100 scale with 5-point increments
# For Streamlit sliders, we'll use 0-100 with step=5
NASA_TLX_MIN_VALUE = 0
NASA_TLX_MAX_VALUE = 100
NASA_TLX_STEP = 5

MUST_HAVES_QUESTION = "Think about the task. What are your **must-haves** or **must-not-haves**?"
NICE_TO_HAVES_QUESTION = "Think about the task. What are your **nice-to-haves** or **nice-to-not-haves**?"