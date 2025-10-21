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
    "technical_competence": [
        "The assistant seemed highly competent at the task.",
        "Working with the assistant was more efficient than using a web browser to solve the task myself.",
        "It took the assistant multiple tries before executing the task correctly.",
    ],
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
