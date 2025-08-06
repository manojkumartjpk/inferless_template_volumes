INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["There is a fine house in the forest"]
    },
    'negative_prompt': {
        'datatype': 'STRING',
        'required': False,
        'example': ["test"],
        'shape': [1]
    },
    'width': {
        'datatype': 'INT8',
        'required': False,
        'example': [ 768 ,  512 ],
        'shape': [2]
    },
}
