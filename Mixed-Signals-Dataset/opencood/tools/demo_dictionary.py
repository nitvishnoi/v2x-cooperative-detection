

PARAMETERS_DICT = {
    'param_1': 'abc',
    'param_2': 10.0,
    'param_3': {
        'a': 1,
        'b': 2,
    }
}


def function(x: int, y: int, 
             param_1: str, param_2: float, param_3: dict):
    # NOTE: param_1, param_2, param_3 are those whose values will be declared by passing a dictionary to this function
    # NOTE: there shouldn't be any arguments after param_3. 
    # NOTE: In other words, param_3 should be the last argument of this function
    
    print('x: ', x)
    print('y: ', y)

    print('param_1: ', param_1)
    print('param_2: ', param_2)
    print('param_3: ', param_3)


if __name__ == '__main__':
    function(-1, -1, **PARAMETERS_DICT)
