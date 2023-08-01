def my_function(param1, param2, param3):
    # Your function code here
    print(f"param1: {param1}, param2: {param2}, param3: {param3}")

# Using a dictionary to pass key-value pairs as arguments
params_dict = {
    'param1': 10,
    'param2': 'Hello',
    'param3': [1, 2, 3]
}

my_function(**params_dict)