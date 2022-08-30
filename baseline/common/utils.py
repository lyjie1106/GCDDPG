import sys


# convert str to boolean
def str_to_bool(str):
    if str == '':
        return None
    elif str == 'True':
        return True
    elif str == 'False':
        return False
    else:
        print('Error in boolean config')
        sys.exit()


# convert str to int
def str_to_int(str):
    if str == '':
        return None
    try:
        r = int(str)
    except BaseException:
        print('Error in int config')
        sys.exit()
    else:
        return r


# convert str to float
def str_to_float(str):
    if str == '':
        return None
    try:
        r = float(str)
    except BaseException:
        print('Error in float config')
        sys.exit()
    else:
        return r
