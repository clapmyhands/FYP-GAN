import time
from torch.nn.init import xavier_uniform, constant, xavier_normal


def initializeWeight(module):
    """
    Initialize weight with Xavier uniform distribution to avoid vanishing gradient problem
    should be enough with this without batch norm since our network is not very deep

    :param module:
    :return:
    """
    classname = module.__class__.__name__
    if(classname.find('Conv') != -1):
        xavier_normal(module.weight)
        constant(module.bias, 0)
    elif(classname.find('Linear') != -1):
        xavier_normal(module.weight)
        constant(module.bias, 0)

## TODO: wrapper for logging time
def epoch_timer(func, *args, **kwargs):
    """
    wrapper to count time taken for function
    might have added logging function for epoch train/test (plotting?)

    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    start_time = time.time()
    #func(*args, **kwargs)
    end_time = time.time()
    # logger.info("".format(end_time-start_time))
    pass