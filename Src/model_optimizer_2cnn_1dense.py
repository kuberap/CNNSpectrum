
def compute_flatten_size(cnn_config, input_length=1024, verbose=True):
    if verbose:
        print(f"input length:{input_length}")
    for i,conf in enumerate(cnn_config):
       in_channels=conf["in_channels"]
       out_channels=conf["out_channels"]
       kernel_size=conf["kernel_size"]
       pool_size=conf["pool_size"]
       input_length=input_length-(kernel_size//2)*2
       if verbose:
            print(f"cnn1 - block:{i}: size:{input_length} x units:{out_channels}")
       input_length = input_length - (kernel_size // 2)*2
       if verbose:
            print(f"cnn2- block:{i}: size:{input_length} x units:{out_channels}")
       input_length//=2
       if verbose:
            print(f"final- block:{i}: size:{input_length} x units:{out_channels}")
    if verbose:
        print(f"Flatten units:{input_length*out_channels}")
    return input_length*out_channels




def train_method(config, data_dir=None):
    CNN_CONF = [
        {"in_channels": 1,
         "out_channels": config["cnn1-out"],
         "kernel_size": config["cnn1-kernel"],
         "pool_size": 2, # division by 2 is universal
         "dropout": config["cnn1-dropout"]
         },
        {"in_channels": config["cnn1-out"],  # must be the same is output from the layer one
         "out_channels": config["cnn2-out"],
         "kernel_size": config["cnn2-kernel"],
         "pool_size": 2, # division by 2 is universal
         "dropout": config["cnn2-dropout"]
         }
    ]
    flatten_size = compute_flatten_size(CNN_CONF)
    DENSE_CONF = [
        {"in": flatten_size,
         "out": 512,
         "activation": True,
         "dropout": 0.01},  # 0.01
        {
            "in": 512,
            "out": 2,
            "activation": False,
            "dropout": None}
    ]



def optimize():

    config={
        "cnn1-out":64,
        "cnn1-kernel":5,
        "cnn1-dropout":0.01,
        "cnn2-out":32,
        "cnn2-kernel":3,
        "cnn2-dropout": 0.01 }

    train_method(config)







if __name__=="__main__":
    optimize()







