import torch

def test_num_gpus(num_gpu):
    assert(torch.cuda.device_count() == num_gpu) 


if __name__ == "__main__":
    test_num_gpus(5)
    
    print("Everything passed.")