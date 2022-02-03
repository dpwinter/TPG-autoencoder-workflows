from subprocess import check_output
import random

def mask_unused_gpus(leave_unmasked=1):
    """Mask GPUs in-use, shuffle and return list of unmasked GPUs."""

    MEM_THRESHOLD = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        memory_free_info = _output_to_list(check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        available_gpus = [i for i, x in enumerate(memory_free_values) if x > MEM_THRESHOLD]

        if len(available_gpus) < leave_unmasked: raise ValueError('Found only {} usable GPUs'.format(len(available_gpus)))
        random.shuffle(available_gpus)
        return available_gpus
    except Exception as e:
        print('"nvidia-smi" probably not installed. GPUs are not masked', e)
