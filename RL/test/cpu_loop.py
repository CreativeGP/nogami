import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process


def worker(i):
    while True:
        pass

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--num-process", type=int, default=1000)
    args = argparse.parse_args()

    processes = []
    for i in range(args.num_process):
        process = Process(target=worker, args=(i,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
