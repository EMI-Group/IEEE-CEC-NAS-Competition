from competition.C10_MOP1 import C10_MOP1
from competition.C10_MOP2 import C10_MOP2
from competition.C10_MOP3 import C10_MOP3
from competition.C10_MOP4 import C10_MOP4
from competition.C10_MOP5 import C10_MOP5
from competition.C10_MOP6 import C10_MOP6
from competition.C10_MOP7 import C10_MOP7
from competition.C10_MOP8 import C10_MOP8
from competition.C10_MOP9 import C10_MOP9
from competition.IN1K_MOP1 import IN1K_MOP1
from competition.IN1K_MOP2 import IN1K_MOP2
from competition.IN1K_MOP3 import IN1K_MOP3
from competition.IN1K_MOP4 import IN1K_MOP4
from competition.IN1K_MOP5 import IN1K_MOP5
from competition.IN1K_MOP6 import IN1K_MOP6
from competition.IN1K_MOP7 import IN1K_MOP7
from competition.IN1K_MOP8 import IN1K_MOP8
from competition.IN1K_MOP9 import IN1K_MOP9

problem_list = [
    C10_MOP1(),
    C10_MOP2(),
    C10_MOP3(),
    C10_MOP4(),
    C10_MOP5(),
    C10_MOP6(),
    C10_MOP7(),
    C10_MOP8(),
    C10_MOP9(),
    IN1K_MOP1(),
    IN1K_MOP2(),
    IN1K_MOP3(),
    IN1K_MOP4(),
    IN1K_MOP5(),
    IN1K_MOP6(),
    IN1K_MOP7(),
    IN1K_MOP8(),
    IN1K_MOP9(),
]


def main():
    from run_file import main, method_list

    for problem in problem_list:
        for method in method_list:
            main(31, problem, method)


def main1():
    from run_file import main as run, method_list
    import multiprocessing
    tasks = [(31, i, j) for i in problem_list for j in method_list]
    process_pool = []
    for args in tasks:
        process = multiprocessing.Process(target=run, args=args)
        process.start()
        process_pool.append(process)
    for process in process_pool:
        process.join()


def main2():
    import argparse
    from run_file import main, method_list
    praser = argparse.ArgumentParser()
    praser.add_argument('num', type=int, choices=range(18))
    praser.add_argument('method', type=str, choices=method_list)
    args = praser.parse_args()
    main(31, problem_list[args.num], args.method)


def main3():
    import argparse
    from run_file import main, method_list
    praser = argparse.ArgumentParser()
    praser.add_argument('num', type=int, choices=range(18))
    args = praser.parse_args()
    for method in method_list:
        main(31, problem_list[args.num], method)


def main4():
    import argparse
    from run_file import main, method_list
    praser = argparse.ArgumentParser()
    praser.add_argument('method', type=str, choices=method_list)
    args = praser.parse_args()
    for i in range(18):
        main(31, problem_list[i], args.method)


if __name__ == '__main__':
    main2()
