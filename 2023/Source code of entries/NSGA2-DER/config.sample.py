# 这个文件在每台电脑/服务器上只需要运行成功一次后面就可以不写了
# 复制这个文件一份并命名为config.py，运行一次

from evoxbench.database.init import config

# make sure you update these two paths accordingly, and the first path should be for database file
config(
    "/opt/data/private/BigFiles/EvoXBench_tutorial_materials_IEEE CEC’2023 Competition on Multiobjective Neural Architecture Search/database",
    "/opt/data/private/BigFiles/EvoXBench_tutorial_materials_IEEE CEC’2023 Competition on Multiobjective Neural Architecture Search/data")
