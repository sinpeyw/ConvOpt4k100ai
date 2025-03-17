#!/bin/bash

# 基本参数
#SBATCH --job-name=job		# 作业名称
#SBATCH --output=output.txt		# 标准输出文件
#SBATCH --error=error.txt		# 错误输出文件

# 指定分区
#SBATCH --partition=wzidnormal		# 提交分区

# 节点、CPU资源和任务参数
#SBATCH --nodes=1			# 使用节点数目
#SBATCH --ntasks=1			# 启动任务（进程）个数
#SBATCH --ntasks-per-node=1		# 每个节点分配任务数
#SBATCH --cpus-per-task=16		# 每个任务使用cpu核心数量,用户最大限制是22，不能超过22。比赛测试应该不能大于8

# dcu资源
#SBATCH --gres=dcu:1			# dcu的个数
#SBATCH --nodelist=xdb2		# 指定dcu序号

# 内存资源
#SBATCH --mem=32G			# 每个节点分配的大小，最大是512G

# 作业时间限制
#SBATCH --time=1-00:00:00		# 作业最多运行时间

# 加载环境
module purge
module load compiler/dtk/24.04

# 编译程序
make clean
make gpu=1 

# 测试数据
./conv2dfp16demo 16 128 64 64 27 3 3 1 1 1 1
#./conv2dfp16demo 16 256 32 32 256 3 3 1 1 1 1
# ./conv2dfp16demo 16 64 128 128 64 3 3 1 1 1 1
# ./conv2dfp16demo 2 1920 32 32 640 3 3 1 1 1 1
# ./conv2dfp16demo 2 640 64 64 640 3 3 1 1 1 1
# ./conv2dfp16demo 2 320 64 64 4 3 3 1 1 1 1
