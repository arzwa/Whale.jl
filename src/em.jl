# Expectation - Maximization algorithm for (WH)ALE (only DL model probably)
# Arthur Zwaenepoel (2019)
# NOTE: Use the BirthDeathProcesses.jl library as much as possible!
# XXX: This won't work with the GBM prior I guess? 


# 1. Count transitions in backtracked rectrees

# 2. Do EM update in postorder, always recompute ϵ
