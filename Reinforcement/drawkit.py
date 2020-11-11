import numpy as np

def show_v_table_small(v_table, env):
    for i in range(env.rewards.shape[0]):        
        print("+----------"*env.rewards.shape[1])
        print("|", end="")
        for j in range(env.rewards.shape[1]):
            print("{0:8.2f}  |".format(v_table[i,j]),end="")
        print()
    print("+----------"*env.rewards.shape[1])

# V table 그리기    
def show_v_table(v_table, env):    
    for i in range(env.rewards.shape[0]):        
        print("+-----------------"*env.rewards.shape[1],end="")
        print("+")
        for k in range(3):
            print("|",end="")
            for j in range(env.rewards.shape[1]):
                if k==0:
                    print("                 |",end="")
                if k==1:
                        print("   {0:8.2f}      |".format(v_table[i,j]),end="")
                if k==2:
                    print("                 |",end="")
            print()
    print("+-----------------"*env.rewards.shape[1],end="")
    print("+")
    
# q_table table 그리기
def show_q_table(q_table, env):
    for i in range(env.rewards.shape[0]):
        print("+-----------------"*env.rewards.shape[1],end="")
        print("+")
        for k in range(3):
            print("|",end="")
            for j in range(env.rewards.shape[1]):
                if k==0:
                    print("{0:10.2f}       |".format(q_table[i,j,0]),end="")
                if k==1:
                    print("{0:6.2f}    {1:6.2f} |".format(q_table[i,j,3],q_table[i,j,1]),end="")
                if k==2:
                    print("{0:10.2f}       |".format(q_table[i,j,2]),end="")
            print()
    print("+-----------------"*env.rewards.shape[1],end="")
    print("+")
    

# 정책 policy 화살표로 그리기
def show_q_table_arrow(q_table, env):
    for i in range(env.rewards.shape[0]):        
        print("+-----------------"*env.rewards.shape[1],end="")
        print("+")
        for k in range(3):
            print("|",end="")
            for j in range(env.rewards.shape[1]):
                if k==0:
                    if np.max(q_table[i,j,:]) == q_table[i,j,0]:
                        print("        ↑        |",end="")
                    else:
                        print("                 |",end="")
                if k==1:                    
                    if np.max(q_table[i,j,:]) == q_table[i,j,1] and np.max(q_table[i,j,:]) == q_table[i,j,3]:
                        print("      ←   →      |",end="")
                    elif np.max(q_table[i,j,:]) == q_table[i,j,1]:
                        print("          →      |",end="")
                    elif np.max(q_table[i,j,:]) == q_table[i,j,3]:
                        print("       ←         |",end="")
                    else:
                        print("                 |",end="")
                if k==2:
                    if np.max(q_table[i,j,:]) == q_table[i,j,2]:
                        print("        ↓        |",end="")
                    else:
                        print("                 |",end="")
            print()
    print("+-----------------"*env.rewards.shape[1],end="")
    print("+")    
    
# 정책 policy 화살표로 그리기
def show_policy_small(policy, env):
    for i in range(env.rewards.shape[0]):        
        print("+----------"*env.rewards.shape[1],end="")
        print("+")
        print("|", end="")
        for j in range(env.rewards.shape[1]):
            if env.reward_list_char[i][j] == "road":
                if policy[i,j] == 0:
                    print("   ↑     |",end="")
                elif policy[i,j] == 1:
                    print("   →     |",end="")
                elif policy[i,j] == 2:
                    print("   ↓     |",end="")
                elif policy[i,j] == 3:
                    print("   ←     |",end="")
            else:
                print("          |",end="")
        print()
    print("+----------"*env.rewards.shape[1],end="")
    print("+")
    
# 정책 policy 화살표로 그리기
def show_policy(policy, env):
    for i in range(env.rewards.shape[0]):        
        print("+-----------------"*env.rewards.shape[1],end="")
        print("+")
        for k in range(3):
            print("|",end="")
            for j in range(env.rewards.shape[1]):
                if k==0:
                    print("                 |",end="")
                if k==1:
                    if policy[i,j] == 0:
                        print("       ↑         |",end="")
                    elif policy[i,j] == 1:
                        print("       →         |",end="")
                    elif policy[i,j] == 2:
                        print("       ↓         |",end="")
                    elif policy[i,j] == 3:
                        print("       ←         |",end="")
                if k==2:
                    print("                 |",end="")
            print()
    print("+-----------------"*env.rewards.shape[1],end="")
    print("+")