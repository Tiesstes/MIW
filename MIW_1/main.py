import numpy as np
import matplotlib.pyplot as pypl




def RSP():

    ai_y = []
    player_y = []
    x = []

    ai_cash = 0
    player_cash = 0

    # possible states
    start = ['rock', 'scissors', 'paper']

    print('Select your move by entering: rock, scissors or paper')

    player_move = str(input())



    # starting probability
    p_start = [0.33, 0.33, 0.34]

    # transformation matrix
    t_matrix = ['rock', 'scissors', 'paper']
    pt_matrix = [[0.33, 0.33, 0.34], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]]

    #number of games
    n = 25

    initial_move = np.random.choice(start, replace=True, p=p_start)
    state = initial_move

    for i in range(1, n+1):

        if state == 'rock' and player_move == 'rock' :

            ai_y.append(ai_cash)
            player_y.append(player_cash)
            x.append(i)

            print('AI chose:', state, 'you chose:', player_move)


            pt_matrix[0][0] = pt_matrix[0][0] - 0.025
            if pt_matrix[0][0] < 0.0:
                pt_matrix[0][0] = 0.0
            elif pt_matrix[0][0] > 1.0:
                pt_matrix[0][0] = 1.0

            pt_matrix[0][1] = pt_matrix[0][1] + 0.050
            if pt_matrix[0][1] < 0.0:
                pt_matrix[0][1] = 0.0
            elif pt_matrix[0][1] > 1.0:
                pt_matrix[0][1] = 1.0

            pt_matrix[0][2] = pt_matrix[0][2] - 0.025
            if pt_matrix[0][2] < 0.0:
                pt_matrix[0][2] = 0.0
            elif pt_matrix[0][2] > 1.0:
                pt_matrix[0][2] = 1.0

            if i == n:
                print('AI cash', ai_cash)
                print('Your cash', player_cash)
                pypl.plot(x, player_y, 'g', label='Player')
                pypl.plot(x, ai_y, 'r', label='AI')
                pypl.show()
                break

            state = np.random.choice(t_matrix, p=pt_matrix[0])

            print('Select your next move by entering: rock, paper or scissors')
            player_move = str(input())


        elif state == 'paper' and player_move == 'paper' :

            ai_y.append(ai_cash)
            player_y.append(player_cash)
            x.append(i)

            print('AI chose:', state, 'you chose:', player_move)

            pt_matrix[1][0] = pt_matrix[1][0] - 0.025
            if pt_matrix[1][0] < 0.0:
                pt_matrix[1][0] = 0.0
            elif pt_matrix[1][0] > 1.0:
                pt_matrix[1][0] = 1.0

            pt_matrix[1][1] = pt_matrix[1][1] - 0.025
            if pt_matrix[1][1] < 0.0:
                pt_matrix[1][1] = 0.0
            elif pt_matrix[1][1] > 1.0:
                pt_matrix[1][1] = 1.0

            pt_matrix[1][2] = pt_matrix[1][2] + 0.050
            if pt_matrix[1][2] < 0.0:
                pt_matrix[1][2] = 0.0
            elif pt_matrix[1][2] > 1.0:
                pt_matrix[1][2] = 1.0

            if i == n:
                print('AI cash', ai_cash)
                print('Your cash', player_cash)
                pypl.plot(x, player_y, 'g', label='Player')
                pypl.plot(x, ai_y, 'r', label='AI')
                pypl.show()
                break


            state = np.random.choice(t_matrix, p=pt_matrix[1])

            print('Select your next move by entering: rock, paper or scissors')
            player_move = str(input())


        elif state == 'scissors' and player_move == 'scissors':

            ai_y.append(ai_cash)
            player_y.append(player_cash)
            x.append(i)

            print('AI chose:', state, 'you chose:', player_move)

            pt_matrix[2][0] = pt_matrix[2][0] + 0.050
            if pt_matrix[2][0] < 0.0:
                pt_matrix[2][0] = 0.0
            elif pt_matrix[2][0] > 1.0:
                pt_matrix[2][0] = 1.0

            pt_matrix[2][1] = pt_matrix[2][1] - 0.025
            if pt_matrix[2][1] < 0.0:
                pt_matrix[2][1] = 0.0
            elif pt_matrix[2][1] > 1.0:
                pt_matrix[2][1] = 1.0

            pt_matrix[2][2] = pt_matrix[2][2] - 0.025
            if pt_matrix[2][2] < 0.0:
                pt_matrix[2][2] = 0.0
            elif pt_matrix[2][2] > 1.0:
                pt_matrix[2][2] = 1.0

            if i == n:
                print('AI cash', ai_cash)
                print('Your cash', player_cash)
                pypl.plot(x, player_y, 'g', label='Player')
                pypl.plot(x, ai_y, 'r', label='AI')
                pypl.show()
                break

            state = np.random.choice(t_matrix, p=pt_matrix[2])

            print('Select your next move by entering: rock, paper or scissors')
            player_move = str(input())


        elif state == 'rock' and player_move == 'paper':
            ai_cash -= 1
            player_cash += 1

            ai_y.append(ai_cash)
            player_y.append(player_cash)
            x.append(i)



            print('AI chose:', state, 'you chose:', player_move)

            pt_matrix[1][0] = pt_matrix[1][0] - 0.025
            if pt_matrix[1][0] < 0.0:
                pt_matrix[1][0] = 0.0
            elif pt_matrix[1][0] > 1.0:
                pt_matrix[1][0] = 1.0

            pt_matrix[1][1] = pt_matrix[1][1] - 0.025
            if pt_matrix[1][1] < 0.0:
                pt_matrix[1][1] = 0.0
            elif pt_matrix[1][1] > 1.0:
                pt_matrix[1][1] = 1.0

            pt_matrix[1][2] = pt_matrix[1][2] + 0.050
            if pt_matrix[1][2] < 0.0:
                pt_matrix[1][2] = 0.0
            elif pt_matrix[1][2] > 1.0:
                pt_matrix[1][2] = 1.0

            if i == n:
                print('AI cash', ai_cash)
                print('Your cash', player_cash)
                pypl.plot(x, player_y, 'g', label='Player')
                pypl.plot(x, ai_y, 'r', label='AI')
                pypl.show()
                break

            state = np.random.choice(t_matrix, p=pt_matrix[1])

            print('Select your next move by entering: rock, paper or scissors')
            player_move = str(input())


        elif state == 'rock' and player_move == 'scissors' :
            ai_cash += 1
            player_cash -= 1

            ai_y.append(ai_cash)
            player_y.append(player_cash)
            x.append(i)



            print('AI chose:', state, 'you chose:', player_move)

            if i == n:
                print('AI cash', ai_cash)
                print('Your cash', player_cash)
                pypl.plot(x, player_y, 'g', label='Player')
                pypl.plot(x, ai_y, 'r', label='AI')
                pypl.show()
                break

            state = np.random.choice(t_matrix, p=pt_matrix[2])

            print('Select your next move by entering: rock, paper or scissors')
            player_move = str(input())


        elif state == 'paper' and player_move == 'rock' :
            ai_cash += 1
            player_cash -= 1

            ai_y.append(ai_cash)
            player_y.append(player_cash)
            x.append(i)



            print('AI chose:', state, 'you chose:', player_move)

            if i == n:
                print('AI cash', ai_cash)
                print('Your cash', player_cash)
                pypl.plot(x, player_y, 'g', label='Player')
                pypl.plot(x, ai_y, 'r', label='AI')
                pypl.show()
                break

            state = np.random.choice(t_matrix, p=pt_matrix[1])

            print('Select your next move by entering: rock, paper or scissors')
            player_move = str(input())


        elif state == 'paper' and player_move == 'scissors' :
            ai_cash -= 1
            player_cash += 1

            ai_y.append(ai_cash)
            player_y.append(player_cash)
            x.append(i)

            print('AI chose:', state, 'you chose:', player_move)

            pt_matrix[2][0] = pt_matrix[2][0] + 0.050
            if pt_matrix[2][0] < 0.0:
                pt_matrix[2][0] = 0.0
            elif pt_matrix[2][0] > 1.0:
                pt_matrix[2][0] = 1.0

            pt_matrix[2][1] = pt_matrix[2][1] - 0.025
            if pt_matrix[2][1] < 0.0:
                pt_matrix[2][1] = 0.0
            elif pt_matrix[2][1] > 1.0:
                pt_matrix[2][1] = 1.0

            pt_matrix[2][2] = pt_matrix[2][2] - 0.025
            if pt_matrix[2][2] < 0.0:
                pt_matrix[2][2] = 0.0
            elif pt_matrix[2][2] > 1.0:
                pt_matrix[2][2] = 1.0

            if i == n:
                print('AI cash', ai_cash)
                print('Your cash', player_cash)
                pypl.plot(x, player_y, 'g', label='Player')
                pypl.plot(x, ai_y, 'r', label='AI')
                pypl.show()
                break

            state = np.random.choice(t_matrix, p=pt_matrix[2])

            print('Select your next move by entering: rock, paper or scissors')
            player_move = str(input())
          #  isinstance(player_move, str)

        elif state == 'scissors' and player_move == 'rock' :

            ai_cash -= 1
            player_cash += 1

            ai_y.append(ai_cash)
            player_y.append(player_cash)
            x.append(i)

            print('AI chose:', state, 'you chose:', player_move)

            pt_matrix[0][0] = pt_matrix[0][0] - 0.025
            if pt_matrix[0][0] < 0.0:
                pt_matrix[0][0] = 0.0
            elif pt_matrix[0][0] > 1.0:
                pt_matrix[0][0] = 1.0

            pt_matrix[0][1] = pt_matrix[0][1] + 0.050
            if pt_matrix[0][1] < 0.0:
                pt_matrix[0][1] = 0.0
            elif pt_matrix[0][1] > 1.0:
                pt_matrix[0][1] = 1.0

            pt_matrix[0][2] = pt_matrix[0][2] - 0.025
            if pt_matrix[0][2] < 0.0:
                pt_matrix[0][2] = 0.0
            elif pt_matrix[0][2] > 1.0:
                pt_matrix[0][2] = 1.0

            if i == n:
                print('AI cash', ai_cash)
                print('Your cash', player_cash)
                pypl.plot(x, player_y, 'g', label='Player')
                pypl.plot(x, ai_y, 'r', label='AI')
                pypl.show()
                break

            state = np.random.choice(t_matrix, p=pt_matrix[0])

            print('Select your next move by entering: rock, paper or scissors')
            player_move = str(input())


        elif state == 'scissors' and player_move == 'paper' :

            ai_cash += 1
            player_cash -= 1

            ai_y.append(ai_cash)
            player_y.append(player_cash)
            x.append(i)

            print('AI chose:', state, 'you chose:', player_move)

            if i == n:
                print('AI cash', ai_cash)
                print('Your cash', player_cash)
                pypl.plot(x, player_y, 'g', label='Player')
                pypl.plot(x, ai_y, 'r', label='AI')
                pypl.show()
                break

            state = np.random.choice(t_matrix, p=pt_matrix[1])

            print('Select your next move by entering: rock, paper or scissors')
            player_move = str(input())



if __name__ == '__main__':
    RSP()
    exit(0)
