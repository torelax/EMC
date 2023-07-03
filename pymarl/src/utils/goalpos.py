

def getvectorofgoal(goal, ranges):
    res = []
    for i in range(ranges * 2):
        res.append((-ranges, -ranges + i))
    for i in range(ranges * 2):
        res.append((-ranges + i, ranges))
    for i in range(ranges * 2):
        res.append((ranges, ranges - i))
    for i in range(ranges * 2):
        res.append((ranges - i, -ranges))
    return res[goal]

def getgoalPos(obs, goal, ranges):
    (vect_x, vect_y) = getvectorofgoal(goal, ranges)
    return [vect_x + obs[0], vect_y + obs[1]]