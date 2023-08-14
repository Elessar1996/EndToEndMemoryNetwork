

def find_greates_divisor(n:int):

    all_divisors = []

    for i in range(1, n):

        if n % i == 0:

            all_divisors.append(i)


    return max(all_divisors)

def find_smallest_divisor(n:int):

    all_divisors = []

    for i in range(1, n):

        if n % i == 0 and i != 1:

            all_divisors.append(i)


    return min(all_divisors)
