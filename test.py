"""Problem available [here](https://projecteuler.net/problem=737)

thetas contains the angles at which the coins are placed, from last to first coin (coins are placed from top to bottom)

centersOfMass[k] contains the center of mass of the stack of coins from coin n (last) to coin n - k - 1
"""
import numpy as np

thetas = []
centersOfMass = []

# place the first 2 coins
thetas.extend([0, - np.pi / 3])
centersOfMass.append( (1 + np.exp(- 1j * np.pi / 3)) / 2 )

# the total variation in angle we want
tgtAngle = 10 * 2 * np.pi

# place recursively all coins until we reach the desired angle for the stack of coins
stackAngle = thetas[0] - thetas[-1]
while stackAngle < tgtAngle:
    # write the center of mass in exponential form
    r = np.abs(centersOfMass[-1])
    alpha = np.angle(centersOfMass[-1])

    # place a new coin below the current stack
    theta = alpha - np.arccos(r / 2)

    # put theta in the correct interval (so that measuring the stack angle is easy)
    p = - np.floor( (theta - thetas[-1]) / (2 * np.pi) ) - 1
    theta += p * 2 * np.pi

    # add the coin
    thetas.append(theta)

    # compute the new center of mass
    centerOfMass = centersOfMass[-1] + (np.exp(1j * theta) - centersOfMass[-1]) / len(thetas)
    centersOfMass.append(centerOfMass)

    # update the stack angle
    stackAngle = thetas[0] - thetas[-1]

nb_coins = len(thetas)
print(f"{nb_coins} coins needed to exceed an angle of {tgtAngle}")
